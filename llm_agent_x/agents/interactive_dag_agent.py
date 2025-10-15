# llm_agent_x/agents/interactive_dag_agent.py

import asyncio
import json
import logging
import os
import threading
import uuid
from collections import deque
from typing import List, Tuple, Union, Dict, Any, Optional, Callable

import pika
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from phoenix.trace.schemas import SpanAttributes
from pydantic import BaseModel
from pydantic_ai.agent import AgentRunResult, Agent, CallToolsNode, ModelRequestNode
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ToolCallPart, ToolReturnPart, ModelResponse, ModelResponsePart, TextPart, \
    SystemPromptPart, UserPromptPart
from pydantic_ai.result import FinalResult
from pydantic_graph import End

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from llm_agent_x.agents.dag_agent import (
    DAGAgent,
    Task,
    TaskRegistry,
    TaskContext,
    ExecutionPlan,
    ProposedSubtask,
    verification,
    RetryDecision,
    UserQuestion,
    ProposalResolutionPlan,
    AdaptiveDecomposerResponse, InformationNeedDecision, DependencySelection, PruningDecision,
    TaskForMerging, MergedTask, MergingDecision, NewSubtask, ContextualAnswer,
    RedundancyDecision,
    ChainedExecutionPlan, TaskChain, TaskDescription, DocumentState, generate_hash,
)

from dotenv import load_dotenv

from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Tracer, Span, StatusCode
from openinference.semconv.trace import SpanAttributes

from llm_agent_x.backend.extract_json_from_text import extract_json

load_dotenv(".env", override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("InteractiveDAGAgent")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")


class InteractiveDAGAgent(DAGAgent):
    def __init__(self, *args, **kwargs):
        # ... (init remains the same)
        self._init_args = args
        self._init_kwargs = kwargs
        kwargs.pop("min_question_priority", None)

        self.max_total_tasks = kwargs.pop("max_total_tasks", 25)
        self.max_dependencies_per_task = kwargs.pop("max_dependencies_per_task", 7)


        super().__init__(*args, **kwargs)

        self.directives_queue = asyncio.Queue()
        self._publisher_connection: Optional[pika.BlockingConnection] = None
        self._publish_channel: Optional[
            pika.adapters.blocking_connection.BlockingChannel
        ] = None
        self.STATE_UPDATES_EXCHANGE = "state_updates_exchange"
        self._consumer_connection_for_thread: Optional[pika.BlockingConnection] = None
        self._consumer_channel_for_thread: Optional[
            pika.adapters.blocking_connection.BlockingChannel
        ] = None
        self.DIRECTIVES_QUEUE = "directives_queue"
        self._shutdown_event = threading.Event()
        self._consumer_thread: Optional[threading.Thread] = None
        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.proposed_tasks_buffer = []
        self.proposed_task_dependencies_buffer = []
        self.update_notebook_tool = self._create_notebook_tool(
            self.registry, self._broadcast_state_update
        )
        self._setup_agent_roles()

    def _setup_agent_roles(self):
        """Initializes or reinitializes all agent roles with the current self._tools_for_agents."""
        llm_model = self.base_llm_model  # Use the stored model string

        planner_system_prompt = (
            "You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. "
            "You will be given the main objective and, crucially, a list of 'AVAILABLE DOCUMENTS' and their global IDs. These documents are your primary data sources."
            "\n\n**YOUR DECISION PROCESS MUST FOLLOW THESE RULES:**"
            "\n1. **CHECK DOCUMENTS FIRST:** Before creating any task, review the list of AVAILABLE DOCUMENTS. Ask yourself: 'Is the information needed for this step already present in one of these documents?'"
            "\n2. **DEPEND, DON'T RE-CREATE:** If a task's primary purpose is to get information from an existing document, you MUST NOT create a new task to 'read' or 'analyze' it. Instead, create a task that USES the information and add the document's global ID directly to its `deps` list."
            "\n3. **LINK SEQUENTIAL STEPS:** Structure your output as 'chains' of tasks. A 'chain' is a list of tasks that must be done sequentially. All chains will run in parallel."
            "\n\n**EXAMPLE:**"
            "\n- Objective: 'Create a summary of Q2 performance.'"
            "\n- AVAILABLE DOCUMENTS: `[{'id': 'doc-abc', 'desc': 'Document: Q2 Financials'}]`"
            "\n- **CORRECT ACTION:** Create a single task like `{'desc': 'Synthesize Q2 Financials into a summary', 'deps': ['doc-abc']}`."
            "\n- **INCORRECT ACTION:** Creating a task like `{'desc': 'Read the Q2 Financials document', 'deps': []}`. This is redundant."
        )

        self.initial_planner = Agent(
            model=llm_model,
            system_prompt=planner_system_prompt,  # Use the new, improved prompt
            output_type=ChainedExecutionPlan,
            tools=self._tools_for_agents,
        )
        self.cycle_breaker = Agent(
            model=llm_model,
            system_prompt=(
                "You are a logical validation expert. Your task is to analyze an execution plan and resolve any circular dependencies (cycles). "
                "A cycle is when Task A depends on B, and Task B depends on A (directly or indirectly). "
                "If you find a cycle, you must remove the *least critical* dependency to break it. Use the 'reason' field for each dependency to decide. "
                "Your final output MUST be the complete, corrected ExecutionPlan. If there are no cycles, return the original plan unchanged."
            ),
            output_type=ExecutionPlan,
            tools=self._tools_for_agents,  # Use the internal tools list
        )

        self.task_merger = Agent(
            model=llm_model,
            system_prompt=(
                "You are an efficiency expert. Your SOLE job is to merge an ordered list of task descriptions "
                "into a single, coherent, and actionable task description that encapsulates the entire sequence. "
                "For example, if you receive ['Research venues', 'Contact venues', 'Book venue'], you should output "
                "something like 'Research, contact, and book a suitable conference venue based on requirements.' "
                "Your output MUST be only the final, merged task description as a single string."
            ),
            output_type=str,  # MODIFIED to output a simple string
            tools=[],
        )

        self.redundancy_checker = Agent(
            model=llm_model,
            system_prompt=(
                "You are a meticulous de-duplication expert. Your job is to check if a list of 'Proposed Tasks' is redundant given the 'Existing Tasks' already in the project plan. "
                "A proposed task is redundant if its objective is already covered by an existing task that is completed, running, proposing, planning, or even cancelled (as it indicates the idea was already considered). "
                "Use semantic understanding: for example, a proposal to 'Finalize the marketing plan' is redundant if there's an existing task to 'Create the marketing plan'. "
                "Your SOLE output must be the list of proposed tasks that are genuinely new and not redundant."
            ),
            output_type=RedundancyDecision,
            tools=[],
        )

        self.adaptive_decomposer = Agent(
            model=llm_model,
            system_prompt=(
                "You are an adaptive expert. Analyze the given task. If it is too complex, break it down into one or more 'chains' of new, more granular sub-tasks. "
                "A 'chain' is a list of tasks that must be done sequentially. You can create multiple parallel chains for independent workstreams."
            ),
            output_type=ChainedExecutionPlan,  # MODIFIED
            tools=self._tools_for_agents,
        )

        # cycle_breaker, redundancy_checker, and conflict_resolver are no longer needed, but we will keep them for now.

        self.dependency_pruner = Agent(
            model=llm_model,
            system_prompt=(
                "You are a ruthless focus expert. A task is about to run but has too many input dependencies, making it difficult to focus. "
                "You will be given the main task's objective and a list of available dependencies. "
                f"Your SOLE job is to select the most critical dependencies that are absolutely essential, up to a maximum of {self.max_dependencies_per_task}. "
                "Return ONLY the IDs of the approved dependencies."
            ),
            output_type=DependencySelection,
            tools=[],
        )

        self.conflict_resolver = Agent(
            model=llm_model,
            system_prompt=f"You are a ruthless but fair project manager. You have been given a list of proposed tasks that exceeds the budget. Analyze the list and their importance scores. You MUST prune the list by removing the LEAST critical tasks until the total number of tasks is no more than {self.global_proposal_limit}. Return only the final, approved list of tasks.",
            output_type=ProposalResolutionPlan,
            tools=self._tools_for_agents,  # Use the internal tools list
            retries=3,
        )

        self.graph_pruner = Agent(
            model=llm_model,
            system_prompt=(
                "You are a strategic and prudent project manager responsible for keeping a project on budget and on track. "
                "The total number of tasks has exceeded the allowed limit. Your job is to carefully select tasks for removal to make space. "
                "You must provide a clear reason for each pruning decision. "
                "\n\nFollow this decision hierarchy strictly:"
                "\n1. First, identify tasks that are clearly redundant or have been made obsolete by the completion of other tasks."
                "\n2. Second, identify low-impact, 'nice-to-have' tasks that are not critical for the main objective."
                "\n3. **Critically, AVOID pruning foundational parent tasks that have many children or downstream dependents.** Pruning a parent task will cancel all its sub-tasks, which is highly destructive. Only do this as an absolute last resort if no other options exist."
                "\nYour goal is to surgically remove tasks with the minimum possible impact on the overall project."
            ),
            output_type=PruningDecision,
            tools=[],
        )

        self.executor_system_prompt = (
            "You are the executor. Your job is to run the given task and return the result. If this task has MCP servers associated with it, you must use them to run the task. If no MCP servers are associated with a task, you must use the base tools.",
        )

        self.executor = Agent(
            model=llm_model,
            system_prompt=self.executor_system_prompt,
            output_type=str,
            tools=self._tools_for_agents,  # Use the internal tools list
            mcp_servers=self.mcp_servers,  # Use the MCP servers list
        )
        self.verifier = Agent(
            model=llm_model,
            system_prompt="You are a meticulous quality assurance verifier. Your job is to check if the provided 'Candidate Result' accurately and completely addresses the 'Task', considering the history and user instructions. Output JSON matching the 'verification' schema. Be strict.",
            output_type=verification,
            tools=self._tools_for_agents,  # Use the internal tools list
        )
        self.retry_analyst = Agent(
            model=llm_model,
            system_prompt=(
                "You are a meticulous quality assurance analyst. Your ONLY job is to decide if another AUTONOMOUS attempt on a failing task is likely to succeed. "
                "Review the task and its verification score history (a score > 5 is a success). "
                "If scores are improving, recommend a retry with a concrete suggestion. "
                "If the task is stagnant or worsening, you MUST recommend giving up. "
                "You cannot ask for help or external information. Your decision is final for the autonomous phase."
            ),
            output_type=RetryDecision,
            tools=self._tools_for_agents,
        )

        self.information_gap_detector = Agent(
            model=llm_model,
            system_prompt=(
                "You are a root cause analyst. A task has failed repeatedly, and an autonomous retry has been ruled out. "
                "Your SOLE job is to determine if the failure is due to a critical, unresolvable lack of information. "
                "Is it IMPOSSIBLE for the agent to proceed without external human input? Be extremely critical. "
                "If the agent *could* make a reasonable assumption or use a hypothetical example, then new information is NOT needed."
            ),
            output_type=InformationNeedDecision,
            tools=[],
        )

        self.question_formulator = Agent(
            model=llm_model,
            system_prompt=(
                "You are an expert communicator. A task is blocked waiting for critical human input. "
                "Your SOLE purpose is to formulate the SINGLE most critical, concise, and clear question for the human operator to unblock the task. "
                "Review the entire task history and context to identify the one piece of missing information. "
                "Do not add any preamble. Just ask the question."
            ),
            output_type=UserQuestion,
            tools=[],
        )

        self.internal_context_researcher = Agent(
            model=llm_model,
            system_prompt=(
                "You are an expert internal researcher. Your SOLE job is to determine if the answer to a given 'Question' "
                "exists within the 'Provided Context' of previously completed tasks. "
                "You must not use external knowledge. If the answer is present, extract it. If not, state that it is missing."
            ),
            output_type=ContextualAnswer,
            tools=[],
        )

        self._agent_role_map = {
            "initial_planner": self.initial_planner,
            "cycle_breaker": self.cycle_breaker,
            "task_merger": self.task_merger,
            "redundancy_checker": self.redundancy_checker,
            "adaptive_decomposer": self.adaptive_decomposer,
            "conflict_resolver": self.conflict_resolver,
            "graph_pruner": self.graph_pruner,
            "dependency_pruner": self.dependency_pruner,
            "executor": self.executor,
            "verifier": self.verifier,
            "retry_analyst": self.retry_analyst,
            "information_gap_detector": self.information_gap_detector,
            "question_formulator": self.question_formulator,
            "internal_context_researcher": self.internal_context_researcher,
        }

    async def _run_stateless_agent(
            self, agent: Agent, user_prompt: str, ctx: TaskContext
    ) -> Union[BaseModel, str, None]:
        """
        Runs a "one-shot" stateless request for decision-making agents.
        This does NOT use the agent's history, preventing context overflow.
        Agent's output type had better be a Pydantic model, or who knows what happens?
        """
        t = ctx.task if ctx else None
        if t:
            logger.info(f"[{t.id}] Running stateless agent: {agent.__class__.__name__}")

        # Prepare the request parameters. Pydantic-AI needs the output schema defined as a tool.
        params = ModelRequestParameters()
        output_is_model = isinstance(agent.output_type, type) and issubclass(agent.output_type, BaseModel)

        # --- MODIFICATION START ---
        # Build a comprehensive instruction list
        all_instructions = list(agent._system_prompts)

        if output_is_model:
            # This part remains the same: it configures the API call for tool use.
            params.output_tools = [
                ToolDefinition(
                    name="output",
                    description="The structured output response.",
                    parameters_json_schema=agent.output_type.model_json_schema()
                )
            ]

            # NEW: Explicitly add the JSON schema to the prompt instructions for the LLM.
            # This is the crucial step to prevent field name hallucination.
            schema_instruction = (
                f"\n\n--- OUTPUT FORMAT ---\n"
                f"You MUST respond using the 'output' tool with a JSON object that strictly adheres to the following JSON Schema:\n"
                f"```json\n{json.dumps(agent.output_type.model_json_schema(), indent=2)}\n```"
            )
            all_instructions.append(schema_instruction)

        # The final instruction to use the tool remains.
        all_instructions.append("\n\nUse the `output` tool.")

        messages = [
            ModelRequest(parts=[
                UserPromptPart(user_prompt),
            # Use the newly constructed, comprehensive instructions.
            ], instructions="\n\n".join(all_instructions)),
        ]
        # --- MODIFICATION END ---


        try:
            # Make the direct, stateless API call
            resp = await model_request(
                agent.model,
                messages,
                model_request_parameters=params
            )

            # Manually add telemetry data since we are not using Agent.run()
            # We create a dummy object with a usage() method to satisfy _add_llm_data_to_span
            class DummyResult:
                def __init__(self, usage_data):
                    self._usage = usage_data

                def usage(self):
                    return self._usage
            if t:
                self._add_llm_data_to_span(t.span, DummyResult(resp.usage), t)

            # Parse the response
            if output_is_model:

                logger.info(f"One shot response: {resp.parts[0]}")

                # For structured output, parse the tool call from the response
                if hasattr(resp.parts[0], "tool_name"):
                    if resp.parts and resp.parts[0].tool_name == "output":
                        args_as_dict = resp.parts[0].args_as_dict()
                        output = agent.output_type(**args_as_dict)
                        logger.info(f"One shot response (pydantic-ified): {output}")
                        return output
                    else:
                        if t:
                            logger.error(f"[{t.id}] Stateless agent failed to produce structured output.")
                        else:
                            logger.error(f"Stateless agent failed to produce structured output.")
                        return None
                else:
                    # Attempt to get a JSON object out of response text, then convert to the specified pydantic model
                    # Response may not be pure JSON, may be wrapped in text
                    response_text = resp.parts[0].content
                    json_ = extract_json(response_text)
                    if json_:
                        output = agent.output_type(**json_)
                        logger.info(f"One shot response (json): {output}")
                        return output


            else:
                # For simple string output
                return resp.parts[0].content if resp.parts else None

        except Exception as e:
            logger.error(f"[{t.id}] Exception in _run_stateless_agent: {e}", exc_info=True)
            if t:
                t.span.record_exception(e)
            return None

    def _inject_and_clear_user_response(self, prompt_parts: List[str], task: Task):
        """
        Checks for a user response on the task, prepends it to the prompt parts,
        and clears it to ensure it's only used once.
        """
        if task.user_response:
            logger.info(f"Injecting user response into prompt for task [{task.id}]")
            response_prompt = (
                f"\n--- CRITICAL INFORMATION PROVIDED BY HUMAN OPERATOR ---\n"
                f"{task.user_response}\n"
                f"----------------------------------------------------------\n"
            )
            prompt_parts.insert(0, response_prompt)  # Prepend to the prompt
            task.user_response = None  # Clear it after use
            self._broadcast_state_update(task)  # Broadcast the cleared state

    def _add_llm_data_to_system_span(self, span: Span, agent_res: AgentRunResult):
        """Adds LLM usage data to a span for a system-level operation without a task context."""
        if not span or not agent_res: return
        usage = agent_res.usage()
        # We don't track cost here as there is no task to assign it to.
        # This is a trade-off for keeping the system logic clean.
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))
    def _create_notebook_tool(
        self, registry: TaskRegistry, broadcast_callback: Callable
    ):
        """
        Factory function to create a callable tool for updating the shared notebook.
        This tool will also trigger a broadcast to the UI when used.
        """

        def update_notebook_tool_instance(task_id: str, updates: Dict[str, Any]) -> str:
            """
            Updates the shared notebook for a given task.
            Use this tool to permanently record important decisions, parameters, or derived facts.
            Provide key-value pairs in the 'updates' dictionary.
            Set a value to 'null' (Python `None`) to delete a key from the notebook.
            """
            task = registry.tasks.get(task_id)
            if not task:
                return f"Error: Task {task_id} not found in registry."

            updated_keys = []
            deleted_keys = []
            for key, value in updates.items():
                if value is None:  # Convention to delete a key
                    if key in task.shared_notebook:
                        del task.shared_notebook[key]
                        deleted_keys.append(key)
                else:
                    task.shared_notebook[key] = value
                    updated_keys.append(key)

            # Trigger broadcast after modification to update UI
            broadcast_callback(task)

            result_msg = f"Notebook for task {task_id} updated. Updated: {', '.join(updated_keys) if updated_keys else 'None'}. Deleted: {', '.join(deleted_keys) if deleted_keys else 'None'}."
            logger.info(result_msg)
            return result_msg

        return update_notebook_tool_instance

    def _get_publisher_channel(
        self,
    ) -> pika.adapters.blocking_connection.BlockingChannel:
        """Ensures a publisher channel is available and open."""
        if self._publisher_connection is None or self._publisher_connection.is_closed:
            logger.info("Establishing new publisher RabbitMQ connection and channel.")
            self._publisher_connection = pika.BlockingConnection(
                pika.ConnectionParameters(RABBITMQ_HOST)
            )
            self._publish_channel = self._publisher_connection.channel()
            self._publish_channel.exchange_declare(
                exchange=self.STATE_UPDATES_EXCHANGE, exchange_type="fanout"
            )
        return self._publish_channel

    def _broadcast_state_update(self, task: Task):
        """Publishes the full state of a task to RabbitMQ."""
        try:
            channel = self._get_publisher_channel()
            # Exclude OpenTelemetry span context from JSON serialization
            message = task.model_dump_json(
                exclude={"span", "otel_context", "last_llm_history"}
            )
            channel.basic_publish(
                exchange=self.STATE_UPDATES_EXCHANGE, routing_key="", body=message
            )
            # logger.debug(f"Broadcasted state update for task {task.id}") # Use debug to avoid log spam
        except Exception as e:
            logger.error(
                f"Failed to broadcast state update for task {task.id}: {e}",
                exc_info=False,
            )

    def _format_stream_node(self, node: Any) -> Optional[Dict[str, Any]]:
        """
        Formats a Pydantic-AI stream node from agent.iter() into a UI-friendly dictionary.
        """

        # Helper function to truncate long strings for UI display
        def truncate(s: Any, length: int = 250) -> str:
            s_str = str(s)
            return s_str if len(s_str) <= length else s_str[:length - 3] + "..."

        # NEW: Check for the high-level node that contains the model's response parts
        if isinstance(node, CallToolsNode):
            # A CallToolsNode contains the model's response. Iterate through its parts.
            for part in node.model_response.parts:
                # CHANGED: Check for the concrete TextPart class
                if isinstance(part, TextPart):
                    content = part.content.strip()
                    # Your original logic to capture "thoughts" was good, we just apply it here.
                    if content:
                        return {"type": "thought", "content": truncate(content)}

                # This check for ToolCallPart is now correctly placed
                elif isinstance(part, ToolCallPart):
                    return {
                        "type": "tool_call",
                        "tool_name": part.tool_name,
                        "args": part.args_as_dict()
                    }

        # NEW: Tool results are found inside the *next* request sent to the model
        elif isinstance(node, ModelRequestNode):
            # Check if this request contains the result of a tool call
            for part in node.request.parts:
                # CHANGED: Use the correct class name `ToolResultPart`
                if isinstance(part, ToolReturnPart):
                    return {
                        "type": "tool_result",
                        "tool_name": part.tool_name,
                        # The result content is in the `content` attribute
                        "result": truncate(part.content)
                    }

        # NEW: Check for the `End` node which contains the `FinalResult`
        elif isinstance(node, End):
            # The `End` node signals the agent is done and is producing the final answer.
            if isinstance(node.data, FinalResult):
                # You can either signal synthesis is starting, or pass the final output.
                # Passing the final output is often more useful.
                return {
                    "type": "final_answer",
                    "content": node.data.output
                }

        # Return None for node types we don't want to display (like UserPromptNode)
        return None

        # --- Override _handle_agent_output for Interactive DAG Agent (from dag_agent.py) ---
    # This ensures consistency for auto-answering and simple type checks.
    # Note: the broadcast_callback is specific to InteractiveDAGAgent.
    async def _handle_agent_output(
            self,
            ctx: TaskContext,
            agent_res: AgentRunResult,
            expected_output_type: Any,
            agent_role_name: str,
    ) -> Tuple[bool, Any]:
        """
        Processes an agent's output. Allows ONLY the 'question_formulator'
        to pause the task by asking a UserQuestion.
        """
        t = ctx.task
        self._add_llm_data_to_span(t.span, agent_res, t)

        t.last_llm_history = agent_res.all_messages()

        if agent_role_name == "executor":
            t.executor_llm_history = agent_res.all_messages_json()

        actual_output = agent_res.output

        # --- NEW TARGETED EXCEPTION LOGIC ---
        if isinstance(actual_output, UserQuestion):
            # ONLY the question_formulator is allowed to pause the system.
            if agent_role_name == "question_formulator":
                logger.info(
                    f"Task [{t.id}] escalating to human with question (Priority: {actual_output.priority}): {actual_output.question[:80]}..."
                )
                t.current_question = actual_output
                t.agent_role_paused = agent_role_name
                t.status = "waiting_for_user_response"
                self._broadcast_state_update(t)  # Broadcast the pause state
                return True, actual_output  # Signal that the task was successfully paused
            else:
                # Any other agent attempting to ask a question is violating the "forced decisiveness" rule.
                logger.warning(
                    f"Agent '{agent_role_name}' for task [{t.id}] attempted to ask a question, which is not allowed. "
                    "This will be treated as an invalid output."
                )
                # Let it fall through to be treated as a schema mismatch / failure.
                pass
        # --- END OF NEW LOGIC ---

        if not isinstance(actual_output, BaseModel) and not isinstance(
                actual_output, str
        ):
            logger.warning(
                f"Task [{t.id}] received an unexpected non-BaseModel/str output from agent '{agent_role_name}'. "
                f"Expected {str(expected_output_type)}, got {type(actual_output).__name__}. This might indicate a schema mismatch or LLM deviation."
            )

        # Default behavior for all non-pausing scenarios.
        return False, actual_output

    async def _prune_task_graph_if_needed(self, required_space: int = 0):
        """
        UPDATED: Checks if the graph is over the limit and prunes tasks.
        This version enforces a strict hierarchy: it will ONLY present safe "leaf" tasks
        to the LLM if they exist. Risky parent tasks are only considered as a last resort.
        It also validates the LLM's response to ensure it only selected valid candidates.
        """
        current_size = len(self.registry.tasks)
        limit = self.max_total_tasks

        num_to_prune = max(0, (current_size + required_space) - limit)

        if num_to_prune == 0:
            return

        logger.warning(
            f"Graph size ({current_size}) + proposed ({required_space}) exceeds limit ({limit}). "
            f"Attempting to prune at least {num_to_prune} task(s)."
        )

        with self.tracer.start_as_current_span("GraphPruning") as span:
            # All tasks that are not active or finished are eligible candidates.
            prunable_statuses = {"pending", "paused_by_human", "failed", "cancelled"}
            all_eligible_tasks = [
                t for t in self.registry.tasks.values()
                if t.status in prunable_statuses and t.parent is not None and (not t.is_critical) and t.counts_toward_limit  # Never prune a root task
            ]

            # Build a map of which tasks depend on which other tasks.
            downstream_dependents: Dict[str, List[str]] = {t.id: [] for t in self.registry.tasks.values()}
            for task in self.registry.tasks.values():
                for dep_id in task.deps:
                    if dep_id in downstream_dependents:
                        downstream_dependents[dep_id].append(task.id)

            # --- NEW STRICT HIERARCHY LOGIC ---
            # Tier 1: Prioritize "leaf" nodes among eligible tasks (tasks that no other task depends on).
            safe_candidates = [
                task for task in all_eligible_tasks
                if not downstream_dependents.get(task.id)
            ]

            pruning_candidates = []
            pruner_prompt = ""

            if safe_candidates:
                logger.info(
                    f"Found {len(safe_candidates)} safe leaf-node tasks to consider for pruning. LLM will only see these.")
                pruning_candidates = safe_candidates
                pruning_candidates_prompt = "\n".join([
                    f"- ID: {t.id}, Status: {t.status}, Description: {t.desc}"
                    for t in pruning_candidates
                ])
                pruner_prompt = (
                    f"The project has too many tasks. You MUST select at least {num_to_prune} of the LEAST critical tasks for removal "
                    f"from the following list of SAFE candidates.\n\n"
                    f"SAFE CANDIDATES FOR PRUNING:\n{pruning_candidates_prompt}"
                )
            else:
                # Tier 2: Fallback to risky pruning only if no safe options exist.
                logger.warning(
                    "No safe leaf-node tasks found for pruning. Falling back to considering all eligible tasks. This is risky.")
                pruning_candidates = all_eligible_tasks
                pruning_candidates_prompt = "\n".join([
                    (f"- ID: {t.id}, Status: {t.status}, Description: {t.desc}, "
                     f"Required by: {downstream_dependents.get(t.id, [])}")
                    for t in pruning_candidates
                ])
                pruner_prompt = (
                    f"The project has too many tasks and NO safe 'leaf' tasks could be found. "
                    f"You MUST select at least {num_to_prune} of the LEAST critical tasks for removal from the following list. "
                    f"Prioritize tasks that are required by the fewest other tasks.\n\n"
                    f"RISKY CANDIDATES FOR PRUNING:\n{pruning_candidates_prompt}"
                )

            if not pruning_candidates:
                logger.error(
                    "Pruning required, but no eligible tasks (safe or risky) were found. Cannot proceed with pruning.")
                return

            # This set will be used to validate the LLM's response.
            valid_candidate_ids = {t.id for t in pruning_candidates}

            pruning_decision = await self._run_stateless_agent(self.graph_pruner, pruner_prompt, ctx=None)

            if not pruning_decision or not pruning_decision.tasks_to_prune:
                logger.error("Graph pruner failed to return a valid decision. No tasks removed.")
                span.set_status(trace.Status(StatusCode.ERROR, "Pruner returned invalid or empty output"))
                return

            # --- NEW: VALIDATION OF LLM RESPONSE ---
            for task_to_prune in pruning_decision.tasks_to_prune:
                if task_to_prune.task_id not in valid_candidate_ids:
                    logger.error(
                        f"LLM proposed pruning an invalid or protected task ID ('{task_to_prune.task_id}'). "
                        f"Rejecting entire pruning plan for this cycle to ensure safety."
                    )
                    span.set_status(trace.Status(StatusCode.ERROR, "LLM proposed invalid task to prune"))
                    return
            # --- END OF VALIDATION ---

            pruned_ids = set()
            for task_to_prune in pruning_decision.tasks_to_prune:
                await self._prune_specific_task(
                    task_id=task_to_prune.task_id,
                    reason=f"Pruned by graph manager: {task_to_prune.reason}",
                    new_status="pruned"
                )
                pruned_ids.add(task_to_prune.task_id)

            await self._prune_orphaned_tasks()

            span.set_attribute("graph.tasks.pruned_count", len(pruned_ids))
            logger.info(
                f"Pruning complete. Removed {len(pruned_ids)} tasks. New graph size: {len(self.registry.tasks)}")

    async def _prune_orphaned_tasks(self):
        """
        Identifies and prunes any tasks that have no downstream dependents (orphans),
        excluding completed tasks and the root task. This is a cleanup utility.
        """
        logger.info("Running orphan task cleanup...")

        # --- START OF MODIFICATION ---
        # The while True loop is the source of the infinite loop. We will remove it
        # and replace it with a single, safer pass.

        all_task_ids = set(self.registry.tasks.keys())

        # 1. Build a set of all tasks that are listed as a dependency by at least one other task.
        tasks_with_dependents = set()
        for task in self.registry.tasks.values():
            tasks_with_dependents.update(task.deps)

        # 2. Identify orphans: tasks that are NOT in the set from step 1
        #    and are NOT currently being executed.
        orphaned_ids = [
            task_id for task_id in all_task_ids
            if task_id not in tasks_with_dependents
               and self.registry.tasks[task_id].status != "complete"
               and self.registry.tasks[task_id].parent is not None
               and task_id not in self.inflight  # CRITICAL CHECK: Do not touch in-flight tasks
        ]

        # 3. If no safe-to-prune orphans are found, the cleanup is done for this cycle.
        if not orphaned_ids:
            logger.info("No orphaned tasks found to prune.")
            return  # Exit the function

        # 4. Prune the found orphans.
        logger.warning(f"Found {len(orphaned_ids)} orphaned tasks to prune: {orphaned_ids}")
        for orphan_id in orphaned_ids:
            # The safety check inside _prune_specific_task is still valuable, but this
            # pre-check prevents the infinite loop condition.
            await self._prune_specific_task(
                task_id=orphan_id,
                reason="Orphaned: No other tasks depend on this task.",
                new_status="cancelled"
            )

    async def _prune_specific_task(self, task_id: str, reason: str, new_status: str = "cancelled", override_critical=False):
        """
        Safely removes a task from the registry and cleans up all references to it.
        Now uses a 'cancelled' status by default for a clearer UI representation.
        """
        if task_id not in self.registry.tasks:
            logger.warning(f"Attempted to prune non-existent task: {task_id}")
            return

        task = self.registry.tasks[task_id]
        logger.info(f"Initiating pruning of task [{task_id}] with status '{new_status}'. Reason: {reason}")

        if task.is_critical and not override_critical:
            logger.error(
                f"FATAL PRUNING ERROR: Attempted to prune task [{task_id}] which is a critical task. "
                f"This action has been BLOCKED."
            )
            return

        UNPRUNABLE_STATUSES = {"running", "proposing", "planning", "complete"}
        if task.status in UNPRUNABLE_STATUSES:
            logger.error(
                f"FATAL PRUNING ERROR: Attempted to prune task [{task_id}] which is in an "
                f"unprunable state ('{task.status}'). This action has been BLOCKED."
            )
            return

        # 1. Recursively prune children first
        children_to_prune = list(task.children)
        for child_id in children_to_prune:
            await self._prune_specific_task(child_id, f"Cascading cancellation from parent {task_id}", new_status)

        # 2. Remove from parent's children list
        if task.parent and task.parent in self.registry.tasks:
            parent = self.registry.tasks[task.parent]
            if task_id in parent.children:
                parent.children.remove(task_id)
                self._broadcast_state_update(parent)

        # 3. Remove from downstream dependencies
        for other_task in self.registry.tasks.values():
            if task_id in other_task.deps:
                other_task.deps.remove(task_id)
                logger.info(f"Removed cancelled task [{task_id}] from dependencies of [{other_task.id}].")
                self._broadcast_state_update(other_task)

        # 4. Cancel any in-flight asyncio execution
        if task_id in self.task_futures:
            self.task_futures[task_id].cancel()

        # 5. Send a final update to UI with the new status
        task.status = new_status
        task.result = f"❌ CANCELLED: {reason}"
        self._broadcast_state_update(task)

        # 6. Finally, remove from registry and inflight set
        if task_id in self.registry.tasks:
            del self.registry.tasks[task_id]
        if task_id in self.inflight:
            self.inflight.discard(task_id)

        logger.info(f"Task [{task_id}] completely pruned from registry.")


    def _listen_for_directives_target(self):
        """
        Target function for the consumer thread. It runs in a separate thread.
        Manages its own RabbitMQ connection.
        """
        logger.info(
            "Consumer thread starting: Connecting to RabbitMQ for directives..."
        )
        try:
            # Establish an independent connection for this thread
            self._consumer_connection_for_thread = pika.BlockingConnection(
                pika.ConnectionParameters(RABBITMQ_HOST)
            )
            self._consumer_channel_for_thread = (
                self._consumer_connection_for_thread.channel()
            )
            self._consumer_channel_for_thread.queue_declare(
                queue=self.DIRECTIVES_QUEUE, durable=True
            )
            self._consumer_channel_for_thread.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body):
                message = json.loads(body)
                logger.debug(f"Received directive from RabbitMQ: {message}")

                # Use the stored main event loop reference
                main_loop = self._main_event_loop
                if (
                    main_loop and main_loop.is_running()
                ):  # Check if it's available and running
                    main_loop.call_soon_threadsafe(
                        self.directives_queue.put_nowait, message
                    )
                else:
                    logger.warning(
                        "Main asyncio loop not running or not set, cannot put directive into queue. Directive dropped. Message: %s",
                        message,
                    )
                ch.basic_ack(delivery_tag=method.delivery_tag)

            self._consumer_channel_for_thread.basic_consume(
                queue=self.DIRECTIVES_QUEUE, on_message_callback=callback
            )
            logger.info("Consumer thread: Starting to consume directives...")

            # Keep consuming until instructed to shut down
            while not self._shutdown_event.is_set():
                # Process events for a short duration, allowing periodic checks for shutdown_event
                # This ensures the thread doesn't block indefinitely on consumption if no messages arrive.
                self._consumer_connection_for_thread.process_data_events(time_limit=0.1)

            logger.info("Consumer thread: Shutdown event received. Stopping consuming.")
            # Ensure consuming stops gracefully if it was active
            self._consumer_channel_for_thread.stop_consuming()

        except pika.exceptions.ConnectionClosedByBroker:
            logger.error(
                "Consumer thread: RabbitMQ connection closed by broker.", exc_info=False
            )
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Consumer thread: AMQP Connection Error: {e}", exc_info=False)
        except Exception as e:
            logger.error(f"Consumer thread stopped unexpectedly: {e}", exc_info=True)
        finally:
            if (
                self._consumer_connection_for_thread
                and self._consumer_connection_for_thread.is_open
            ):
                logger.info("Consumer thread: Closing RabbitMQ connection.")
                self._consumer_connection_for_thread.close()
            self._consumer_connection_for_thread = None
            self._consumer_channel_for_thread = None
            logger.info("Consumer thread exited.")

    async def _handle_directive(self, directive: dict):
        command = directive.get("command")
        task_id = directive.get("task_id")
        payload = directive.get("payload")
        task = self.registry.tasks.get(task_id)
        # ADD_ROOT_TASK logic remains the same...
        if command == "ADD_ROOT_TASK":
            payload = directive.get("payload", {})
            desc = payload.get("desc")
            if not desc:
                logger.warning("ADD_ROOT_TASK directive received with no description.")
                return

            mcp_servers_config = payload.get("mcp_servers", [])

            new_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=desc,
                needs_planning=payload.get("needs_planning", True),
                status="pending",
                mcp_servers=mcp_servers_config,
            )
            self.registry.add_task(new_task)
            logger.info(
                f"Added new root task from directive: {new_task.id} - {new_task.desc}"
            )
            self._broadcast_state_update(new_task)
            return

        if command == "ADD_DOCUMENT":
            payload = directive.get("payload", {})
            name = payload.get("name")
            content = payload.get("content")
            if not name or content is None:
                logger.warning("ADD_DOCUMENT directive missing name or content.")
                return

            doc_state = DocumentState(content=content)
            new_doc_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=f"Document: {name}",
                task_type="document",
                document_state=doc_state,
                status="complete",  # Documents are considered 'complete' by default
                is_critical=True,
                counts_toward_limit=False,  # Documents don't count towards the task limit
            )
            self.registry.add_task(new_doc_task)
            logger.info(f"Added new document node: {new_doc_task.id} - {name}")
            self._broadcast_state_update(new_doc_task)
            return

        elif command == "UPDATE_DOCUMENT":
            if task and task.task_type == "document" and task.document_state:
                new_name = payload.get("name")
                new_content = payload.get("content")

                if new_name:
                    task.desc = f"Document: {new_name}"

                if new_content is not None and generate_hash(new_content) != task.document_state.content_hash:
                    logger.info(
                        f"Updating content for document {task.id}. Version {task.document_state.version} -> {task.document_state.version + 1}")
                    task.document_state = DocumentState(
                        content=new_content,
                        version=task.document_state.version + 1
                    )

                self._broadcast_state_update(task)
            else:
                logger.warning(f"UPDATE_DOCUMENT for invalid/non-document task {task_id}")
            return

        elif command == "DELETE_DOCUMENT":
            if task and task.task_type == "document":
                logger.info(f"Pruning document node {task.id} by operator directive.")
                await self._prune_specific_task(
                    task_id=task.id,
                    reason="Document deleted by operator.",
                    new_status="pruned",
                    override_critical=True,
                )
            else:
                logger.warning(f"DELETE_DOCUMENT for invalid/non-document task {task_id}")
            return

        # --- NEW LOGIC FOR CANCEL ---
        if command == "CANCEL":
            if task:
                logger.info(f"Soft cancelling task {task_id} by operator directive.")
                task.status = "cancelled"
                task.result = f"Cancelled by operator: {payload or 'No reason given.'}"
                # Cancel in-flight asyncio task but do not delete from registry
                if task_id in self.task_futures:
                    self.task_futures[task_id].cancel()
                self._broadcast_state_update(task)
                # Invalidate downstream tasks so they can be re-evaluated
                self._reset_downstream_tasks(task_id)
            else:
                logger.warning(f"Directive 'CANCEL' for unknown task {task_id} ignored.")
            return

        if command == "PRUNE_TASK":
            reason = payload or "Pruned by operator."
            await self._prune_specific_task(task_id, reason=reason, new_status="pruned")
            await self._prune_orphaned_tasks()
            return

        if command == "RESET_STATE":
            logger.warning("Received RESET_STATE directive. Wiping all current tasks and loading new state.")
            payload = directive.get("payload", {})
            if not isinstance(payload, dict):
                logger.error("RESET_STATE payload is not a dictionary. Aborting.")
                return

            # 1. Cancel and clear all in-flight operations
            for future in self.task_futures.values():
                future.cancel()
            self.task_futures.clear()
            self.inflight.clear()

            # 2. Wipe the current registry
            self.registry.tasks.clear()

            # 3. Load the new state from the payload
            for task_id, task_data in payload.items():
                try:
                    # Re-create Task objects from the dictionary data
                    new_task = Task(**task_data)
                    self.registry.add_task(new_task)
                except Exception as e:
                    logger.error(f"Failed to load task {task_id} from state file: {e}")

            # 4. Broadcast all loaded tasks to update the UI
            logger.info(f"Successfully loaded {len(self.registry.tasks)} tasks. Broadcasting state to UI.")
            for task in self.registry.tasks.values():
                self._broadcast_state_update(task)

            return

            # ----------------------------

        # All other directives require the task to exist

        if not task:
            logger.warning(f"Directive '{command}' for unknown task {task_id} ignored.")
            return

        logger.info(f"Handling command '{command}' for task {task_id}")


        # Cancel in-flight task future if a directive changes its state
        if task_id in self.task_futures and not self.task_futures[task_id].done():
            self.task_futures[task_id].cancel()
            logger.info(
                f"Cancelled in-flight asyncio.Task for {task_id} due to directive."
            )

        original_status = task.status
        broadcast_needed = False
        reset_dependents = False

        if command == "PAUSE":
            task.status = "paused_by_human"
            broadcast_needed = True
        elif command == "RESUME":
            if task.status == "paused_by_human":
                task.status = "pending"
                broadcast_needed = True
        elif command == "TERMINATE":
            task.status = "failed"
            task.result = f"Terminated by operator: {payload}"
            reset_dependents = True
            broadcast_needed = True
        elif command == "REDIRECT":
            task.human_directive = payload
            task.status = "pending"
            reset_dependents = True
            broadcast_needed = True
        elif command == "MANUAL_OVERRIDE":
            task.status = "complete"
            task.result = payload
            reset_dependents = True
            broadcast_needed = True
        elif command == "ANSWER_QUESTION":
            if task.status == "waiting_for_user_response":
                logger.info(f"Task {task_id} received answer: {str(payload)[:50]}...")
                task.user_response = str(payload)
                # Do NOT change status to pending here. Let main loop pick it up based on user_response presence.
                # Just clear the question flags.
                task.current_question = None
                task.agent_role_paused = None
                broadcast_needed = True
            else:
                logger.warning(f"Task {task_id} not waiting for a question, ignoring ANSWER_QUESTION.")

        # Handle state resets for Redirect/Override to ensure fresh execution
        if command in ["REDIRECT", "MANUAL_OVERRIDE"]:
             # Clear execution history to force fresh thinking based on new input/result
             task.fix_attempts = 0
             task.grace_attempts = 0
             task.verification_scores = []
             task.last_llm_history = None
             task.execution_log = []

        if reset_dependents:
             self._reset_downstream_tasks(task_id)

        if broadcast_needed or task.status != original_status:
            self._broadcast_state_update(task)

    async def _reset_downstream_tasks(self, task_id: str):
        """
        Recursively finds all downstream tasks that depend on the given task_id
        and resets their state to 'pending', forcing re-evaluation.
        """
        logger.info(f"Resetting downstream tasks of {task_id}")

        downstream_ids_to_reset = set()
        q = deque([task_id])

        # Build a complete set of all tasks that need resetting
        while q:
            current_id = q.popleft()
            for task in self.registry.tasks.values():
                if current_id in task.deps and task.id not in downstream_ids_to_reset:
                    downstream_ids_to_reset.add(task.id)
                    q.append(task.id)

        if not downstream_ids_to_reset:
            return

        logger.info(f"Found {len(downstream_ids_to_reset)} downstream tasks to reset: {downstream_ids_to_reset}")

        for downstream_id in downstream_ids_to_reset:
            task_to_reset = self.registry.tasks.get(downstream_id)
            if task_to_reset and task_to_reset.status != "pending":
                logger.info(f"Resetting task {downstream_id} to 'pending' state.")
                task_to_reset.status = "pending"
                task_to_reset.result = None
                task_to_reset.fix_attempts = 0
                task_to_reset.grace_attempts = 0
                task_to_reset.verification_scores = []
                task_to_reset.human_directive = None
                task_to_reset.user_response = None
                task_to_reset.current_question = None
                task_to_reset.last_llm_history = None
                task_to_reset.agent_role_paused = None

                self._broadcast_state_update(task_to_reset)


    def _reset_task_and_dependents(self, task_id: str):
        """Implements cascading state invalidation."""
        q = deque([task_id])
        dependents = set()
        while q:
            current_id = q.popleft()
            for t in self.registry.tasks.values():
                if current_id in t.deps and t.id not in dependents:
                    dependents.add(t.id)
                    q.append(t.id)

        if not dependents:
            return
        logger.info(f"Cascading invalidation will reset tasks: {dependents}")
        for tid in dependents:
            task = self.registry.tasks[tid]
            if task.status in [
                "complete",
                "failed",
                "running",
                "paused_by_human",
                "waiting_for_user_response",
                "planning",
                "proposing",
                "waiting_for_children",
            ]:
                task.status = "pending"
                task.result = None
                task.fix_attempts = 0
                task.grace_attempts = 0
                task.verification_scores = []
                task.human_directive = None
                task.user_response = None
                task.current_question = None
                task.last_llm_history = None
                task.agent_role_paused = None
                task.shared_notebook = {}  # Clear notebook on reset - ADDED
                self._broadcast_state_update(task)  # ADDED

    async def run(self):
        # Capture the main event loop reference *before* starting any new threads
        self._main_event_loop = asyncio.get_running_loop()

        # Start the consumer thread
        self._consumer_thread = threading.Thread(
            target=self._listen_for_directives_target, daemon=True
        )
        self._consumer_thread.start()

        # Initialize publisher connection for the main thread
        try:
            self._publisher_connection = pika.BlockingConnection(
                pika.ConnectionParameters(RABBITMQ_HOST)
            )
            self._publish_channel = self._publisher_connection.channel()
            self._publish_channel.exchange_declare(
                exchange=self.STATE_UPDATES_EXCHANGE, exchange_type="fanout"
            )
            self._publish_channel.queue_declare(
                queue=self.DIRECTIVES_QUEUE, durable=True
            )
            logger.info("Publisher RabbitMQ connection and channel initialized.")
        except Exception as e:
            logger.critical(
                f"Failed to initialize publisher RabbitMQ connection, agent cannot communicate state: {e}",
                exc_info=True,
            )
            self._shutdown_event.set()  # Signal consumer thread to stop
            raise  # Propagate critical error to prevent agent from running silently.

        logger.info("Broadcasting initial state of all tasks (if any)...")
        for task in self.registry.tasks.values():
            self._broadcast_state_update(task)

        try:
            with self.tracer.start_as_current_span("InteractiveDAGAgent.run"):
                while True:
                    await self._prune_task_graph_if_needed()

                    while not self.directives_queue.empty():
                        directive = await self.directives_queue.get()
                        await self._handle_directive(directive)

                    # --- START OF THE FIX ---
                    # Check for cascading failures from upstream dependencies.
                    for task in list(self.registry.tasks.values()):
                        if task.status == "failed":
                            continue

                        # Correctly check the status of each dependency task object.
                        # This generator expression yields True for each failed dependency.
                        failed_deps_exist = (
                            dep_task.status == "failed"
                            for dep_id in task.deps
                            if (dep_task := self.registry.tasks.get(dep_id)) is not None
                        )

                        if any(failed_deps_exist):
                            task.status = "failed"
                            task.result = "Upstream dependency failed."
                            self._broadcast_state_update(task)
                            task.last_llm_history = None
                            task.agent_role_paused = None
                    # --- END OF THE FIX ---

                    all_task_ids = set(self.registry.tasks.keys())
                    executable_statuses = {
                        "pending", "running", "planning", "proposing", "waiting_for_children",
                    }
                    non_executable_tasks = {
                        t.id for t in self.registry.tasks.values()
                        if t.status not in executable_statuses
                           and not (t.status == "waiting_for_user_response" and t.user_response is not None)
                    }

                    pending_tasks_for_scheduling = all_task_ids - non_executable_tasks

                    ready_to_run_ids = set()
                    for tid in pending_tasks_for_scheduling:
                        task = self.registry.tasks.get(tid)
                        if not task: continue

                        if task.status == "waiting_for_user_response" and task.user_response is not None:
                            ready_to_run_ids.add(tid)

                        elif task.status == "pending" and all(
                                (dep_task := self.registry.tasks.get(d)) is not None and dep_task.status == "complete"
                                for d in task.deps
                        ):
                            ready_to_run_ids.add(tid)

                        elif task.status == "waiting_for_children" and all(
                                (child_task := self.registry.tasks.get(c)) is not None and child_task.status in [
                                    "complete", "failed"]
                                for c in task.children
                        ):
                            if not any((child_task := self.registry.tasks.get(
                                    c)) is not None and child_task.status == "failed" for c in task.children):
                                ready_to_run_ids.add(tid)
                                task.status = "pending"
                                self._broadcast_state_update(task)

                    ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

                    has_pending_interaction = any(
                        t.status == "waiting_for_user_response" and t.user_response is None
                        for t in self.registry.tasks.values()
                    )

                    if (
                            not ready and not self.inflight and not has_pending_interaction and self.directives_queue.empty()):
                        if (self._shutdown_event.is_set() or not self._consumer_thread.is_alive()):
                            logger.info("Interactive DAG execution complete.")
                            break
                        else:
                            logger.debug("Agent idle, but consumer thread active. Waiting for directives...")

                    for tid in ready:
                        if tid in self.registry.tasks:
                            self.inflight.add(tid)
                            self.task_futures[tid] = asyncio.create_task(self._run_taskflow(tid))
                            self._broadcast_state_update(self.registry.tasks[tid])

                    if self.task_futures:
                        done, _ = await asyncio.wait(
                            list(self.task_futures.values()),
                            timeout=0.1,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for fut in done:
                            tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                            if tid:
                                self.inflight.discard(tid)
                                del self.task_futures[tid]
                                try:
                                    fut.result()
                                except asyncio.CancelledError:
                                    logger.info(f"Task [{tid}] was cancelled by operator or system.")
                                except Exception as e:
                                    logger.error(f"Task [{tid}] future failed: {e}", exc_info=True)
                                    t = self.registry.tasks.get(tid)
                                    if t and t.status != "failed":
                                        t.status = "failed"
                                        t.result = f"Execution failed: {e}"
                                        self._broadcast_state_update(t)
                                        t.last_llm_history = None
                                        t.agent_role_paused = None

                    if self.proposed_tasks_buffer:
                        await self._run_global_resolution()

                    await asyncio.sleep(0.05)
        finally:
            logger.info(
                "Agent run loop is shutting down. Signaling consumer thread to stop and closing publisher connection.")
            self._shutdown_event.set()
            if self._consumer_thread and self._consumer_thread.is_alive():
                self._consumer_thread.join(timeout=5)
            if self._publisher_connection and self._publisher_connection.is_open:
                logger.info("Closing publisher RabbitMQ connection.")
                self._publisher_connection.close()

    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)

            if t.status == "paused_by_human" or (t.status == "waiting_for_user_response" and t.user_response is None):
                # logger.info(f"[{t.id}] Task is {t.status}, skipping execution until directive received.")
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                def update_status_and_broadcast(new_status):
                    if t.status != new_status:
                        t.status = new_status
                        self._broadcast_state_update(t)

                if t.status == "waiting_for_user_response" and t.user_response is not None:
                    logger.info(f"[{t.id}] Resuming task with user response.")
                    if t.agent_role_paused == "initial_planner":
                        update_status_and_broadcast("planning")
                    elif t.agent_role_paused == "adaptive_decomposer":
                        update_status_and_broadcast("proposing")
                    else:  # Covers 'executor' and 'question_formulator' which resume execution
                        update_status_and_broadcast("running")

                # Simplified, linear state progression
                if t.status == 'waiting_for_children':
                    update_status_and_broadcast("running")
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    update_status_and_broadcast("planning")
                    await self._run_initial_planning(ctx)
                    if t.status not in ["paused_by_human", "waiting_for_user_response"]:
                        t.already_planned = True
                        update_status_and_broadcast(
                            "waiting_for_children" if t.children else "complete"
                        )
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    update_status_and_broadcast("proposing")

                    was_frozen = await self._run_adaptive_decomposition(ctx)
                    if not was_frozen:
                        if t.status not in ["paused_by_human", "waiting_for_user_response"]:
                            update_status_and_broadcast("waiting_for_children")
                elif t.status in ["pending", "running"]:
                    update_status_and_broadcast("running")
                    await self._run_task_execution(ctx)
                else:
                    logger.warning(
                        f"[{t.id}] Task in unhandled status '{t.status}'. Skipping."
                    )

                # Final status check
                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    update_status_and_broadcast("complete")
                    t.last_llm_history = None
                    t.agent_role_paused = None

            except asyncio.CancelledError:
                logger.info(f"Task [{t.id}] asyncio future was cancelled.")
                if t.status not in ["waiting_for_user_response"]:  # Don't override pause from question
                    t.status = "paused_by_human"
                self._broadcast_state_update(t)
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                self._broadcast_state_update(t)
                t.last_llm_history = None
                t.agent_role_paused = None
                raise
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", t.status)
                self._broadcast_state_update(t)

    async def _run_initial_planning(self, ctx: TaskContext):
        """
        OVERRIDDEN: Implements the new "Plan Chains -> Merge Chains -> Commit Tasks" workflow.
        """
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning (chain-based) for: {t.desc}")

        # (The first part of getting the prompt is the same as the base class)
        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context_str = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt_parts = [
            f"Objective: {t.desc}",
            f"\nAvailable completed data sources:\n{context_str}",
            f"\n--- Current Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
        ]
        self._inject_and_clear_user_response(prompt_parts, t)
        full_prompt = "\n".join(prompt_parts)

        plan_res = await self.initial_planner.run(user_prompt=full_prompt, message_history=t.last_llm_history)
        is_paused, chained_plan = await self._handle_agent_output(
            ctx=ctx, agent_res=plan_res, expected_output_type=ChainedExecutionPlan, agent_role_name="initial_planner"
        )

        if is_paused or not chained_plan or not chained_plan.task_chains:
            if not is_paused: logger.info(f"[{t.id}] Planning complete. No subtasks needed. Proceeding to execution.")
            return

        logger.info(f"[{t.id}] Planner proposed {len(chained_plan.task_chains)} chain(s). Consolidating...")

        final_task_descriptions = []
        for chain_obj in chained_plan.task_chains:
            if not chain_obj.chain: continue

            # --- START OF LOGICAL FIX ---
            # Collect all unique dependencies from every task in the chain.
            all_deps_in_chain = set()
            for task_desc in chain_obj.chain:
                all_deps_in_chain.update(task_desc.deps)
            # --- END OF LOGICAL FIX ---

            if len(chain_obj.chain) == 1:
                # If a chain has only one step, use it directly (it already has its deps).
                final_task_descriptions.append(chain_obj.chain[0])
            else:
                # If a chain has multiple steps, merge them into a single task.
                descriptions_to_merge = [td.desc for td in chain_obj.chain]
                merger_prompt = (
                    "Combine the following sequential steps into a single, comprehensive task description:\n"
                    f"{json.dumps(descriptions_to_merge, indent=2)}"
                )
                merged_desc = await self._run_stateless_agent(self.task_merger, merger_prompt, ctx)

                if merged_desc:
                    # A merged task should be decomposable if any of its parts were.
                    should_decompose_further = any(td.can_request_new_subtasks for td in chain_obj.chain)

                    # --- CRUCIAL CHANGE: Create the new TaskDescription with the aggregated dependencies.
                    final_task_descriptions.append(
                        TaskDescription(
                            local_id=f"merged_{chain_obj.chain[0].local_id}",
                            desc=merged_desc,
                            can_request_new_subtasks=should_decompose_further,
                            deps=list(all_deps_in_chain)  # Assign the collected deps here.
                        )
                    )
                else:
                    logger.warning(f"[{t.id}] Task merger failed for a chain. Discarding chain.")

        if final_task_descriptions:
            logger.info(f"[{t.id}] Committing {len(final_task_descriptions)} consolidated sub-tasks.")
            self._commit_task_descriptions(ctx, final_task_descriptions)
            t.status = "waiting_for_children"
        else:
            logger.info(f"[{t.id}] No valid sub-tasks remained after consolidation.")

    async def _process_initial_planning_output(
        self, ctx: TaskContext, plan: ExecutionPlan
    ):
        """Helper to encapsulate common logic after initial planning produces an ExecutionPlan."""
        t = ctx.task
        if not plan.needs_subtasks:
            return

        local_to_global_id_map = {}
        for sub in plan.subtasks:
            new_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=sub.desc,
                parent=t.id,
                can_request_new_subtasks=sub.can_request_new_subtasks,
                mcp_servers=t.mcp_servers,
            )
            self.registry.add_task(new_task)
            t.children.append(new_task.id)
            self.registry.add_dependency(t.id, new_task.id)
            local_to_global_id_map[sub.local_id] = new_task.id
            self._broadcast_state_update(new_task)

        for sub in plan.subtasks:
            new_global_id = local_to_global_id_map.get(sub.local_id)
            if not new_global_id:
                continue
            for dep in sub.deps:
                dep_global_id = local_to_global_id_map.get(dep.local_id) or (
                    dep.local_id if dep.local_id in self.registry.tasks else None
                )
                if dep_global_id:
                    self.registry.add_dependency(new_global_id, dep_global_id)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        """
        OVERRIDDEN: Implements the new "Plan Chains -> Merge Chains -> Commit Tasks" workflow
        for dynamic, bottom-up decomposition.
        """
        t = ctx.task
        # (Same graph size check as before)
        if len(self.registry.tasks) >= self.max_total_tasks:
            logger.warning(f"[{t.id}] Task graph is full. Adaptive decomposition is FROZEN.")
            t.status = "running"
            self._broadcast_state_update(t)
            return True

        # (Same prompt setup as before)
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt_parts = [
            f"Your current complex task is: {t.desc}",
            f"\nResults from dependencies:\n{json.dumps(t.dep_results, indent=2)}",
            "\nBreak this down into smaller steps if necessary.",
        ]
        self._inject_and_clear_user_response(prompt_parts, t)
        full_prompt = "\n".join(prompt_parts)

        proposals_res = await self.adaptive_decomposer.run(user_prompt=full_prompt, message_history=t.last_llm_history)
        is_paused, chained_plan = await self._handle_agent_output(
            ctx=ctx, agent_res=proposals_res, expected_output_type=ChainedExecutionPlan,
            agent_role_name="adaptive_decomposer"
        )

        if is_paused or not chained_plan or not chained_plan.task_chains:
            if not is_paused: logger.info(f"[{t.id}] Adaptive decomposer proposed no new tasks.")
            return False

        logger.info(f"[{t.id}] Decomposer proposed {len(chained_plan.task_chains)} new chain(s). Consolidating...")

        # (Same consolidation logic as in _run_initial_planning)
        final_task_descriptions = []
        for chain_obj in chained_plan.task_chains:
            if not chain_obj.chain: continue

            # --- START OF LOGICAL FIX (IDENTICAL TO THE ONE ABOVE) ---
            # Collect all unique dependencies from every task in the chain.
            all_deps_in_chain = set()
            for task_desc in chain_obj.chain:
                all_deps_in_chain.update(task_desc.deps)
            # --- END OF LOGICAL FIX ---

            if len(chain_obj.chain) == 1:
                final_task_descriptions.append(chain_obj.chain[0])
            else:
                descriptions_to_merge = [td.desc for td in chain_obj.chain]
                merger_prompt = f"Combine these steps into one task: {json.dumps(descriptions_to_merge)}"
                merged_desc = await self._run_stateless_agent(self.task_merger, merger_prompt, ctx)
                if merged_desc:
                    should_decompose_further = any(td.can_request_new_subtasks for td in chain_obj.chain)

                    # --- CRUCIAL CHANGE: Create the new TaskDescription with the aggregated dependencies.
                    final_task_descriptions.append(
                        TaskDescription(
                            local_id=f"merged_{chain_obj.chain[0].local_id}",
                            desc=merged_desc,
                            can_request_new_subtasks=should_decompose_further,
                            deps=list(all_deps_in_chain)  # Assign the collected deps here.
                        )
                    )

        if final_task_descriptions:
            logger.info(
                f"[{t.id}] Committing {len(final_task_descriptions)} consolidated sub-tasks from decomposition.")
            self._commit_task_descriptions(ctx, final_task_descriptions)

        return False

    def _commit_task_descriptions(self, ctx: TaskContext, task_descriptions: List[TaskDescription]):
        """
        MODIFIED HELPER: A clean way to add a list of final, consolidated tasks to the registry,
        now with support for dependencies on existing nodes.
        """
        t = ctx.task
        for desc in task_descriptions:
            new_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=desc.desc,
                parent=t.id,
                can_request_new_subtasks=desc.can_request_new_subtasks,
                mcp_servers=t.mcp_servers,
                # --- NEW: Apply dependencies specified by the planner ---
                deps=set(desc.deps),
            )
            self.registry.add_task(new_task)
            t.children.append(new_task.id)
            # The parent task automatically depends on all its new children
            # self.registry.add_dependency(t.id, new_task.id)
            self._broadcast_state_update(new_task)

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running task execution for: {t.desc}")

        # >>> FIX 2: RESTORE THE DEPENDENCY PRUNING LOGIC <<<
        # This entire block was missing from the execution flow.
        children_ids = set(t.children)
        prunable_deps_ids = t.deps - children_ids
        final_deps_to_use = children_ids.copy()

        # Check if the number of other dependencies exceeds the configured limit.
        if len(prunable_deps_ids) > self.max_dependencies_per_task:
            logger.warning(
                f"Task [{t.id}] has {len(prunable_deps_ids)} prunable dependencies, exceeding limit of {self.max_dependencies_per_task}. Initiating selection..."
            )
            t.span.add_event("DependencyPruningTriggered", {"dependency_count": len(prunable_deps_ids)})

            pruning_candidates_prompt = "\n".join(
                [f"- ID: {dep_id}, Description: {self.registry.tasks[dep_id].desc}" for dep_id in prunable_deps_ids]
            )

            pruner_prompt = (
                f"Your main objective is: '{t.desc}'\n\n"
                f"From the following list of available data dependencies, you must select ONLY the most critical ones "
                f"to achieve this objective. You cannot select more than {self.max_dependencies_per_task}.\n\n"
                f"Available Dependencies:\n{pruning_candidates_prompt}"
            )

            # Use the stateless agent for a robust, one-shot decision.
            selection = await self._run_stateless_agent(self.dependency_pruner, pruner_prompt, ctx)

            if selection and selection.approved_dependency_ids:
                approved_ids = set(selection.approved_dependency_ids)
                final_deps_to_use.update(approved_ids)
                logger.info(f"Pruned dependencies down to {len(approved_ids)} for this run. Reasoning: {selection.reasoning}")
                t.span.set_attribute("dependencies.pruned_count", len(prunable_deps_ids) - len(approved_ids))
            else:
                logger.error(f"[{t.id}] Dependency pruner failed. Using a random subset of dependencies as a fallback.")
                import random
                sample_size = min(self.max_dependencies_per_task, len(prunable_deps_ids))
                fallback_deps = set(random.sample(list(prunable_deps_ids), sample_size))
                final_deps_to_use.update(fallback_deps)
        else:
            # If within limits, use all dependencies.
            final_deps_to_use.update(prunable_deps_ids)
        # --- END OF RESTORED LOGIC ---


        # The rest of the function now correctly uses the pruned `final_deps_to_use` set
        child_results = {
            self.registry.tasks[cid].desc: self.registry.tasks[cid].result
            for cid in t.children
            if self.registry.tasks[cid].status == "complete" and self.registry.tasks[cid].result
        }

        pruned_dep_ids = final_deps_to_use - children_ids
        dep_results = {
            self.registry.tasks[did].desc: self.registry.tasks[did].result
            for did in pruned_dep_ids
            if self.registry.tasks[did].status == "complete" and self.registry.tasks[did].result
        }

        prompt_lines = [f"Your task is: {t.desc}\n"]
        if child_results:
            prompt_lines.append("\nSynthesize the results from your sub-tasks into a final answer:\n")
            for desc, res in child_results.items():
                prompt_lines.append(f"- From sub-task '{desc}':\n{res}\n\n")
        if dep_results:
            prompt_lines.append("\nUse data from the following critical dependencies to inform your answer:\n")
            for desc, res in dep_results.items():
                prompt_lines.append(f"- From dependency '{desc}':\n{res}\n\n")

        prompt_base_content = "".join(prompt_lines)

        # --- EXECUTOR AND MCP SETUP ---
        task_specific_mcp_clients = []
        task_specific_tools = list(self._tools_for_agents)

        if t.mcp_servers:
            logger.info(f"Task [{t.id}] has {len(t.mcp_servers)} MCP servers. Initializing clients.")
            for server_config in t.mcp_servers:
                address = server_config.get("address")
                server_type = server_config.get("type")
                server_name = server_config.get("name", address)
                if not address:
                    logger.warning(f"Task [{t.id}]: MCP server config missing address. Skipping.")
                    continue
                try:
                    if server_type == "streamable_http":
                        client = MCPServerStreamableHTTP(address)
                        task_specific_mcp_clients.append(client)
                        logger.info(f"[{t.id}]: Initialized StreamableHTTP MCP client for '{server_name}' at {address}")
                    else:
                        logger.warning(
                            f"[{t.id}]: Unknown MCP server type '{server_type}' for '{server_name}'. Skipping.")
                except Exception as e:
                    logger.error(f"[{t.id}]: Failed to create MCP client for '{server_name}' at {address}: {e}")

        task_executor = Agent(
            model=self.base_llm_model,
            system_prompt=self.executor_system_prompt,
            output_type=str,
            tools=task_specific_tools,
            mcp_servers=task_specific_mcp_clients,
        )

        # --- EXECUTION AND RETRY LOOP ---
        while True:
            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})

            t.execution_log = []
            self._broadcast_state_update(t)

            current_prompt_parts = [
                prompt_base_content,
                f"\n\n--- Current Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
                self._get_notebook_tool_guidance(t, "executor"),
            ]
            if t.human_directive:
                current_prompt_parts.insert(
                    0,
                    f"--- CRITICAL GUIDANCE FROM OPERATOR ---\n{t.human_directive}\n--------------------------------------\n\n",
                )
                t.human_directive = None
                self._broadcast_state_update(t)

            current_prompt = "".join(current_prompt_parts)

            logger.info(f"[{t.id}] Attempt {current_attempt}: Streaming executor actions...")

            exec_res = None
            try:
                async with task_executor.iter(user_prompt=current_prompt,
                                              message_history=t.last_llm_history) as agent_run:
                    async for node in agent_run:
                        formatted_node = self._format_stream_node(node)

                        if formatted_node:
                            logger.info(f"   > Stream [{t.id}]: {formatted_node}")
                            t.execution_log.append(formatted_node)
                            self._broadcast_state_update(t)

                exec_res = agent_run.result

            except Exception as e:
                logger.error(f"[{t.id}] Exception during agent stream: {e}", exc_info=True)
                t.execution_log.append({"type": "error", "content": f"An error occurred during execution: {e}"})
                self._broadcast_state_update(t)
                result = None

            if exec_res:
                is_paused, result = await self._handle_agent_output(
                    ctx=ctx,
                    agent_res=exec_res,
                    expected_output_type=str,
                    agent_role_name="executor",
                )
            else:
                result = None

            if result is None:
                t.human_directive = "Your last execution attempt failed or produced no output. Please re-evaluate and try again."
                t.fix_attempts += 1
                logger.warning(
                    f"[{t.id}] Executor stream (attempt {current_attempt}) produced no result. Triggering retry."
                )
                self._broadcast_state_update(t)
                if (t.fix_attempts + t.grace_attempts) > (
                        t.max_fix_attempts + self.max_grace_attempts
                ):
                    t.status = "failed"
                    self._broadcast_state_update(t)
                    raise Exception(f"Exceeded max attempts for task '{t.id}' after empty executor result.")
                continue

            await self._process_executor_output_for_verification(ctx, result)

            # --- CRITICAL FIX ---
            # After handling verification, if the task was paused or has reached a terminal state,
            # we must exit this execution loop to prevent re-running immediately.
            if t.status in ["complete", "failed", "paused_by_human", "waiting_for_user_response"]:
                return

    async def _run_global_resolution(self):
        """
        UPDATED: System-level operation to evaluate buffered task proposals.
        It now acts as a strict gatekeeper, calculating total required space,
        triggering a coordinated prune, and discarding all proposals if pruning fails
        to create sufficient space, thus preventing uncontrolled decomposition.
        """
        with self.tracer.start_as_current_span("GlobalConflictResolution") as span:
            num_proposals = len(self.proposed_tasks_buffer)
            logger.info(f"GLOBAL RESOLUTION: Evaluating {num_proposals} proposed sub-tasks.")
            span.set_attribute("proposals.count", num_proposals)

            try:
                approved_proposals = [p[0] for p in self.proposed_tasks_buffer]
                parent_map = {p[0].local_id: p[1] for p in self.proposed_tasks_buffer}

                # Step 1: Resolve conflicts if proposal count > limit.
                if len(approved_proposals) > self.global_proposal_limit:
                    logger.warning(
                        f"Proposal buffer ({len(approved_proposals)}) exceeds proposal limit ({self.global_proposal_limit}). Engaging resolver."
                    )
                    prompt_list = [f"local_id: {p.local_id}, importance: {p.importance}, desc: {p.desc}" for p in
                                   approved_proposals]
                    resolver_prompt = f"Prune this list to {self.global_proposal_limit} items: {prompt_list}"
                    resolved_plan = await self._run_stateless_agent(self.conflict_resolver, resolver_prompt, ctx=None)
                    if resolved_plan and isinstance(resolved_plan, ProposalResolutionPlan):
                        approved_proposals = resolved_plan.approved_tasks
                    else:
                        logger.error("Conflict resolver failed. Discarding all proposals.")
                        approved_proposals = []

                # Step 2: De-duplicate proposals against existing tasks.
                if approved_proposals:
                    statuses_to_check = {"completed", "running", "proposing", "planning", "cancelled"}
                    existing_tasks_context = "\n".join([
                        f"- ID: {t.id}, Status: {t.status}, Description: {t.desc}"
                        for t in self.registry.tasks.values() if t.status in statuses_to_check
                    ])
                    proposals_json = json.dumps([p.model_dump() for p in approved_proposals], indent=2)

                    redundancy_prompt = (
                        f"Here are the tasks already in the project:\n--- EXISTING TASKS ---\n{existing_tasks_context}\n\n"
                        f"Now, review the following 'Proposed Tasks' and return ONLY the ones that are not redundant.\n"
                        f"--- PROPOSED TASKS ---\n{proposals_json}"
                    )

                    deduplication_result = await self._run_stateless_agent(self.redundancy_checker, redundancy_prompt,
                                                                           ctx=None)

                    if deduplication_result and isinstance(deduplication_result, RedundancyDecision):
                        original_count = len(approved_proposals)
                        approved_proposals = deduplication_result.non_redundant_tasks
                        new_count = len(approved_proposals)
                        if original_count != new_count:
                            logger.info(
                                f"Redundancy check complete. Removed {original_count - new_count} redundant proposals.")
                            span.set_attribute("proposals.redundant_removed", original_count - new_count)
                    else:
                        logger.warning("Redundancy checker failed. Proceeding with potentially duplicate tasks.")

                # --- START: NEW COORDINATED PRUNING AND SAFETY CHECK ---
                # Step 3: Check if space is needed and trigger a coordinated prune.
                num_to_commit = len(approved_proposals)
                if num_to_commit > 0:
                    await self._prune_task_graph_if_needed(required_space=num_to_commit)

                    # Step 4: CRITICAL SAFETY CHECK. If pruning failed, discard everything for this cycle.
                    if (len(self.registry.tasks) + num_to_commit) > self.max_total_tasks:
                        logger.error(
                            f"Pruning did not create enough space for {num_to_commit} new tasks. "
                            "Discarding proposals for this cycle to prevent graph overgrowth."
                        )
                        approved_proposals = []  # This is the key change
                        span.set_status(trace.Status(StatusCode.ERROR, "Pruning failed to create sufficient space"))
                # --- END: NEW COORDINATED PRUNING AND SAFETY CHECK ---

                # Step 5: Filter out proposals whose parents were pruned.
                final_proposals_to_commit = []
                if approved_proposals:
                    for proposal in approved_proposals:
                        parent_id = parent_map.get(proposal.local_id)
                        if parent_id and parent_id in self.registry.tasks:
                            final_proposals_to_commit.append(proposal)
                        else:
                            logger.warning(
                                f"Discarding proposal '{proposal.desc[:50]}...' because its parent task '{parent_id}' was pruned."
                            )
                    approved_proposals = final_proposals_to_commit

                # Step 6: Commit the final, filtered list.
                if approved_proposals:
                    logger.info(f"Committing {len(approved_proposals)} approved sub-tasks to the graph.")
                    self._commit_proposals(approved_proposals, parent_map)
                    span.set_attribute("proposals.approved_count", len(approved_proposals))
                else:
                    logger.info("No proposals were approved or committed in this cycle.")

            finally:
                self.proposed_tasks_buffer = []

    async def _process_executor_output_for_verification(
        self, ctx: TaskContext, result: str
    ):
        """
        Implements the multi-stage failure analysis chain.
        MODIFIED: If all autonomous attempts (standard + grace) are exhausted,
        it will ALWAYS escalate to the user with a question instead of failing.
        """
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)

        if not verify_task_result:
            # Handle cases where the verifier itself failed
            t.status = "failed"
            t.result = "Verification agent failed to produce a valid assessment."
            self._broadcast_state_update(t)
            return

        if verify_task_result.get_successful():
            t.result = result
            logger.info(f"COMPLETED [{t.id}]")
            t.status = "complete"
            self._broadcast_state_update(t)
            t.last_llm_history, t.agent_role_paused = None, None
            return

        # --- Standard Retry Logic (Unchanged) ---
        t.fix_attempts += 1
        self._broadcast_state_update(t)

        if t.fix_attempts < t.max_fix_attempts:
            t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
            logger.info(
                f"[{t.id}] Retrying execution (Attempt {t.fix_attempts + 1}). Feedback: {verify_task_result.reason[:50]}...")
            self._broadcast_state_update(t)
            return

        # --- Grace Attempt Logic (Unchanged) ---
        if t.grace_attempts < self.max_grace_attempts:
            logger.info(
                f"[{t.id}] Max standard attempts reached. Consulting retry analyst for a grace attempt...")
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\n\n{str(t.executor_llm_history)}\n\nDecide if one final autonomous grace attempt is viable."

            decision = await self._run_stateless_agent(self.retry_analyst, analyst_prompt, ctx)

            if decision and decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion
                logger.info(f"[{t.id}] Grace attempt granted by analyst. Next step: {decision.next_step_suggestion}")
                self._broadcast_state_update(t)
                return  # Exit to perform the grace attempt.

        # --- NEW: MANDATORY ESCALATION TO HUMAN ---
        # This code is now the final step, reached only when all autonomous attempts are exhausted.
        # It replaces the previous logic that would mark the task as failed.
        logger.warning(
            f"[{t.id}] All autonomous attempts exhausted. Escalating to human operator for guidance."
        )
        t.span.add_event("AllAutonomousAttemptsExhausted", {"reason": "Escalating to human."})

        # Formulate the question for the user.
        formulator_prompt = (
            f"Context: The task '{t.desc}' has failed after multiple autonomous attempts with these verification scores: {t.verification_scores}. "
            f"The final reason for failure was: '{verify_task_result.reason}'.\n\n"
            f"Your job is to formulate the SINGLE most critical, concise, and clear question for the human operator that will unblock this task. "
            f"Do not add any preamble. Just ask the question."
        )

        question_output = await self._run_stateless_agent(self.question_formulator, formulator_prompt, ctx)

        if isinstance(question_output, UserQuestion):
            # Successfully formulated a question, now pause the task.
            logger.info(f"[{t.id}] Escalation successful. Task pausing for user response.")
            t.current_question = question_output
            t.agent_role_paused = "question_formulator"
            t.status = "waiting_for_user_response"
            self._broadcast_state_update(t)
            return
        else:
            # This is the new TRUE failure condition: the agent couldn't even formulate a question.
            error_msg = "All recovery options exhausted, and failed to formulate a question for the operator."
            logger.error(f"[{t.id}] {error_msg}")
            t.span.set_status(trace.Status(StatusCode.ERROR, error_msg))
            t.status = "failed"
            t.result = error_msg
            self._broadcast_state_update(t)
            t.last_llm_history, t.agent_role_paused = None, None
            return

    async def _verify_task(self, t: Task, candidate_result: str) -> Optional[verification]:
        prompt_parts = [
            f"Your job is to verify if the 'Candidate Result' accurately and completely addresses the 'Original Task'.",
            f"\n--- Original Task ---\n{t.desc}"
        ]

        if t.human_directive:
            prompt_parts.append(f"\n--- Operator's Corrective Directive ---\n{t.human_directive}")
        if t.user_response:
            prompt_parts.append(f"\n--- Operator's Answer to a Question ---\n{t.user_response}")

        prompt_parts.append(f"\n--- Candidate Result ---\n{candidate_result}")
        prompt_parts.append(
            "\n\nDoes the result, considering any operator directives, fully and accurately complete the task? Be strict.")

        ver_prompt = "\n".join(prompt_parts)

        vout = await self._run_stateless_agent(self.verifier, ver_prompt, TaskContext(t.id, self.registry))

        if not isinstance(vout, verification):
            logger.error(f"[{t.id}] Verifier returned an invalid type or None. Assigning score 0.")
            return verification(
                reason="Verifier agent failed to produce a valid 'verification' model output.",
                message_for_user="Verification failed due to an internal agent error.",
                score=0,
            )

        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(
            f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'"
        )
        return vout
