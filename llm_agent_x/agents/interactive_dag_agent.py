# llm_agent_x/agents/interactive_dag_agent.py

import asyncio
import heapq
import json
import logging
import os
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import List, Tuple, Union, Dict, Any, Optional, Callable

import pika
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from phoenix.trace.schemas import SpanAttributes
from pydantic import BaseModel
from pydantic_ai.agent import AgentRunResult, Agent, CallToolsNode, ModelRequestNode
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerSSE
from pydantic_ai.messages import (
    ToolCallPart,
    ToolReturnPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    SystemPromptPart,
    UserPromptPart,
)
from pydantic_ai.result import FinalResult
from pydantic_graph import End

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext

from dotenv import load_dotenv

from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Tracer, Span, StatusCode
from openinference.semconv.trace import SpanAttributes

from llm_agent_x.agents.dag_agent import (
    DAGAgent,
)
from llm_agent_x.backend.utils import generate_hash
from llm_agent_x.core import (
    Task,
    UserQuestion,
    DocumentState,
    verification,
    RetryDecision,
    InformationNeedDecision,
    PruningDecision,
    DependencySelection,
    NewSubtask,
    ExecutionPlan,
    ProposedSubtask,
    AdaptiveDecomposerResponse,
    TaskForMerging,
    MergedTask,
    MergingDecision,
    ProposalResolutionPlan,
    ContextualAnswer,
    RedundancyDecision,
    TaskDescription,
    TaskChain,
    ChainedExecutionPlan,
    HumanInjectedDependency,
    Interrupt,
    HumanDirectiveInterrupt,
    AgentMessageInterrupt,
)
from llm_agent_x.managers import CommunicationManager
from llm_agent_x.state_manager.abstract_state_manager import TaskContext

from llm_agent_x.backend.extract_json_from_text import extract_json
from llm_agent_x.state_manager import InMemoryStateManager, AbstractStateManager

load_dotenv(".env", override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("InteractiveDAGAgent")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")


def agent_pika_heartbeat_thread(connection_ref, lock_ref, shutdown_event_ref):
    """Heartbeat thread specifically for the agent's publisher connection."""
    while not shutdown_event_ref.is_set():
        try:
            connection = connection_ref()
            if connection and connection.is_open:
                with lock_ref:
                    connection.process_data_events()
            time.sleep(10)
        except Exception as e:
            logger.error(f"Error in Agent Pika heartbeat thread: {e}")
            break
    logger.info("Agent Pika heartbeat thread shutting down.")


class InteractiveDAGAgent(DAGAgent):
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        kwargs.pop("min_question_priority", None)

        self.max_total_tasks = kwargs.pop("max_total_tasks", 25)
        self.max_dependencies_per_task = kwargs.pop("max_dependencies_per_task", 7)

        state_manager = InMemoryStateManager(broadcast_callback=self._broadcast_state_update)

        super().__init__(*args, state_manager=state_manager, setup_agent_roles=False, **kwargs)

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
        self._publisher_connection_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._consumer_thread: Optional[threading.Thread] = None
        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.proposed_tasks_buffer = []
        self.proposed_task_dependencies_buffer = []

        self.comms_manager = CommunicationManager(
            state_manager=self.state_manager,
        )

        self.update_notebook_tool = self._create_notebook_tool(
            self.state_manager, self._broadcast_state_update
        )
        self._setup_agent_roles()

    def _setup_agent_roles(self):
        """Initializes or reinitializes all agent roles with the current self._tools_for_agents."""
        llm_model = self.base_llm_model

        planner_system_prompt = (
            "You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. "
            "Crucially, you must also assign `tags` to each new task to define its capabilities and allow other agents to communicate with it."
            "\n\n**TAGGING RULES:**"
            "\n1. **Be Specific:** Tags should describe the task's function (e.g., 'research', 'writing', 'outreach')."
            "\n2. **Group Related Tasks:** Tasks that need to collaborate or share information should have a common, unique tag (e.g., 'venue_selection', 'budget_approval_q2')."
            "\n3. **Reuse Existing Tags:** A list of 'EXISTING TAGS' is provided. Reuse these where appropriate to maintain consistency."
            "\n\n**PLANNING RULES:**"
            "\n1. **Use Documents:** Check 'AVAILABLE DOCUMENTS' first. If a document provides needed info, create a task that USES it and add the document's ID to `deps`."
            "\n2. **Create Chains:** Group sequential steps into a 'chain'. All chains run in parallel."
            "\n\n**EXAMPLE:**"
            "\n- Objective: 'Plan the conference venue and marketing.'"
            "\n- EXISTING TAGS: ['finance', 'planning']"
            "\n- **CORRECT ACTION:** Create a plan with tasks like:"
            "\n  - `{'desc': 'Research potential venues', 'tags': ['planning', 'venue_selection']}`"
            "\n  - `{'desc': 'Finalize venue choice', 'tags': ['planning', 'venue_selection']}` (depends on the first task)"
            "\n  - `{'desc': 'Draft marketing copy', 'tags': ['marketing', 'copywriting']}` (runs in parallel)"
        )

        self.initial_planner = Agent(
            model=llm_model,
            system_prompt=planner_system_prompt,
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
            tools=self._tools_for_agents,
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
            output_type=str,
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
            output_type=ChainedExecutionPlan,
            tools=self._tools_for_agents,
        )

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
            tools=self._tools_for_agents,
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
            tools=self._tools_for_agents,
            mcp_servers=self.mcp_servers,
        )

        self.verifier = Agent(
            model=llm_model,
            system_prompt="You are a meticulous quality assurance verifier. Your job is to check if the provided 'Candidate Result' accurately and completely addresses the 'Task', considering the history and user instructions. Output JSON matching the 'verification' schema. Be strict.",
            output_type=verification,
            tools=self._tools_for_agents,
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
        t = ctx.task if ctx else None
        if t:
            logger.info(f"[{t.id}] Running stateless agent: {agent.__class__.__name__}")

        params = ModelRequestParameters()
        output_is_model = isinstance(agent.output_type, type) and issubclass(
            agent.output_type, BaseModel
        )

        all_instructions = list(agent._system_prompts)

        if output_is_model:
            params.output_tools = [
                ToolDefinition(
                    name="output",
                    description="The structured output response.",
                    parameters_json_schema=agent.output_type.model_json_schema(),
                )
            ]
            schema_instruction = (
                f"\n\n--- OUTPUT FORMAT ---\n"
                f"You MUST respond using the 'output' tool with a JSON object that strictly adheres to the following JSON Schema:\n"
                f"```json\n{json.dumps(agent.output_type.model_json_schema(), indent=2)}\n```"
            )
            all_instructions.append(schema_instruction)

        all_instructions.append("\n\nUse the `output` tool.")

        messages = [
            ModelRequest(
                parts=[UserPromptPart(user_prompt)],
                instructions="\n\n".join(all_instructions),
            ),
        ]

        try:
            resp = await model_request(
                agent.model, messages, model_request_parameters=params
            )

            class DummyResult:
                def __init__(self, usage_data): self._usage = usage_data
                def usage(self): return self._usage

            if t:
                self._add_llm_data_to_span(t.span, DummyResult(resp.usage), t)

            if output_is_model:
                logger.info(f"One shot response: {resp.parts[0]}")
                if hasattr(resp.parts[0], "tool_name"):
                    if resp.parts and resp.parts[0].tool_name == "output":
                        args_as_dict = resp.parts[0].args_as_dict()
                        output = agent.output_type(**args_as_dict)
                        logger.info(f"One shot response (pydantic-ified): {output}")
                        return output
                    else:
                        logger.error(f"[{t.id if t else 'System'}] Stateless agent failed to produce structured output.")
                        return None
                else:
                    response_text = resp.parts[0].content
                    json_ = extract_json(response_text)
                    if json_:
                        output = agent.output_type(**json_)
                        logger.info(f"One shot response (json): {output}")
                        return output
            else:
                return resp.parts[0].content if resp.parts else None

        except Exception as e:
            logger.error(f"[{t.id if t else 'System'}] Exception in _run_stateless_agent: {e}", exc_info=True)
            if t:
                t.span.record_exception(e)
            return None

    def _add_llm_data_to_system_span(self, span: Span, agent_res: AgentRunResult):
        if not span or not agent_res:
            return
        usage = agent_res.usage()
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))

    def _create_notebook_tool(self, state_manager: AbstractStateManager, broadcast_callback: Callable):
            # --- THE FIX: Make the tool function `async def` ---
            async def update_notebook_tool_instance(task_id: str, updates: Dict[str, Any]) -> str:
                """
    Updates the shared notebook for a given task.
    Provide key-value pairs in the 'updates' dictionary.
    Set a value to 'null' (Python `None`) to delete a key from the notebook.
                """
                task = state_manager.get_task(task_id)
                if not task:
                    return f"Error: Task {task_id} not found."

                updated_keys, deleted_keys = [], []
                for key, value in updates.items():
                    if value is None:
                        if key in task.shared_notebook:
                            del task.shared_notebook[key]
                            deleted_keys.append(key)
                    else:
                        task.shared_notebook[key] = value
                        updated_keys.append(key)

                state_manager.upsert_task(task)

                result_msg = f"Notebook for task {task_id} updated. Updated: {', '.join(updated_keys) or 'None'}. Deleted: {', '.join(deleted_keys) or 'None'}."
                logger.info(result_msg)
                return result_msg

            return update_notebook_tool_instance

    def _get_publisher_channel(self) -> pika.adapters.blocking_connection.BlockingChannel:
        if self._publisher_connection is None or self._publisher_connection.is_closed:
            logger.info("Establishing new publisher RabbitMQ connection and channel.")
            self._publisher_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST, heartbeat=60))
            self._publish_channel = self._publisher_connection.channel()
            self._publish_channel.exchange_declare(exchange=self.STATE_UPDATES_EXCHANGE, exchange_type="fanout")
        return self._publish_channel

    def _broadcast_state_update(self, task: Task):
        try:
            with self._publisher_connection_lock:
                channel = self._get_publisher_channel()
                message = task.model_dump_json(exclude={"span", "otel_context", "last_llm_history", "executor_llm_history"})
                channel.basic_publish(exchange=self.STATE_UPDATES_EXCHANGE, routing_key="", body=message)
        except Exception as e:
            logger.error(f"Failed to broadcast state update for task {task.id}: {e}", exc_info=False)

    def _broadcast_execution_log_update(self, task_id: str, log_entry: Dict[str, Any]):
        """Helper to broadcast a single execution log entry."""
        task = self.state_manager.get_task(task_id)
        if task:
            log_entry['timestamp'] = datetime.now(timezone.utc).isoformat()

            task.execution_log.append(log_entry)
            # We broadcast the whole task so the UI can update its state.
            # This is more robust than sending partial updates.
            self._broadcast_state_update(task)

    def _format_stream_node(self, node: Any) -> Optional[Dict[str, Any]]:
        def truncate(s: Any, length: int = 250) -> str:
            s_str = str(s)
            return s_str if len(s_str) <= length else s_str[: length - 3] + "..."

        if isinstance(node, CallToolsNode):
            for part in node.model_response.parts:
                if isinstance(part, TextPart):
                    content = part.content.strip()
                    if content:
                        return {"type": "thought", "content": truncate(content)}
                elif isinstance(part, ToolCallPart):
                    return {"type": "tool_call", "tool_name": part.tool_name, "args": part.args_as_dict()}
        elif isinstance(node, ModelRequestNode):
            for part in node.request.parts:
                if isinstance(part, ToolReturnPart):
                    return {"type": "tool_result", "tool_name": part.tool_name, "result": truncate(part.content)}
        elif isinstance(node, End):
            if isinstance(node.data, FinalResult):
                return {"type": "final_answer", "content": node.data.output}
        return None

    async def _handle_agent_output(self, ctx: TaskContext, agent_res: AgentRunResult, expected_output_type: Any,
                                   agent_role_name: str) -> Tuple[bool, Any]:
        t = ctx.task
        self._add_llm_data_to_span(t.span, agent_res, t)
        t.last_llm_history = agent_res.all_messages()
        if agent_role_name == "executor":
            t.executor_llm_history = agent_res.all_messages_json()
        actual_output = agent_res.output

        if isinstance(actual_output, UserQuestion):
            logger.info(
                f"Task [{t.id}] agent '{agent_role_name}' is asking a human question: {actual_output.question[:80]}...")
            t.current_question = actual_output
            t.agent_role_paused = agent_role_name
            t.status = "waiting_for_user_response"
            self._broadcast_state_update(t)
            return True, actual_output  # Return True to indicate the task is paused

        if not isinstance(actual_output, (BaseModel, str)):
            logger.warning(
                f"Task [{t.id}] received unexpected output from '{agent_role_name}'. Expected {expected_output_type}, got {type(actual_output).__name__}.")

        return False, actual_output

    async def _escalate_to_human_question(self, ctx: TaskContext, final_failure_reason: str):
        """Handles the final escalation path when all autonomous attempts are exhausted."""
        t = ctx.task
        logger.warning(
            f"[{t.id}] All autonomous attempts exhausted. Escalating to human. Reason: {final_failure_reason}")

        # Use the last verification result if available for better context
        last_reason = t.verification_history[-1].reason if t.verification_history else final_failure_reason

        scores_only = [v.score for v in t.verification_history]

        formulator_prompt = f"Context: Task '{t.desc}' failed with scores {scores_only}. Final failure reason: '{last_reason}'. Formulate the SINGLE most critical question for the human operator to unblock this."
        question_output = await self._run_stateless_agent(self.question_formulator, formulator_prompt, ctx)

        if isinstance(question_output, UserQuestion):
            t.current_question = question_output
            t.agent_role_paused = "question_formulator"
            t.status = "waiting_for_user_response"
        else:
            error_msg = "All recovery options exhausted, and failed to formulate a question for the operator."
            t.status = "failed"
            t.result = error_msg
            t.agent_role_paused = None

        self.state_manager.upsert_task(t)

    async def _prune_task_graph_if_needed(self, required_space: int = 0):
        tasks = self.state_manager.get_all_tasks()
        current_size = len(tasks)
        limit = self.max_total_tasks
        num_to_prune = max(0, (current_size + required_space) - limit)

        if num_to_prune == 0:
            return

        logger.warning(f"Graph size ({current_size}) + proposed ({required_space}) exceeds limit ({limit}). Attempting to prune {num_to_prune} task(s).")

        with self.tracer.start_as_current_span("GraphPruning") as span:
            prunable_statuses = {"pending", "paused_by_human", "failed", "cancelled"}
            all_eligible_tasks = [t for t in tasks.values() if t.status in prunable_statuses and t.parent is not None and not t.is_critical and t.counts_toward_limit]

            downstream_dependents: Dict[str, List[str]] = {t.id: [] for t in tasks.values()}
            for task in tasks.values():
                for dep_id in task.deps:
                    if dep_id in downstream_dependents:
                        downstream_dependents[dep_id].append(task.id)

            safe_candidates = [task for task in all_eligible_tasks if not downstream_dependents.get(task.id)]
            pruning_candidates, pruner_prompt = [], ""

            if safe_candidates:
                pruning_candidates = safe_candidates
                pruner_prompt = f"Select at least {num_to_prune} LEAST critical tasks for removal from SAFE candidates:\n" + "\n".join([f"- ID: {t.id}, Status: {t.status}, Desc: {t.desc}" for t in pruning_candidates])
            else:
                logger.warning("No safe leaf-node tasks for pruning. Falling back to risky pruning.")
                pruning_candidates = all_eligible_tasks
                pruner_prompt = f"NO SAFE TASKS. Select at least {num_to_prune} LEAST critical tasks for removal from RISKY candidates:\n" + "\n".join([f"- ID: {t.id}, Status: {t.status}, Desc: {t.desc}, Required by: {downstream_dependents.get(t.id, [])}" for t in pruning_candidates])

            if not pruning_candidates:
                logger.error("Pruning required, but no eligible tasks found.")
                return

            valid_candidate_ids = {t.id for t in pruning_candidates}
            pruning_decision = await self._run_stateless_agent(self.graph_pruner, pruner_prompt, ctx=None)

            if not pruning_decision or not isinstance(pruning_decision, PruningDecision) or not pruning_decision.tasks_to_prune:
                logger.error("Graph pruner failed to return a valid decision.")
                span.set_status(trace.Status(StatusCode.ERROR, "Pruner returned invalid output"))
                return

            for task_to_prune in pruning_decision.tasks_to_prune:
                if task_to_prune.task_id not in valid_candidate_ids:
                    logger.error(f"LLM proposed pruning invalid task ID '{task_to_prune.task_id}'. Rejecting plan.")
                    span.set_status(trace.Status(StatusCode.ERROR, "LLM proposed invalid task to prune"))
                    return

            pruned_ids = set()
            for task_to_prune in pruning_decision.tasks_to_prune:
                await self._prune_specific_task(task_id=task_to_prune.task_id, reason=f"Graph manager: {task_to_prune.reason}", new_status="pruned")
                pruned_ids.add(task_to_prune.task_id)

            await self._prune_orphaned_tasks()
            span.set_attribute("graph.tasks.pruned_count", len(pruned_ids))
            logger.info(f"Pruning complete. Removed {len(pruned_ids)} tasks.")

    async def _prune_orphaned_tasks(self):
        logger.info("Running orphan task cleanup...")
        tasks = self.state_manager.get_all_tasks()
        all_task_ids = set(tasks.keys())
        tasks_with_dependents = set()
        for task in tasks.values():
            tasks_with_dependents.update(task.deps)

        orphaned_ids = [tid for tid in all_task_ids if tid not in tasks_with_dependents and (task := tasks[tid]) and task.status != "complete" and task.parent is not None and tid not in self.inflight]

        if not orphaned_ids:
            logger.info("No orphaned tasks found to prune.")
            return

        logger.warning(f"Found {len(orphaned_ids)} orphaned tasks to prune: {orphaned_ids}")
        for orphan_id in orphaned_ids:
            await self._prune_specific_task(task_id=orphan_id, reason="Orphaned: No tasks depend on this.", new_status="cancelled")

    async def _prune_specific_task(self, task_id: str, reason: str, new_status: str = "cancelled", override_critical=False):
        task = self.state_manager.get_task(task_id)
        if not task:
            logger.warning(f"Attempted to prune non-existent task: {task_id}")
            return

        logger.info(f"Pruning task [{task_id}] with status '{new_status}'. Reason: {reason}")
        if task.is_critical and not override_critical:
            logger.error(f"FATAL: Attempted to prune critical task [{task_id}]. BLOCKED.")
            return

        UNPRUNABLE_STATUSES = {"running", "proposing", "planning", "complete"}
        if task.status in UNPRUNABLE_STATUSES:
            logger.error(f"FATAL: Attempted to prune task [{task_id}] in unprunable state '{task.status}'. BLOCKED.")
            return

        for child_id in list(task.children):
            await self._prune_specific_task(child_id, f"Cascading from parent {task_id}", new_status)

        if task.parent and (parent := self.state_manager.get_task(task.parent)):
            if task_id in parent.children:
                parent.children.remove(task_id)
                self.state_manager.upsert_task(parent)

        for other_task in self.state_manager.get_all_tasks().values():
            if task_id in other_task.deps:
                other_task.deps.remove(task_id)
                self.state_manager.upsert_task(other_task)

        if task_id in self.task_futures:
            self.task_futures[task_id].cancel()

        task.status, task.result = new_status, f"❌ CANCELLED: {reason}"
        self._broadcast_state_update(task) # Send final status update

        self.state_manager.delete_task(task_id)
        self.inflight.discard(task_id)
        logger.info(f"Task [{task_id}] completely pruned.")

    def _listen_for_directives_target(self):
        logger.info("Consumer thread starting for directives...")
        try:
            connection_params = pika.ConnectionParameters(RABBITMQ_HOST, heartbeat=60)
            self._consumer_connection_for_thread = pika.BlockingConnection(connection_params)
            self._consumer_channel_for_thread = self._consumer_connection_for_thread.channel()
            self._consumer_channel_for_thread.queue_declare(queue=self.DIRECTIVES_QUEUE, durable=True)
            self._consumer_channel_for_thread.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body):
                message = json.loads(body)
                if self._main_event_loop and self._main_event_loop.is_running():
                    self._main_event_loop.call_soon_threadsafe(self.directives_queue.put_nowait, message)
                ch.basic_ack(delivery_tag=method.delivery_tag)

            self._consumer_channel_for_thread.basic_consume(queue=self.DIRECTIVES_QUEUE, on_message_callback=callback)
            logger.info("Consumer thread now consuming directives.")
            self._consumer_channel_for_thread.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Consumer thread AMQP Connection Error: {e}")
        except Exception as e:
            logger.error(f"Consumer thread stopped unexpectedly: {e}", exc_info=True)
        finally:
            if self._consumer_connection_for_thread and self._consumer_connection_for_thread.is_open:
                self._consumer_connection_for_thread.close()
            logger.info("Consumer thread exited.")

    async def _handle_directive(self, directive: dict):
        command = directive.get("command")
        task_id = directive.get("task_id")
        payload = directive.get("payload")

        logger.info(f"Dispatcher received command '{command}' for task '{task_id}'")

        # --- System-Level Commands (Graph/State Manipulation) ---
        # These are handled directly by the worker and do not go into the task's interrupt queue.

        if command == "ADD_ROOT_TASK":
            payload = directive.get("payload", {})
            if desc := payload.get("desc"):
                new_task = Task(
                    id=str(uuid.uuid4())[:8],
                    desc=desc,
                    needs_planning=payload.get("needs_planning", True),
                    status="pending",
                    mcp_servers=payload.get("mcp_servers", []),
                    tags=set(payload.get("tags", [])), # <-- Handle tags on creation
                )
                self.state_manager.upsert_task(new_task)
                logger.info(f"Added new root task: {new_task.id} - {new_task.desc}")
            return

        if command == "ADD_DOCUMENT":
            payload = directive.get("payload", {})
            if (name := payload.get("name")) and (content := payload.get("content")) is not None:
                doc_state = DocumentState(content=content)
                new_doc_task = Task(
                    id=str(uuid.uuid4())[:8],
                    desc=f"Document: {name}",
                    task_type="document",
                    document_state=doc_state,
                    status="complete",
                    is_critical=True,
                    counts_toward_limit=False,
                )
                self.state_manager.upsert_task(new_doc_task)
                logger.info(f"Added new document node: {new_doc_task.id} - {name}")
            return

        if command == "RESET_STATE":
            logger.warning("Received RESET_STATE directive. Wiping all tasks.")
            for future in self.task_futures.values(): future.cancel()
            self.task_futures.clear()
            self.inflight.clear()
            current_tasks = list(self.state_manager.get_all_tasks().keys())
            for tid in current_tasks:
                self.state_manager.delete_task(tid)
            for tid, task_data in payload.items():
                try:
                    self.state_manager.upsert_task(Task(**task_data))
                except Exception as e:
                    logger.error(f"Failed to load task {tid} from state file: {e}")
            logger.info(f"Successfully loaded {len(payload)} tasks. Broadcasting all.")
            for task in self.state_manager.get_all_tasks().values():
                self._broadcast_state_update(task)
            return

        # --- All subsequent commands require a valid task_id ---
        task = self.state_manager.get_task(task_id)
        if not task:
            logger.warning(f"Directive '{command}' for unknown task '{task_id}' ignored.")
            return

        if command == "UPDATE_DOCUMENT" and task.task_type == "document" and task.document_state:
            if new_name := payload.get("name"):
                task.desc = f"Document: {new_name}"
            if (new_content := payload.get("content")) is not None and generate_hash(new_content) != task.document_state.content_hash:
                task.document_state = DocumentState(content=new_content, version=task.document_state.version + 1)
            self.state_manager.upsert_task(task)
            return

        if command == "DELETE_DOCUMENT" and task.task_type == "document":
            await self._prune_specific_task(task_id, reason="Document deleted by operator.", new_status="pruned", override_critical=True)
            return

        if command == "PAUSE":
            task.status = "paused_by_human"
            if task_id in self.task_futures and not self.task_futures[task_id].done():
                self.task_futures[task_id].cancel()
            self.state_manager.upsert_task(task)
            return

        if command == "RESUME":
            if task.status == "paused_by_human":
                task.status = "pending" # Reset to be picked up by scheduler
                self.state_manager.upsert_task(task)
            return

        if command == "CANCEL":
            task.status, task.result = "cancelled", f"Cancelled by operator: {payload or 'No reason given.'}"
            if task_id in self.task_futures: self.task_futures[task_id].cancel()
            self.state_manager.upsert_task(task)
            await self._reset_downstream_tasks(task_id)
            return

        if command == "TERMINATE":
            task.status, task.result = "failed", f"Terminated by operator: {payload}"
            if task_id in self.task_futures: self.task_futures[task_id].cancel()
            self.state_manager.upsert_task(task)
            await self._reset_downstream_tasks(task_id)
            return

        if command == "PRUNE_TASK":
            await self._prune_specific_task(task_id, reason=payload or "Pruned by operator.", new_status="pruned")
            await self._prune_orphaned_tasks()
            return

        if command == "MANUAL_OVERRIDE":
            task.status, task.result = "complete", payload
            if task_id in self.task_futures: self.task_futures[task_id].cancel()
            task.fix_attempts = task.grace_attempts = 0
            task.verification_history, task.last_llm_history, task.execution_log = [], None, []
            self.state_manager.upsert_task(task)
            await self._reset_downstream_tasks(task_id)
            return

        if command == "INJECT_DEPENDENCY":
            if task.status == "waiting_for_user_response":
                source_task_id = payload.get("source_task_id")
                depth = payload.get("depth", "shallow")
                if source_task_id and (source_task := self.state_manager.get_task(source_task_id)):
                    if source_task.status == 'complete':
                        task.human_injected_deps.append(HumanInjectedDependency(source_task_id=source_task_id, depth=depth))
                        task.status = "pending" # Reset to be picked up by scheduler
                        task.current_question = task.agent_role_paused = None
                        logger.info(f"Injected dependency {source_task_id} into task {task_id}. Resetting.")
                    else:
                        logger.warning(f"Cannot inject from non-complete task {source_task_id}.")
                else:
                    logger.warning(f"Could not find source task {source_task_id}.")
                self.state_manager.upsert_task(task)
            else:
                logger.warning(f"Cannot inject dependency into task {task_id} not waiting for user response.")
            return

        # --- Task-Level Interrupts (Context for the Task's LLM) ---
        # These are converted into Interrupt objects and placed in the task's queue.

        if command in ["REDIRECT", "ANSWER_QUESTION"]:
            if task.status in ["complete", "failed", "cancelled", "pruned"]:
                logger.warning(f"Cannot send directive '{command}' to inactive task {task_id}.")
                return

            interrupt = HumanDirectiveInterrupt(command=command, payload=payload)
            heapq.heappush(task.interrupt_queue, interrupt)
            logger.info(f"Queued '{command}' interrupt for task {task_id}.")

            # If the task was waiting for an answer, this new context unblocks it.
            if task.status == "waiting_for_user_response":
                task.status = "pending"
                task.current_question = None
                task.agent_role_paused = None

            self.state_manager.upsert_task(task)
            return

        logger.warning(f"Unknown or unhandled command '{command}' for task {task_id}.")

    async def _reset_downstream_tasks(self, task_id: str):
        logger.info(f"Resetting downstream tasks of {task_id}")
        downstream_ids_to_reset, q = set(), deque([task_id])
        tasks = self.state_manager.get_all_tasks()

        while q:
            current_id = q.popleft()
            for task in tasks.values():
                if current_id in task.deps and task.id not in downstream_ids_to_reset:
                    downstream_ids_to_reset.add(task.id)
                    q.append(task.id)

        if not downstream_ids_to_reset: return

        for tid in downstream_ids_to_reset:
            if (task_to_reset := self.state_manager.get_task(tid)) and task_to_reset.status != "pending":
                task_to_reset.status = "pending"
                task_to_reset.result = None
                task_to_reset.fix_attempts = task_to_reset.grace_attempts = 0
                task_to_reset.verification_history = []
                # task_to_reset.human_directive = task_to_reset.user_response = task_to_reset.current_question = task_to_reset.last_llm_history = task_to_reset.agent_role_paused = None
                self.state_manager.upsert_task(task_to_reset)

    def _get_all_existing_tags(self) -> List[str]:
        """Scans all tasks and returns a sorted list of unique tags."""
        all_tags = set()
        for task in self.state_manager.get_all_tasks().values():
            all_tags.update(task.tags)
        return sorted(list(all_tags))

    async def run(self):
        self._main_event_loop = asyncio.get_running_loop()
        self._consumer_thread = threading.Thread(target=self._listen_for_directives_target, daemon=True)
        self._consumer_thread.start()

        try:
            self._publisher_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST, heartbeat=60))
            self._publish_channel = self._publisher_connection.channel()
            self._publish_channel.exchange_declare(exchange=self.STATE_UPDATES_EXCHANGE, exchange_type="fanout")
            logger.info("Publisher RabbitMQ connection initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize publisher RabbitMQ: {e}", exc_info=True)
            self._shutdown_event.set()
            return

        for task in self.state_manager.get_all_tasks().values():
            self._broadcast_state_update(task)

        try:
            with self.tracer.start_as_current_span("InteractiveDAGAgent.run"):
                while not self._shutdown_event.is_set():
                    if not self._consumer_thread.is_alive():
                        logger.error("Consumer thread has died. Shutting down agent worker.")
                        break

                    while not self.directives_queue.empty():
                        await self._handle_directive(await self.directives_queue.get())

                    tasks = self.state_manager.get_all_tasks()
                    all_task_ids = set(tasks.keys())

                    executable_statuses = {"pending", "running", "planning", "proposing", "waiting_for_children"}
                    non_executable_tasks = {t.id for t in tasks.values() if t.status not in executable_statuses and not (t.status == "waiting_for_user_response")}

                    pending_tasks_for_scheduling = all_task_ids - non_executable_tasks
                    ready_to_run_ids = set()

                    for tid in pending_tasks_for_scheduling:
                        task = self.state_manager.get_task(tid)
                        if not task: continue

                        # Condition 1: Is it a 'pending' task whose dependencies are met?
                        if task.status == "pending" and all(
                                (dep_task := self.state_manager.get_task(d)) and dep_task.status == "complete" for d in
                                task.deps):
                            ready_to_run_ids.add(tid)

                        # --- THIS IS THE RESTORED BLOCK ---
                        # Condition 2: Is it a parent task waiting for its children to finish?
                        elif task.status == "waiting_for_children" and all(
                                (child_task := self.state_manager.get_task(c)) and child_task.status in ["complete",
                                                                                                         "failed"] for c
                                in task.children):
                            # Check if any child failed.
                            if any((child_task := self.state_manager.get_task(c)) and child_task.status == "failed" for
                                   c in task.children):
                                if task.status != "failed":
                                    task.status, task.result = "failed", "A child task failed, cannot synthesize results."
                                    self.state_manager.upsert_task(task)
                            else:
                                # All children succeeded. Wake up the parent to run its own execution (synthesis).
                                ready_to_run_ids.add(tid)
                                task.status = "pending"  # Set to pending so it gets picked up to run.
                                self.state_manager.upsert_task(task)

                    ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

                    if not ready and not self.inflight and self.directives_queue.empty():
                        await asyncio.sleep(0.1)
                        continue

                    for tid in ready:
                        if self.state_manager.get_task(tid):
                            self.inflight.add(tid)
                            self.task_futures[tid] = asyncio.create_task(self._run_taskflow(tid))

                    if self.task_futures:
                        done, _ = await asyncio.wait(list(self.task_futures.values()), timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                        for fut in done:
                            tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                            if tid:
                                self.inflight.discard(tid)
                                del self.task_futures[tid]
                                try: fut.result()
                                except asyncio.CancelledError: logger.info(f"Task [{tid}] was cancelled.")
                                except Exception as e:
                                    logger.error(f"Task [{tid}] future failed: {e}", exc_info=True)
                                    if t := self.state_manager.get_task(tid):
                                        if t.status != "failed":
                                            t.status, t.result = "failed", f"Execution failed: {e}"
                                            self.state_manager.upsert_task(t)
                    await asyncio.sleep(0.05)
        finally:
            logger.info("Agent run loop shutting down.")
            self._shutdown_event.set()
            if self._publisher_connection and self._publisher_connection.is_open:
                self._publisher_connection.close()
            if self._consumer_thread and self._consumer_thread.is_alive():
                if self._consumer_connection_for_thread and self._consumer_connection_for_thread.is_open:
                    self._consumer_connection_for_thread.close()
                self._consumer_thread.join(timeout=2)

    async def _run_taskflow(self, tid: str):
        t = self.state_manager.get_task(tid)
        if not t: return
        ctx = TaskContext(tid, self.state_manager)

        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)

            if t.status in ["paused_by_human", "waiting_for_user_response"]:
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                def update_status(new_status):
                    if t.status != new_status:
                        t.status = new_status
                        self.state_manager.upsert_task(t)

                if t.status == "waiting_for_children":
                    update_status("running")
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    update_status("planning")
                    await self._run_initial_planning(ctx)
                    if t.status not in ["paused_by_human", "waiting_for_user_response"]:
                        t.already_planned = True
                        update_status("waiting_for_children" if t.children else "complete")
                elif t.can_request_new_subtasks and t.status != "proposing":
                    update_status("proposing")
                    was_frozen = await self._run_adaptive_decomposition(ctx)
                    if not was_frozen and t.status not in ["paused_by_human", "waiting_for_user_response"]:
                        update_status("waiting_for_children")
                elif t.status in ["pending", "running"]:
                    update_status("running")
                    await self._run_task_execution(ctx)

                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    update_status("complete")
                    t.last_llm_history, t.agent_role_paused = None, None

            except asyncio.CancelledError:
                if t.status not in ["waiting_for_user_response"]:
                    t.status = "paused_by_human"
                self.state_manager.upsert_task(t)
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                t.last_llm_history = t.agent_role_paused = None
                self.state_manager.upsert_task(t)
                raise
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", t.status)
                self.state_manager.upsert_task(t) # Final state save

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning for: {t.desc}")

        all_tasks = self.state_manager.get_all_tasks()
        context_str = "\n".join([f"- ID: {tk.id} Desc: {tk.desc}" for tk in all_tasks.values() if tk.status == "complete" and tk.id != t.id])

        existing_tags = self._get_all_existing_tags()
        tags_str = f"\nEXISTING TAGS (reuse these if possible:\n{', '.join(existing_tags)}" if existing_tags else "\nNo existing tags."

        prompt_parts = [f"Objective: {t.desc}", tags_str, f"\nAvailable completed data sources:\n{context_str}", f"\n--- Shared Notebook ---\n{self._format_notebook_for_llm(t)}"]
        # self._inject_and_clear_user_response(prompt_parts, t)

        plan_res = await self.initial_planner.run(user_prompt="\n".join(prompt_parts), message_history=t.last_llm_history)
        is_paused, chained_plan = await self._handle_agent_output(ctx, plan_res, ChainedExecutionPlan, "initial_planner")

        if is_paused or not chained_plan or not chained_plan.task_chains:
            if not is_paused: logger.info(f"[{t.id}] Planning complete. No subtasks needed.")
            return

        logger.info(f"[{t.id}] Planner proposed {len(chained_plan.task_chains)} chain(s).")
        local_to_global_id_map, all_new_task_descriptions = {}, []

        for chain_obj in chained_plan.task_chains:
            previous_task_global_id_in_chain = None
            for task_desc in chain_obj.chain:
                all_new_task_descriptions.append(task_desc)
                new_task = Task(id=str(uuid.uuid4())[:8], desc=task_desc.desc, parent=t.id, can_request_new_subtasks=task_desc.can_request_new_subtasks, mcp_servers=t.mcp_servers, tags=task_desc.tags)
                self.state_manager.upsert_task(new_task)
                t.children.append(new_task.id)
                local_to_global_id_map[task_desc.local_id] = new_task.id

                if previous_task_global_id_in_chain:
                    self.state_manager.add_dependency(new_task.id, previous_task_global_id_in_chain)
                previous_task_global_id_in_chain = new_task.id

        for task_desc in all_new_task_descriptions:
            if new_task_global_id := local_to_global_id_map.get(task_desc.local_id):
                for dep_id in task_desc.deps:
                    if dep_global_id := local_to_global_id_map.get(dep_id) or (dep_id if self.state_manager.get_task(dep_id) else None):
                        self.state_manager.add_dependency(new_task_global_id, dep_global_id)
                    else:
                        logger.warning(f"[{t.id}] Planner specified unknown dependency '{dep_id}'. Ignored.")

        t.status = "waiting_for_children"
        self.state_manager.upsert_task(t)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task
        if len(self.state_manager.get_all_tasks()) >= self.max_total_tasks:
            logger.warning(f"[{t.id}] Task graph is full. Adaptive decomposition is FROZEN.")
            t.status = "running"
            self.state_manager.upsert_task(t)
            return True

        t.dep_results = {d: (dep.result if (dep := self.state_manager.get_task(d)) else None) for d in t.deps}
        prompt_parts = [f"Your complex task is: {t.desc}", f"\nDependency Results:\n{json.dumps(t.dep_results, indent=2)}", "\nBreak this down if necessary."]
        # self._inject_and_clear_user_response(prompt_parts, t)

        proposals_res = await self.adaptive_decomposer.run(user_prompt="\n".join(prompt_parts), message_history=t.last_llm_history)
        is_paused, chained_plan = await self._handle_agent_output(ctx, proposals_res, ChainedExecutionPlan, "adaptive_decomposer")

        if is_paused or not chained_plan or not chained_plan.task_chains:
            if not is_paused: logger.info(f"[{t.id}] Decomposer proposed no new tasks.")
            return False

        logger.info(f"[{t.id}] Decomposer proposed {len(chained_plan.task_chains)} new chain(s).")
        for chain_obj in chained_plan.task_chains:
            previous_task_id_in_chain = None
            for task_desc in chain_obj.chain:
                new_task = Task(id=str(uuid.uuid4())[:8], desc=task_desc.desc, parent=t.id, can_request_new_subtasks=task_desc.can_request_new_subtasks, mcp_servers=t.mcp_servers, deps=set(task_desc.deps))
                self.state_manager.upsert_task(new_task)
                t.children.append(new_task.id)
                if previous_task_id_in_chain:
                    self.state_manager.add_dependency(new_task.id, previous_task_id_in_chain)
                previous_task_id_in_chain = new_task.id
        self.state_manager.upsert_task(t)
        return False

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running task execution for: {t.desc}")

        # --- 1. CONSTRUCT BASE PROMPT (UNCHANGED) ---
        # This section builds the initial context from dependencies and sub-tasks.
        # It runs only once at the beginning of the execution flow.
        children_ids = set(t.children)
        prunable_deps_ids = t.deps - children_ids
        final_deps_to_use = children_ids.copy()
        if len(prunable_deps_ids) > self.max_dependencies_per_task:
            logger.warning(
                f"Task [{t.id}] has {len(prunable_deps_ids)} prunable dependencies, exceeding limit. Selecting...")
            pruning_candidates_prompt = "\n".join(
                [f"- ID: {dep_id}, Description: {self.state_manager.get_task(dep_id).desc}" for dep_id in
                 prunable_deps_ids if self.state_manager.get_task(dep_id)])
            pruner_prompt = f"Objective: '{t.desc}'\n\nSelect ONLY the most critical dependencies (max {self.max_dependencies_per_task}):\n{pruning_candidates_prompt}"
            selection = await self._run_stateless_agent(self.dependency_pruner, pruner_prompt, ctx)
            if selection and selection.approved_dependency_ids:
                final_deps_to_use.update(selection.approved_dependency_ids)
                logger.info(
                    f"Pruned dependencies to {len(selection.approved_dependency_ids)}. Reasoning: {selection.reasoning}")
            else:
                import random
                logger.error(f"[{t.id}] Dependency pruner failed. Using random subset.")
                final_deps_to_use.update(
                    random.sample(list(prunable_deps_ids), min(self.max_dependencies_per_task, len(prunable_deps_ids))))
        else:
            final_deps_to_use.update(prunable_deps_ids)

        child_results = {child.desc: child.result for cid in t.children if
                         (child := self.state_manager.get_task(cid)) and child.status == 'complete' and child.result}
        pruned_dep_ids = final_deps_to_use - children_ids
        dep_results = {}
        for did in pruned_dep_ids:
            dep_task = self.state_manager.get_task(did)
            if not dep_task or dep_task.status != 'complete':
                continue
            if dep_task.task_type == "document" and dep_task.document_state:
                dep_results[dep_task.desc] = dep_task.document_state.content
            elif dep_task.result is not None:
                dep_results[dep_task.desc] = dep_task.result

        prompt_lines = [f"Your task is: {t.desc}\n"]

        if t.human_injected_deps:
            prompt_lines.append("\n--- CONTEXT MANUALLY PROVIDED BY OPERATOR ---\n")
            logger.info(f"Task [{t.id}] has {len(t.human_injected_deps)} injected dependencies.")
            for injected_dep in t.human_injected_deps:
                source_task = self.state_manager.get_task(injected_dep.source_task_id)
                if not source_task: continue
                if injected_dep.depth == "shallow":
                    prompt_lines.append(
                        f"- From injected shallow dependency '{source_task.desc}':\n{source_task.result}\n")
                elif injected_dep.depth == "deep":
                    history_str = json.dumps(source_task.executor_llm_history,
                                             indent=2) if source_task.executor_llm_history else "Not available."
                    prompt_lines.append(
                        f"- From injected DEEP dependency '{source_task.desc}':\n  - Result: {source_task.result}\n  - Execution History: {history_str}\n")
            prompt_lines.append("-------------------------------------------\n\n")
            t.human_injected_deps = []
            self.state_manager.upsert_task(t)

        if child_results:
            prompt_lines.append("\nSynthesize results from sub-tasks:\n" + "\n".join(
                [f"- From '{desc}':\n{res}\n" for desc, res in child_results.items()]))
        if dep_results:
            prompt_lines.append("\nUse data from critical dependencies:\n" + "\n".join(
                [f"- From '{desc}':\n{res}\n" for desc, res in dep_results.items()]))
        prompt_base_content = "".join(prompt_lines)

        # --- 2. SETUP EXECUTOR AGENT (UNCHANGED) ---
        task_specific_mcp_clients = []
        if t.mcp_servers:
            for server_config in t.mcp_servers:
                try:
                    if server_config.get("type") == "streamable_http":
                        task_specific_mcp_clients.append(MCPServerStreamableHTTP(server_config["address"]))
                except Exception as e:
                    logger.error(f"[{t.id}]: Failed to create MCP client for {server_config.get('name')}: {e}")

        task_executor = Agent(
            model=self.base_llm_model,
            system_prompt=self.executor_system_prompt,
            output_type=str,
            tools=list(self._tools_for_agents),
            mcp_servers=task_specific_mcp_clients,
        )
        self.comms_manager.apply_to_agent(task_executor)

        # --- 3. SMARTLY CLEAR LOGS ---
        # **FIX**: Only clear logs on a true "cold start" (no previous LLM history), not on every retry.
        if not t.last_llm_history:
            t.execution_log = []
            self.state_manager.upsert_task(t)

        # --- 4. MAIN EXECUTION & RETRY LOOP ---
        while True:
            t.comm_token_bucket.refill()
            self.state_manager.upsert_task(t)

            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})
            attempt_marker = {"type": "thought", "content": f"--- Starting Execution Attempt #{current_attempt} ---"}
            self._broadcast_execution_log_update(t.id, attempt_marker)

            current_prompt_parts = [prompt_base_content]

            # **FIX**: Process interrupts from the queue to form the next prompt's context.
            if t.interrupt_queue:
                interrupt = heapq.heappop(t.interrupt_queue)
                logger.info(f"Task [{t.id}] processing interrupt (Priority: {interrupt.priority}).")
                current_prompt_parts.insert(0, f"{interrupt.get_interrupt_prompt()}\n\n")
                # This clears the queue for this turn and is persisted
                self.state_manager.upsert_task(t)

            # Legacy support for the old directive system
            elif t.human_directive:
                current_prompt_parts.insert(0,
                                            f"--- HIGH PRIORITY CONTEXT ---\n{t.human_directive}\n------------------------\n\n")
                t.human_directive = None

            current_prompt = "".join(current_prompt_parts)
            logger.info(f"[{t.id}] Attempt {current_attempt}: Streaming executor actions...")
            exec_res, result = None, None

            try:
                # **FIX**: The agent now runs with the memory of its last attempt.
                async with task_executor.iter(
                        user_prompt=current_prompt,
                        message_history=t.last_llm_history,
                        deps=ctx
                ) as agent_run:
                    async for node in agent_run:
                        if formatted_node := self._format_stream_node(node):
                            self._broadcast_execution_log_update(t.id, formatted_node)

                # If the loop completes without exception, the run was successful.
                exec_res = agent_run.result

            except Exception as e:
                logger.error(f"[{t.id}] Exception during agent.iter: {e}", exc_info=True)
                error_node = {"type": "error",
                              "content": f"Execution error: {str(e)}\n\nTraceback: {traceback.format_exc()}"}
                self._broadcast_execution_log_update(t.id, error_node)

            # **FIX**: ALWAYS capture the latest history, no matter the outcome.
            if exec_res:
                t.last_llm_history = exec_res.all_messages()
                self.state_manager.upsert_task(t)  # Persist the memory

                is_paused, output = await self._handle_agent_output(ctx, exec_res, Union[str, UserQuestion], "executor")
                if is_paused:
                    return  # Exit if a question was asked
                result = output

            if result is None:
                # This path is taken if exec_res was None (due to an error) or if the agent returned nothing.
                t.human_directive = "Your last attempt failed or produced no output. Please re-evaluate your approach and try again, paying close attention to the original goal and provided context."
                t.fix_attempts += 1
                self.state_manager.upsert_task(t)
                if (t.fix_attempts + t.grace_attempts) > (t.max_fix_attempts + self.max_grace_attempts):
                    await self._escalate_to_human_question(ctx, "Executor returned no valid output after all retries.")
                    if t.status in ["failed", "waiting_for_user_response"]:
                        return
                continue  # Retry, carrying over the last_llm_history.

            # If we have a result, proceed to verification.
            await self._process_executor_output_for_verification(ctx, result)

            # If verification passed or the task failed/paused, exit the loop.
            if t.status in ["complete", "failed", "paused_by_human", "waiting_for_user_response"]:
                return

    async def _process_executor_output_for_verification(self, ctx: TaskContext, result: str):
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)

        if not verify_task_result:
            t.status, t.result = "failed", "Verification agent failed."
            self.state_manager.upsert_task(t)
            return

        if verify_task_result.get_successful():
            t.result, t.status = result, "complete"
            t.agent_role_paused = None
            self.state_manager.upsert_task(t)
            return

        t.fix_attempts += 1
        self.state_manager.upsert_task(t)

        if t.fix_attempts < t.max_fix_attempts:
            t.human_directive = f"Last answer insufficient. Reason: {verify_task_result.reason}\nRetry."
            self.state_manager.upsert_task(t)
            return

        if t.grace_attempts < self.max_grace_attempts:
            score_history_text = [f"Score: {v.score}, Reason: {v.reason}" for v in t.verification_history]
            analyst_prompt = f"Task: '{t.desc}'\nScores: {score_history_text}\nHistory: {t.executor_llm_history}\n\nDecide if one final autonomous grace attempt is viable."
            decision = await self._run_stateless_agent(self.retry_analyst, analyst_prompt, ctx)
            if decision and decision.should_retry:
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion
                self.state_manager.upsert_task(t)
                return

        # Replace the old question formulator logic with a call to the new helper
        await self._escalate_to_human_question(ctx, verify_task_result.reason)

    async def _verify_task(self, t: Task, candidate_result: str) -> Optional[verification]:
        prompt_parts = [f"Job: Verify if 'Candidate Result' completes 'Original Task'.", f"\n--- Original Task ---\n{t.desc}"]

        prompt_parts.append(f"\n--- Candidate Internal Log ---\n{t.executor_llm_history}")
        if t.human_directive: prompt_parts.append(f"\n--- Operator's Directive ---\n{t.human_directive}")
        # if t.user_response: prompt_parts.append(f"\n--- Operator's Answer ---\n{t.user_response}")
        prompt_parts.extend([f"\n--- Candidate Result ---\n{candidate_result}", "\n\nDoes the result, considering directives, complete the task? Be strict."])

        vout = await self._run_stateless_agent(self.verifier, "\n".join(prompt_parts), TaskContext(t.id, self.state_manager))

        if not isinstance(vout, verification):
            logger.error(f"[{t.id}] Verifier returned invalid type or None. Score 0.")
            return verification(reason="Verifier agent failed.", message_for_user="Verification failed internally.", score=0)

        t.verification_history.append(vout)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout