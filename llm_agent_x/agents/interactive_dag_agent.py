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
from pydantic_ai.messages import ToolCallPart, ToolReturnPart, ModelResponse, ModelResponsePart, TextPart
from pydantic_ai.result import FinalResult
from pydantic_graph import End

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
)

from dotenv import load_dotenv

from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Tracer, Span, StatusCode
from openinference.semconv.trace import SpanAttributes

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

        self.initial_planner = Agent(
            model=llm_model,
            system_prompt="You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. You can link tasks to pre-existing completed tasks. For each new sub-task, decide if it is complex enough to merit further dynamic decomposition by setting `can_request_new_subtasks` to true.",
            output_type=ExecutionPlan,  # MODIFIED: Removed Union[..., UserQuestion]
            tools=self._tools_for_agents,  # Use the internal tools list
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
        self.adaptive_decomposer = Agent(
            model=llm_model,
            system_prompt="You are an adaptive expert. Analyze the given task and results of its dependencies. If the task is still too complex, propose a list of new, more granular sub-tasks to achieve it. You MUST provide an `importance` score (1-100) for each proposal, reflecting how critical it is.",
            output_type=AdaptiveDecomposerResponse,  # MODIFIED: Removed Union[..., UserQuestion]
            tools=self._tools_for_agents,  # Use the internal tools list
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
            tools=self._tools_for_agents,  # Use the internal tools list
            retries=3,
        )

        self.graph_pruner = Agent(
            model=llm_model,
            system_prompt=(
                "You are a ruthless project manager responsible for keeping a project on budget. The total number of tasks has exceeded the limit. "
                "You will be given a list of all PENDING tasks. Your sole job is to identify and select the LEAST critical, most redundant, or lowest-impact "
                "tasks for removal. You must provide a clear reason for each pruning decision."
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

        self._agent_role_map = {
            "initial_planner": self.initial_planner,
            "cycle_breaker": self.cycle_breaker,
            "adaptive_decomposer": self.adaptive_decomposer,
            "conflict_resolver": self.conflict_resolver,
            "graph_pruner": self.graph_pruner,
            "dependency_pruner": self.dependency_pruner,
            "executor": self.executor,
            "verifier": self.verifier,
            "retry_analyst": self.retry_analyst,
            "information_gap_detector": self.information_gap_detector,
            "question_formulator": self.question_formulator,
        }

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

        # ADD_ROOT_TASK logic is separate and remains the same
        if command == "ADD_ROOT_TASK":
            # ... (no changes here) ...
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

        # All other directives require a task_id
        task = self.registry.tasks.get(task_id)
        if not task:
            logger.warning(f"Directive for unknown task {task_id} ignored.")
            return

        logger.info(f"Handling command '{command}' for task {task_id}")
        payload = directive.get("payload")

        # Cancel in-flight task future if a directive changes its state
        if task_id in self.task_futures and not self.task_futures[task_id].done():
            self.task_futures[task_id].cancel()
            logger.info(
                f"Cancelled in-flight asyncio.Task for {task_id} due to directive."
            )

        original_status = task.status
        if command == "PAUSE":
            task.status = "paused_by_human"
        elif command == "RESUME":
            task.status = "pending"  # Simply set to pending, the loop will pick it up
        elif command == "TERMINATE":
            task.status = "failed"
            task.result = f"Terminated by operator: {payload}"
        elif command == "REDIRECT":
            task.status = "pending"
            task.human_directive = payload
        elif command == "MANUAL_OVERRIDE":
            task.status = "complete"
            task.result = payload
        elif command == "ANSWER_QUESTION":
            if task.status == "waiting_for_user_response":
                logger.info(f"Task {task_id} received answer: {str(payload)[:50]}...")
                # --- THIS IS THE KEY CHANGE ---
                # Place the answer into the human_directive field.
                task.human_directive = str(payload)
                # Reset the task to be re-run by the main scheduler.
                task.status = "pending"
                # Clear the question-related fields.
                task.current_question = None
                task.agent_role_paused = None
            else:
                logger.warning(f"Task {task_id} not waiting for a question, ignoring ANSWER_QUESTION.")

        # Consolidate state reset and broadcast logic
        if command in ["TERMINATE", "REDIRECT", "MANUAL_OVERRIDE"]:
             self._reset_task_and_dependents(task_id)
             task.result = payload if command == "MANUAL_OVERRIDE" else task.result
             task.human_directive = payload if command == "REDIRECT" else None
             task.current_question = None
             task.agent_role_paused = None
             task.last_llm_history = None

        if task.status != original_status or command in ["REDIRECT", "ANSWER_QUESTION", "MANUAL_OVERRIDE"]:
            self._broadcast_state_update(task)

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
                    # Process any directives received from the consumer thread
                    while not self.directives_queue.empty():
                        directive = await self.directives_queue.get()
                        await self._handle_directive(directive)

                    all_task_ids = set(self.registry.tasks.keys())
                    executable_statuses = {
                        "pending",
                        "running",
                        "planning",
                        "proposing",
                        "waiting_for_children",
                    }
                    # Include 'waiting_for_user_response' for scheduling purposes when a user_response is present
                    # This only happens if a human explicitly answered a question via a directive.
                    non_executable_tasks = {
                        t.id
                        for t in self.registry.tasks.values()
                        if t.status not in executable_statuses
                        and not (
                            t.status == "waiting_for_user_response"
                            and t.user_response is not None
                        )
                    }

                    for tid in all_task_ids - non_executable_tasks:
                        task = self.registry.tasks[tid]
                        if any(
                            self.registry.tasks.get(d, {}).status == "failed"
                            for d in task.deps
                        ):
                            if task.status != "failed":
                                task.status, task.result = (
                                    "failed",
                                    "Upstream dependency failed.",
                                )
                                self._broadcast_state_update(task)
                                task.last_llm_history = None
                                task.agent_role_paused = None

                    pending_tasks_for_scheduling = {
                        t.id
                        for t in self.registry.tasks.values()
                        if t.status in executable_statuses
                        or (
                            t.status == "waiting_for_user_response"
                            and t.user_response is not None
                        )
                    }

                    ready_to_run_ids = set()
                    for tid in pending_tasks_for_scheduling:
                        task = self.registry.tasks[tid]
                        # Condition 1: Task is waiting for user response, but a response is present (human-provided)
                        if (
                            task.status == "waiting_for_user_response"
                            and task.user_response is not None
                            and task.last_llm_history is not None
                            and task.agent_role_paused is not None
                        ):
                            ready_to_run_ids.add(tid)
                        # Condition 2: Task is pending AND all dependencies are complete (ready for initial run or next step after planning)
                        elif task.status == "pending" and all(
                            self.registry.tasks.get(d, {}).status == "complete"
                            for d in task.deps
                        ):
                            ready_to_run_ids.add(tid)
                        # Condition 3: Parent task waiting for children, and all children are complete/failed
                        elif task.status == "waiting_for_children" and all(
                            self.registry.tasks[c].status in ["complete", "failed"]
                            for c in task.children
                        ):
                            if any(
                                self.registry.tasks[c].status == "failed"
                                for c in task.children
                            ):
                                task.status, task.result = (
                                    "failed",
                                    "A child task failed, cannot synthesize.",
                                )
                                self._broadcast_state_update(task)
                                task.last_llm_history = None
                                task.agent_role_paused = None
                            else:
                                ready_to_run_ids.add(tid)
                                task.status = "pending"  # Transition to pending for execution/synthesis
                                self._broadcast_state_update(task)

                    ready = [
                        tid for tid in ready_to_run_ids if tid not in self.inflight
                    ]

                    # Check for graceful shutdown conditions.
                    # The loop continues if there's work for agents, or if the consumer thread is still active
                    # (meaning more directives could arrive).
                    has_pending_interaction = any(
                        t.status == "waiting_for_user_response"
                        and t.user_response is None
                        for t in self.registry.tasks.values()
                    )
                    if (
                        not ready
                        and not self.inflight
                        and not has_pending_interaction
                        and self.directives_queue.empty()
                    ):  # Check internal queue for directives processed in this loop

                        # If no internal work or pending interaction, check if external input is still possible.
                        if (
                            self._shutdown_event.is_set()
                            or not self._consumer_thread.is_alive()
                        ):
                            # Consumer thread is either signaled to stop, or already dead.
                            # So no new directives will come in. It's safe to shut down.
                            logger.info(
                                "Interactive DAG execution complete (no pending, no inflight, no user interaction, no directives, consumer thread inactive/stopping)."
                            )
                            break
                        else:
                            # Agent is idle, but consumer thread is still alive and waiting for external directives.
                            # So, we should continue looping and waiting.
                            logger.debug(
                                "Agent idle, but consumer thread active. Waiting for directives..."
                            )

                    for tid in ready:
                        self.inflight.add(tid)
                        self.task_futures[tid] = asyncio.create_task(
                            self._run_taskflow(tid)
                        )
                        self._broadcast_state_update(self.registry.tasks[tid])

                    if self.task_futures:
                        done, pending_futures = await asyncio.wait(
                            list(self.task_futures.values()),
                            timeout=0.1,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for fut in done:
                            tid = next(
                                (
                                    tid
                                    for tid, f in self.task_futures.items()
                                    if f == fut
                                ),
                                None,
                            )
                            if tid:
                                self.inflight.discard(tid)
                                del self.task_futures[tid]
                                try:
                                    fut.result()
                                except asyncio.CancelledError:
                                    logger.info(
                                        f"Task [{tid}] was cancelled by operator or system."
                                    )
                                except Exception as e:
                                    logger.error(f"Task [{tid}] future failed: {e}")
                                    t = self.registry.tasks.get(tid)
                                    if t and t.status != "failed":
                                        t.status = "failed"
                                        self._broadcast_state_update(t)
                                        t.last_llm_history = None
                                        t.agent_role_paused = None

                    if self.proposed_tasks_buffer:
                        await self._run_global_resolution()

                    await asyncio.sleep(0.05)
        finally:
            logger.info(
                "Agent run loop is shutting down. Signaling consumer thread to stop and closing publisher connection."
            )

            # Signal consumer thread to stop
            self._shutdown_event.set()
            if self._consumer_thread and self._consumer_thread.is_alive():
                self._consumer_thread.join(
                    timeout=5
                )  # Wait for consumer to finish gracefully
                if self._consumer_thread.is_alive():
                    logger.warning(
                        "Consumer thread did not shut down gracefully after 5 seconds."
                    )

            # Close publisher connection
            if self._publisher_connection and self._publisher_connection.is_open:
                logger.info("Closing publisher RabbitMQ connection.")
                self._publisher_connection.close()
            self._publisher_connection = None
            self._publish_channel = None

    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)

            if t.status in ["paused_by_human", "waiting_for_user_response"]:
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution until directive received.")
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                def update_status_and_broadcast(new_status):
                    if t.status != new_status:
                        t.status = new_status
                        self._broadcast_state_update(t)

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
                    await self._run_adaptive_decomposition(ctx)
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
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning for: {t.desc}")

        completed_tasks = [
            tk
            for tk in self.registry.tasks.values()
            if tk.status == "complete" and tk.id != t.id
        ]
        context_str = "\n".join(
            f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks
        )
        prompt_parts = [
            f"Objective: {t.desc}",
            f"\n\nAvailable completed data sources:\n{context_str}",
            f"\n\n--- Current Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
            self._get_notebook_tool_guidance(t, "initial_planner"),
        ]

        if t.human_directive:
            prompt_parts.append(
                f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            )
            t.human_directive = None
            self._broadcast_state_update(t)  # ADDED: Broadcast cleared directive

        full_prompt = "\n".join(prompt_parts)

        plan_res = await self.initial_planner.run(
            user_prompt=full_prompt, message_history=t.last_llm_history
        )
        # _handle_agent_output no longer pauses for UserQuestion.
        # If the LLM produces unexpected output, it will be treated as non-successful.
        is_paused, initial_plan = await self._handle_agent_output(
            ctx=ctx,
            agent_res=plan_res,
            expected_output_type=ExecutionPlan,
            agent_role_name="initial_planner",
        )
        if initial_plan is None:  # Treat invalid/empty plan as a failure
            t.status = "failed"
            self._broadcast_state_update(t)
            logger.error(
                f"[{t.id}] Initial planner returned no plan or invalid output."
            )
            return

        if initial_plan.needs_subtasks and initial_plan.subtasks:
            fixer_prompt = f"Analyze and fix cycles in this plan:\n\n{initial_plan.model_dump_json(indent=2)}"
            # Cycle breaker does not get notebook update guidance
            fixed_plan_res = await self.cycle_breaker.run(
                user_prompt=fixer_prompt, message_history=t.last_llm_history
            )
            # MODIFIED: is_paused_cb check removed here
            is_paused_cb, plan = await self._handle_agent_output(
                ctx=ctx,
                agent_res=fixed_plan_res,
                expected_output_type=ExecutionPlan,
                agent_role_name="cycle_breaker",
            )
            if plan is None:  # Treat invalid/empty fixed plan as a failure
                t.status = "failed"
                self._broadcast_state_update(t)
                logger.error(
                    f"[{t.id}] Cycle breaker returned no plan or invalid output."
                )
                return
        else:
            plan = initial_plan

        if not plan.needs_subtasks:
            t.status = "complete"
            t.already_planned = True
            self._broadcast_state_update(t)
            t.last_llm_history = None
            t.agent_role_paused = None
            return

        await self._process_initial_planning_output(ctx, plan)
        # MODIFIED: Removed "waiting_for_user_response"
        if t.status not in ["paused_by_human"]:
            t.already_planned = True
            t.status = "waiting_for_children" if t.children else "complete"
            if t.status == "complete":
                t.last_llm_history = None
                t.agent_role_paused = None
            self._broadcast_state_update(t)

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
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        available_tasks = str(
            [
                {"id": task.id, "desc": task.desc}
                for task in self.registry.tasks.values()
            ]
        )
        prompt_parts = [
            f"Task: {t.desc}",
            f"\n\nResults from dependencies:\n{t.dep_results}",
            f"\n\n--- Available tasks to request results from: {available_tasks}",
        ]

        if t.human_directive:
            prompt_parts.append(
                f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            )
            t.human_directive = None
            self._broadcast_state_update(t)

        full_prompt = "\n".join(prompt_parts)

        proposals_res = await self.adaptive_decomposer.run(
            user_prompt=full_prompt, message_history=t.last_llm_history
        )
        is_paused, proposals_response = await self._handle_agent_output(
            ctx=ctx,
            agent_res=proposals_res,
            expected_output_type=AdaptiveDecomposerResponse,
            agent_role_name="adaptive_decomposer",
        )
        if proposals_response is None:
            t.status = "failed"
            self._broadcast_state_update(t)
            logger.error(
                f"[{t.id}] Adaptive decomposer returned no proposals or invalid output."
            )
            return

        if proposals_response.tasks:
            # FIXED: Get length from the .tasks attribute
            logger.info(
                f"Task [{t.id}] proposing {len(proposals_response.tasks)} new sub-tasks."
            )
            # FIXED: Iterate over the .tasks attribute
            for sub in proposals_response.tasks:
                self.proposed_tasks_buffer.append((sub, t.id))

        if proposals_response.dep_requests:
            logger.info(
                f"Task [{t.id}] requesting results from {len(proposals_response.dep_requests)} new dependencies."
            )
            for dep in proposals_response.dep_requests:
                self.proposed_task_dependencies_buffer.append((dep, t.id))

        if t.status not in ["paused_by_human"]:
            t.status = "waiting_for_children"
            self._broadcast_state_update(t)

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running task execution for: {t.desc}")

        # --- DEPENDENCY PRUNING LOGIC ---
        children_ids = set(t.children)
        prunable_deps_ids = t.deps - children_ids
        final_deps_to_use = children_ids.copy()

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

            pruner_res = await self.dependency_pruner.run(user_prompt=pruner_prompt)
            selection = pruner_res.output

            if selection and selection.approved_dependency_ids:
                approved_ids = set(selection.approved_dependency_ids)
                final_deps_to_use.update(approved_ids)
                logger.info(
                    f"Pruned dependencies down to {len(approved_ids)} for this run. Reasoning: {selection.reasoning}")
                t.span.set_attribute("dependencies.pruned_count", len(prunable_deps_ids) - len(approved_ids))
            else:
                logger.error(f"[{t.id}] Dependency pruner failed. Using a random subset of dependencies as a fallback.")
                import random
                # Ensure we don't try to sample more items than exist
                sample_size = min(self.max_dependencies_per_task, len(prunable_deps_ids))
                fallback_deps = set(random.sample(list(prunable_deps_ids), sample_size))
                final_deps_to_use.update(fallback_deps)
        else:
            final_deps_to_use.update(prunable_deps_ids)

        # --- PROMPT BUILDING ---
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

    async def _process_executor_output_for_verification(
        self, ctx: TaskContext, result: str
    ):
        """
        Overrides the base verification logic to implement a multi-stage, interactive failure analysis chain,
        prioritizing human escalation before autonomous retries.
        """
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)

        if verify_task_result.get_successful():
            t.result = result
            logger.info(f"COMPLETED [{t.id}]")
            t.status = "complete"
            self._broadcast_state_update(t)
            t.last_llm_history, t.agent_role_paused = None, None
            return

        t.fix_attempts += 1

        # Are we still in the standard, pre-analysis retry phase?
        if (t.fix_attempts < t.max_fix_attempts):
            t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
            logger.info(f"[{t.id}] Retrying execution (Attempt {t.fix_attempts + 1}). Feedback: {verify_task_result.reason[:50]}...")
            self._broadcast_state_update(t)
            return

        # --- BEGIN FAILURE ANALYSIS CHAIN (triggered after max standard attempts) ---
        logger.info(f"[{t.id}] Max standard attempts reached. Initiating failure analysis chain...")

        # STEP A: Check for Information Need (First Priority)
        logger.info(f"[{t.id}] (Step A) Consulting information gap detector...")
        detector_prompt = f"Task History:\nTask: '{t.desc}'\nScores: {t.verification_scores}\nIs this failure fundamentally due to missing info that a human must provide?"
        detector_res = await self.information_gap_detector.run(user_prompt=detector_prompt, message_history=t.last_llm_history)
        info_decision = detector_res.output

        if info_decision and info_decision.is_needed:
            logger.info(f"[{t.id}] Information gap detected. Formulating question for operator...")
            formulator_prompt = f"Context: The task '{t.desc}' has failed with scores {t.verification_scores}. A root cause analysis determined it's missing key information. Formulate the single, most critical question to ask the human."
            question_res = await self.question_formulator.run(user_prompt=formulator_prompt, message_history=t.last_llm_history)

            is_paused, _ = await self._handle_agent_output(
                ctx=ctx, agent_res=question_res, expected_output_type=UserQuestion, agent_role_name="question_formulator"
            )
            if is_paused:
                logger.info(f"[{t.id}] Escalation successful. Task is now 'waiting_for_user_response'.")
                return # SUCCESSFUL EXIT: Task is paused awaiting human input.

        # STEP B: Decide on Autonomous Retry (Second Priority)
        # This code only runs if the information detector said 'is_needed: False'.
        if t.grace_attempts < self.max_grace_attempts:
            logger.info(f"[{t.id}] (Step B) Information gap ruled out. Consulting retry analyst for grace attempt...")
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\nAn information gap was ruled out. Decide if one final autonomous grace attempt is viable."
            decision_res = await self.retry_analyst.run(user_prompt=analyst_prompt, message_history=t.last_llm_history)
            decision = decision_res.output

            if decision and decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion
                logger.info(f"[{t.id}] Grace attempt granted by analyst. Next step: {decision.next_step_suggestion}")
                self._broadcast_state_update(t)
                return  # SUCCESSFUL EXIT: Perform the grace attempt.

        # STEP C: Final Failure (Last Resort)
        # This code only runs if Step A was False AND Step B was False.
        error_msg = f"Exceeded max attempts for task '{t.id}'. All recovery options exhausted."
        t.span.set_status(trace.Status(StatusCode.ERROR, error_msg))
        t.status = "failed"
        self._broadcast_state_update(t)
        t.last_llm_history, t.agent_role_paused = None, None
        raise Exception(error_msg)

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
        # Verifier does not get notebook update guidance

        vres = await self.verifier.run(
            user_prompt=ver_prompt, message_history=t.last_llm_history
        )
        # _handle_agent_output no longer pauses for UserQuestion.
        is_paused, vout = await self._handle_agent_output(
            ctx=TaskContext(t.id, self.registry),
            agent_res=vres,
            expected_output_type=verification,
            agent_role_name="verifier",
        )
        if vout is None:  # Treat invalid/empty verification as score 0 (worst)
            logger.error(
                f"[{t.id}] Verifier returned no output or invalid output. Assigning score 0."
            )
            return verification(
                reason="Verifier returned no valid output.",
                message_for_user="Verification failed due to internal error.",
                score=0,
            )

        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(
            f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'"
        )
        return vout
