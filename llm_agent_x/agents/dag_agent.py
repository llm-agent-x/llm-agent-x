import asyncio
import hashlib
import uuid
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from hashlib import md5
from os import getenv
import json  # NEW: For pretty-printing tool schemas
from typing import Set, Dict, Any, Optional, List, Tuple, Union, Callable, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.mcp import (
    MCPServerStreamableHTTP,
)  # NEW: Import MCPServerStreamableHTTP
from dotenv import load_dotenv

# --- OpenTelemetry Imports ---
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Tracer, Span, StatusCode
from openinference.semconv.trace import SpanAttributes

# --- Basic Setup ---
load_dotenv(".env", override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DAGAgent")


def get_cost(usage) -> float:
    if not usage:
        return 0.0
    # Using GPT-4o-mini pricing for example
    input_cost = 0.15 / 1_000_000
    output_cost = 0.60 / 1_000_000
    return (input_cost * getattr(usage, "request_tokens", 0)) + (
        output_cost * getattr(usage, "response_tokens", 0)
    )


# --- Pydantic Models ---
class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best).")

    def get_successful(self):
        return self.score > 5


class RetryDecision(BaseModel):
    """The decision on whether to attempt another fix for a failing task."""

    should_retry: bool = Field(
        description="Set to true if the trend of scores suggests success is likely."
    )
    reason: str = Field(
        description="A brief explanation for the decision, citing the score trend."
    )
    next_step_suggestion: str = Field(
        description="A specific, actionable suggestion for the next attempt to improve the result."
    )


class InformationNeedDecision(BaseModel):
    """The decision on whether a task's failure is due to a critical lack of information."""

    is_needed: bool = Field(
        description="Set to true ONLY if the task is fundamentally impossible to complete without specific external information from a human."
    )
    reason: str = Field(
        description="A brief explanation for the decision, justifying why the information gap is or is not the root cause."
    )


class TaskToPrune(BaseModel):
    """A task that is a candidate for pruning."""

    task_id: str = Field(description="The unique ID of the task to be pruned.")
    reason: str = Field(
        description="A brief justification for why this task is the least critical."
    )


class PruningDecision(BaseModel):
    """The final decision on which tasks to prune from the graph."""

    tasks_to_prune: List[TaskToPrune] = Field(
        description="A list of tasks that have been selected for removal."
    )


class UserQuestion(BaseModel):
    """
    A specific output type for agents to ask clarifying questions to the human operator.
    The task will pause until the human provides an answer.
    """

    question: str = Field(
        description="The clarifying question to ask the human operator."
    )
    priority: int = Field(
        description="A numerical score from 1 (lowest) to 10 (highest) representing the urgency or criticality of getting an answer.",
        ge=1,
        le=10,
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional context or structured data relevant to the question.",
    )


# Models for Initial, Top-Down Planning (The "Architect")
class Dependency(BaseModel):
    reason: str
    local_id: str


class DependencySelection(BaseModel):
    """The decision on which dependencies are most critical for a task."""

    reasoning: str = Field(description="A brief explanation of the selection strategy.")
    approved_dependency_ids: List[str] = Field(
        description="A list of the most critical dependency task IDs to use."
    )


class NewSubtask(BaseModel):
    local_id: str
    desc: str
    deps: List[Dependency] = Field(default_factory=list)
    can_request_new_subtasks: bool = Field(
        False,
        description="Set to true ONLY for complex, integrative, or uncertain tasks that might need further dynamic decomposition later.",
    )


class ExecutionPlan(BaseModel):
    needs_subtasks: bool
    subtasks: List[NewSubtask] = Field(default_factory=list)


# Models for Adaptive, Bottom-Up Decomposition (The "Explorer")
class ProposedSubtask(BaseModel):
    local_id: str
    desc: str
    importance: int = Field(
        description="An integer score from 1 (least critical) to 100 (most critical) representing this task's necessity.",
        ge=1,
        le=100,
    )
    deps: List[str] = Field(
        default_factory=list,
        description="A list of local_ids of other PROPOSED tasks that this one depends on.",
    )


class AdaptiveDecomposerResponse(BaseModel):
    """
    The structured response from the adaptive_decomposer agent.
    It can propose new tasks to be created and/or request dependencies on existing tasks.
    """

    tasks: List[ProposedSubtask] = Field(
        default_factory=list,
        description="A list of new, granular sub-tasks to be created to achieve the parent objective.",
    )
    dep_requests: List[str] = Field(
        default_factory=list,
        description="A list of IDs of EXISTING tasks whose results are needed as new dependencies for the current task.",
    )


class TaskForMerging(BaseModel):
    """Represents a single task in a plan being evaluated for merging."""

    local_id: str
    desc: str
    deps: List[str] = Field(description="List of local_ids this task depends on.")


class MergedTask(BaseModel):
    """Represents a new, consolidated task that replaces one or more original tasks."""

    new_local_id: str = Field(
        description="A new, descriptive local ID for the merged task (e.g., 'plan_and_book_venue')."
    )
    new_desc: str = Field(
        description="A new, comprehensive description for the merged task."
    )
    subsumed_task_ids: List[str] = Field(
        description="A list of the original local_ids that this new task replaces."
    )


class MergingDecision(BaseModel):
    """The plan for merging overly granular or redundant tasks."""

    merged_tasks: List[MergedTask] = Field(
        description="A list of new tasks that consolidate others."
    )
    kept_task_ids: List[str] = Field(
        description="A list of local_ids for tasks that were NOT merged and should be kept as-is."
    )


class ProposalResolutionPlan(BaseModel):
    """The final, pruned list of approved sub-tasks."""

    approved_tasks: List[ProposedSubtask]


class ContextualAnswer(BaseModel):
    """The result of searching internal task history for an answer."""

    is_found: bool = Field(
        description="True if a definitive answer was found in the provided context."
    )
    answer: Optional[str] = Field(
        None, description="The answer found in the context, if any."
    )
    source_task_id: Optional[str] = Field(
        None, description="The ID of the task that contained the answer."
    )
    reasoning: str = Field(
        description="A brief explanation of why the context is or is not sufficient to answer the question."
    )


class RedundancyDecision(BaseModel):
    """The decision on which proposed tasks are not redundant and should proceed."""

    non_redundant_tasks: List[ProposedSubtask] = Field(
        description="A list of the tasks from the proposal that are unique and not covered by existing work."
    )


class TaskDescription(BaseModel):
    """A single, discrete step in a plan."""

    local_id: str = Field(
        description="A temporary, unique identifier for this task within the current plan."
    )
    desc: str = Field(description="A clear and concise description of the task.")
    can_request_new_subtasks: bool = Field(
        False,
        description="Set to true ONLY for complex, integrative, or uncertain tasks that might need further dynamic decomposition later.",
    )
    deps: List[str] = Field(
        default_factory=list,
        description="A list of GLOBAL IDs of existing tasks or documents this new task depends on.",
    )


class TaskChain(BaseModel):
    """Represents a sequence of tasks that MUST be executed in a specific order."""

    chain: List[TaskDescription] = Field(
        description="An ordered list of tasks forming a sequential chain."
    )


class ChainedExecutionPlan(BaseModel):
    """
    The full execution plan, composed of one or more parallel chains of tasks.
    Each chain represents a sequence of dependent tasks. Different chains can be executed in parallel.
    """

    task_chains: List[TaskChain] = Field(
        description="A list of task chains. All chains can run in parallel."
    )


def generate_hash(content: str) -> str:
    """Generates a SHA256 hash for a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DocumentState(BaseModel):
    """Holds the versioned content of a document."""

    content: str
    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.content_hash:
            self.content_hash = generate_hash(self.content)


# The main Task model, enhanced for the new architecture
class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = (
        "pending"  # can be: pending, planning, proposing, waiting_for_children, running, complete, failed, cancelled, pruned, paused_by_human, waiting_for_user_response
    )
    is_critical: bool = Field(
        False,
        description="If True, this task cannot be automatically pruned by the graph manager.",
    )

    counts_toward_limit: bool = Field(
        True,
        description="If False, this task does not count towards the limit of tasks that can exist",
    )
    task_type: Literal["task", "document"] = "task"
    document_state: Optional[DocumentState] = None

    result: Optional[str] = None
    cost: float = 0.0
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    dep_results: Dict[str, Any] = Field(default_factory=dict)
    shared_notebook: Dict[str, Any] = Field(
        default_factory=dict,
        description="A key-value store visible to the agent for persisting important state across interactions.",
    )  # NEW: Shared Notebook

    # --- HYBRID PLANNING & ADAPTATION FIELDS ---
    needs_planning: bool = False
    already_planned: bool = False
    can_request_new_subtasks: bool = False

    # --- RETRY & TRACING FIELDS ---
    fix_attempts: int = 0
    max_fix_attempts: int = 2
    verification_scores: List[float] = Field(default_factory=list)
    grace_attempts: int = 0
    otel_context: Optional[Any] = None
    span: Optional[Span] = None

    human_directive: Optional[str] = Field(
        None, description="A direct, corrective instruction from the operator."
    )
    current_question: Optional[UserQuestion] = Field(
        None,
        description="The question an agent is currently asking the human operator.",
    )
    user_response: Optional[str] = Field(
        None, description="The human operator's response to an agent's question."
    )

    # --- NEW CONTEXT FIELDS ---
    last_llm_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Stores the full message history of the last LLM interaction for resuming context.",
    )
    executor_llm_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Stores the full message history of the last LLM interaction for the executor of this task.",
    )

    execution_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="A log of real-time execution steps (tool calls, thoughts) for the UI.",
    )

    agent_role_paused: Optional[str] = Field(
        None,
        description="Stores the name of the agent role that paused for human input.",
    )

    # --- MCP FIEDS ---
    mcp_servers: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class TaskContext:
    def __init__(self, task_id: str, state_manager: AbstractStateManager):
        self.task_id = task_id
        self.state_manager = state_manager
        self.task = self.state_manager.get_task(task_id)


class DAGAgent:
    def __init__(
        self,
        state_manager: Optional[AbstractStateManager] = None,
        llm_model: str = "gpt-4o-mini",
        tracer: Optional[Tracer] = None,
        tools: Optional[List[Any]] = None,
        global_proposal_limit: int = 5,
        max_grace_attempts: int = 1,
        min_question_priority: int = 1,
        mcp_server_url: Optional[str] = None,  # NEW: MCP server URL parameter
    ):
        self.state_manager = state_manager or InMemoryStateManager()
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}
        self.base_llm_model = llm_model
        self.tracer = tracer or trace.get_tracer(__name__)
        self.global_proposal_limit = global_proposal_limit
        self.max_grace_attempts = max_grace_attempts
        self.min_question_priority = min_question_priority

        self._tools_for_agents: List[Any] = tools or []
        self._agent_role_map: Dict[str, Agent] = {}
        self.mcp_tool_schemas_str = "No remote tools are available for planning."  # NEW: To hold tool schemas for planner

        self.proposed_tasks_buffer: List[Tuple[ProposedSubtask, str]] = []
        self.mcp_servers = []  # For future use

        self._setup_agent_roles()

    def _setup_agent_roles(self):
        """Initializes or reinitializes all agent roles."""
        llm_model = self.base_llm_model

        # NEW: Create a dynamic system prompt for the planner
        planner_system_prompt = (
            "You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. "
            "Structure your output as 'chains' of tasks. "
            "A 'chain' is a list of tasks that must be done sequentially. "
            "You can have multiple chains, and all chains will be executed in parallel. "
            "Group related sequential steps into a single chain. Use parallel chains for independent streams of work. "
            f"The executor has access to the following remote tools. You can create sub-tasks that use these tools, but you cannot execute them yourself:\n"
            f"--- AVAILABLE REMOTE TOOLS ---\n{self.mcp_tool_schemas_str}\n------------------------------"
        )

        self.initial_planner = Agent(
            model=llm_model,
            system_prompt=planner_system_prompt,
            output_type=ChainedExecutionPlan,  # MODIFIED
            tools=[],
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
            tools=[],  # Planner-related agents should not have execution tools
        )
        self.adaptive_decomposer = Agent(
            model=llm_model,
            system_prompt=(
                "You are an adaptive expert. Analyze the given task and results of its dependencies. If the task is still too complex, break it down into one or more 'chains' of new, more granular sub-tasks. "
                "A 'chain' is a list of tasks that must be done sequentially. You can create multiple parallel chains for independent workstreams."
            ),
            output_type=ChainedExecutionPlan,  # MODIFIED
            tools=self._tools_for_agents,
        )
        self.conflict_resolver = Agent(
            model=llm_model,
            system_prompt=f"You are a ruthless but fair project manager. You have been given a list of proposed tasks that exceeds the budget. Analyze the list and their importance scores. You MUST prune the list by removing the LEAST critical tasks until the total number of tasks is no more than {self.global_proposal_limit}. Return only the final, approved list of tasks.",
            output_type=ProposalResolutionPlan,
            tools=[],  # Planner-related agents should not have execution tools
            retries=3,
        )
        # MODIFIED: Executor now gets the full list of tools, including the McpClient
        self.executor = Agent(
            model=llm_model,
            output_type=Union[str, UserQuestion],
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
                "You are a meticulous quality assurance analyst. Your job is to decide if a failing task is worth retrying based on its progress. "
                "You will be given the task's original goal and a history of its verification scores. "
                "If the scores are generally increasing, it is worth retrying. If scores are stagnant or decreasing, it is not. "
                "You must provide a concrete and actionable 'next_step_suggestion' for the next autonomous attempt."
            ),
            output_type=RetryDecision,  # It ONLY outputs a RetryDecision
            tools=self._tools_for_agents,
        )
        self._agent_role_map = {
            "initial_planner": self.initial_planner,
            "cycle_breaker": self.cycle_breaker,
            "adaptive_decomposer": self.adaptive_decomposer,
            "conflict_resolver": self.conflict_resolver,
            "executor": self.executor,
            "verifier": self.verifier,
            "retry_analyst": self.retry_analyst,
        }

    def _add_llm_data_to_system_span(self, span: Span, agent_res: AgentRunResult):
        """Adds LLM usage data to a span for a system-level operation without a task context."""
        if not span or not agent_res:
            return
        usage = agent_res.usage()
        # We don't track cost here as there is no task to assign it to.
        # This is a trade-off for keeping the system logic clean.
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0)
        )
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            getattr(usage, "response_tokens", 0),
        )

    def _add_llm_data_to_span(self, span: Span, agent_res: AgentRunResult, task: Task):
        if not span or not agent_res:
            return
        usage, cost = agent_res.usage(), get_cost(agent_res.usage())
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0)
        )
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            getattr(usage, "response_tokens", 0),
        )
        task.cost += cost

    def _format_notebook_for_llm(self, task: Task, max_len_per_entry: int = 200) -> str:
        if not task.shared_notebook:
            return "The shared notebook is currently empty."
        formatted_entries = []
        for key, value in task.shared_notebook.items():
            truncated_value = str(value)
            if len(truncated_value) > max_len_per_entry:
                truncated_value = truncated_value[: max_len_per_entry - 3] + "..."
            formatted_entries.append(f"- {key}: {truncated_value}")
        return "\n".join(formatted_entries)

    def _get_notebook_tool_guidance(self, task: Task, agent_role_name: str) -> str:
        just_talked_to_human = (
            task.status == "pending" and task.user_response is not None
        )
        is_executor_phase = agent_role_name == "executor"
        can_update_this_turn = is_executor_phase or (
            just_talked_to_human
            and agent_role_name in ["initial_planner", "adaptive_decomposer"]
        )
        if can_update_this_turn:
            return (
                f"\n\n--- Shared Notebook Update Guidance ---\n"
                f"To record critical information permanently for Task {task.id}, use `update_notebook_tool`.\n"
                f"Call it like: `await update_notebook_tool(task_id='{task.id}', updates={{'Key Name': 'Value'}})`\n"
                f"Set value to `None` to delete a key.\n-------------------------------------"
            )
        return ""

    async def _handle_agent_output(
        self,
        ctx: TaskContext,
        agent_res: AgentRunResult,
        expected_output_type: Any,
        agent_role_name: str,
    ) -> Tuple[bool, Any]:
        t = ctx.task
        self._add_llm_data_to_span(t.span, agent_res, t)
        t.last_llm_history = agent_res.all_messages()
        actual_output = agent_res.output
        if isinstance(actual_output, UserQuestion):
            if actual_output.priority < self.min_question_priority:
                t.current_question = actual_output
                t.user_response = "No specific input, proceed with best judgment."
                t.agent_role_paused = agent_role_name
                logger.info(
                    f"Task [{t.id}] low-priority question auto-answered. Task will resume."
                )
                return True, None
            else:
                logger.info(
                    f"Task [{t.id}] asking human question (Priority: {actual_output.priority}): {actual_output.question[:80]}..."
                )
                t.current_question = actual_output
                t.agent_role_paused = agent_role_name
                t.status = "waiting_for_user_response"
                return True, None
        else:
            if not isinstance(actual_output, BaseModel) and not isinstance(
                actual_output, str
            ):
                logger.warning(
                    f"Task [{t.id}] received unexpected output. Expected {str(expected_output_type)} or UserQuestion, got {type(actual_output).__name__}."
                )
            return False, actual_output

    async def run(self):
        with self.tracer.start_as_current_span("DAGAgent.run") as root_span:
            while True:
                all_task_ids = set(self.state_manager.tasks.keys())
                completed_or_failed_or_paused = {
                    t.id
                    for t in self.state_manager.tasks.values()
                    if t.status
                    in (
                        "complete",
                        "failed",
                        "paused_by_human",
                        "waiting_for_user_response",
                    )
                }
                for tid in all_task_ids - completed_or_failed_or_paused:
                    task = self.state_manager.tasks[tid]
                    if any(
                        self.state_manager.tasks.get(d, {}).status == "failed"
                        for d in task.deps
                    ):
                        if task.status != "failed":
                            task.status, task.result = (
                                "failed",
                                "Upstream dependency failed.",
                            )
                            task.last_llm_history, task.agent_role_paused = None, None
                pending_tasks = all_task_ids - completed_or_failed_or_paused
                ready_to_run_ids = {
                    tid
                    for tid in pending_tasks
                    if self.state_manager.tasks[tid].status == "pending"
                    and all(
                        self.state_manager.tasks.get(d, {}).status == "complete"
                        for d in self.state_manager.tasks[tid].deps
                    )
                }
                for tid in pending_tasks:
                    task = self.state_manager.tasks[tid]
                    if task.status == "waiting_for_children" and all(
                        self.state_manager.tasks[c].status in ["complete", "failed"]
                        for c in task.children
                    ):
                        if any(
                            self.state_manager.tasks[c].status == "failed"
                            for c in task.children
                        ):
                            task.status, task.result = (
                                "failed",
                                "A child task failed, cannot synthesize.",
                            )
                            task.last_llm_history, task.agent_role_paused = None, None
                        else:
                            ready_to_run_ids.add(tid)
                            task.status = "pending"
                ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]
                if not ready and not self.inflight and pending_tasks:
                    logger.error("Stalled DAG!")
                    self.state_manager.print_status_tree()
                    root_span.set_status(trace.Status(StatusCode.ERROR, "DAG Stalled"))
                    break
                if not pending_tasks and not self.inflight:
                    logger.info("DAG execution complete.")
                    break
                for tid in ready:
                    self.inflight.add(tid)
                    self.task_futures[tid] = asyncio.create_task(
                        self._run_taskflow(tid)
                    )
                if not self.task_futures:
                    continue
                done, _ = await asyncio.wait(
                    self.task_futures.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.1,
                )
                for fut in done:
                    tid = next(
                        (tid for tid, f in self.task_futures.items() if f == fut), None
                    )
                    if tid:
                        self.inflight.discard(tid)
                        del self.task_futures[tid]
                        try:
                            fut.result()
                        except asyncio.CancelledError:
                            logger.info(f"Task [{tid}] was cancelled.")
                        except Exception as e:
                            logger.error(f"Task [{tid}] future failed: {e}")
                            t = self.state_manager.tasks.get(tid)
                            if t and t.status != "failed":
                                t.status = "failed"
                                t.last_llm_history, t.agent_role_paused = None, None
                if self.proposed_tasks_buffer:
                    await self._run_global_resolution()

    async def _run_global_resolution(self):
        # This is now a standalone system operation, not a task.
        with self.tracer.start_as_current_span("GlobalConflictResolution") as span:
            logger.info(
                f"GLOBAL RESOLUTION: Evaluating {len(self.proposed_tasks_buffer)} proposed sub-tasks."
            )
            span.set_attribute("proposals.count", len(self.proposed_tasks_buffer))

            # Use a try...finally block to guarantee the buffer is cleared
            try:
                approved_proposals = [p[0] for p in self.proposed_tasks_buffer]
                parent_map = {p[0].local_id: p[1] for p in self.proposed_tasks_buffer}

                if len(approved_proposals) > self.global_proposal_limit:
                    logger.warning(
                        f"Proposal buffer ({len(approved_proposals)}) exceeds limit ({self.global_proposal_limit}). Engaging resolver."
                    )
                    span.add_event("Conflict detected, engaging resolver.")
                    prompt_list = [
                        f"local_id: {p.local_id}, importance: {p.importance}, desc: {p.desc}"
                        for p in approved_proposals
                    ]

                    # Directly call the agent without a task context
                    resolver_res = await self.conflict_resolver.run(
                        user_prompt=f"Prune this list to {self.global_proposal_limit} items: {prompt_list}"
                    )

                    # Manually handle telemetry and output
                    self._add_llm_data_to_system_span(span, resolver_res)

                    resolved_plan = resolver_res.output
                    if resolved_plan and isinstance(
                        resolved_plan, ProposalResolutionPlan
                    ):
                        approved_proposals = resolved_plan.approved_tasks
                        span.set_attribute(
                            "proposals.approved_count", len(approved_proposals)
                        )
                        span.set_status(StatusCode.OK)
                    else:
                        # If the agent returns None or an invalid type, fail safely
                        logger.error(
                            "Conflict resolver failed to return a valid plan. Discarding all proposals for this cycle."
                        )
                        approved_proposals = []
                        span.set_status(
                            trace.Status(
                                StatusCode.ERROR, "Resolver returned invalid output"
                            )
                        )

                if approved_proposals:
                    logger.info(
                        f"Committing {len(approved_proposals)} approved sub-tasks to the graph."
                    )
                    self._commit_proposals(approved_proposals, parent_map)
                else:
                    logger.info(
                        "No proposals were approved or committed in this cycle."
                    )

            finally:
                # This is critical to prevent infinite loops
                self.proposed_tasks_buffer = []

    def _commit_proposals(
        self,
        approved_proposals: List[ProposedSubtask],
        proposal_to_parent_map: Dict[str, str],
    ):
        local_to_global_id_map = {}
        for proposal in approved_proposals:
            parent_id = proposal_to_parent_map.get(proposal.local_id)
            if not parent_id:
                continue
            new_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=proposal.desc,
                parent=parent_id,
                can_request_new_subtasks=False,
            )
            self.state_manager.add_task(new_task)
            self.state_manager.tasks[parent_id].children.append(new_task.id)
            self.state_manager.add_dependency(parent_id, new_task.id)
            local_to_global_id_map[proposal.local_id] = new_task.id
        for proposal in approved_proposals:
            new_task_global_id = local_to_global_id_map.get(proposal.local_id)
            if not new_task_global_id:
                continue
            for dep_local_id in proposal.deps:
                dep_global_id = local_to_global_id_map.get(dep_local_id) or (
                    dep_local_id if dep_local_id in self.state_manager.tasks else None
                )
                if dep_global_id:
                    self.state_manager.add_dependency(new_task_global_id, dep_global_id)

    async def _run_taskflow(self, tid: str):
        ctx, t = TaskContext(tid, self.state_manager), self.state_manager.tasks[tid]
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span, span.set_attribute("dag.task.id", t.id), span.set_attribute(
                "dag.task.status", t.status
            )
            if t.status in ["paused_by_human", "waiting_for_user_response"]:
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution.")
                return
            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))
                if t.status == "waiting_for_children":
                    t.status = "running"
                    logger.info(f"SYNTHESIS PHASE for [{t.id}].")
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    t.status = "planning"
                    logger.info(f"ARCHITECT PHASE for [{t.id}].")
                    await self._run_initial_planning(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        t.already_planned = True
                        t.status = "waiting_for_children" if t.children else "complete"
                elif t.can_request_new_subtasks and t.status != "proposing":
                    t.status = "proposing"
                    logger.info(f"EXPLORER PHASE for [{t.id}].")
                    await self._run_adaptive_decomposition(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        t.status = "waiting_for_children"
                else:
                    t.status = "running"
                    logger.info(f"WORKER PHASE for [{t.id}].")
                    await self._run_task_execution(ctx)
                if t.status not in [
                    "waiting_for_children",
                    "failed",
                    "paused_by_human",
                    "waiting_for_user_response",
                ]:
                    t.status = "complete"
                    t.last_llm_history, t.agent_role_paused = None, None
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                t.last_llm_history, t.agent_role_paused = None, None
                raise
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", t.status)

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        completed_tasks = [
            tk
            for tk in self.state_manager.tasks.values()
            if tk.status == "complete" and tk.id != t.id
        ]
        context = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt_parts = [
            f"Objective: {t.desc}",
            f"\nAvailable completed data sources:\n{context}",
            f"\n--- Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
            self._get_notebook_tool_guidance(t, "initial_planner"),
        ]
        if t.human_directive:
            prompt_parts.append(
                f"\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            )
            t.human_directive = None
        plan_res = await self.initial_planner.run(
            user_prompt="\n".join(prompt_parts), message_history=t.last_llm_history
        )
        is_paused, plan = await self._handle_agent_output(
            ctx=ctx,
            agent_res=plan_res,
            expected_output_type=ChainedExecutionPlan,
            agent_role_name="initial_planner",
        )
        if is_paused or not plan:
            return
        await self._process_chained_plan_output(ctx, plan)

    async def _process_chained_plan_output(
        self, ctx: TaskContext, plan: ChainedExecutionPlan
    ):
        """
        Processes a ChainedExecutionPlan by creating tasks and auto-linking them within each chain.
        This is the simpler "Path A" implementation for the base agent.
        """
        t = ctx.task
        if not plan.task_chains:
            return

        for chain in plan.task_chains:
            previous_task_id_in_chain = None
            for task_desc in chain.chain:
                new_task = Task(
                    id=str(uuid.uuid4())[:8],
                    desc=task_desc.desc,
                    parent=t.id,
                    can_request_new_subtasks=task_desc.can_request_new_subtasks,
                )
                self.state_manager.add_task(new_task)
                t.children.append(new_task.id)
                self.state_manager.add_dependency(t.id, new_task.id)

                # Auto-link to the previous task in the same chain
                if previous_task_id_in_chain:
                    self.state_manager.add_dependency(
                        new_task.id, previous_task_id_in_chain
                    )

                previous_task_id_in_chain = new_task.id

    async def _process_initial_planning_output(
        self, ctx: TaskContext, plan: ExecutionPlan
    ):
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
            )
            self.state_manager.add_task(new_task)
            t.children.append(new_task.id)
            self.state_manager.add_dependency(t.id, new_task.id)
            local_to_global_id_map[sub.local_id] = new_task.id
        for sub in plan.subtasks:
            new_global_id = local_to_global_id_map.get(sub.local_id)
            if not new_global_id:
                continue
            for dep in sub.deps:
                dep_global_id = local_to_global_id_map.get(dep.local_id) or (
                    dep.local_id if dep.local_id in self.state_manager.tasks else None
                )
                if dep_global_id:
                    self.state_manager.add_dependency(new_global_id, dep_global_id)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.state_manager.tasks[d].result for d in t.deps}
        prompt_parts = [
            f"Task: {t.desc}",
            f"\nResults from dependencies:\n{t.dep_results}",
            f"\n--- Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
            self._get_notebook_tool_guidance(t, "adaptive_decomposer"),
        ]
        if t.human_directive:
            prompt_parts.append(
                f"\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            )
            t.human_directive = None
        proposals_res = await self.adaptive_decomposer.run(
            user_prompt="\n".join(prompt_parts), message_history=t.last_llm_history
        )
        is_paused, plan = await self._handle_agent_output(
            ctx=ctx,
            agent_res=proposals_res,
            expected_output_type=ChainedExecutionPlan,
            agent_role_name="adaptive_decomposer",
        )
        if is_paused or not plan:
            return
        # Since this is the base agent, we don't buffer proposals. We just process them directly.
        if plan.task_chains:
            logger.info(
                f"Task [{t.id}] is decomposing into {len(plan.task_chains)} new chain(s)."
            )
            await self._process_chained_plan_output(ctx, plan)

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running task execution for: {t.desc}")

        child_results = {
            cid: self.state_manager.tasks[cid].result
            for cid in t.children
            if self.state_manager.tasks[cid].status == "complete"
        }
        dep_results = {
            did: self.state_manager.tasks[did].result
            for did in t.deps
            if self.state_manager.tasks[did].status == "complete"
            and did not in child_results
        }

        prompt_lines = [f"Your task is: {t.desc}\n"]
        if child_results:
            prompt_lines.append(
                "\nSynthesize the results from your sub-tasks into a final answer:\n"
            )
            for cid, res in child_results.items():
                prompt_lines.append(
                    f"- From sub-task '{self.state_manager.tasks[cid].desc}':\n{res}\n\n"
                )
        elif dep_results:
            prompt_lines.append("\nUse data from dependencies to inform your answer:\n")
            for did, res in dep_results.items():
                prompt_lines.append(
                    f"- From dependency '{self.state_manager.tasks[did].desc}':\n{res}\n\n"
                )

        prompt_base_content = "".join(
            prompt_lines
        )  # FIXED: Use "" to join to avoid double newlines

        task_specific_mcp_clients = []
        task_specific_tools = list(self._tools_for_agents)

        if t.mcp_servers:
            logger.info(
                f"Task [{t.id}] has {len(t.mcp_servers)} MCP servers. Initializing clients."
            )
            for server_config in t.mcp_servers:
                address = server_config.get("address")
                server_type = server_config.get("type")
                server_name = server_config.get("name", address)
                if not address:
                    logger.warning(
                        f"Task [{t.id}]: MCP server config missing address. Skipping."
                    )
                    continue
                try:
                    if server_type == "streamable_http":
                        client = MCPServerStreamableHTTP(address)
                        task_specific_mcp_clients.append(client)
                        logger.info(
                            f"[{t.id}]: Initialized StreamableHTTP MCP client for '{server_name}' at {address}"
                        )
                    else:
                        logger.warning(
                            f"[{t.id}]: Unknown MCP server type '{server_type}' for '{server_name}'. Skipping."
                        )
                except Exception as e:
                    logger.error(
                        f"[{t.id}]: Failed to create MCP client for '{server_name}' at {address}: {e}"
                    )

        task_executor = Agent(
            model=self.base_llm_model,
            output_type=Union[str, UserQuestion],
            tools=task_specific_tools,
            mcp_servers=task_specific_mcp_clients,
        )

        while True:
            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})
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

            current_prompt = "".join(current_prompt_parts)  # FIXED: Use "" to join

            logger.info(f"[{t.id}] Attempt {current_attempt}: Calling executor LLM.")
            exec_res = await task_executor.run(
                user_prompt=current_prompt, message_history=t.last_llm_history
            )
            is_paused, result = await self._handle_agent_output(
                ctx=ctx,
                agent_res=exec_res,
                expected_output_type=str,
                agent_role_name="executor",
            )

            if is_paused:
                return
            if result is None:
                t.human_directive = "Your last output was empty or invalid. Please re-evaluate and try again."
                t.fix_attempts += 1
                logger.warning(
                    f"[{t.id}] Executor (attempt {current_attempt}) returned no result. Triggering retry."
                )
                if (t.fix_attempts + t.grace_attempts) > (
                    t.max_fix_attempts + self.max_grace_attempts
                ):
                    t.status = "failed"
                    raise Exception(
                        f"Exceeded max attempts for task '{t.id}' after empty/invalid executor result."
                    )
                continue

            await self._process_executor_output_for_verification(ctx, result)
            if t.status in [
                "complete",
                "failed",
                "waiting_for_user_response",
                "paused_by_human",
            ]:
                return

    async def _process_executor_output_for_verification(
        self, ctx: TaskContext, result: str
    ):
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)
        if verify_task_result.get_successful():
            t.result = result
            logger.info(f"COMPLETED [{t.id}]")
            t.status = "complete"
            t.last_llm_history, t.agent_role_paused = None, None
            return

        t.fix_attempts += 1

        # Consult the analyst only when max standard attempts are reached.
        if (
            t.fix_attempts >= t.max_fix_attempts
            and t.grace_attempts < self.max_grace_attempts
        ):
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\nScore > 5 is a success. Should we retry autonomously?"
            decision_res = await self.retry_analyst.run(user_prompt=analyst_prompt)

            # Here, we don't need the complex _handle_agent_output since this agent can't ask questions.
            decision = decision_res.output

            if decision and decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion
                logger.info(
                    f"[{t.id}] Grace attempt granted. Next step: {decision.next_step_suggestion}"
                )
                return  # Return to the execution loop for the grace attempt.

        # Final failure condition if grace attempt is denied or not applicable.
        if (t.fix_attempts + t.grace_attempts) >= (
            t.max_fix_attempts + self.max_grace_attempts
        ):
            error_msg = f"Exceeded max attempts for task '{t.id}'"
            t.span.set_status(trace.Status(StatusCode.ERROR, error_msg))
            t.status = "failed"
            t.last_llm_history, t.agent_role_paused = None, None
            raise Exception(error_msg)

        # Default action for standard retries.
        t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
        logger.info(
            f"[{t.id}] Retrying execution. Feedback: {verify_task_result.reason[:50]}..."
        )

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
        vres = await self.verifier.run(
            user_prompt=ver_prompt, message_history=t.last_llm_history
        )
        is_paused, vout = await self._handle_agent_output(
            ctx=TaskContext(t.id, self.state_manager),
            agent_res=vres,
            expected_output_type=verification,
            agent_role_name="verifier",
        )
        if is_paused:
            return verification(
                reason="Verifier paused to ask question.", message_for_user="", score=0
            )
        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(
            f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'"
        )
        return vout


if __name__ == "__main__":
    try:
        from nest_asyncio import apply

        apply()
    except ImportError:
        pass
    logger.info(
        "==== BEGIN DEMO: HYBRID PLANNING AGENT (WITH RETRIES & CYCLE BREAKING) ===="
    )
    reg = TaskRegistry()
    doc_ids = [
        reg.add_document(name, content)
        for name, content in [
            (
                "Financials Q2 Revenue",
                "Apple Inc. (AAPL) Q2 2024 Revenue: $94.5B. iPhones: $50.5B. Services: $22.0B.",
            ),
            (
                "Financials Q2 Profit",
                "Apple Inc. (AAPL) Q2 2024 Net Income: $25.1B. EPS: $1.55.",
            ),
            (
                "Mgmt Outlook Q2",
                "Guidance: Q3 revenue to decline slightly. Risks: Supply chain.",
            ),
            (
                "Market Intel Q2",
                "Analyst Rating: Strong Buy. Rationale: Bullish on services growth.",
            ),
            ("Irrelevant Memo", "The company picnic is next Friday."),
        ]
    ]
    root_task = Task(
        id="ROOT_INVESTOR_BRIEFING",
        desc=(
            "Create a comprehensive investor briefing for Apple's Q2 2024 performance. "
            "First, plan to consolidate all relevant data. Then, synthesize the information into a report. "
            "Ignore irrelevant data. If unsure about audience focus, ask the manager."
        ),
        needs_planning=True,
    )
    reg.add_task(root_task)
    agent = DAGAgent(
        registry=reg,
        llm_model="gpt-4o-mini",
        tracer=trace.get_tracer("hybrid_dag_demo"),
        global_proposal_limit=2,
        max_grace_attempts=1,
        mcp_server_url=getenv("MCP_SERVER_URL"),  # NEW: Pass the MCP server URL
    )

    async def main():
        print("\n--- INITIAL TASK STATUS TREE ---")
        reg.print_status_tree()
        print("\n===== EXECUTING DAG AGENT =====\n")
        await agent.run()
        print("\n===== DAG EXECUTION COMPLETE =====\n")
        print("\n--- FINAL TASK STATUS TREE ---")
        reg.print_status_tree()
        print("\n--- Final Output for Root Task ---\n")
        root_result = reg.tasks.get("ROOT_INVESTOR_BRIEFING")
        if root_result:
            print(f"Final Result (Status: {root_result.status}):\n{root_result.result}")
        print(f"\nTotal estimated cost: ${sum(t.cost for t in reg.tasks.values()):.4f}")

    asyncio.run(main())
