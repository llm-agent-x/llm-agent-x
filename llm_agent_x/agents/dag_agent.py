import asyncio
import uuid
import logging
from collections import defaultdict, deque
from hashlib import md5
from os import getenv
from typing import Set, Dict, Any, Optional, List, Tuple, Union, Callable  # Added Callable and Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from dotenv import load_dotenv

# --- OpenTelemetry Imports ---
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Tracer, Span, StatusCode
from openinference.semconv.trace import SpanAttributes

# --- rich imports removed as they are no longer needed ---

# --- Basic Setup ---
load_dotenv(".env", override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DAGAgent")


def get_cost(usage) -> float:
    if not usage: return 0.0
    # Using GPT-4o-mini pricing for example
    input_cost = 0.15 / 1_000_000
    output_cost = 0.60 / 1_000_000
    return (input_cost * getattr(usage, "request_tokens", 0)) + (output_cost * getattr(usage, "response_tokens", 0))


# --- Pydantic Models ---
class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best).")

    def get_successful(self): return self.score > 5


class RetryDecision(BaseModel):
    """The decision on whether to attempt another fix for a failing task."""
    should_retry: bool = Field(description="Set to true if the trend of scores suggests success is likely.")
    reason: str = Field(description="A brief explanation for the decision, citing the score trend.")
    next_step_suggestion: str = Field(
        description="A specific, actionable suggestion for the next attempt to improve the result.")


# NEW: Model for asking clarifying questions
class UserQuestion(BaseModel):
    """
    A specific output type for agents to ask clarifying questions to the human operator.
    The task will pause until the human provides an answer.
    """
    question: str = Field(description="The clarifying question to ask the human operator.")
    priority: int = Field(
        description="A numerical score from 1 (lowest) to 10 (highest) representing the urgency or criticality of getting an answer.",
        ge=1, le=10)
    details: Dict[str, Any] = Field(default_factory=dict,
                                    description="Any additional context or structured data relevant to the question.")


# Models for Initial, Top-Down Planning (The "Architect")
class Dependency(BaseModel):
    reason: str
    local_id: str


class NewSubtask(BaseModel):
    local_id: str
    desc: str
    deps: List[Dependency] = Field(default_factory=list)
    can_request_new_subtasks: bool = Field(False,
                                           description="Set to true ONLY for complex, integrative, or uncertain tasks that might need further dynamic decomposition later.")


class ExecutionPlan(BaseModel):
    needs_subtasks: bool
    subtasks: List[NewSubtask] = Field(default_factory=list)


# Models for Adaptive, Bottom-Up Decomposition (The "Explorer")
class ProposedSubtask(BaseModel):
    local_id: str
    desc: str
    importance: int = Field(
        description="An integer score from 1 (least critical) to 100 (most critical) representing this task's necessity.",
        ge=1, le=100)
    deps: List[str] = Field(default_factory=list,
                            description="A list of local_ids of other PROPOSED tasks that this one depends on.")


class ProposalResolutionPlan(BaseModel):
    """The final, pruned list of approved sub-tasks."""
    approved_tasks: List[ProposedSubtask]


# The main Task model, enhanced for the new architecture
class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"  # can be: pending, planning, proposing, waiting_for_children, running, complete, failed, paused_by_human, waiting_for_user_response
    result: Optional[str] = None
    cost: float = 0.0
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    dep_results: Dict[str, Any] = Field(default_factory=dict)
    shared_notebook: Dict[str, Any] = Field(default_factory=dict,
                                            description="A key-value store visible to the agent for persisting important state across interactions.")  # NEW: Shared Notebook

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

    # --- NEW INTERACTIVITY FIELDS ---
    human_directive: Optional[str] = Field(None, description="A direct, corrective instruction from the operator.")
    current_question: Optional[UserQuestion] = Field(None,
                                                     description="The question an agent is currently asking the human operator.")
    user_response: Optional[str] = Field(None, description="The human operator's response to an agent's question.")
    last_llm_history: Optional[List[Dict[str, Any]]] = Field(None,
                                                             description="Stores the full message history of the last LLM interaction for resuming context.")
    agent_role_paused: Optional[str] = Field(None,
                                             description="Stores the name of the agent role that paused for human input.")

    class Config:
        arbitrary_types_allowed = True


class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        if task.id in self.tasks: raise ValueError(f"Task {task.id} already exists")
        self.tasks[task.id] = task

    def add_document(self, document_name: str, content: Dict[str, str]) -> str:
        id = md5(f"{document_name}{content}".encode()).hexdigest()
        print(f"Adding document {document_name} with id {id}")
        print(f"Document content: {content}")
        self.tasks[id] = Task(id=id, deps=set(), desc=f"Document: {document_name}", status="complete", result=content)
        return id

    def add_dependency(self, task_id: str, dep_id: str):
        if task_id not in self.tasks or dep_id not in self.tasks: return
        self.tasks[task_id].deps.add(dep_id)

    def print_status_tree(self):
        logger.info("--- CURRENT TASK STATUS TREE ---")
        root_nodes = [t for t in self.tasks.values() if not t.parent]
        for root in root_nodes: self._print_node(root, 0)
        logger.info("------------------------------------")

    def _print_node(self, task: Task, level: int):
        prefix = "  " * level
        status_color = {"complete": "✅", "failed": "❌", "running": "⏳", "planning": "📝", "proposing": "💡",
                        "waiting_for_children": "⏸️", "pending": "📋", "paused_by_human": "🛑",
                        "waiting_for_user_response": "❓"}
        status_icon = status_color.get(task.status, "❓")
        logger.info(f"{prefix}- {status_icon} {task.id[:8]}: {task.status.upper()} | {task.desc[:60]}...")
        for child_id in task.children:
            if child_id in self.tasks: self._print_node(self.tasks[child_id], level + 1)


class TaskContext:
    def __init__(self, task_id: str, registry: TaskRegistry):
        self.task_id = task_id
        self.registry = registry
        self.task = self.registry.tasks[task_id]


class DAGAgent:
    def __init__(
            self,
            registry: TaskRegistry,
            llm_model: str = "gpt-4o-mini",
            tracer: Optional[Tracer] = None,
            tools: Optional[List[Any]] = None,
            global_proposal_limit: int = 5,
            max_grace_attempts: int = 1,
            min_question_priority: int = 1,
    ):
        self.registry = registry
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}
        self.base_llm_model = llm_model
        self.tracer = tracer or trace.get_tracer(__name__)
        self.global_proposal_limit = global_proposal_limit
        self.max_grace_attempts = max_grace_attempts
        self.min_question_priority = min_question_priority

        self._tools_for_agents: List[Any] = tools or []
        self._agent_role_map: Dict[str, Agent] = {}

        self.proposed_tasks_buffer: List[Tuple[ProposedSubtask, str]] = []

        self._setup_agent_roles()

    def _setup_agent_roles(self):
        """Initializes or reinitializes all agent roles."""
        llm_model = self.base_llm_model

        self.initial_planner = Agent(
            model=llm_model,
            system_prompt="You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. You can link tasks to pre-existing completed tasks. For each new sub-task, decide if it is complex enough to merit further dynamic decomposition by setting `can_request_new_subtasks` to true. If you need a crucial piece of information from the human manager to proceed with planning, you may output a `UserQuestion`.",
            output_type=Union[ExecutionPlan, UserQuestion],
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
        self.adaptive_decomposer = Agent(
            model=llm_model,
            system_prompt="You are an adaptive expert. Analyze the given task and results of its dependencies. If the task is still too complex, propose a list of new, more granular sub-tasks to achieve it. You MUST provide an `importance` score (1-100) for each proposal, reflecting how critical it is. If you need a crucial piece of information from the human manager to proceed with decomposition, you may output a `UserQuestion`.",
            output_type=Union[List[ProposedSubtask], UserQuestion],
            tools=self._tools_for_agents,
        )
        self.conflict_resolver = Agent(
            model=llm_model,
            system_prompt=f"You are a ruthless but fair project manager. You have been given a list of proposed tasks that exceeds the budget. Analyze the list and their importance scores. You MUST prune the list by removing the LEAST critical tasks until the total number of tasks is no more than {self.global_proposal_limit}. Return only the final, approved list of tasks.",
            output_type=ProposalResolutionPlan,
            tools=self._tools_for_agents,
            retries=3,
        )
        self.executor = Agent(model=llm_model, output_type=Union[str, UserQuestion], tools=self._tools_for_agents)
        self.verifier = Agent(
            model=llm_model,
            system_prompt="You are a meticulous quality assurance verifier. Your job is to check if the provided 'Candidate Result' accurately and completely addresses the 'Task', considering the history and user instructions. Output JSON matching the 'verification' schema. Be strict.",
            output_type=verification,
            tools=self._tools_for_agents,
        )
        self.retry_analyst = Agent(
            model=llm_model,
            system_prompt=(
                "You are a meticulous quality assurance analyst. Your job is to decide if a failing task is worth retrying. "
                "You will be given the task's original goal and a history of its verification scores. "
                "If the scores are generally increasing and approaching the success threshold of 6, it is worth retrying. "
                "If scores are stagnant or decreasing, it is not. "
                "Crucially, you must provide a concrete and actionable 'next_step_suggestion'."
            ),
            output_type=RetryDecision,
            tools=self._tools_for_agents,
        )
        self._agent_role_map = {
            "initial_planner": self.initial_planner, "cycle_breaker": self.cycle_breaker,
            "adaptive_decomposer": self.adaptive_decomposer, "conflict_resolver": self.conflict_resolver,
            "executor": self.executor, "verifier": self.verifier, "retry_analyst": self.retry_analyst,
        }

    def _add_llm_data_to_span(self, span: Span, agent_res: AgentRunResult, task: Task):
        if not span or not agent_res: return
        usage, cost = agent_res.usage(), get_cost(agent_res.usage())
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))
        task.cost += cost

    def _format_notebook_for_llm(self, task: Task, max_len_per_entry: int = 200) -> str:
        if not task.shared_notebook: return "The shared notebook is currently empty."
        formatted_entries = []
        for key, value in task.shared_notebook.items():
            truncated_value = str(value)
            if len(truncated_value) > max_len_per_entry:
                truncated_value = truncated_value[:max_len_per_entry - 3] + "..."
            formatted_entries.append(f"- {key}: {truncated_value}")
        return "\n".join(formatted_entries)

    def _get_notebook_tool_guidance(self, task: Task, agent_role_name: str) -> str:
        just_talked_to_human = task.status == "pending" and task.user_response is not None
        is_executor_phase = agent_role_name == "executor"
        can_update_this_turn = is_executor_phase or (
                just_talked_to_human and agent_role_name in ["initial_planner", "adaptive_decomposer"])
        if can_update_this_turn:
            return (
                f"\n\n--- Shared Notebook Update Guidance ---\n"
                f"To record critical information permanently for Task {task.id}, use `update_notebook_tool`.\n"
                f"Call it like: `await update_notebook_tool(task_id='{task.id}', updates={{'Key Name': 'Value'}})`\n"
                f"Set value to `None` to delete a key.\n-------------------------------------"
            )
        return ""

    async def _handle_agent_output(self, ctx: TaskContext, agent_res: AgentRunResult, expected_output_type: Any,
                                   agent_role_name: str) -> Tuple[bool, Any]:
        t = ctx.task
        self._add_llm_data_to_span(t.span, agent_res, t)
        t.last_llm_history = agent_res.all_messages()
        actual_output = agent_res.output
        if isinstance(actual_output, UserQuestion):
            if actual_output.priority < self.min_question_priority:
                t.current_question = actual_output
                t.user_response = "No specific input, proceed with best judgment."
                t.agent_role_paused = agent_role_name
                logger.info(f"Task [{t.id}] low-priority question auto-answered. Task will resume.")
                return True, None
            else:
                logger.info(
                    f"Task [{t.id}] asking human question (Priority: {actual_output.priority}): {actual_output.question[:80]}...")
                t.current_question = actual_output
                t.agent_role_paused = agent_role_name
                t.status = "waiting_for_user_response"
                return True, None
        else:
            if not isinstance(actual_output, BaseModel) and not isinstance(actual_output, str):
                logger.warning(
                    f"Task [{t.id}] received unexpected output. Expected {str(expected_output_type)} or UserQuestion, got {type(actual_output).__name__}.")
            return False, actual_output

    async def run(self):
        with self.tracer.start_as_current_span("DAGAgent.run") as root_span:
            while True:
                all_task_ids = set(self.registry.tasks.keys())
                completed_or_failed_or_paused = {t.id for t in self.registry.tasks.values() if
                                                 t.status in ('complete', 'failed', 'paused_by_human',
                                                              'waiting_for_user_response')}
                for tid in all_task_ids - completed_or_failed_or_paused:
                    task = self.registry.tasks[tid]
                    if any(self.registry.tasks.get(d, {}).status == "failed" for d in task.deps):
                        if task.status != "failed":
                            task.status, task.result = "failed", "Upstream dependency failed."
                            task.last_llm_history, task.agent_role_paused = None, None
                pending_tasks = all_task_ids - completed_or_failed_or_paused
                ready_to_run_ids = {
                    tid for tid in pending_tasks if
                    self.registry.tasks[tid].status == "pending" and
                    all(self.registry.tasks.get(d, {}).status == "complete" for d in self.registry.tasks[tid].deps)}
                for tid in pending_tasks:
                    task = self.registry.tasks[tid]
                    if task.status == 'waiting_for_children' and all(
                            self.registry.tasks[c].status in ['complete', 'failed'] for c in task.children):
                        if any(self.registry.tasks[c].status == 'failed' for c in task.children):
                            task.status, task.result = "failed", "A child task failed, cannot synthesize."
                            task.last_llm_history, task.agent_role_paused = None, None
                        else:
                            ready_to_run_ids.add(tid);
                            task.status = "pending"
                ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]
                if not ready and not self.inflight and pending_tasks:
                    logger.error("Stalled DAG!"); self.registry.print_status_tree()
                    root_span.set_status(trace.Status(StatusCode.ERROR, "DAG Stalled")); break
                if not pending_tasks and not self.inflight:
                    logger.info("DAG execution complete."); break
                for tid in ready:
                    self.inflight.add(tid); self.task_futures[tid] = asyncio.create_task(self._run_taskflow(tid))
                if not self.task_futures: continue
                done, _ = await asyncio.wait(self.task_futures.values(), return_when=asyncio.FIRST_COMPLETED,
                                             timeout=0.1)
                for fut in done:
                    tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                    if tid:
                        self.inflight.discard(tid); del self.task_futures[tid]
                        try:
                            fut.result()
                        except asyncio.CancelledError:
                            logger.info(f"Task [{tid}] was cancelled.")
                        except Exception as e:
                            logger.error(f"Task [{tid}] future failed: {e}")
                            t = self.registry.tasks.get(tid)
                            if t and t.status != "failed":
                                t.status = "failed"; t.last_llm_history, t.agent_role_paused = None, None
                if self.proposed_tasks_buffer: await self._run_global_resolution()

    async def _run_global_resolution(self):
        logger.info(f"GLOBAL RESOLUTION: Evaluating {len(self.proposed_tasks_buffer)} proposed sub-tasks.")
        approved_proposals, parent_map = [p[0] for p in self.proposed_tasks_buffer], {p[0].local_id: p[1] for p in self.proposed_tasks_buffer}
        if len(approved_proposals) > self.global_proposal_limit:
            logger.warning(
                f"Proposal buffer ({len(approved_proposals)}) exceeds limit ({self.global_proposal_limit}). Engaging resolver.")
            prompt_list = [f"local_id: {p.local_id}, importance: {p.importance}, desc: {p.desc}" for p in
                           approved_proposals]
            global_resolver_task_id = "GLOBAL_RESOLVER"
            if global_resolver_task_id not in self.registry.tasks:
                self.registry.add_task(
                    Task(id=global_resolver_task_id, desc="Internal task for global conflict resolution.",
                         status="pending"))
            global_resolver_task = self.registry.tasks[global_resolver_task_id]
            global_resolver_ctx = TaskContext(global_resolver_task_id, self.registry)
            resolver_res = await self.conflict_resolver.run(
                user_prompt=f"Prune this list to {self.global_proposal_limit} items: {prompt_list}",
                message_history=global_resolver_task.last_llm_history)
            is_paused, resolved_plan = await self._handle_agent_output(ctx=global_resolver_ctx, agent_res=resolver_res,
                                                                       expected_output_type=ProposalResolutionPlan,
                                                                       agent_role_name="conflict_resolver")
            if is_paused:
                logger.warning(
                    f"Conflict resolver paused. Discarding current proposals; they will be regenerated later.")
                self.proposed_tasks_buffer = []; return
            if resolved_plan is None and global_resolver_task.status == "waiting_for_user_response":
                logger.warning("Conflict resolver paused, no plan resolved yet."); return
            approved_proposals = resolved_plan.approved_tasks
        logger.info(f"Committing {len(approved_proposals)} approved sub-tasks to the graph.")
        self._commit_proposals(approved_proposals, parent_map); self.proposed_tasks_buffer = []

    def _commit_proposals(self, approved_proposals: List[ProposedSubtask], proposal_to_parent_map: Dict[str, str]):
        local_to_global_id_map = {}
        for proposal in approved_proposals:
            parent_id = proposal_to_parent_map.get(proposal.local_id)
            if not parent_id: continue
            new_task = Task(id=str(uuid.uuid4())[:8], desc=proposal.desc, parent=parent_id, can_request_new_subtasks=False)
            self.registry.add_task(new_task); self.registry.tasks[parent_id].children.append(new_task.id)
            self.registry.add_dependency(parent_id, new_task.id); local_to_global_id_map[proposal.local_id] = new_task.id
        for proposal in approved_proposals:
            new_task_global_id = local_to_global_id_map.get(proposal.local_id)
            if not new_task_global_id: continue
            for dep_local_id in proposal.deps:
                dep_global_id = local_to_global_id_map.get(dep_local_id) or (dep_local_id if dep_local_id in self.registry.tasks else None)
                if dep_global_id: self.registry.add_dependency(new_task_global_id, dep_global_id)

    async def _run_taskflow(self, tid: str):
        ctx, t = TaskContext(tid, self.registry), self.registry.tasks[tid]
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span, span.set_attribute("dag.task.id", t.id), span.set_attribute("dag.task.status", t.status)
            if t.status in ["paused_by_human", "waiting_for_user_response"]:
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution."); return
            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))
                if t.status == 'waiting_for_children':
                    t.status = 'running'; logger.info(f"SYNTHESIS PHASE for [{t.id}]."); await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    t.status = 'planning'; logger.info(f"ARCHITECT PHASE for [{t.id}]."); await self._run_initial_planning(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        t.already_planned = True; t.status = "waiting_for_children" if t.children else "complete"
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    t.status = 'proposing'; logger.info(f"EXPLORER PHASE for [{t.id}]."); await self._run_adaptive_decomposition(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]: t.status = "waiting_for_children"
                else:
                    t.status = 'running'; logger.info(f"WORKER PHASE for [{t.id}]."); await self._run_task_execution(ctx)
                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    t.status = "complete"; t.last_llm_history, t.agent_role_paused = None, None
            except Exception as e:
                span.record_exception(e); span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"; t.last_llm_history, t.agent_role_paused = None, None; raise
            finally:
                otel_context.detach(otel_ctx_token); span.set_attribute("dag.task.status", t.status)

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task; completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt_parts = [f"Objective: {t.desc}", f"\nAvailable completed data sources:\n{context}",
                        f"\n--- Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
                        self._get_notebook_tool_guidance(t, "initial_planner")]
        if t.human_directive:
            prompt_parts.append(f"\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"); t.human_directive = None
        plan_res = await self.initial_planner.run(user_prompt="\n".join(prompt_parts), message_history=t.last_llm_history)
        is_paused, initial_plan = await self._handle_agent_output(ctx=ctx, agent_res=plan_res,
                                                                  expected_output_type=ExecutionPlan, agent_role_name="initial_planner")
        if is_paused: return
        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
            fixed_plan_res = await self.cycle_breaker.run(
                user_prompt=f"Analyze and fix cycles in this plan:\n{initial_plan.model_dump_json(indent=2)}",
                message_history=t.last_llm_history)
            is_paused, plan = await self._handle_agent_output(ctx=ctx, agent_res=fixed_plan_res,
                                                              expected_output_type=ExecutionPlan, agent_role_name="cycle_breaker")
            if is_paused: return
        else:
            plan = initial_plan
        if not plan.needs_subtasks: return
        await self._process_initial_planning_output(ctx, plan)

    async def _process_initial_planning_output(self, ctx: TaskContext, plan: ExecutionPlan):
        t = ctx.task;
        if not plan.needs_subtasks: return
        local_to_global_id_map = {}
        for sub in plan.subtasks:
            new_task = Task(id=str(uuid.uuid4())[:8], desc=sub.desc, parent=t.id,
                            can_request_new_subtasks=sub.can_request_new_subtasks)
            self.registry.add_task(new_task); t.children.append(new_task.id); self.registry.add_dependency(t.id, new_task.id)
            local_to_global_id_map[sub.local_id] = new_task.id
        for sub in plan.subtasks:
            new_global_id = local_to_global_id_map.get(sub.local_id)
            if not new_global_id: continue
            for dep in sub.deps:
                dep_global_id = local_to_global_id_map.get(dep.local_id) or (dep.local_id if dep.local_id in self.registry.tasks else None)
                if dep_global_id: self.registry.add_dependency(new_global_id, dep_global_id)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task; t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt_parts = [f"Task: {t.desc}", f"\nResults from dependencies:\n{t.dep_results}",
                        f"\n--- Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
                        self._get_notebook_tool_guidance(t, "adaptive_decomposer")]
        if t.human_directive:
            prompt_parts.append(f"\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"); t.human_directive = None
        proposals_res = await self.adaptive_decomposer.run(user_prompt="\n".join(prompt_parts), message_history=t.last_llm_history)
        is_paused, proposals = await self._handle_agent_output(ctx=ctx, agent_res=proposals_res,
                                                               expected_output_type=List[ProposedSubtask],
                                                               agent_role_name="adaptive_decomposer")
        if is_paused: return
        if proposals:
            logger.info(f"Task [{t.id}] proposing {len(proposals)} new sub-tasks.")
            for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        child_results = {cid: self.registry.tasks[cid].result for cid in t.children if
                         self.registry.tasks[cid].status == 'complete'}
        dep_results = {did: self.registry.tasks[did].result for did in t.deps if
                       self.registry.tasks[did].status == 'complete' and did not in child_results}
        prompt_lines = [f"Task: {t.desc}"]
        if child_results:
            prompt_lines.append("\nSynthesize the results from your sub-tasks into a final answer:\n")
            for cid, res in child_results.items(): prompt_lines.append(f"- From sub-task '{self.registry.tasks[cid].desc}':\n{res}\n")
        elif dep_results:
            prompt_lines.append("\nUse data from dependencies to inform your answer:\n")
            for did, res in dep_results.items(): prompt_lines.append(f"- From dependency '{self.registry.tasks[did].desc}':\n{res}\n")
        prompt_base_content = "\n".join(prompt_lines)
        while True:
            current_attempt = t.fix_attempts + t.grace_attempts
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt + 1})
            current_prompt_parts = [prompt_base_content, f"\n--- Shared Notebook for Task {t.id} ---\n{self._format_notebook_for_llm(t)}",
                                    self._get_notebook_tool_guidance(t, "executor")]
            if t.human_directive:
                current_prompt_parts.insert(0, f"--- CRITICAL GUIDANCE FROM OPERATOR ---\n{t.human_directive}\n------------------------------\n")
                t.human_directive = None
            exec_res = await self.executor.run(user_prompt="\n".join(current_prompt_parts), message_history=t.last_llm_history)
            is_paused, result = await self._handle_agent_output(ctx=ctx, agent_res=exec_res, expected_output_type=str,
                                                                agent_role_name="executor")
            if is_paused: return
            await self._process_executor_output_for_verification(ctx, result)
            if t.status in ["complete", "failed", "waiting_for_user_response", "paused_by_human"]: return

    async def _process_executor_output_for_verification(self, ctx: TaskContext, result: str):
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)
        if verify_task_result.get_successful():
            t.result = result; logger.info(f"COMPLETED [{t.id}]"); t.status = "complete"
            t.last_llm_history, t.agent_role_paused = None, None; return
        t.fix_attempts += 1
        if t.fix_attempts >= t.max_fix_attempts and t.grace_attempts < self.max_grace_attempts:
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\nScore > 5 is a success. Should we retry?"
            decision_res = await self.retry_analyst.run(user_prompt=analyst_prompt, message_history=t.last_llm_history)
            is_paused, decision = await self._handle_agent_output(ctx=ctx, agent_res=decision_res,
                                                                  expected_output_type=RetryDecision,
                                                                  agent_role_name="retry_analyst")
            if is_paused: return
            if decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason}); t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion
                logger.info(f"[{t.id}] Grace attempt granted. Next step: {decision.next_step_suggestion}"); return
        if (t.fix_attempts + t.grace_attempts) >= (t.max_fix_attempts + self.max_grace_attempts):
            error_msg = f"Exceeded max attempts for task '{t.id}'"; t.span.set_status(trace.Status(StatusCode.ERROR, error_msg))
            t.status = "failed"; t.last_llm_history, t.agent_role_paused = None, None; raise Exception(error_msg)
        t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
        logger.info(f"[{t.id}] Retrying execution. Feedback: {verify_task_result.reason[:50]}...")

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
        vres = await self.verifier.run(user_prompt=ver_prompt, message_history=t.last_llm_history)
        is_paused, vout = await self._handle_agent_output(ctx=TaskContext(t.id, self.registry), agent_res=vres,
                                                          expected_output_type=verification, agent_role_name="verifier")
        if is_paused: return verification(reason="Verifier paused to ask question.", message_for_user="", score=0)
        t.verification_scores.append(vout.score); t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout


if __name__ == "__main__":
    try: from nest_asyncio import apply; apply()
    except ImportError: pass
    logger.info("==== BEGIN DEMO: HYBRID PLANNING AGENT (WITH RETRIES & CYCLE BREAKING) ====")
    reg = TaskRegistry()
    doc_ids = [reg.add_document(name, content) for name, content in [
        ("Financials Q2 Revenue", "Apple Inc. (AAPL) Q2 2024 Revenue: $94.5B. iPhones: $50.5B. Services: $22.0B."),
        ("Financials Q2 Profit", "Apple Inc. (AAPL) Q2 2024 Net Income: $25.1B. EPS: $1.55."),
        ("Mgmt Outlook Q2", "Guidance: Q3 revenue to decline slightly. Risks: Supply chain."),
        ("Market Intel Q2", "Analyst Rating: Strong Buy. Rationale: Bullish on services growth."),
        ("Irrelevant Memo", "The company picnic is next Friday.")
    ]]
    root_task = Task(
        id="ROOT_INVESTOR_BRIEFING",
        desc=("Create a comprehensive investor briefing for Apple's Q2 2024 performance. "
              "First, plan to consolidate all relevant data. Then, synthesize the information into a report. "
              "Ignore irrelevant data. If unsure about audience focus, ask the manager."),
        needs_planning=True,
    )
    reg.add_task(root_task)
    agent = DAGAgent(
        registry=reg, llm_model="gpt-4o-mini", tracer=trace.get_tracer("hybrid_dag_demo"),
        global_proposal_limit=2, max_grace_attempts=1
    )
    async def main():
        print("\n--- INITIAL TASK STATUS TREE ---"); reg.print_status_tree()
        print("\n===== EXECUTING DAG AGENT =====\n"); await agent.run()
        print("\n===== DAG EXECUTION COMPLETE =====\n"); print("\n--- FINAL TASK STATUS TREE ---")
        reg.print_status_tree()
        print("\n--- Final Output for Root Task ---\n")
        root_result = reg.tasks.get("ROOT_INVESTOR_BRIEFING")
        if root_result:
            print(f"Final Result (Status: {root_result.status}):\n{root_result.result}")
        print(f"\nTotal estimated cost: ${sum(t.cost for t in reg.tasks.values()):.4f}")
    asyncio.run(main())