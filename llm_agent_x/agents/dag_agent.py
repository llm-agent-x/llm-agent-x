import asyncio
import uuid
import logging
from collections import defaultdict, deque
from hashlib import md5
from os import getenv
from typing import Set, Dict, Any, Optional, List, Tuple, Union
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
    # task_id is implicitly available from the task context, no need to include here.


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

    human_directive: Optional[str] = Field(None, description="A direct, corrective instruction from the operator.")
    current_question: Optional[UserQuestion] = Field(None,
                                                     description="The question an agent is currently asking the human operator.")
    user_response: Optional[str] = Field(None, description="The human operator's response to an agent's question.")

    # --- NEW CONTEXT FIELDS ---
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

    # --- REMOVED: topological_layers method is no longer needed ---


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
    ):
        self.registry = registry
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}
        self.tracer = tracer or trace.get_tracer(__name__)
        self.max_grace_attempts = max_grace_attempts
        self.tools = tools or []

        self.global_proposal_limit = global_proposal_limit
        self.proposed_tasks_buffer: List[Tuple[ProposedSubtask, str]] = []

        # --- AGENT ROLES ---
        # Modify output_type to include UserQuestion
        self.initial_planner = Agent(
            model=llm_model,
            system_prompt="You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. You can link tasks to pre-existing completed tasks. For each new sub-task, decide if it is complex enough to merit further dynamic decomposition by setting `can_request_new_subtasks` to true. If you need a crucial piece of information from the human manager to proceed with planning, you may output a `UserQuestion`.",
            output_type=Union[ExecutionPlan, UserQuestion],
            tools=self.tools,
        )
        self.cycle_breaker = Agent(  # Cycle breaker rarely needs to ask questions, but included for completeness
            model=llm_model,
            system_prompt=(
                "You are a logical validation expert. Your task is to analyze an execution plan and resolve any circular dependencies (cycles). "
                "A cycle is when Task A depends on B, and Task B depends on A (directly or indirectly). "
                "If you find a cycle, you must remove the *least critical* dependency to break it. Use the 'reason' field for each dependency to decide. "
                "Your final output MUST be the complete, corrected ExecutionPlan. If there are no cycles, return the original plan unchanged. If you need a crucial piece of information from the human manager to resolve a complex cycle, you may output a `UserQuestion`."
            ),
            output_type=Union[ExecutionPlan, UserQuestion]
        )
        self.adaptive_decomposer = Agent(
            model=llm_model,
            system_prompt="You are an adaptive expert. Analyze the given task and results of its dependencies. If the task is still too complex, propose a list of new, more granular sub-tasks to achieve it. You MUST provide an `importance` score (1-100) for each proposal, reflecting how critical it is. If you need a crucial piece of information from the human manager to proceed with decomposition, you may output a `UserQuestion`.",
            output_type=Union[List[ProposedSubtask], UserQuestion],
            tools=self.tools,
        )
        self.conflict_resolver = Agent(
            # Conflict resolver rarely needs to ask questions, but included for completeness
            model=llm_model,
            system_prompt=f"You are a ruthless but fair project manager. You have been given a list of proposed tasks that exceeds the budget. Analyze the list and their importance scores. You MUST prune the list by removing the LEAST critical tasks until the total number of tasks is no more than {self.global_proposal_limit}. Return only the final, approved list of tasks. If you need a crucial piece of information from the human manager to resolve a conflict, you may output a `UserQuestion`.",
            output_type=Union[ProposalResolutionPlan, UserQuestion]
        )
        self.executor = Agent(model=llm_model, output_type=Union[str, UserQuestion], tools=self.tools)
        self.verifier = Agent(model=llm_model, output_type=Union[verification, UserQuestion])
        self.retry_analyst = Agent(
            model=llm_model,
            system_prompt=(
                "You are a meticulous quality assurance analyst. Your job is to decide if a failing task is worth retrying. "
                "You will be given the task's original goal and a history of its verification scores. "
                "If the scores are generally increasing and approaching the success threshold of 6, it is worth retrying. "
                "If scores are stagnant or decreasing, it is not. "
                "Crucially, you must provide a concrete and actionable 'next_step_suggestion'. If you need a crucial piece of information from the human manager to make a decision, you may output a `UserQuestion`."
            ),
            output_type=Union[RetryDecision, UserQuestion],
        )

    def _add_llm_data_to_span(self, span: Span, agent_res: AgentRunResult, task: Task):
        if not span or not agent_res: return
        usage, cost = agent_res.usage(), get_cost(agent_res.usage())
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))
        task.cost += cost

    async def _handle_agent_output(self, ctx: TaskContext, agent_res: AgentRunResult, expected_output_type: Any,
                                   agent_role_name: str) -> Tuple[bool, Any]:
        """
        Processes an agent's output. If it's a UserQuestion, pauses the task.
        Stores the full message history of the agent run.
        Returns (is_paused, actual_output).
        """
        t = ctx.task
        self._add_llm_data_to_span(t.span, agent_res, t)

        # Store the full message history for context preservation
        t.last_llm_history = agent_res.all_messages()
        actual_output = agent_res.output

        if isinstance(actual_output, UserQuestion):
            logger.info(
                f"Task [{t.id}] asking human question (Priority: {actual_output.priority}): {actual_output.question[:80]}...")
            t.current_question = actual_output
            t.agent_role_paused = agent_role_name  # Store which agent role asked the question
            t.status = "waiting_for_user_response"
            return True, None  # Indicate paused
        elif isinstance(actual_output, expected_output_type):
            return False, actual_output  # Not paused, here's the normal output
        else:
            logger.warning(
                f"Task [{t.id}] received unexpected output type from agent. Expected {expected_output_type.__name__} or UserQuestion, got {type(actual_output).__name__}.")
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
                            # Clear LLM history if task fails due to upstream
                            task.last_llm_history = None
                            task.agent_role_paused = None

                pending_tasks = all_task_ids - completed_or_failed_or_paused
                ready_to_run_ids = {
                    tid for tid in pending_tasks if
                    self.registry.tasks[tid].status == "pending" and
                    all(self.registry.tasks.get(d, {}).status == "complete" for d in self.registry.tasks[tid].deps)
                }

                for tid in pending_tasks:
                    task = self.registry.tasks[tid]
                    if task.status == 'waiting_for_children' and all(
                            self.registry.tasks[c].status in ['complete', 'failed'] for c in task.children):
                        if any(self.registry.tasks[c].status == 'failed' for c in task.children):
                            task.status, task.result = "failed", "A child task failed, cannot synthesize."
                            # Clear LLM history if child tasks fail
                            task.last_llm_history = None
                            task.agent_role_paused = None
                        else:
                            # Child tasks completed, parent is ready to run (e.g., synthesize)
                            ready_to_run_ids.add(tid)
                            # Also important: reset its status to pending so it can be picked up
                            task.status = "pending"  # Ensures _run_taskflow picks it up for execution/synthesis

                ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

                if not ready and not self.inflight and pending_tasks:
                    logger.error("Stalled DAG!");
                    self.registry.print_status_tree()
                    root_span.set_status(trace.Status(StatusCode.ERROR, "DAG Stalled"));
                    break
                if not pending_tasks and not self.inflight:
                    logger.info("DAG execution complete.");
                    break

                for tid in ready:
                    self.inflight.add(tid)
                    self.task_futures[tid] = asyncio.create_task(self._run_taskflow(tid))

                if not self.task_futures: continue

                # Wait for any task to complete or for a short timeout to check for new directives/tasks
                done, _ = await asyncio.wait(self.task_futures.values(), return_when=asyncio.FIRST_COMPLETED,
                                             timeout=0.1)

                for fut in done:
                    tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                    if tid:
                        self.inflight.discard(tid)
                        del self.task_futures[tid]
                        try:
                            fut.result()
                        except asyncio.CancelledError:
                            logger.info(f"Task [{tid}] was cancelled.")
                        except Exception as e:
                            logger.error(f"Task [{tid}] future failed: {e}")
                            # Ensure task status is updated to failed here if it wasn't already
                            t = self.registry.tasks.get(tid)
                            if t and t.status != "failed":
                                t.status = "failed"
                                t.last_llm_history = None  # Clear history on failure
                                t.agent_role_paused = None

                if self.proposed_tasks_buffer:
                    await self._run_global_resolution()

    async def _run_global_resolution(self):
        logger.info(f"GLOBAL RESOLUTION: Evaluating {len(self.proposed_tasks_buffer)} proposed sub-tasks.")
        approved_proposals = [p[0] for p in self.proposed_tasks_buffer]
        parent_map = {p[0].local_id: p[1] for p in self.proposed_tasks_buffer}  # store parent_id

        if len(approved_proposals) > self.global_proposal_limit:
            logger.warning(
                f"Proposal buffer ({len(approved_proposals)}) exceeds limit ({self.global_proposal_limit}). Engaging resolver."
            )
            prompt_list = [f"local_id: {p.local_id}, importance: {p.importance}, desc: {p.desc}" for p in
                           approved_proposals]

            # Dummy task for global resolver's context and history
            global_resolver_task_id = "GLOBAL_RESOLVER"
            if global_resolver_task_id not in self.registry.tasks:
                self.registry.add_task(
                    Task(id=global_resolver_task_id, desc="Internal task for global conflict resolution.",
                         status="pending"))
            global_resolver_task = self.registry.tasks[global_resolver_task_id]
            global_resolver_ctx = TaskContext(global_resolver_task_id, self.registry)

            resolver_res = await self.conflict_resolver.run(
                user_prompt=f"Prune this list to {self.global_proposal_limit} items: {prompt_list}",
                message_history=global_resolver_task.last_llm_history  # Pass history
            )
            is_paused, resolved_plan = await self._handle_agent_output(
                ctx=global_resolver_ctx,  # Dummy context for logging cost and history
                agent_res=resolver_res,
                expected_output_type=ProposalResolutionPlan,
                agent_role_name="conflict_resolver"  # New: pass agent role name
            )

            # If the conflict resolver asked a question, we cannot proceed with global resolution now.
            # The global_resolver_task's status will be 'waiting_for_user_response'.
            # We clear the buffer so they can be re-proposed later.
            if is_paused:
                logger.warning(
                    f"Conflict resolver for {global_resolver_task_id} paused for human input. Discarding current proposals for now. They will be re-generated later.")
                self.proposed_tasks_buffer = []
                return

            # If the resolver was *just* resumed, the actual result will be in `resolved_plan`.
            # If it had previously been paused and resumed, `global_resolver_task.user_response` would have been used.
            # We need to explicitly check if resolved_plan is None and task is waiting for user response,
            # indicating a pause, or if the agent produced a valid plan.
            if resolved_plan is None and global_resolver_task.status == "waiting_for_user_response":
                logger.warning("Conflict resolver paused, no plan resolved yet.")
                return  # Do not commit proposals yet, wait for human response

            approved_proposals = resolved_plan.approved_tasks

        logger.info(f"Committing {len(approved_proposals)} approved sub-tasks to the graph.")
        self._commit_proposals(approved_proposals, parent_map)
        self.proposed_tasks_buffer = []

    def _commit_proposals(self, approved_proposals: List[ProposedSubtask], proposal_to_parent_map: Dict[str, str]):
        local_to_global_id_map = {}
        for proposal in approved_proposals:
            parent_id = proposal_to_parent_map.get(proposal.local_id)
            if not parent_id: continue
            new_task = Task(id=str(uuid.uuid4())[:8], desc=proposal.desc, parent=parent_id,
                            can_request_new_subtasks=False)  # Newly proposed tasks don't get dynamic decomposition by default
            self.registry.add_task(new_task)
            self.registry.tasks[parent_id].children.append(new_task.id)
            self.registry.add_dependency(parent_id, new_task.id)  # Parent will wait for children for synthesis
            local_to_global_id_map[proposal.local_id] = new_task.id

        for proposal in approved_proposals:
            new_task_global_id = local_to_global_id_map.get(proposal.local_id)
            if not new_task_global_id: continue
            for dep_local_id in proposal.deps:
                dep_global_id = local_to_global_id_map.get(dep_local_id) or (
                    dep_local_id if dep_local_id in self.registry.tasks else None)  # Check if it's an existing global ID
                if dep_global_id: self.registry.add_dependency(new_task_global_id, dep_global_id)

    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)  # Initial status for the span

            # If the task is paused by human or waiting for user response, just return.
            # It will be picked up again when its status changes.
            if t.status in ["paused_by_human", "waiting_for_user_response"]:
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution for now.")
                return

            try:
                # Store parent context for trace continuity if needed within sub-functions
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                if t.status == 'waiting_for_children':
                    # This implies all children have completed, and now this parent needs to synthesize.
                    # It transitions from 'waiting_for_children' to 'running' for execution.
                    t.status = 'running'
                    logger.info(f"SYNTHESIS PHASE for [{t.id}].")
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    t.status = 'planning'
                    logger.info(f"ARCHITECT PHASE for [{t.id}].")
                    await self._run_initial_planning(ctx)
                    # After planning, if it generated children, it waits for them.
                    # If not, it means the planner decided no subtasks were needed, so it's complete.
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        t.already_planned = True  # Only mark as planned if not paused for a question
                        t.status = "waiting_for_children" if t.children else "complete"
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    t.status = 'proposing'
                    logger.info(f"EXPLORER PHASE for [{t.id}].")
                    await self._run_adaptive_decomposition(ctx)
                    # After proposing, it will wait for children (if proposals are approved)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        t.status = "waiting_for_children"
                else:
                    # Regular execution (for leaf tasks, or tasks after waiting for children)
                    t.status = 'running'
                    logger.info(f"WORKER PHASE for [{t.id}].")
                    await self._run_task_execution(ctx)

                # Final status update for the task itself (if not already paused or failed)
                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    t.status = "complete"
                    # Clear LLM history on successful completion
                    t.last_llm_history = None
                    t.agent_role_paused = None

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                t.last_llm_history = None  # Clear history on failure
                t.agent_role_paused = None
                raise
            finally:
                otel_context.detach(otel_ctx_token)  # Detach the context
                span.set_attribute("dag.task.status", t.status)  # Final status for the span

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt = f"Objective: {t.desc}\n\nAvailable completed data sources:\n{context}"

        # Inject human directive if present, *before* passing to LLM
        if t.human_directive:
            prompt += f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            t.human_directive = None  # Clear after use

        plan_res = await self.initial_planner.run(
            user_prompt=prompt,
            message_history=t.last_llm_history  # Pass the full message history
        )
        is_paused, initial_plan = await self._handle_agent_output(
            ctx=ctx, agent_res=plan_res, expected_output_type=ExecutionPlan, agent_role_name="initial_planner"
        )
        if is_paused: return  # Task is paused for user input

        # After initial planning, if subtasks are identified, check for cycles
        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
            fixer_prompt = f"Analyze and fix cycles in this plan:\n\n{initial_plan.model_dump_json(indent=2)}"

            fixed_plan_res = await self.cycle_breaker.run(
                user_prompt=fixer_prompt,
                message_history=t.last_llm_history  # Pass the full message history
            )
            is_paused, plan = await self._handle_agent_output(
                ctx=ctx, agent_res=fixed_plan_res, expected_output_type=ExecutionPlan, agent_role_name="cycle_breaker"
            )
            if is_paused: return  # Task is paused for user input
        else:
            plan = initial_plan  # No subtasks or no issues

        if not plan.needs_subtasks: return

        # Delegate the actual commitment to a new helper method
        await self._process_initial_planning_output(ctx, plan)

    async def _process_initial_planning_output(self, ctx: TaskContext, plan: ExecutionPlan):
        """
        Helper method to process the output of initial planning and commit subtasks.
        This is separated so it can be called directly by InteractiveDAGAgent's resume logic.
        """
        t = ctx.task
        if not plan.needs_subtasks: return

        local_to_global_id_map = {}
        for sub in plan.subtasks:
            new_task = Task(id=str(uuid.uuid4())[:8], desc=sub.desc, parent=t.id,
                            can_request_new_subtasks=sub.can_request_new_subtasks)
            self.registry.add_task(new_task)
            t.children.append(new_task.id)
            self.registry.add_dependency(t.id, new_task.id)
            local_to_global_id_map[sub.local_id] = new_task.id

        for sub in plan.subtasks:
            new_global_id = local_to_global_id_map.get(sub.local_id)
            if not new_global_id: continue
            for dep in sub.deps:
                dep_global_id = local_to_global_id_map.get(dep.local_id) or (
                    dep.local_id if dep.local_id in self.registry.tasks else None)
                if dep_global_id: self.registry.add_dependency(new_global_id, dep_global_id)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt = f"Task: {t.desc}\n\nResults from dependencies:\n{t.dep_results}"

        # Inject human directive if present
        if t.human_directive:
            prompt += f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            t.human_directive = None

        proposals_res = await self.adaptive_decomposer.run(
            user_prompt=prompt,
            message_history=t.last_llm_history  # Pass the full message history
        )
        is_paused, proposals = await self._handle_agent_output(
            ctx=ctx, agent_res=proposals_res, expected_output_type=List[ProposedSubtask],
            agent_role_name="adaptive_decomposer"
        )
        if is_paused: return  # Task is paused for user input

        if proposals:
            logger.info(f"Task [{t.id}] proposing {len(proposals)} new sub-tasks.")
            for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        child_results = {cid: self.registry.tasks[cid].result for cid in t.children if
                         self.registry.tasks[cid].status == 'complete'}
        dep_results = {did: self.registry.tasks[did].result for did in t.deps if
                       self.registry.tasks[did].status == 'complete' and did not in child_results}

        prompt_base = f"Task: {t.desc}\n"
        if child_results:
            prompt_base += "\nSynthesize the results from your sub-tasks into a final answer:\n"
            for cid, res in child_results.items():
                prompt_base += f"- From sub-task '{self.registry.tasks[cid].desc}':\n{res}\n\n"
        elif dep_results:
            prompt_base += "\nUse data from dependencies to inform your answer:\n"
            for did, res in dep_results.items():
                prompt_base += f"- From dependency '{self.registry.tasks[did].desc}':\n{res}\n\n"

        while True:
            current_attempt = t.fix_attempts + t.grace_attempts
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt + 1})

            # Start with base prompt and add any current human directive or previous feedback
            current_prompt = prompt_base
            if t.human_directive:
                current_prompt = f"--- CRITICAL GUIDANCE FROM OPERATOR ---\n{t.human_directive}\n------------------------------\n" + current_prompt
                t.human_directive = None  # Clear after use

            exec_res = await self.executor.run(
                user_prompt=current_prompt,
                message_history=t.last_llm_history  # Pass the full message history
            )
            is_paused, result = await self._handle_agent_output(
                ctx=ctx, agent_res=exec_res, expected_output_type=str, agent_role_name="executor"
            )
            if is_paused: return  # Task is paused for user input

            await self._process_executor_output_for_verification(ctx, result)
            if t.status in ["complete", "failed", "waiting_for_user_response", "paused_by_human"]:
                return

            # If the task status is still "pending" or "running" after verification,
            # it means a retry is in order. The loop will continue.
            # The feedback for the retry is now managed by `t.human_directive` set in `_process_executor_output_for_verification`.

    async def _process_executor_output_for_verification(self, ctx: TaskContext, result: str):
        """Helper to handle verification and retry logic after executor runs."""
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)
        is_successful = verify_task_result.get_successful()

        if is_successful:
            t.result = result
            logger.info(f"COMPLETED [{t.id}]");
            t.status = "complete"  # Explicitly set status to complete
            self._broadcast_state_update(t)
            t.last_llm_history = None  # Clear history on success
            t.agent_role_paused = None
            return

        # If not successful, proceed with retry logic
        t.fix_attempts += 1
        if t.fix_attempts >= t.max_fix_attempts and t.grace_attempts < self.max_grace_attempts:
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\n" \
                             f"Score > 5 is a success. Should we retry?"

            decision_res = await self.retry_analyst.run(
                user_prompt=analyst_prompt,
                message_history=t.last_llm_history
            )
            is_paused, decision = await self._handle_agent_output(
                ctx=ctx, agent_res=decision_res, expected_output_type=RetryDecision, agent_role_name="retry_analyst"
            )
            if is_paused: return  # Task is paused for user input

            if decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion  # Inject suggestion for next attempt
                logger.info(f"[{t.id}] Grace attempt granted. Next step: {decision.next_step_suggestion}")
                # Status remains "running" or "pending" to cause a retry with the new directive
                return

        # If max attempts (including grace) are exceeded
        if (t.fix_attempts + t.grace_attempts) >= (t.max_fix_attempts + self.max_grace_attempts):
            error_msg = f"Exceeded max attempts for task '{t.id}'"
            t.span.set_status(trace.Status(StatusCode.ERROR, error_msg));
            t.status = "failed"
            self._broadcast_state_update(t)
            t.last_llm_history = None  # Clear history on failure
            t.agent_role_paused = None
            raise Exception(error_msg)

        # If just a normal retry (within initial max_fix_attempts)
        # Add feedback as a human directive for the next executor run
        t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
        logger.info(f"[{t.id}] Retrying execution. Feedback: {verify_task_result.reason[:50]}...")
        # Status remains "running" or "pending" to trigger next iteration of _run_task_execution loop.

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."

        vres = await self.verifier.run(
            user_prompt=ver_prompt,
            message_history=t.last_llm_history  # Pass the full message history
        )
        is_paused, vout = await self._handle_agent_output(
            ctx=TaskContext(t.id, self.registry), agent_res=vres, expected_output_type=verification,
            agent_role_name="verifier"
        )
        if is_paused:
            # If verifier asks a question, we must pause the *current* execution path.
            # The execution loop in _run_task_execution will handle this return.
            return verification(reason="Verifier asked a question, pausing.", message_for_user="", score=0)

        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score);
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout

# --- REMOVED: print_topo_table function is no longer needed ---

if __name__ == "__main__":
    try:
        from nest_asyncio import apply;

        apply()
    except ImportError:
        pass
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
        registry=reg,
        llm_model="gpt-4o-mini",
        tracer=trace.get_tracer("hybrid_dag_demo"),
        global_proposal_limit=2,
        max_grace_attempts=1
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