import asyncio
import uuid
import logging
from collections import defaultdict, deque
from hashlib import md5
from os import getenv
from typing import Set, Dict, Any, Optional, List, Tuple
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
    status: str = "pending"  # can be: pending, planning, proposing, waiting_for_children, running, complete, failed
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
        self.tasks[id] = Task(id=id, deps=set(), desc=f"Document: {document_name}", status="complete",
                              result=content.get('markdown_content'))
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
                        "waiting_for_children": "⏸️", "pending": "📋"}
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
        self.initial_planner = Agent(
            model=llm_model,
            system_prompt="You are a master project planner. Your job is to break down a complex objective into a series of smaller, actionable sub-tasks. You can link tasks to pre-existing completed tasks. For each new sub-task, decide if it is complex enough to merit further dynamic decomposition by setting `can_request_new_subtasks` to true.",
            output_type=ExecutionPlan,
            tools=self.tools,
        )
        self.cycle_breaker = Agent(
            model=llm_model,
            system_prompt=(
                "You are a logical validation expert. Your task is to analyze an execution plan and resolve any circular dependencies (cycles). "
                "A cycle is when Task A depends on B, and Task B depends on A (directly or indirectly). "
                "If you find a cycle, you must remove the *least critical* dependency to break it. Use the 'reason' field for each dependency to decide. "
                "Your final output MUST be the complete, corrected ExecutionPlan. If there are no cycles, return the original plan unchanged."
            ),
            output_type=ExecutionPlan
        )
        self.adaptive_decomposer = Agent(
            model=llm_model,
            system_prompt="You are an adaptive expert. Analyze the given task and results of its dependencies. If the task is still too complex, propose a list of new, more granular sub-tasks to achieve it. You MUST provide an `importance` score (1-100) for each proposal, reflecting how critical it is.",
            output_type=List[ProposedSubtask],
            tools=self.tools,
        )
        self.conflict_resolver = Agent(
            model=llm_model,
            system_prompt=f"You are a ruthless but fair project manager. You have been given a list of proposed tasks that exceeds the budget. Analyze the list and their importance scores. You MUST prune the list by removing the LEAST critical tasks until the total number of tasks is no more than {self.global_proposal_limit}. Return only the final, approved list of tasks.",
            output_type=ProposalResolutionPlan
        )
        self.executor = Agent(model=llm_model, output_type=str, tools=self.tools)
        self.verifier = Agent(model=llm_model, output_type=verification)
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
        )

    def _add_llm_data_to_span(self, span: Span, agent_res: AgentRunResult, task: Task):
        if not span or not agent_res: return
        usage, cost = agent_res.usage(), get_cost(agent_res.usage())
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))
        task.cost += cost

    async def run(self):
        with self.tracer.start_as_current_span("DAGAgent.run") as root_span:
            while True:
                all_task_ids = set(self.registry.tasks.keys())
                completed_or_failed = {t.id for t in self.registry.tasks.values() if t.status in ('complete', 'failed')}

                for tid in all_task_ids - completed_or_failed:
                    task = self.registry.tasks[tid]
                    if any(self.registry.tasks.get(d, {}).status == "failed" for d in task.deps):
                        task.status, task.result = "failed", "Upstream dependency failed."

                pending_tasks = all_task_ids - completed_or_failed
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
                        else:
                            ready_to_run_ids.add(tid)

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

                done, _ = await asyncio.wait(self.task_futures.values(), return_when=asyncio.ALL_COMPLETED)

                for fut in done:
                    tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                    if tid:
                        self.inflight.discard(tid)
                        del self.task_futures[tid]
                        try:
                            fut.result()
                        except Exception as e:
                            logger.error(f"Task [{tid}] future failed: {e}")

                if self.proposed_tasks_buffer:
                    await self._run_global_resolution()

    async def _run_global_resolution(self):
        logger.info(f"GLOBAL RESOLUTION: Evaluating {len(self.proposed_tasks_buffer)} proposed sub-tasks.")
        approved_proposals = [p[0] for p in self.proposed_tasks_buffer]
        if len(approved_proposals) > self.global_proposal_limit:
            logger.warning(
                f"Proposal buffer ({len(approved_proposals)}) exceeds limit ({self.global_proposal_limit}). Engaging resolver.")
            prompt_list = [f"local_id: {p.local_id}, importance: {p.importance}, desc: {p.desc}" for p in
                           approved_proposals]
            resolver_res = await self.conflict_resolver.run(
                user_prompt=f"Prune this list to {self.global_proposal_limit} items: {prompt_list}")
            approved_proposals = resolver_res.output.approved_tasks

        logger.info(f"Committing {len(approved_proposals)} approved sub-tasks to the graph.")
        self._commit_proposals(approved_proposals)
        self.proposed_tasks_buffer = []

    def _commit_proposals(self, approved_proposals: List[ProposedSubtask]):
        local_to_global_id_map, proposal_to_parent_map = {}, {p[0].local_id: p[1] for p in self.proposed_tasks_buffer}
        for proposal in approved_proposals:
            parent_id = proposal_to_parent_map.get(proposal.local_id)
            if not parent_id: continue
            new_task = Task(id=str(uuid.uuid4())[:8], desc=proposal.desc, parent=parent_id)
            self.registry.add_task(new_task)
            self.registry.tasks[parent_id].children.append(new_task.id)
            self.registry.add_dependency(parent_id, new_task.id)
            local_to_global_id_map[proposal.local_id] = new_task.id

        for proposal in approved_proposals:
            new_task_global_id = local_to_global_id_map.get(proposal.local_id)
            if not new_task_global_id: continue
            for dep_local_id in proposal.deps:
                dep_global_id = local_to_global_id_map.get(dep_local_id)
                if dep_global_id: self.registry.add_dependency(new_task_global_id, dep_global_id)

    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)
            try:
                if t.status == 'waiting_for_children':
                    t.status = 'running';
                    logger.info(f"SYNTHESIS PHASE for [{t.id}].")
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    t.status = 'planning';
                    logger.info(f"ARCHITECT PHASE for [{t.id}].")
                    await self._run_initial_planning(ctx)
                    t.already_planned = True
                    t.status = "waiting_for_children" if t.children else "complete"
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    t.status = 'proposing';
                    logger.info(f"EXPLORER PHASE for [{t.id}].")
                    await self._run_adaptive_decomposition(ctx)
                    t.status = "waiting_for_children"
                else:
                    t.status = 'running';
                    logger.info(f"WORKER PHASE for [{t.id}].")
                    await self._run_task_execution(ctx)

                if t.status not in ["waiting_for_children", "failed"]: t.status = "complete"
            except Exception as e:
                span.record_exception(e);
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed";
                raise
            finally:
                span.set_attribute("dag.task.status", t.status)

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt = f"Objective: {t.desc}\n\nAvailable completed data sources:\n{context}"

        plan_res = await self.initial_planner.run(user_prompt=prompt)
        self._add_llm_data_to_span(t.span, plan_res, t)
        initial_plan = plan_res.output

        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
            fixer_prompt = f"Analyze and fix cycles in this plan:\n\n{initial_plan.model_dump_json(indent=2)}"
            fixed_plan_res = await self.cycle_breaker.run(user_prompt=fixer_prompt)
            self._add_llm_data_to_span(t.span, fixed_plan_res, t)
            plan = fixed_plan_res.output
        else:
            plan = initial_plan

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
        proposals_res = await self.adaptive_decomposer.run(user_prompt=prompt)
        self._add_llm_data_to_span(t.span, proposals_res, t)
        if proposals := proposals_res.output:
            logger.info(f"Task [{t.id}] proposing {len(proposals)} new sub-tasks.")
            for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        child_results = {cid: self.registry.tasks[cid].result for cid in t.children if
                         self.registry.tasks[cid].status == 'complete'}
        dep_results = {did: self.registry.tasks[did].result for did in t.deps if
                       self.registry.tasks[did].status == 'complete' and did not in child_results}

        prompt = f"Task: {t.desc}\n"
        if child_results:
            prompt += "\nSynthesize the results from your sub-tasks into a final answer:\n"
            for cid, res in child_results.items():
                prompt += f"- From sub-task '{self.registry.tasks[cid].desc}':\n{res}\n\n"
        elif dep_results:
            prompt += "\nUse data from dependencies to inform your answer:\n"
            for did, res in dep_results.items():
                prompt += f"- From dependency '{self.registry.tasks[did].desc}':\n{res}\n\n"

        while True:
            current_attempt = t.fix_attempts + t.grace_attempts
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt + 1})
            exec_res = await self.executor.run(user_prompt=prompt)
            self._add_llm_data_to_span(t.span, exec_res, t)
            result = exec_res.output

            if await self._verify_task(t, result):
                t.result = result;
                logger.info(f"COMPLETED [{t.id}]");
                return

            t.fix_attempts += 1
            if t.fix_attempts >= t.max_fix_attempts and t.grace_attempts < self.max_grace_attempts:
                analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\n" \
                                 f"Score > 5 is a success. Should we retry?"
                decision_res = await self.retry_analyst.run(user_prompt=analyst_prompt)
                self._add_llm_data_to_span(t.span, decision_res, t)
                decision = decision_res.output

                if decision.should_retry:
                    t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                    t.grace_attempts += 1
                    prompt += f"\n\n--- GRACE ATTEMPT GRANTED ---\nAnalyst Suggestion: {decision.next_step_suggestion}\n"
                    continue

            if (t.fix_attempts + t.grace_attempts) >= (t.max_fix_attempts + self.max_grace_attempts):
                error_msg = f"Exceeded max attempts for task '{t.id}'"
                t.span.set_status(trace.Status(StatusCode.ERROR, error_msg));
                raise Exception(error_msg)

            prompt += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\nYour last answer was insufficient. Re-evaluate and try again."

    async def _verify_task(self, t: Task, candidate_result: str) -> bool:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
        vres = await self.verifier.run(user_prompt=ver_prompt)
        self._add_llm_data_to_span(t.span, vres, t);
        vout = vres.output
        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score);
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout.get_successful()


# --- REMOVED: print_topo_table function is no longer needed ---

if __name__ == "__main__":
    try:
        from nest_asyncio import apply; apply()
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
              "Ignore irrelevant data."),
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