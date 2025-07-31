import asyncio
import uuid
import logging
from typing import Set, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from dotenv import load_dotenv

# --- NEW: OpenTelemetry Imports ---
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Tracer, Span
from openinference.semconv.trace import SpanAttributes

load_dotenv(".env", override=True)

# --- Basic Setup (Unchanged) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DAGAgent")


def get_cost(usage) -> float:
    if not usage: return 0.0
    input_cost = 0.15 / 1_000_000
    output_cost = 0.60 / 1_000_000
    return (input_cost * getattr(usage, "request_tokens", 0)) + (output_cost * getattr(usage, "response_tokens", 0))


class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best).")

    def get_successful(self): return self.score > 5


class Dependency(BaseModel):
    local_id: str = Field(description="The local_id of the task that must be completed first.")
    reason: str = Field(description="A clear, concise explanation for why this dependency exists.")


class NewSubtask(BaseModel):
    local_id: str = Field(description="A temporary, unique ID for this new task (e.g., 'task1', 'research_flights').")
    desc: str = Field(description="The detailed description of this specific sub-task.")
    deps: List[Dependency] = Field(default_factory=list,
                                   description="A list of dependencies, with reasons, for this sub-task.")


class ExecutionPlan(BaseModel):
    needs_subtasks: bool = Field(description="Set to true if the task is complex and should be broken down.")
    subtasks: List[NewSubtask] = Field(default_factory=list, description="A list of new sub-tasks to be executed.")


class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"
    result: Optional[str] = None
    needs_planning: bool = False
    already_planned: bool = False
    fix_attempts: int = 0
    max_fix_attempts: int = 2
    cost: float = 0.0
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    siblings: List[str] = Field(default_factory=list)
    dep_results: Dict[str, Any] = Field(default_factory=dict)

    # --- NEW: Tracing Fields ---
    otel_context: Optional[Any] = None  # Stores the parent's OTel context for linking
    span: Optional[Span] = None  # Holds the active span for this task during its run

    class Config:
        arbitrary_types_allowed = True


class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        if task.id in self.tasks: raise ValueError(f"Task {task.id} already exists")
        self.tasks[task.id] = task

    def add_dependency(self, task_id: str, dep_id: str):
        if task_id not in self.tasks: raise ValueError(f"Task {task_id} does not exist")
        if dep_id not in self.tasks: raise ValueError(f"Dependency {dep_id} does not exist")
        self.tasks[task_id].deps.add(dep_id)

    def all_tasks(self):
        return list(self.tasks.values())

    def print_status_tree(self):
        logger.info("--- CURRENT TASK REGISTRY STATUS ---")
        root_nodes = [t for t in self.tasks.values() if not t.parent]
        for root in root_nodes: self._print_node(root, 0)
        logger.info("------------------------------------")

    def _print_node(self, task: Task, level: int):
        prefix = "  " * level
        logger.info(f"{prefix}- {task.id}: {task.status} | {task.desc[:60]}... | deps: {list(task.deps)}")
        for child_id in task.children:
            if child_id in self.tasks: self._print_node(self.tasks[child_id], level + 1)


class TaskContext:
    def __init__(self, task_id: str, registry: TaskRegistry):
        self.task_id = task_id
        self.registry = registry
        self.task = self.registry.tasks[task_id]

    def add_dependency(self, dep_id: str):
        self.registry.add_dependency(self.task_id, dep_id)

    # --- MODIFIED: To accept and store the parent's tracing context ---
    def inject_task(self, desc: str, parent: Optional[str] = None, parent_context: Optional[Any] = None) -> str:
        sub_id = str(uuid.uuid4())[:8]
        task = Task(id=sub_id, desc=desc, parent=parent, otel_context=parent_context)
        self.registry.add_task(task)
        if parent: self.registry.tasks[parent].children.append(sub_id)
        return sub_id


class DAGAgent:
    def __init__(
            self,
            registry: TaskRegistry,
            llm_model: str = "gpt-4o-mini",
            tools: List = [],
            tracer: Optional[Tracer] = None
    ):
        self.registry = registry
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}
        # --- MODIFIED: Use injected tracer or get a default one ---
        self.tracer = tracer or trace.get_tracer(__name__)

        # Agents remain the same
        self.planner = Agent(model=llm_model, system_prompt=..., output_type=ExecutionPlan)
        self.cycle_breaker = Agent(model=llm_model, system_prompt=..., output_type=ExecutionPlan)
        self.executor = Agent(model=llm_model, output_type=str, tools=tools)
        self.verifier = Agent(model=llm_model, output_type=verification)

    # --- NEW: Helper for consistently adding LLM data to spans ---
    def _add_llm_data_to_span(self, span: Span, agent_res: AgentRunResult):
        if not span or not agent_res: return
        usage = agent_res.usage()
        cost = get_cost(usage)

        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, getattr(usage, "total_tokens", 0))
        span.set_attribute("llm.cost", cost)

        # Find the task associated with this span's context and add the cost
        span_ctx = trace.get_current_span().get_span_context()
        for task in self.registry.all_tasks():
            if task.span and task.span.get_span_context() == span_ctx:
                task.cost += cost
                break

    async def run(self):
        # --- MODIFIED: Wrap the entire run in a root span ---
        with self.tracer.start_as_current_span("DAGAgent.run") as root_span:
            root_span.set_attribute("dag.task_count.initial", len(self.registry.tasks))

            pending = set(self.registry.tasks.keys())
            while pending or self.inflight:
                all_task_ids = set(self.registry.tasks.keys())

                # Failure Propagation
                current_pending = all_task_ids - {t.id for t in self.registry.tasks.values() if
                                                  t.status in ('complete', 'failed')}
                for tid in list(current_pending):
                    task = self.registry.tasks[tid]
                    failed_deps = [d for d in task.deps if self.registry.tasks.get(d, {}).status == "failed"]
                    if failed_deps:
                        logger.warning(f"Task [{tid}] is failing because its dependencies {failed_deps} failed.")
                        task.status = "failed"
                        task.result = f"Upstream dependency failure: {failed_deps}"

                pending = all_task_ids - {t.id for t in self.registry.tasks.values() if
                                          t.status in ('complete', 'failed')}
                ready = [tid for tid in pending if all(self.registry.tasks.get(d, {}).status == "complete" for d in
                                                       self.registry.tasks[tid].deps) and tid not in self.inflight]

                if not ready and not self.inflight and pending:
                    logger.error("Stalled DAG (cycle/unsatisfiable dependency):")
                    self.registry.print_status_tree()
                    root_span.add_event("DAG Stalled", {"pending_tasks": list(pending)})
                    root_span.set_status(trace.Status(trace.StatusCode.ERROR, "DAG Stalled"))
                    break

                for tid in ready:
                    self.inflight.add(tid)
                    ctx = TaskContext(tid, self.registry)
                    task_future = asyncio.create_task(self._run_taskflow(tid, ctx))
                    self.task_futures[tid] = task_future

                if self.inflight:
                    done, _ = await asyncio.wait([self.task_futures[tid] for tid in list(self.inflight)],
                                                 return_when=asyncio.FIRST_COMPLETED)
                    for fut in done:
                        tid_of_future = next((tid for tid, task_fut in self.task_futures.items() if task_fut == fut),
                                             None)
                        if tid_of_future:
                            try:
                                fut.result()  # Check for exceptions
                            except Exception as e:
                                logger.error(f"Task [{tid_of_future}] failed with exception: {e}")
                                task = self.registry.tasks.get(tid_of_future)
                                if task:
                                    task.status = "failed"
                                    # --- MODIFIED: Record exception on the correct span ---
                                    if task.span:
                                        task.span.record_exception(e)
                                        task.span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                            finally:
                                self.inflight.discard(tid_of_future)
                                del self.task_futures[tid_of_future]

                await asyncio.sleep(0.01)

            root_span.set_attribute("dag.task_count.final", len(self.registry.tasks))

    async def _run_taskflow(self, tid: str, ctx: TaskContext):
        t = ctx.task
        # --- MODIFIED: Create a span for the entire task lifecycle, using stored context ---
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}...", context=t.otel_context) as span:
            t.span = span  # Store the active span on the task object
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.description", t.desc)
            span.set_attribute("dag.task.parent_id", t.parent or "None")
            span.set_attribute("dag.task.status", t.status)

            try:
                # --- PHASE 1: DYNAMIC PLANNING ---
                if not t.already_planned and t.needs_planning:
                    with self.tracer.start_as_current_span("Phase: Planning") as planning_span:
                        t.status = "planning"
                        span.set_attribute("dag.task.status", t.status)
                        logger.info(f"PLANNING for [{t.id}]: {t.desc}")

                        with self.tracer.start_as_current_span("Agent: Planner") as planner_span:
                            plan_res = await self.planner.run(user_prompt=t.desc)
                            self._add_llm_data_to_span(planner_span, plan_res)
                        initial_plan = plan_res.output
                        t.already_planned = True

                        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
                            planning_span.add_event("Initial plan generated",
                                                    {"subtask_count": len(initial_plan.subtasks)})

                            with self.tracer.start_as_current_span("Agent: Cycle Breaker") as cb_span:
                                fixer_prompt = f"Please analyze...{initial_plan.model_dump_json(indent=2)}"
                                fixed_plan_res = await self.cycle_breaker.run(user_prompt=fixer_prompt)
                                self._add_llm_data_to_span(cb_span, fixed_plan_res)
                            final_plan = fixed_plan_res.output

                            planning_span.add_event("Plan validated by Cycle Breaker",
                                                    {"final_subtask_count": len(final_plan.subtasks)})

                            # --- CRITICAL: Capture parent context before creating children ---
                            parent_context = otel_context.get_current()

                            local_to_global_id_map: Dict[str, str] = {}
                            for sub in final_plan.subtasks:
                                new_task_id = ctx.inject_task(sub.desc, parent=t.id, parent_context=parent_context)
                                local_to_global_id_map[sub.local_id] = new_task_id
                                ctx.add_dependency(new_task_id)

                            for sub in final_plan.subtasks:
                                new_task_global_id = local_to_global_id_map[sub.local_id]
                                for dep in sub.deps:
                                    if dep.local_id in local_to_global_id_map:
                                        global_dep_id = local_to_global_id_map[dep.local_id]
                                        self.registry.add_dependency(new_task_global_id, global_dep_id)

                            t.status = "waiting"
                            self.inflight.discard(tid)
                            logger.info(f"Finished PLANNING for [{t.id}]. Now waiting for {len(t.deps)} children.")
                            return  # Exit early, this task is now a manager

                # --- PHASE 2: EXECUTION / SYNTHESIS ---
                with self.tracer.start_as_current_span("Phase: Execution") as exec_phase_span:
                    t.status = "running"
                    span.set_attribute("dag.task.status", t.status)
                    logger.info(f"EXECUTING [{t.id}]: {t.desc}")
                    await self._run_task_execution(ctx)

                self.inflight.discard(tid)

            except Exception as e:
                # Catch-all for any unhandled exception within the taskflow
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, f"Task failed in execution: {e}"))
                t.status = "failed"
                raise  # Re-raise to be handled by the main loop's exception handler
            finally:
                span.set_attribute("dag.task.status", t.status)
                span.set_attribute("dag.task.result_preview", (t.result or "")[:200])

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt = f"Task: {t.desc}\n..."  # Prompt building logic is the same

        worked = False
        fix_attempt = 0
        result = None
        while not worked and fix_attempt <= t.max_fix_attempts:
            t.span.add_event(f"Execution Attempt", {"attempt": fix_attempt + 1})
            with self.tracer.start_as_current_span(f"Agent: Executor (Attempt {fix_attempt + 1})") as exec_span:
                agent_res = await self.executor.run(user_prompt=prompt)
                self._add_llm_data_to_span(exec_span, agent_res)
            result = agent_res.output

            if await self._verify_task(t, result):
                worked = True
                t.span.add_event("Execution Succeeded and Verified")
            else:
                prompt += f"\n\n--- PREVIOUS ATTEMPT FAILED ---"
                t.span.add_event("Execution Attempt Failed Verification")
                result = None
                fix_attempt += 1

        if not worked:
            t.status = "failed"
            t.result = "Failed to produce a satisfactory result after multiple attempts."
            logger.error(f"Task [{t.id}] failed after {t.max_fix_attempts + 1} attempts.")
            t.span.set_status(trace.Status(trace.StatusCode.ERROR, "Exceeded max fix attempts"))
            raise Exception(f"Exceeded max fix attempts for {t.desc}")

        t.result = result
        t.status = "complete"
        logger.info(f"COMPLETED [{t.id}]")

    async def _verify_task(self, t: Task, candidate_result: str) -> bool:
        with self.tracer.start_as_current_span("Agent: Verifier") as verifier_span:
            ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\n..."
            vres = await self.verifier.run(user_prompt=ver_prompt)
            self._add_llm_data_to_span(verifier_span, vres)
            vout = vres.output

            verifier_span.set_attribute("verification.score", vout.score)
            verifier_span.set_attribute("verification.reason", vout.reason)

            logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
            return vout.get_successful()


if __name__ == "__main__":
    from llm_agent_x.tools.brave_web_search import brave_web_search
    from nest_asyncio import apply

    apply()

    # --- NEW: OpenTelemetry SDK Boilerplate Setup ---
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    # Get a tracer for the main execution context
    main_tracer = trace.get_tracer("dag_agent_main")


    def build_dynamic_planning_dag():
        # ... (function is unchanged)
        reg = TaskRegistry()
        root_task = Task(id="ROOT_PLANNER", desc=..., needs_planning=True)
        reg.add_task(root_task)
        return reg


    # ... (dummy tool functions are unchanged) ...
    tools = [brave_web_search, ...]


    async def main():
        # --- MODIFIED: Use the tracer ---
        with main_tracer.start_as_current_span("main_execution"):
            reg = build_dynamic_planning_dag()
            runner = DAGAgent(reg, tools=tools, tracer=main_tracer)

            print("\n====== BEGIN DYNAMIC PLANNING EXECUTION ======")
            reg.print_status_tree()

            await runner.run()

            print("\n====== ALL DONE =============")
            print("Final state:")
            reg.print_status_tree()

            print("\n--- FINAL RESULTS ---")
            root_result_task = runner.registry.tasks.get("ROOT_PLANNER")
            if root_result_task:
                print(f"** Final Itinerary for ROOT_PLANNER ({root_result_task.status}) **\n{root_result_task.result}")

            total_cost = sum(t.cost for t in reg.all_tasks())
            print(f"\nTotal estimated cost: ${total_cost:.4f}")


    asyncio.run(main())