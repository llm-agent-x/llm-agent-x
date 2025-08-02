import asyncio
import uuid
import logging
from collections import defaultdict, deque
from hashlib import md5
from rich.table import Table
from os import getenv
from typing import Set, Dict, Any, Optional, List
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
from rich.console import Console

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


class Dependency(BaseModel):
    """Represents a dependency link between two new sub-tasks."""
    reason: str = Field(description="A clear, concise explanation for why this dependency exists.")
    local_id: str = Field(description="The local_id of the task that must be completed first.")


class NewSubtask(BaseModel):
    """Defines a new sub-task to be added to the execution graph."""
    local_id: str = Field(
        description="A temporary, unique ID for this new task (e.g., 'task1', 'research_flights'). Used to define dependencies among other new tasks.")
    desc: str = Field(description="The detailed description of this specific sub-task.")
    deps: List[Dependency] = Field(default_factory=list,
                                   description="A list of dependencies, with reasons, for this sub-task.")


class ExecutionPlan(BaseModel):
    """The execution plan for a complex task. It can either be a simple task or be broken down into sub-tasks."""
    needs_subtasks: bool = Field(description="Set to true if the task is complex and should be broken down.")
    subtasks: List[NewSubtask] = Field(default_factory=list, description="A list of new sub-tasks to be executed.")


class RetryDecision(BaseModel):
    """The decision on whether to attempt another fix for a failing task."""
    should_retry: bool = Field(description="Set to true if the trend of scores suggests success is likely.")
    reason: str = Field(description="A brief explanation for the decision, citing the score trend.")
    next_step_suggestion: str = Field(
        description="A specific, actionable suggestion for the next attempt to improve the result.")


# Update the Task model
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
    # --- NEW: Fields for Adaptive Retries ---
    verification_scores: List[float] = Field(default_factory=list)
    grace_attempts: int = 0

    # Tracing Fields
    otel_context: Optional[Any] = None
    span: Optional[Span] = None

    class Config:
        arbitrary_types_allowed = True


from hashlib import md5

class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
    def add_task(self, task: Task):
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists")
        self.tasks[task.id] = task
    def add_document(self, document_name: str, content: str) -> str:
        if content is None:
            raise ValueError("Document content cannot be None")
        # Always hash bytes
        id = md5(f"{document_name}{content}".encode()).hexdigest()
        self.tasks[id] = Task(
            id=id,
            deps=set(),
            desc=f"Document: {document_name}",
            status="complete",
            result=content
        )
        return id
    def add_dependency(self, task_id: str, dep_id: str):
        if task_id not in self.tasks: raise ValueError(f"Task {task_id} does not exist")
        if dep_id not in self.tasks: raise ValueError(f"Dependency {dep_id} does not exist")
        self.tasks[task_id].deps.add(dep_id)

    def all_tasks(self):
        return list(self.tasks.values())

    def print_status_tree(self):
        logger.info("--- CURRENT TASK REGISTRY STATUS ---")
        root_nodes = [t for t in self.tasks.values() if not t.parent]
        for root in root_nodes:
            self._print_node(root, 0)
        logger.info("------------------------------------")

    def _print_node(self, task: Task, level: int):
        prefix = "  " * level
        logger.info(f"{prefix}- {task.id}: {task.status} | {task.desc[:60]}... | deps: {list(task.deps)}")
        for child_id in task.children:
            if child_id in self.tasks:
                self._print_node(self.tasks[child_id], level + 1)\


    def topological_layers(self) -> List[List[Task]]:
        """
        Returns a list of lists (layers/waves) of Task, where each sublist contains tasks
        that can be executed in parallel after previous waves have finished.
        """
        in_degree = defaultdict(int)
        children = defaultdict(list)
        for t in self.tasks.values():
            for dep in t.deps:
                in_degree[t.id] += 1
                children[dep].append(t.id)
            # Ensure all tasks are in in_degree
            if t.id not in in_degree:
                in_degree[t.id] = 0

        ready = deque([tid for tid, deg in in_degree.items() if deg == 0])
        seen = set()
        levels = []  # Each layer is a list of tasks

        while ready:
            layer = []
            for _ in range(len(ready)):
                tid = ready.popleft()
                if tid in seen:
                    continue
                seen.add(tid)
                task = self.tasks[tid]
                layer.append(task)
                for child_id in children[tid]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        ready.append(child_id)
            if layer:
                levels.append(layer)

        return levels


class TaskContext:
    def __init__(self, task_id: str, registry: TaskRegistry):
        self.task_id = task_id
        self.registry = registry
        self.task = self.registry.tasks[task_id]

    def add_dependency(self, dep_id: str):
        self.registry.add_dependency(self.task_id, dep_id)

    def inject_task(self, desc: str, parent: Optional[str] = None, parent_context: Optional[Any] = None) -> str:
        sub_id = str(uuid.uuid4())[:8]
        task = Task(id=sub_id, desc=desc, parent=parent, otel_context=parent_context)
        self.registry.add_task(task)
        if parent: self.registry.tasks[parent].children.append(sub_id)
        return sub_id


# --- The Corrected and Instrumented DAGAgent ---
class DAGAgent:
    def __init__(
            self,
            registry: TaskRegistry,
            llm_model: str = "gpt-4o-mini",
            tools: List = [],
            tracer: Optional[Tracer] = None,
            max_grace_attempts: int = 2
    ):
        self.registry = registry
        self.max_grace_attempts = max_grace_attempts
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}
        self.tracer = tracer or trace.get_tracer(__name__)

        # --- BUG FIX: Restored the full system prompts ---
        self.planner = Agent(
            model=llm_model,
            system_prompt=(
                "You are an expert project planner. Your job is to break down a complex task into a series of smaller, actionable sub-tasks. "
                "Define a clear plan with dependencies. For each dependency, you MUST provide a 'reason' explaining why it is necessary. "
                "For example, a 'Book Hotel' task should depend on a 'Research Hotels' task, with the reason being 'The hotel to book is determined by the research results.' "
                "Do not worry about creating perfect, cycle-free graphs; focus on capturing all logical connections."
            ),
            output_type=ExecutionPlan
        )

        self.cycle_breaker = Agent(
            model=llm_model,
            system_prompt=(
                "You are a logical validation expert. Your task is to analyze an execution plan and resolve any circular dependencies (cycles). "
                "A cycle is when Task A depends on B, and Task B depends on A (directly or indirectly). "
                "You will be given a plan as a JSON object. Analyze the 'deps' for each task. "
                "If you find a cycle, you must remove the *least critical* dependency to break the cycle. Use the 'reason' field for each dependency to decide which one is less important. "
                "Your final output MUST be the complete, corrected ExecutionPlan, with the cycle-causing dependency removed. Do not add or remove any tasks. "
                "If there are no cycles, return the original plan unchanged."
            ),
            output_type=ExecutionPlan
        )

        self.executor = Agent(model=llm_model, output_type=str, tools=tools)
        self.verifier = Agent(model=llm_model, output_type=verification)

        self.retry_analyst = Agent(
            model=llm_model,
            system_prompt=(
                "You are a meticulous quality assurance analyst. Your job is to decide if a failing task is worth retrying. "
                "You will be given the task's original goal and a history of its verification scores from past attempts. "
                "Analyze the trend of these scores. If the scores are generally increasing and approaching the success threshold of 6, it is worth retrying. "
                "If the scores are stagnant, decreasing, or very low, it is not worth retrying. "
                "Crucially, you must provide a concrete and actionable 'next_step_suggestion' for the next attempt to guide it towards success."
            ),
            output_type=RetryDecision,
        )

    def _add_llm_data_to_span(self, span: Span, agent_res: AgentRunResult, task: Task):
        if not span or not agent_res: return
        usage = agent_res.usage()
        cost = get_cost(usage)

        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, getattr(usage, "request_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, getattr(usage, "response_tokens", 0))
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, getattr(usage, "total_tokens", 0))
        span.set_attribute("llm.cost", cost)

        # This logic is a bit tricky with async contexts. We find the task that owns this span.
        task.cost += cost

    async def run(self):
        with self.tracer.start_as_current_span("DAGAgent.run") as root_span:
            root_span.set_attribute("dag.task_count.initial", len(self.registry.tasks))

            pending = set(self.registry.tasks.keys())
            while pending or self.inflight:
                all_task_ids = set(self.registry.tasks.keys())

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
                    root_span.set_status(trace.Status(StatusCode.ERROR, "DAG Stalled"))
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
                                fut.result()
                            except Exception as e:
                                # --- THIS IS THE FIX ---
                                # The exception was already recorded on the span in _run_taskflow.
                                # The main loop's only job is to log it and update the task's state in the registry.
                                logger.error(f"Task [{tid_of_future}]'s future completed with exception: {e}")
                                task = self.registry.tasks.get(tid_of_future)
                                if task:
                                    # The status is already set to 'failed' where the exception was raised,
                                    # but we can ensure it here as a safeguard.
                                    task.status = "failed"
                                # --- REMOVED THE FOLLOWING LINES ---
                                # if task.span:
                                #     task.span.record_exception(e)
                                #     task.span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                            finally:
                                self.inflight.discard(tid_of_future)
                                del self.task_futures[tid_of_future]

            root_span.set_attribute("dag.task_count.final", len(self.registry.tasks))

    async def _run_taskflow(self, tid: str, ctx: TaskContext):
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}...", context=t.otel_context) as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.description", t.desc)
            span.set_attribute("dag.task.parent_id", t.parent or "None")
            span.set_attribute("dag.task.status", t.status)
            try:
                if not t.already_planned and t.needs_planning:
                    with self.tracer.start_as_current_span("Phase: Planning") as planning_span:
                        t.status = "planning";
                        span.set_attribute("dag.task.status", t.status)
                        logger.info(f"PLANNING for [{t.id}]: {t.desc}")

                        # --- BEGIN: Gather completed tasks context ---
                        completed_tasks = [
                            task for task in self.registry.tasks.values()
                            if task.status == "complete" and task.id != t.id
                        ]
                        if completed_tasks:
                            completed_tasks_info = "\n".join(
                                f"- [{task.id[:8]}] {task.desc[:70]} (Result: {(task.result[:60] + '...') if task.result and len(task.result) > 60 else (task.result or 'n/a')})"
                                for task in completed_tasks
                            )
                            completed_tasks_text = (
                                "You may use these already-completed tasks as dependencies to avoid repeating work:\n"
                                f"{completed_tasks_info}\n"
                                "If needed, explicitly state the task id when using as a dependency."
                            )
                        else:
                            completed_tasks_text = "No pre-existing completed tasks are available to use as dependencies."

                        planning_prompt = (
                            f"{t.desc.strip()}\n\n"
                            f"{completed_tasks_text}\n"
                        )
                        # --- END: Gather completed tasks context ---

                        with self.tracer.start_as_current_span("Agent: Planner") as planner_span:
                            plan_res = await self.planner.run(user_prompt=planning_prompt)
                            self._add_llm_data_to_span(planner_span, plan_res, t)
                        initial_plan = plan_res.output
                        t.already_planned = True

                        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
                            planning_span.add_event("Initial plan generated",
                                                    {"subtask_count": len(initial_plan.subtasks)})
                            with self.tracer.start_as_current_span("Agent: Cycle Breaker") as cb_span:
                                fixer_prompt = (
                                    f"Please analyze the following execution plan for cycles and correct it:\n\n{initial_plan.model_dump_json(indent=2)}"
                                )
                                fixed_plan_res = await self.cycle_breaker.run(user_prompt=fixer_prompt)
                                self._add_llm_data_to_span(cb_span, fixed_plan_res, t)
                            final_plan = fixed_plan_res.output
                            planning_span.add_event("Plan validated", {"final_subtask_count": len(final_plan.subtasks)})
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
                                        self.registry.add_dependency(new_task_global_id,
                                                                     local_to_global_id_map[dep.local_id])
                            t.status = "waiting"
                            self.inflight.discard(tid)
                            logger.info(f"Finished PLANNING for [{t.id}]. Now waiting for {len(t.deps)} children.")
                            return
                with self.tracer.start_as_current_span("Phase: Execution") as exec_phase_span:
                    t.status = "running"
                    span.set_attribute("dag.task.status", t.status)
                    logger.info(f"EXECUTING [{t.id}]: {t.desc}")
                    await self._run_task_execution(ctx)
                self.inflight.discard(tid)
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, f"Task failed: {e}"))
                t.status = "failed"
                raise
            finally:
                span.set_attribute("dag.task.status", t.status)
                span.set_attribute("dag.task.result_preview", (t.result or "")[:200])

    # In the DAGAgent class
    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt = f"Task: {t.desc}\n"
        if t.dep_results:
            prompt += "\nUse the following data from completed dependencies to inform your answer:\n"
            for dep_id, res in t.dep_results.items():
                prompt += f"- Result from sub-task '{self.registry.tasks[dep_id].desc}':\n{str(res)}\n\n"
        if t.children:
            prompt += "Your main goal is to synthesize the results from your sub-tasks into a final, cohesive answer."
        else:
            prompt += "Please execute this task directly and provide a complete answer."

        worked = False
        result = None

        while True:  # Use a while-true loop for more complex exit conditions
            current_attempt = t.fix_attempts + t.grace_attempts
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt + 1})
            with self.tracer.start_as_current_span(f"Agent: Executor (Attempt {current_attempt + 1})") as exec_span:
                agent_res = await self.executor.run(user_prompt=prompt)
                self._add_llm_data_to_span(exec_span, agent_res, t)
            result = agent_res.output

            if await self._verify_task(t, result):
                worked = True
                t.span.add_event("Execution Succeeded and Verified")
                break  # Exit the loop on success

            # --- Verification Failed: Adaptive Retry Logic ---
            t.fix_attempts += 1

            # Check if we've exhausted standard attempts and have grace attempts left
            if t.fix_attempts >= t.max_fix_attempts and t.grace_attempts < self.max_grace_attempts:
                with self.tracer.start_as_current_span("Agent: Retry Analyst") as analyst_span:
                    analyst_span.set_attribute("task.scores_history", str(t.verification_scores))

                    analyst_prompt = (
                        f"Task: '{t.desc}'\n"
                        f"Failed Attempts have yielded verification scores: {t.verification_scores}\n"
                        f"A score > 5 is a success. Based on this trend, is another attempt likely to succeed? "
                        f"Provide a specific suggestion for the next attempt."
                    )
                    decision_res = await self.retry_analyst.run(user_prompt=analyst_prompt)
                    self._add_llm_data_to_span(analyst_span, decision_res, t)
                    decision = decision_res.output

                    analyst_span.set_attribute("decision.should_retry", decision.should_retry)
                    analyst_span.set_attribute("decision.reason", decision.reason)

                    if decision.should_retry:
                        t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                        t.grace_attempts += 1
                        prompt += (f"\n\n--- GRACE ATTEMPT GRANTED ---\n"
                                   f"Analyst Suggestion: {decision.next_step_suggestion}\n"
                                   f"Please re-evaluate and try again, incorporating this suggestion.")
                        continue  # Continue to the next iteration of the loop

            # If we reach here, it means we're either out of all attempts or the analyst denied a retry
            if (t.fix_attempts + t.grace_attempts) >= (t.max_fix_attempts + self.max_grace_attempts):
                t.status = "failed"
                t.result = "Failed to produce a satisfactory result after multiple standard and grace attempts."
                error_msg = f"Exceeded max attempts ({t.max_fix_attempts} standard, {t.grace_attempts} grace) for {t.desc}"
                t.span.set_status(trace.Status(StatusCode.ERROR, error_msg))
                raise Exception(error_msg)

            # Standard retry: update prompt and continue
            prompt += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\nYour last answer was insufficient. Re-evaluate and try again."

        # This part only runs if the loop was broken by success
        t.result = result
        t.status = "complete"
        logger.info(f"COMPLETED [{t.id}]")

    async def _verify_task(self, t: Task, candidate_result: str) -> bool:
        with self.tracer.start_as_current_span("Agent: Verifier") as verifier_span:
            ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
            vres = await self.verifier.run(user_prompt=ver_prompt)
            self._add_llm_data_to_span(verifier_span, vres, t)
            vout = vres.output

            # --- NEW: Store the score for the analyst ---
            t.verification_scores.append(vout.score)

            verifier_span.set_attribute("verification.score", vout.score)
            verifier_span.set_attribute("verification.reason", vout.reason)
            logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
            return vout.get_successful()

def print_topo_table(layers, show_status=False):

    console = Console()
    table = Table(title="Topological Task Waves")
    for i in range(len(layers)):
        table.add_column(f"Wave {i}")

    if not layers:
        print("No tasks found.")
        return
    max_len = max(len(layer) for layer in layers)
    for row in range(max_len):
        row_cells = []
        for wave in layers:
            if row < len(wave):
                t = wave[row]
                s = f"{t.id[:6]}: {t.desc[:20]}"
                if show_status:
                    s += f" [{t.status}]"
                row_cells.append(s)
            else:
                row_cells.append("")
        table.add_row(*row_cells)
    console.print(table)


if __name__ == "__main__":
    import sys
    import random
    from rich.table import Table
    from rich.console import Console
    from nest_asyncio import apply

    apply()
    logger.info("==== BEGIN DEMO FOR DOCUMENT-DAG INTEGRATION ====")

    # --- Build registry and add documents
    reg = TaskRegistry()
    doc1_content = (
        "Apple Inc. (AAPL) Q2 2024 Financial Report\n\n"
        "1. **Revenue**: \n"
        "- Total Revenue: $94.5 billion (up 5% year-over-year)\n"
        "- Product Segments:\n"
        "  - iPhones: $50.5 billion (up 8% due to strong demand for the iPhone 14)\n"
        "  - Mac: $10.5 billion (down 2% due to supply chain issues)\n"
        "  - iPads: $8.0 billion (up 10% driven by education sales)\n"
        "  - Services: $22.0 billion (up 15%, a new record for this segment)\n"
        "\n2. **Net Income**:\n"
        "- Net Income: $25.1 billion\n"
        "- Earnings Per Share (EPS): $1.55\n"
        "\n3. **Operating Expenses**:\n"
        "- Total Operating Expenses: $65.3 billion\n"
        "- R&D Expenses: $19.0 billion\n"
        "- SG&A Expenses: $46.3 billion\n"
        "\n4. **Cash Flow**:\n"
        "- Free Cash Flow: $27.5 billion\n"
        "- Cash on Hand: $104.6 billion\n"
        "\n5. **Forward Guidance**:\n"
        "- Expect Q3 revenue to decline slightly due to macroeconomic headwinds, but services revenue is expected to continue growing.\n"
        "\n6. **Market Position**:\n"
        "- Apple remains the market leader in premium smartphones with a share of 50% in the high-end segment.\n"
        "- Strong brand loyalty and robust ecosystem continue to drive engagement.\n"
        "\n7. **Risks**:\n"
        "- Potential supply chain disruptions due to geopolitical tensions.\n"
        "- Inflation concerns affecting consumer discretionary spending."
    )

    doc2_content = (
        "Current Market Sentiment for Apple Inc. (AAPL)\n\n"
        "1. **Analyst Sentiment**:\n"
        "- Overall Analyst Rating: **Buy**\n"
        "- Analysts remain bullish on Apple's potential, citing strong demand for the iPhone and robust growth in services.\n"
        "\n2. **Recent Events**:\n"
        "- Recent reports highlight Apple's strong performance in China, outpacing competitors.\n"
        "- Positive reception of the new iPhone model launching in early Q3 2024.\n"
        "\n3. **Competitor Insights**:\n"
        "- Competitors are struggling with inventory issues, whereas Apple has maintained positive supply chain management.\n"
        "- Samsung's Galaxy S series saw decreased sales, leaving a gap that Apple is poised to fill.\n"
        "\n4. **Consumer Insights**:\n"
        "- Surveys show that 68% of current iPhone users plan to upgrade to the latest model.\n"
        "- Increasing interest in Apple's services (such as Apple TV+ and Apple Music) contributes to overall consumer satisfaction.\n"
        "\n5. **Economic Factors**:\n"
        "- Analysts caution that inflation rates are high and suggest a possible economic slowdown could dampen sales in Q4 2024.\n"
        "- The Federal Reserve's stance on interest rates could impact tech stock valuations."
    )

    # Adding documents to registry
    doc1_id = reg.add_document("Apple Q2 2024 Financial Report", doc1_content)
    doc2_id = reg.add_document("Current Market Sentiment Analysis", doc2_content)

    # --- Root task that should plan to use these documents
    root_task = Task(
        id="ROOT_PROJECT_ANALYSIS",
        desc="Perform project analysis using any relevant document data in the system. " +
             "Summarize sections of the Q2 2024 Financial Report and any useful numerical info from the Market Sentiment Analysis. " +
             "Where possible, reuse completed tasks/documents as dependencies.",
        needs_planning=True,
    )
    reg.add_task(root_task)

    # --- Optionally: Show topo layers before execution
    print("\n--- TOPOLOGICAL TABLE BEFORE DAG EXECUTION ---")
    layers = reg.topological_layers()
    print_topo_table(layers, show_status=True)

    # ---- Create mock or real agent; use a mock agent for testing/demo --------
    # If running for real, you can use your DAGAgent as-is (below).
    from opentelemetry import trace as ot_trace

    # Simple agent with minimal retry to show document re-use.
    agent = DAGAgent(
        registry=reg,
        llm_model="gpt-4o-mini",  # (or None if using a local test/mocked agent)
        tools=[],  # Add any relevant tools you want to use
        tracer=ot_trace.get_tracer("doc_dag_demo"),
        max_grace_attempts=1
    )


    # ---- Run the DAG agent
    async def main():
        print("\n===== EXECUTING DAG AGENT =====\n")
        reg.print_status_tree()
        await agent.run()
        print("\n===== DAG EXECUTION COMPLETE =====\n")
        reg.print_status_tree()
        print("\n--- TOPOLOGICAL TABLE AFTER DAG EXECUTION ---")
        layers = reg.topological_layers()
        print_topo_table(layers, show_status=True)
        print("\n--- Final Output for Root Task ---\n")
        root_result_task = reg.tasks.get("ROOT_PROJECT_ANALYSIS")
        if root_result_task:
            print(f"Result ({root_result_task.status}):\n{root_result_task.result}")
        print(f"\nTotal estimated cost: ${sum(t.cost for t in reg.all_tasks()):.4f}")


    asyncio.run(main())