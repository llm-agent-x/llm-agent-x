import asyncio
import hashlib
import uuid
import logging
from typing import Set, Dict, Any, Optional, List, Callable, Union
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from dotenv import load_dotenv
load_dotenv(".env", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DAGAgent")
def get_cost(usage) -> float:
    if not usage: return 0.0
    return 0.00001 * (getattr(usage, "request_tokens", 0) + getattr(usage, "response_tokens", 0))

def ic_dev(*args, **kwargs): pass

def evaluate_prompt(p):    # Replace with your real complexity model
    class Eval: prompt_complexity_score=[0.5]; domain_knowledge=[0.5]
    return Eval()

class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best).")
    def get_successful(self): return self.score > 5
class SplitTask(BaseModel):
    needs_subtasks: bool
    subtasks: List[str] = []
    evaluation: Optional[Any] = None
    def __bool__(self): return self.needs_subtasks

class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"
    result: Optional[str] = None
    split_me: bool = False
    fix_attempts: int = 0
    max_fix_attempts: int = 2
    already_split: bool = False
    cost: float = 0.0
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    siblings: List[str] = Field(default_factory=list)
    dep_results: Dict[str, Any] = Field(default_factory=dict)
    def to_dict_status(self):
        return dict(id=self.id, desc=self.desc, status=self.status, deps=list(self.deps),
                    parent=self.parent, children=list(self.children), siblings=list(self.siblings),
                    result_preview=(self.result[:70] if self.result else None))
    class Config:
        arbitrary_types_allowed = True

class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
    def add_task(self, task: Task):
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists")
        self.tasks[task.id] = task
    def add_dependency(self, task_id: str, dep_id: str):
        if dep_id not in self.tasks: raise ValueError("Dependency does not exist")
        orig_deps = set(self.tasks[task_id].deps)
        self.tasks[task_id].deps.add(dep_id)
        if self.detect_cycle():
            self.tasks[task_id].deps = orig_deps
            raise ValueError("Adding dependency would create cycle")
    def all_tasks(self):
        return list(self.tasks.values())
    def detect_cycle(self) -> bool:
        visited, stack = set(), set()
        def visit(tid):
            if tid in stack: return True
            if tid in visited: return False
            stack.add(tid)
            task = self.tasks[tid]
            for dep in task.deps:
                if dep in self.tasks and visit(dep): return True
            stack.remove(tid); visited.add(tid)
            return False
        return any(visit(tid) for tid in self.tasks)
    def print_status_tree(self):
        for t in self.tasks.values():
            print(f" {t.id}: {t.to_dict_status()}")
    def dependency_columns(self) -> List[Set[str]]:
        """
        Returns a list of sets of task ids representing columns (levels) for parallel execution:
        - All tasks in a column have dependencies only in columns to the left.
        - All dependencies are satisfied for all tasks in a column.
        """
        # Compute in-degree for each task
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.deps:
                if dep in in_degree:
                    in_degree[task.id] += 1

        # Nodes to process for current level ("frontier")
        unassigned = set(self.tasks)
        result: List[Set[str]] = []

        while unassigned:
            # Find tasks with in-degree 0 (no unmet dependencies)
            ready = {tid for tid in unassigned if in_degree[tid] == 0}
            if not ready:
                raise RuntimeError("Cycle detected (should not happen if graph is a DAG)")
            result.append(ready)
            # Remove these from the unassigned set and decrement in-degree of their dependents
            for tid in ready:
                unassigned.remove(tid)
                for t in self.tasks.values():
                    if tid in t.deps and t.id in in_degree:
                        in_degree[t.id] -= 1
        return result

class TaskContext:
    def __init__(self, task_id: str, registry: TaskRegistry, runner: "DAGAgent"):
        self.task_id = task_id
        self.registry = registry
        self.runner = runner
        self.task = self.registry.tasks[task_id]
    def get_dep_result(self, dep_id: str) -> Any:
        task = self.registry.tasks.get(dep_id)
        if not task or task.status != "complete":
            raise KeyError(f"Dependency {dep_id} has not completed.")
        return task.result
    def add_dependency(self, dep_id: str):
        self.registry.add_dependency(self.task_id, dep_id)
    def inject_task(self, desc: str, parent: Optional[str] = None) -> str:
        sub_id = str(uuid.uuid4())
        task = Task(id=sub_id, desc=desc, deps=set(), parent=parent)
        self.registry.add_task(task)
        if parent:
            self.registry.tasks[parent].children.append(sub_id)
            task.siblings = list(self.registry.tasks[parent].children)
        return sub_id
    def set_already_split(self):
        self.task.already_split = True
    def mark_result(self, result: str):
        self.task.result = result

class DAGAgent:
    def __init__(
        self,
        registry: TaskRegistry,
        llm_model: str = "gpt-4o-mini",
        max_fix_attempts: int = 2,
        allow_tools: bool = False,
        mcp_servers: Optional[List[Any]] = [],
        tools: Optional[List[Any]] = []
    ):
        self.registry = registry
        self.max_fix_attempts = max_fix_attempts
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}
        self.splitter = Agent(
            model=llm_model,
            system_prompt="Split the task below into a list of subtasks. If the task is atomic, output []. Output schema: needs_subtasks:bool, subtasks:list[str].",
            output_type=SplitTask
        )
        self.executor = Agent(
            model=llm_model,
            system_prompt="You are tasked to execute or answer the following task with any data from dependencies as context.",
            output_type=str,
            mcp_servers=mcp_servers, tools=tools
        )
        self.aggregator = Agent(
            model=llm_model,
            system_prompt="Summarize or combine the following results into a single, complete answer.",
            output_type=str,
        )
        self.verifier = Agent(
            model=llm_model,
            system_prompt="You must judge if the result given truly and fully answers the task, given all instructions and dependency information.",
            output_type=verification
        )
    async def run(self):
        finished = {}
        pending = set(self.registry.tasks.keys())
        costs = {}
        while pending or self.inflight:
            ready = [
                tid for tid in pending
                if all(self.registry.tasks[d].status == "complete"
                       for d in self.registry.tasks[tid].deps)
                and self.registry.tasks[tid].status not in ("complete","failed","running")
            ]
            if not ready and not self.inflight and pending:
                logger.error("Stalled DAG (cycle/unsatisfiable dependency):")
                for tid in pending:
                    logger.error(f"  - {tid} (waiting on {self.registry.tasks[tid].deps})")
                raise Exception("DAG is not runnable (cycle?)")
            for tid in ready:
                t = self.registry.tasks[tid]
                dep_results = {d: self.registry.tasks[d].result for d in t.deps}
                t.dep_results = dep_results
                t.status = "running"
                ctx = TaskContext(tid, self.registry, self)
                task_future = asyncio.create_task(self._run_taskflow(tid, ctx))
                self.task_futures[tid] = task_future
                self.inflight.add(tid)
            for tid in ready: pending.remove(tid)
            if self.inflight:
                done, _ = await asyncio.wait(
                    [self.task_futures[tid] for tid in list(self.inflight)],
                    return_when=asyncio.FIRST_COMPLETED
                )
            await asyncio.sleep(0.001)
    async def _run_taskflow(self, tid: str, ctx: TaskContext):
        t = ctx.task
        logger.info(f"RUN [{tid}] {t.desc}")
        # PHASE 1: SPLITTING (if not yet split and flagged to split)
        if (t.split_me or not t.already_split) and not t.already_split:
            eval = evaluate_prompt(t.desc)
            if eval.prompt_complexity_score[0] > 0.6:
                logger.info(f" [{tid}] Asking LLM to split (complexity={eval.prompt_complexity_score[0]})")
                # async with self.splitter.run_mcp_servers():
                split_task = await self.splitter.run(user_prompt=t.desc)
                if split_task is None:
                    raise Exception(f"Splitter LLM agent returned None for {tid}")
                if split_task.usage() is not None:
                    t.cost += get_cost(split_task.usage())
                split_result = split_task.output
                if split_result and getattr(split_result, 'needs_subtasks', False) and getattr(split_result, 'subtasks', []):
                    logger.info(f" [{tid}] Split into {len(split_result.subtasks)} subtasks")
                    for subdesc in split_result.subtasks:
                        sub_id = ctx.inject_task(subdesc, parent=tid)
                        ctx.add_dependency(sub_id)
                    ctx.set_already_split()
                    t.status = "waiting"
                    self.inflight.discard(tid)
                    return
                ctx.set_already_split()
        # PHASE 2: EXECUTION
        await self._run_task_execution(ctx)
        t.status = "complete"
        self.inflight.discard(tid)
    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        history = self._build_context(t)
        prompt = f"Task: {t.desc}\n"
        if t.dep_results:
            prompt += f"\nDependency Results:\n" + "\n".join(
                f"- [{dep}]: {str(res)[:100]}" for dep, res in t.dep_results.items()
            )
        prompt += f"\n\nContext chain:\n{history}\n"
        prompt += "If this task is to compile or synthesize from subtasks/children, do so; otherwise, answer directly."
        worked = False
        fix_attempt = 0
        result = None
        while not worked and fix_attempt <= t.max_fix_attempts:
            logger.info(f"   > Executing LLM for {t.desc} (attempt={fix_attempt})")
            print(">>> about to call self.executor.run()")
            async with self.executor.run_mcp_servers():
                agent_res = await self.executor.run(user_prompt=prompt)
            print(">>> finished calling self.executor.run()")
            if agent_res is None:
                raise Exception("LLM agent executor returned None")
            if agent_res.usage() is not None:
                t.cost += get_cost(agent_res.usage())
            result = agent_res.output
            if result is None:
                raise Exception("Agent output is None")
            if await self._verify_task(t, result, ctx):
                worked = True
            else:
                fix_prompt = f"{prompt}\n\nPrevious answer failed: {str(result)}\nPlease retry and fix. ({fix_attempt+1}/{t.max_fix_attempts})"
                result = None
                fix_attempt += 1
                prompt = fix_prompt
        if not worked:
            t.status = "failed"
            t.result = None
            raise Exception(f"Exceeded max fix attempts for {t.desc}")
        t.result = result
        logger.info(f"   > OK: [{t.id}]: {str(result)[:100]}")
    async def _verify_task(self, t: Task, candidate_result: str, ctx: TaskContext) -> bool:
        ver_prompt = (
            f"Task: {t.desc}\nResult: {candidate_result}\n"
            f"Context: {self._build_context(t)}\n\n"
            "Only approve if it truly fully satisfies the task."
        )
        # async with self.verifier.run_mcp_servers():
        vres = await self.verifier.run(user_prompt=ver_prompt)
        if vres is None:
            raise Exception("Verifier LLM agent returned None")
        if vres.usage() is not None:
            t.cost += get_cost(vres.usage())
        vout = vres.output
        if vout is None:
            raise Exception("Verifier output is None")
        logger.info(f"   > Verification: score={vout.score}, reason={vout.reason[:30]}")
        return vout.get_successful()
    def _build_context(self, t: Task) -> str:
        lines = []
        anc = []
        curr = t
        while curr.parent:
            parent = self.registry.tasks.get(curr.parent)
            if not parent: break
            anc.append(f"Parent: {parent.desc}")
            curr = parent
        anc.reverse()
        if anc: lines += anc
        if t.siblings:
            sibs = [self.registry.tasks[s].desc for s in t.siblings if s and s in self.registry.tasks and s != t.id]
            if sibs:
                lines.append("Siblings: " + ", ".join(sibs))
        return "\n".join(lines or ["(No parent/siblings)"])
    def print_tree_statuses(self):
        for t in self.registry.all_tasks():
            logger.info(f" {t.id} :: {t.status} :: cost={t.cost:.4f} :: preview={str(t.result)[:80] if t.result else None}")

if __name__ == "__main__":
    import asyncio
    import logging

    # Optional: Set log to DEBUG for more insight
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DAGAgent")

    # Build a DAG with only a single, large, hard task as the root.
    def build_root_only_registry():
        reg = TaskRegistry()
        root_task = Task(
            id="ROOT",
            desc=(
                "Write a comprehensive research review on the recent advances in AI for drug discovery. "
                "Include background, main achievements, key challenges, and suggest promising future directions. "
                "Aggregate all findings into a cohesive, structured document."
            ),
            split_me=True,  # Strong suggestion to split!
        )
        reg.add_task(root_task)
        return reg

    def print_final_results(reg: TaskRegistry):
        print("\nFinal results:")
        for t in reg.all_tasks():
            print(f"{t.id} :: {t.status} :: {str(t.result)[:150] if t.result else None}")
        print("\nRegistry layout:")
        reg.print_status_tree()

    reg = build_root_only_registry()
    runner = DAGAgent(reg)
    print("\n====== BEGIN ADVANCED ROOT-ONLY EXECUTION ======")
    asyncio.run(runner.run())
    print("\n====== ALL DONE =============")
    runner.print_tree_statuses()
    print_final_results(reg)