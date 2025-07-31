import asyncio
import uuid
import logging
from typing import Set, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# --- Basic Setup (Unchanged) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DAGAgent")


def get_cost(usage) -> float:
    if not usage: return 0.0
    # Using GPT-4o-mini pricing for example
    input_cost = 0.15 / 1_000_000
    output_cost = 0.60 / 1_000_000
    return (input_cost * getattr(usage, "request_tokens", 0)) + (output_cost * getattr(usage, "response_tokens", 0))


class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best).")

    def get_successful(self): return self.score > 5


# --- NEW/MODIFIED Pydantic Models for Explained Dependencies and Cycle Breaking ---

class Dependency(BaseModel):
    """Represents a dependency link between two new sub-tasks."""
    local_id: str = Field(description="The local_id of the task that must be completed first.")
    reason: str = Field(description="A clear, concise explanation for why this dependency exists.")


class NewSubtask(BaseModel):
    """Defines a new sub-task to be added to the execution graph."""
    local_id: str = Field(
        description="A temporary, unique ID for this new task (e.g., 'task1', 'research_flights'). Used to define dependencies among other new tasks.")
    desc: str = Field(description="The detailed description of this specific sub-task.")
    deps: List[Dependency] = Field(default_factory=list,
                                   description="A list of dependencies, each with a reason, for this sub-task.")


class ExecutionPlan(BaseModel):
    """The execution plan for a complex task. It can either be a simple task or be broken down into sub-tasks."""
    needs_subtasks: bool = Field(description="Set to true if the task is complex and should be broken down.")
    subtasks: List[NewSubtask] = Field(default_factory=list, description="A list of new sub-tasks to be executed.")


# --- Models for Core DAG Structure (Unchanged) ---

class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"  # pending, planning, waiting, running, complete, failed
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

    def to_dict_status(self):
        return dict(id=self.id, desc=self.desc[:50] + "...", status=self.status, deps=list(self.deps),
                    parent=self.parent, children=list(self.children),
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
                self._print_node(self.tasks[child_id], level + 1)


class TaskContext:
    def __init__(self, task_id: str, registry: TaskRegistry):
        self.task_id = task_id
        self.registry = registry
        self.task = self.registry.tasks[task_id]

    def add_dependency(self, dep_id: str):
        self.registry.add_dependency(self.task_id, dep_id)

    def inject_task(self, desc: str, parent: Optional[str] = None) -> str:
        sub_id = str(uuid.uuid4())[:8]
        task = Task(id=sub_id, desc=desc, parent=parent)
        self.registry.add_task(task)
        if parent:
            self.registry.tasks[parent].children.append(sub_id)
        return sub_id


# --- THE REWRITTEN DAGAgent WITH CYCLE BREAKER ---

class DAGAgent:
    def __init__(
            self,
            registry: TaskRegistry,
            llm_model: str = "gpt-4o-mini",
            tools: List = [],
    ):
        self.registry = registry
        self.inflight = set()
        self.task_futures: Dict[str, asyncio.Task] = {}

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

        # --- NEW "Cycle Breaker" Agent ---
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

    async def run(self):
        pending = set(self.registry.tasks.keys())
        while pending or self.inflight:
            all_task_ids = set(self.registry.tasks.keys())

            # --- Failure Propagation ---
            current_pending = all_task_ids - {t.id for t in self.registry.tasks.values() if
                                              t.status in ('complete', 'failed')}
            for tid in list(current_pending):
                task = self.registry.tasks[tid]
                failed_deps = [d for d in task.deps if self.registry.tasks.get(d, {}).status == "failed"]
                if failed_deps:
                    logger.warning(f"Task [{tid}] is failing because its dependencies {failed_deps} failed.")
                    task.status = "failed"
                    task.result = f"Upstream dependency failure: {failed_deps}"

            # Re-evaluate ready tasks in every loop
            pending = all_task_ids - {t.id for t in self.registry.tasks.values() if t.status in ('complete', 'failed')}
            ready = [
                tid for tid in pending
                if all(self.registry.tasks.get(d, {}).status == "complete" for d in self.registry.tasks[tid].deps)
                   and tid not in self.inflight
            ]

            if not ready and not self.inflight and pending:
                logger.error("Stalled DAG (cycle/unsatisfiable dependency):")
                self.registry.print_status_tree()
                # Instead of raising an exception, we can just break the loop
                # This is useful if we want to inspect the final state without a crash
                break

            for tid in ready:
                self.inflight.add(tid)
                ctx = TaskContext(tid, self.registry)
                task_future = asyncio.create_task(self._run_taskflow(tid, ctx))
                self.task_futures[tid] = task_future

            if self.inflight:
                done, _ = await asyncio.wait(
                    [self.task_futures[tid] for tid in list(self.inflight)],
                    return_when=asyncio.FIRST_COMPLETED
                )
                # Handle exceptions for better debugging
                for fut in done:
                    tid_of_future = next((tid for tid, task_fut in self.task_futures.items() if task_fut == fut), None)
                    if tid_of_future:
                        try:
                            fut.result()
                        except Exception as e:
                            logger.error(f"Task [{tid_of_future}] failed with exception: {e}")
                            self.registry.tasks[tid_of_future].status = "failed"
                        del self.task_futures[tid_of_future]

            await asyncio.sleep(0.01)

    async def _run_taskflow(self, tid: str, ctx: TaskContext):
        t = ctx.task

        # --- PHASE 1: DYNAMIC PLANNING (NOW WITH A FIXER STEP) ---
        if not t.already_planned and t.needs_planning:
            t.status = "planning"
            logger.info(f"PLANNING for [{t.id}]: {t.desc}")

            # Step 1: Generate the initial, potentially flawed plan
            plan_res = await self.planner.run(user_prompt=t.desc)
            t.cost += get_cost(plan_res.usage())
            initial_plan = plan_res.output
            t.already_planned = True

            if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
                logger.info(
                    f"Initial plan for [{t.id}] creates {len(initial_plan.subtasks)} sub-tasks. Validating for cycles...")

                # Step 2: Send the plan to the cycle breaker for validation and fixing
                fixer_prompt = (
                    "Please analyze the following execution plan for cycles and correct it:\n\n"
                    f"{initial_plan.model_dump_json(indent=2)}"
                )
                fixed_plan_res = await self.cycle_breaker.run(user_prompt=fixer_prompt)
                t.cost += get_cost(fixed_plan_res.usage())
                final_plan = fixed_plan_res.output

                logger.info(f"Cycle breaker has validated the plan. Proceeding with {len(final_plan.subtasks)} tasks.")

                # You can set a breakpoint here to inspect the initial vs. final plan!
                # print("--- INITIAL PLAN ---\n", initial_plan.model_dump_json(indent=2))
                # print("--- FINAL PLAN ---\n", final_plan.model_dump_json(indent=2))
                # import pdb; pdb.set_trace()

                # Step 3: Inject the validated tasks into the DAG
                local_to_global_id_map: Dict[str, str] = {}
                for sub in final_plan.subtasks:
                    new_task_id = ctx.inject_task(sub.desc, parent=t.id)
                    local_to_global_id_map[sub.local_id] = new_task_id
                    ctx.add_dependency(new_task_id)

                for sub in final_plan.subtasks:
                    new_task_global_id = local_to_global_id_map[sub.local_id]
                    for dep in sub.deps:  # Iterate through Dependency objects
                        local_dep_id = dep.local_id
                        if local_dep_id in local_to_global_id_map:
                            global_dep_id = local_to_global_id_map[local_dep_id]
                            self.registry.add_dependency(new_task_global_id, global_dep_id)

                t.status = "waiting"
                self.inflight.discard(tid)
                logger.info(f"Finished PLANNING for [{t.id}]. Now waiting for {len(t.deps)} children.")
                self.registry.print_status_tree()
                return

        # --- PHASE 2: EXECUTION / SYNTHESIS (Unchanged) ---
        t.status = "running"
        logger.info(f"EXECUTING [{t.id}]: {t.desc}")
        await self._run_task_execution(ctx)
        self.inflight.discard(tid)  # Remove from inflight after execution attempt

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}

        prompt = f"Task: {t.desc}\n"
        if t.dep_results:
            prompt += "\nUse the following data from completed dependencies to inform your answer:\n"
            for dep_id, res in t.dep_results.items():
                dep_desc = self.registry.tasks[dep_id].desc
                prompt += f"- Result from sub-task '{dep_desc}':\n{str(res)}\n\n"

        if t.children:
            prompt += "Your main goal is to synthesize the results from your sub-tasks into a final, cohesive answer for your original task description."
        else:
            prompt += "Please execute this task directly and provide a complete answer."

        worked = False
        fix_attempt = 0
        result = None
        while not worked and fix_attempt <= t.max_fix_attempts:
            logger.info(f"   > LLM execution for [{t.id}] (attempt={fix_attempt})")
            agent_res = await self.executor.run(user_prompt=prompt)
            t.cost += get_cost(agent_res.usage())
            result = agent_res.output

            if await self._verify_task(t, result):
                worked = True
            else:
                prompt += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\nYour last answer was deemed insufficient. Please re-evaluate and try again."
                result = None
                fix_attempt += 1

        if not worked:
            t.status = "failed"
            t.result = "Failed to produce a satisfactory result after multiple attempts."
            logger.error(f"Task [{t.id}] failed after {t.max_fix_attempts + 1} attempts.")
            raise Exception(f"Exceeded max fix attempts for {t.desc}")

        t.result = result
        t.status = "complete"  # Set status to complete only on success
        logger.info(f"COMPLETED [{t.id}]")

    async def _verify_task(self, t: Task, candidate_result: str) -> bool:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
        vres = await self.verifier.run(user_prompt=ver_prompt)
        t.cost += get_cost(vres.usage())
        vout = vres.output
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout.get_successful()


if __name__ == "__main__":

    from llm_agent_x.tools.brave_web_search import brave_web_search
    from nest_asyncio import apply

    apply()


    def build_dynamic_planning_dag():
        reg = TaskRegistry()
        root_task = Task(
            id="ROOT_PLANNER",
            desc=(
                "Plan and book a 3-day research trip to San Francisco for a conference on AI. "
                "The output should be a complete itinerary including flights, hotel, and a daily schedule."
            ),
            needs_planning=True,
        )
        reg.add_task(root_task)
        return reg


    # --- Dummy Tool Functions (Unchanged) ---
    def book_one_way_flight(origin: str, destination: str, datetime: str):
        """Books a one-way flight ticket."""
        print(f"Booking a one-way flight from {origin} to {destination} on {datetime}")
        return f"Successfully booked a one-way flight... Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_round_trip_flight(origin: str, destination: str, departure_datetime: str, return_datetime: str):
        """Books a round-trip flight ticket."""
        print(
            f"Booking a round-trip flight from {origin} to {destination} on {departure_datetime} and {return_datetime}")
        return f"Successfully booked a round-trip flight... Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_hotel(location: str, check_in_date: str, check_out_date: str, room_type: str = "Standard King"):
        """Books a hotel stay for a given location and date range."""
        print(f"Booking a hotel stay at {location} from {check_in_date} to {check_out_date}")
        return f"Successfully booked a {room_type} room at {location}... Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_meal(restaurant_name: str, party_size: int, datetime: str):
        """Makes a meal reservation."""
        print(f"Booking a meal at {restaurant_name} for {party_size} people on {datetime}")
        return f"Successfully booked a table... Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_transportation(service: str, origin: str, destination: str, datetime: str):
        """Books ground transportation."""
        print(f"Booking {service} to {destination} from {origin} on {datetime}")
        return f"Successfully booked {service}... Driver will be dispatched."


    def book_event(event_name: str, location: str, datetime: str, quantity: int = 1):
        """Books a ticket for a specific event."""
        print(f"Booking {quantity} ticket(s) for {event_name} at {location} on {datetime}")
        return f"Successfully booked {quantity} ticket(s)... Tickets sent to registered email."


    tools = [
        brave_web_search,
        book_one_way_flight,
        book_round_trip_flight,
        book_hotel,
        book_meal,
        book_transportation,
        book_event,
    ]


    async def main():
        reg = build_dynamic_planning_dag()
        runner = DAGAgent(reg, tools=tools)

        print("\n====== BEGIN DYNAMIC PLANNING EXECUTION ======")
        print("Initial state:")
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