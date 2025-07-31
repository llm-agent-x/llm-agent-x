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


# --- NEW Pydantic Models for Dynamic Planning ---

class NewSubtask(BaseModel):
    """Defines a new sub-task to be added to the execution graph."""
    local_id: str = Field(
        description="A temporary, unique ID for this new task (e.g., 'task1', 'research_flights'). Used to define dependencies among other new tasks.")
    desc: str = Field(description="The detailed description of this specific sub-task.")
    deps: List[str] = Field(default_factory=list,
                            description="A list of local_ids of other *new* sub-tasks that this one depends on.")


class ExecutionPlan(BaseModel):
    """The execution plan for a complex task. It can either be a simple task or be broken down into sub-tasks."""
    needs_subtasks: bool = Field(description="Set to true if the task is complex and should be broken down.")
    subtasks: List[NewSubtask] = Field(default_factory=list, description="A list of new sub-tasks to be executed.")


# --- Models for Core DAG Structure (with modifications) ---

class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"  # pending, planning, waiting, running, complete, failed
    result: Optional[str] = None

    # --- MODIFIED/NEW Fields for Two-Phase Execution ---
    needs_planning: bool = False
    already_planned: bool = False

    # --- Unchanged Fields ---
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


# TaskRegistry and TaskContext are largely unchanged, but we'll include them for completeness.
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
        # Find root nodes (tasks with no parents)
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


# --- THE REWRITTEN DAGAgent ---

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

        # --- NEW "Planner" Agent ---
        self.planner = Agent(
            model=llm_model,
            system_prompt=(
                "You are an expert project planner. Your job is to break down a complex task into a series of smaller, actionable sub-tasks. "
                "Define a clear plan with dependencies. Use local_ids to define dependencies between the new tasks you create. "
                "If the task is simple and requires no further breakdown, set 'needs_subtasks' to false."
            ),
            output_type=ExecutionPlan
        )
        # --- Other Agents (Unchanged) ---
        self.executor = Agent(model=llm_model, output_type=str, tools=tools)
        self.verifier = Agent(model=llm_model, output_type=verification)

    async def run(self):
        pending = set(self.registry.tasks.keys())
        while pending or self.inflight:
            # Re-evaluate ready tasks in every loop, as the graph can change
            all_task_ids = set(self.registry.tasks.keys())
            pending = all_task_ids - {t.id for t in self.registry.tasks.values() if t.status in ('complete', 'failed')}

            ready = [
                tid for tid in pending
                if all(self.registry.tasks.get(d, {}).status == "complete" for d in self.registry.tasks[tid].deps)
                   and tid not in self.inflight
            ]

            if not ready and not self.inflight and pending:
                logger.error("Stalled DAG (cycle/unsatisfiable dependency):")
                self.registry.print_status_tree()
                raise Exception("DAG is not runnable (cycle?)")

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
            await asyncio.sleep(0.01)  # Small sleep to prevent busy-waiting

    async def _run_taskflow(self, tid: str, ctx: TaskContext):
        t = ctx.task

        # --- PHASE 1: DYNAMIC PLANNING ---
        # If the task hasn't been planned yet and is flagged for planning.
        if not t.already_planned and t.needs_planning:
            t.status = "planning"
            logger.info(f"PLANNING for [{t.id}]: {t.desc}")

            plan_res = await self.planner.run(user_prompt=t.desc)
            t.cost += get_cost(plan_res.usage())
            plan = plan_res.output

            t.already_planned = True  # Mark as planned even if no subtasks are created

            if plan and plan.needs_subtasks and plan.subtasks:
                logger.info(f"Plan for [{t.id}] creates {len(plan.subtasks)} new sub-tasks.")

                local_to_global_id_map: Dict[str, str] = {}

                # First pass: Create all tasks to get their global IDs
                for sub in plan.subtasks:
                    new_task_id = ctx.inject_task(sub.desc, parent=t.id)
                    local_to_global_id_map[sub.local_id] = new_task_id
                    # The parent (current task) now depends on this new child
                    ctx.add_dependency(new_task_id)

                # Second pass: Add dependencies between the new tasks
                for sub in plan.subtasks:
                    new_task_global_id = local_to_global_id_map[sub.local_id]
                    for local_dep_id in sub.deps:
                        if local_dep_id in local_to_global_id_map:
                            global_dep_id = local_to_global_id_map[local_dep_id]
                            self.registry.add_dependency(new_task_global_id, global_dep_id)

                t.status = "waiting"  # The parent now waits for its new children
                self.inflight.discard(tid)
                logger.info(f"Finished PLANNING for [{t.id}]. Now waiting for {len(t.deps)} children.")
                self.registry.print_status_tree()  # Show the new graph
                return  # Yield control back to the main run loop

        # --- PHASE 2: EXECUTION / SYNTHESIS ---
        t.status = "running"
        logger.info(f"EXECUTING [{t.id}]: {t.desc}")
        await self._run_task_execution(ctx)  # This contains the verify/fix loop
        t.status = "complete"
        self.inflight.discard(tid)
        logger.info(f"COMPLETED [{t.id}]")

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        # Refresh dependency results right before execution
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

        # Verify/Fix loop (mostly unchanged)
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
                prompt += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\nYour last answer was deemed insufficient. Please re-evaluate the task and the provided data to generate a better, more complete response. (Attempt {fix_attempt + 1}/{t.max_fix_attempts})"
                result = None
                fix_attempt += 1

        if not worked:
            t.status = "failed"
            t.result = "Failed to produce a satisfactory result after multiple attempts."
            raise Exception(f"Exceeded max fix attempts for {t.desc}")

        t.result = result

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
            # This flag is the entry point to the new dynamic behavior
            needs_planning=True,
        )
        reg.add_task(root_task)
        return reg


    def book_one_way_flight(origin: str, destination: str, datetime: str):
        """
        Books a one-way flight ticket from an origin to a destination.

        Args:
            origin (str): The starting city or airport code (e.g., 'SFO' for San Francisco).
            destination (str): The destination city or airport code (e.g., 'JFK' for New York).
            datetime (str): The desired departure date and time in a clear, standard format (e.g., '2024-09-15 14:30').

        Returns:
            str: A confirmation message including the flight booking details.
        """
        print(f"Booking a one-way flight from {origin} to {destination} on {datetime}")
        return f"Successfully booked a one-way flight from {origin} to {destination} on {datetime}. Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_round_trip_flight(origin: str, destination: str, departure_datetime: str, return_datetime: str):
        """
        Books a round-trip flight ticket, including a departure and a return flight.

        Args:
            origin (str): The starting city or airport code (e.g., 'SFO').
            destination (str): The destination city or airport code (e.g., 'LAX').
            departure_datetime (str): The desired departure date and time (e.g., '2024-10-20 09:00').
            return_datetime (str): The desired return date and time (e.g., '2024-10-25 18:00').

        Returns:
            str: A confirmation message with the round-trip booking details.
        """
        print(
            f"Booking a round-trip flight from {origin} to {destination} on {departure_datetime} and {return_datetime}")
        return f"Successfully booked a round-trip flight from {origin} to {destination}, departing on {departure_datetime} and returning on {return_datetime}. Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_hotel(location: str, check_in_date: str, check_out_date: str, room_type: str = "Standard King"):
        """
        Books a hotel stay for a given location and date range. Use the brave_web_search tool first to find suitable hotel names.

        Args:
            location (str): The name of the hotel to book (e.g., 'Marriott Marquis San Francisco').
            check_in_date (str): The date for checking into the hotel (e.g., '2024-09-15').
            check_out_date (str): The date for checking out of the hotel (e.g., '2024-09-18').
            room_type (str): The desired type of room (e.g., 'Standard King', 'Queen with View'). Defaults to 'Standard King'.

        Returns:
            str: A confirmation of the hotel booking.
        """
        print(f"Booking a hotel stay at {location} from {check_in_date} to {check_out_date}")
        return f"Successfully booked a {room_type} room at {location} from {check_in_date} to {check_out_date}. Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_meal(restaurant_name: str, party_size: int, datetime: str):
        """
        Makes a meal reservation at a specific restaurant.

        Args:
            restaurant_name (str): The name of the restaurant for the reservation.
            party_size (int): The number of people in the party.
            datetime (str): The date and time for the reservation (e.g., '2024-09-16 19:30').

        Returns:
            str: A confirmation of the meal reservation.
        """
        print(f"Booking a meal at {restaurant_name} for {party_size} people on {datetime}")
        return f"Successfully booked a table for {party_size} at {restaurant_name} on {datetime}. Confirmation ID: {str(uuid.uuid4())[:8].upper()}"


    def book_transportation(service: str, origin: str, destination: str, datetime: str):
        """
        Books ground transportation, such as a taxi or rideshare.

        Args:
            service (str): The type of transportation service (e.g., 'Taxi', 'Uber', 'Lyft').
            origin (str): The pickup location (e.g., 'San Francisco International Airport').
            destination (str): The drop-off location (e.g., 'Marriott Marquis San Francisco').
            datetime (str): The date and time for the pickup (e.g., '2024-09-15 16:00').

        Returns:
            str: A confirmation of the transportation booking.
        """
        print(f"Booking {service} to {destination} from {origin} on {datetime}")
        return f"Successfully booked {service} from {origin} to {destination} for {datetime}. Driver will be dispatched."


    def book_event(event_name: str, location: str, datetime: str, quantity: int = 1):
        """
        Books a ticket for a specific event, such as a conference, show, or museum.

        Args:
            event_name (str): The name of the event (e.g., 'AI Global Summit 2024').
            location (str): The venue of the event (e.g., 'Moscone Center, San Francisco').
            datetime (str): The date and time of the event.
            quantity (int): The number of tickets to book. Defaults to 1.

        Returns:
            str: A confirmation of the event ticket booking.
        """
        print(f"Booking {quantity} ticket(s) for {event_name} at {location} on {datetime}")
        return f"Successfully booked {quantity} ticket(s) for {event_name} at {location} on {datetime}. Tickets sent to registered email."


    # The list of tools provided to the agent. It includes a general-purpose search
    # tool and the specific booking tools defined above.
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
        root_result = runner.registry.tasks["ROOT_PLANNER"].result
        print(f"** Final Itinerary for ROOT_PLANNER **\n{root_result}")

        total_cost = sum(t.cost for t in reg.all_tasks())
        print(f"\nTotal estimated cost: ${total_cost:.4f}")


    asyncio.run(main())