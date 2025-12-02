# llm_agent_x/runtime/local_scheduler.py
import asyncio
import logging
from typing import Set

from llm_agent_x.state_manager import AbstractStateManager
from llm_agent_x.runtime.task_processor import TaskProcessor
from llm_agent_x.runtime.base_scheduler import BaseScheduler

logger = logging.getLogger("LocalScheduler")

class Scheduler(BaseScheduler): # <-- Inherits from BaseScheduler
    """A simple, in-process scheduler that runs a continuous loop."""

    def __init__(self, state_manager: AbstractStateManager, task_processor: TaskProcessor):
        self.state_manager = state_manager
        self.processor = task_processor
        self.inflight_tasks: Set[str] = set()
        self.task_futures: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    # The get_ready_tasks() method remains identical to the previous version.
    def get_ready_tasks(self) -> Set[str]:
        tasks = self.state_manager.get_all_tasks()
        all_task_ids = set(tasks.keys())
        executable_statuses = {"pending", "running", "planning", "proposing", "waiting_for_children"}
        non_executable_tasks = {t.id for t in tasks.values() if t.status not in executable_statuses}
        pending_tasks_for_scheduling = all_task_ids - non_executable_tasks
        ready_to_run_ids = set()
        for tid in pending_tasks_for_scheduling:
            task = self.state_manager.get_task(tid)
            if not task: continue
            if task.status == "pending" and all(
                (dep_task := self.state_manager.get_task(d)) and dep_task.status == "complete"
                for d in task.deps
            ):
                ready_to_run_ids.add(tid)
            elif task.status == "waiting_for_children" and all(
                (child_task := self.state_manager.get_task(c)) and child_task.status in ["complete", "failed"]
                for c in task.children
            ):
                if any((child := self.state_manager.get_task(c)) and child.status == "failed" for c in task.children):
                    task.status, task.result = "failed", "A child task failed, cannot synthesize."
                    self.state_manager.upsert_task(task)
                else:
                    ready_to_run_ids.add(tid)
                    task.status = "pending"
                    self.state_manager.upsert_task(task)
        return ready_to_run_ids

    async def run(self):
        """The main loop of the scheduler."""
        logger.info("LocalScheduler starting its run loop.")
        while not self._shutdown_event.is_set():
            ready_tasks = self.get_ready_tasks()

            for task_id in ready_tasks:
                if task_id not in self.inflight_tasks:
                    self.inflight_tasks.add(task_id)
                    future = asyncio.create_task(self.processor.process_task(task_id))
                    self.task_futures[task_id] = future

            if not self.task_futures:
                await asyncio.sleep(0.1) # Prevent busy-waiting
                continue

            done, _ = await asyncio.wait(
                list(self.task_futures.values()),
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED
            )

            for fut in done:
                tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                if tid:
                    self.inflight_tasks.discard(tid)
                    del self.task_futures[tid]
                    try:
                        fut.result()
                    except asyncio.CancelledError:
                        logger.info(f"Scheduled task [{tid}] was cancelled.")
                    except Exception as e:
                        logger.error(f"Scheduled task [{tid}] failed: {e}", exc_info=True)
                        if t := self.state_manager.get_task(tid):
                            if t.status != "failed":
                                t.status, t.result = "failed", f"Execution failed: {e}"
                                self.state_manager.upsert_task(t)

            await asyncio.sleep(0.05)

        logger.info("LocalScheduler has shut down.")


    def shutdown(self):
        self._shutdown_event.set()