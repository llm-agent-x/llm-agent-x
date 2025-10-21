from typing import Dict, Optional, Callable
from llm_agent_x.core import Task
from llm_agent_x.state_manager.abstract_state_manager import AbstractStateManager

class InMemoryStateManager(AbstractStateManager):
    """
An in-memory implementation of the state manager using a simple dictionary.
This serves as the default state backend for the open-source version.
    """
    def __init__(self, broadcast_callback: Optional[Callable[[Task], None]] = None):
        super().__init__(broadcast_callback)
        self._tasks: Dict[str, Task] = {}

    def upsert_task(self, task: Task):
        """Adds a new task or updates an existing one in the dictionary."""
        self._tasks[task.id] = task
        self._broadcast(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a single task by its ID from the dictionary."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, Task]:
        """Returns the entire dictionary of tasks."""
        return self._tasks

    def delete_task(self, task_id: str):
        """Deletes a task from the dictionary if it exists."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            # Note: A broadcast for deletion might be a 'task_removed' event
            # For now, the existing 'pruned' status handles UI removal.

    def add_dependency(self, task_id: str, dep_id: str):
        """Adds a dependency link between two tasks in the dictionary."""
        task = self.get_task(task_id)
        dependency_task = self.get_task(dep_id)

        if not task or not dependency_task:
            return

        # Add the dependency link for the scheduler
        if dep_id not in task.deps:
            task.deps.add(dep_id)

        # Add the child link for graph traversal/visualization
        if task_id not in dependency_task.children:
            dependency_task.children.append(task_id)

        # Save and broadcast the changes
        self.upsert_task(task)
        self.upsert_task(dependency_task)