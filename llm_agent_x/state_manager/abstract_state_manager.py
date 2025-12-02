from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_agent_x.core import Task

class AbstractStateManager(ABC):
    """
Abstract base class for state management.

This defines the contract for creating, retrieving, updating, and deleting
tasks in the agent's state graph, decoupling the agent logic from the
underlying storage mechanism (e.g., in-memory dictionary, database).
"""

    def __init__(self, broadcast_callback: Optional[Callable[["Task"], None]] = None):
        """
Initializes the state manager.

Args:
broadcast_callback: An optional function to call whenever a task's
state is updated, used to notify UIs.
        """
        self._broadcast = broadcast_callback or (lambda task: None)

    @abstractmethod
    def upsert_task(self, task: "Task"):
        """Adds a new task or updates an existing one."""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional["Task"]:
        """Retrieves a single task by its ID."""
        pass

    @abstractmethod
    def get_all_tasks(self) -> Dict[str, "Task"]:
        """Retrieves all tasks in the current state."""
        pass

    @abstractmethod
    def delete_task(self, task_id: str):
        """Permanently deletes a task from the state."""
        pass

    @abstractmethod
    def add_dependency(self, task_id: str, dep_id: str):
        """
Adds a dependency between two tasks.
This is a high-level graph operation that modifies two tasks.
        """
        pass


