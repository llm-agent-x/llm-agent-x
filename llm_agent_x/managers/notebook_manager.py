# llm_agent_x/managers/notebook_manager.py

import logging
from typing import Callable, Dict, Any, Optional

from pydantic_ai import Agent, RunContext
from llm_agent_x.state_manager import AbstractStateManager
from llm_agent_x.core import TaskContext

logger = logging.getLogger(__name__)


class NotebookManager:
    """
    Manages all interactions with a task's shared notebook,
    using the StateManager as the persistence layer.
    """

    def __init__(self, state_manager: AbstractStateManager):
        self.state_manager = state_manager
        logger.info("NotebookManager initialized, using StateManager for persistence.")

    # --- CORE LOGIC METHODS ---

    async def get_notebook(self, task_id: str) -> Dict[str, Any]:
        """Retrieves the notebook for a specific task from the state manager."""
        task = self.state_manager.get_task(task_id)
        return task.shared_notebook if task else {}

    async def update_notebook(self, task_id: str, updates: Dict[str, Any]) -> str:
        """Updates a task's notebook and persists the change via the state manager."""
        task = self.state_manager.get_task(task_id)
        if not task:
            return f"Error: Task {task_id} not found."

        updated_keys, deleted_keys = [], []
        for key, value in updates.items():
            if value is None:
                if key in task.shared_notebook:
                    del task.shared_notebook[key]
                    deleted_keys.append(key)
            else:
                task.shared_notebook[key] = value
                updated_keys.append(key)

        self.state_manager.upsert_task(task)  # Persist the entire updated task object

        return (f"Notebook updated. "
                f"Set: {updated_keys or 'None'}. "
                f"Deleted: {deleted_keys or 'None'}.")

    # --- AGENT INTEGRATION METHODS ---

    def apply_tools(self, agent: Agent):
        """Applies the write-access tool (`update_notebook_tool`) to an agent."""
        agent.tool(self._create_notebook_tool())
        logger.info(f"Notebook tool (write access) applied to agent.")

    def apply_instructions(self, agent: Agent):
        """Applies the read-only notebook context as a dynamic instruction to an agent."""

        @agent.instructions
        async def add_notebook_context(ctx: RunContext[TaskContext]) -> str:
            task_id = ctx.deps.task.id
            notebook = await self.get_notebook(task_id)
            if not notebook:
                return ""  # Don't clutter the prompt if empty

            notebook_content = self._format_notebook_dict_for_llm(notebook)
            return (
                "\n--- CONTEXT FROM SHARED NOTEBOOK ---\n"
                "Use this key information recorded by other tasks to inform your decisions.\n"
                f"{notebook_content}\n"
                "------------------------------------"
            )

        logger.info(f"Notebook instructions (read access) applied to agent.")

    # --- PRIVATE HELPERS ---

    def _create_notebook_tool(self) -> Callable:
        """Creates the `update_notebook_tool` as a thin wrapper around the manager's logic."""

        async def update_notebook_tool_instance(ctx: RunContext[TaskContext], updates: Dict[str, Any]) -> str:
            """
            Updates the shared notebook for the current task. Provide key-value pairs in 'updates'.
            Set a value to 'null' (Python `None`) to delete a key.
            """
            task_id = ctx.deps.task.id
            # The tool now calls the manager's core logic method
            return await self.update_notebook(task_id, updates)

        return update_notebook_tool_instance

    def _format_notebook_dict_for_llm(self, notebook: Dict[str, Any], max_len: int = 200) -> str:
        """Helper to format a notebook dictionary for an LLM prompt."""
        if not notebook:
            return "The shared notebook is currently empty."

        entries = [f"- {key}: {str(value)[:max_len]}" for key, value in notebook.items()]
        return "\n".join(entries)