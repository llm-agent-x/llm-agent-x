
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_agent_x.core import Task, TaskContext, AgentMessageInterrupt
from llm_agent_x.managers.notebook_manager import NotebookManager
from llm_agent_x.managers.communication_manager import CommunicationManager
from llm_agent_x.state_manager import InMemoryStateManager

class TestManagers(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.state_manager = InMemoryStateManager()
        self.notebook_manager = NotebookManager(self.state_manager)
        self.comm_manager = CommunicationManager(self.state_manager)

        self.mock_agent = MagicMock()
        self.mock_agent.tool = MagicMock()
        self.mock_agent.instructions = MagicMock()

        self.task1 = Task(id="task-1", desc="Task One", tags={"research"})
        self.task2 = Task(id="task-2", desc="Task Two", tags={"writing", "research"})
        self.task3 = Task(id="task-3", desc="Task Three", tags={"review"})
        self.state_manager.upsert_task(self.task1)
        self.state_manager.upsert_task(self.task2)
        self.state_manager.upsert_task(self.task3)

        self.mock_task_context = TaskContext(self.task1.id, self.state_manager)
        self.mock_run_context = MagicMock()
        self.mock_run_context.deps = self.mock_task_context

    async def test_notebook_manager_update_and_get(self):
        await self.notebook_manager.update_notebook("task-1", {"key1": "value1", "key2": 123})
        notebook = await self.notebook_manager.get_notebook("task-1")
        self.assertEqual(notebook, {"key1": "value1", "key2": 123})
        self.assertEqual(self.state_manager.get_task("task-1").shared_notebook, {"key1": "value1", "key2": 123})
        print("\n[OK] test_managers: NotebookManager can update and retrieve data.")

    async def test_notebook_manager_tool(self):
        self.notebook_manager.apply_tools(self.mock_agent)
        self.assertTrue(self.mock_agent.tool.called)
        update_tool_func = self.mock_agent.tool.call_args[0][0]
        result = await update_tool_func(self.mock_run_context, updates={"status": "in_progress"})
        self.assertIn("Notebook updated", result)
        updated_task = self.state_manager.get_task("task-1")
        self.assertEqual(updated_task.shared_notebook, {"status": "in_progress"})
        print("\n[OK] test_managers: NotebookManager tool correctly updates task state.")

    def test_comm_manager_broadcast_tool(self): # <-- FIX: Removed async
        """Tests that the broadcast tool sends interrupts to tasks with matching tags."""
        # Arrange
        self.comm_manager.apply_to_agent(self.mock_agent)

        broadcast_tool_func = None
        for call in self.mock_agent.tool.call_args_list:
            func = call.args[0]
            if func.__name__ == 'broadcast':
                broadcast_tool_func = func
                break
        self.assertIsNotNone(broadcast_tool_func, "Could not find 'broadcast' tool.")

        # Act: Broadcast to the 'research' tag
        # --- FIX: Removed await ---
        broadcast_tool_func(self.mock_run_context, message="Hello researchers", target_tags=["research"])

        # Assert
        task1_updated = self.state_manager.get_task("task-1")
        task2_updated = self.state_manager.get_task("task-2")
        task3_updated = self.state_manager.get_task("task-3")

        self.assertEqual(len(task1_updated.interrupt_queue), 0)
        self.assertEqual(len(task2_updated.interrupt_queue), 1)
        self.assertIsInstance(task2_updated.interrupt_queue[0], AgentMessageInterrupt)
        self.assertEqual(task2_updated.interrupt_queue[0].message, "Hello researchers")
        self.assertEqual(len(task3_updated.interrupt_queue), 0)
        print("\n[OK] test_managers: CommunicationManager broadcast tool correctly sends interrupts.")

    @patch("llm_agent_x.managers.communication_manager.Agent")
    async def test_comm_manager_send_message_to_completed_task(self, MockSpiritAgent):
        mock_response = MagicMock()
        mock_response.output = "The final result was 42."
        mock_spirit_instance = MockSpiritAgent.return_value
        mock_spirit_instance.run = AsyncMock(return_value=mock_response)

        completed_task = Task(id="task-comp", desc="Completed Task", status="complete", result="The answer is 42.")
        self.state_manager.upsert_task(completed_task)

        self.comm_manager.apply_to_agent(self.mock_agent)
        send_message_tool_func = None
        for call in self.mock_agent.tool.call_args_list:
            if call.args[0].__name__ == 'send_direct_message':
                send_message_tool_func = call.args[0]
                break
        self.assertIsNotNone(send_message_tool_func, "Could not find 'send_direct_message' tool.")

        response = await send_message_tool_func(self.mock_run_context, message="What was your final result?", target_task_id="task-comp")

        MockSpiritAgent.assert_called_once_with(model="gpt-4o-mini", output_type=str)
        mock_spirit_instance.run.assert_awaited_once()
        self.assertIn("The final result was 42.", response)
        print("\n[OK] test_managers: CommunicationManager correctly mocks and queries completed tasks.")

if __name__ == "__main__":
    unittest.main()