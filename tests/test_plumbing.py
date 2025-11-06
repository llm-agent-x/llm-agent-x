import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from llm_agent_x.agents.interactive_dag_agent import InteractiveDAGAgent
from llm_agent_x.core import Task
from llm_agent_x.state_manager import InMemoryStateManager

class TestAgentPlumbing(unittest.IsolatedAsyncioTestCase):

    # No more decorators on setUp!
    def setUp(self):
        """
Set up a fresh agent for each test using dependency injection.
Mocks will be handled by decorators on the test methods themselves.
        """
        # We will create the agent differently in each test based on its needs.
        self.agent = None

    @patch("llm_agent_x.agents.interactive_dag_agent.InteractiveDAGAgent._setup_agent_roles")
    async def test_handle_add_root_task_directive(self, mock_setup_agent_roles):
        """
Tests if the agent correctly processes an ADD_ROOT_TASK directive.
        """
        # --- Test-specific setup ---
        mock_broadcast = MagicMock()
        state_manager = InMemoryStateManager(broadcast_callback=mock_broadcast)
        self.agent = InteractiveDAGAgent(state_manager=state_manager)
        self.agent._listen_for_directives_target = MagicMock()
        # --- End setup ---

        directive = {"command": "ADD_ROOT_TASK", "payload": {"desc": "This is a test root task."}}

        await self.agent._handle_directive(directive)

        tasks = self.agent.state_manager.get_all_tasks()
        self.assertEqual(len(tasks), 1)
        task = list(tasks.values())[0]
        self.assertEqual(task.desc, "This is a test root task.")

        mock_broadcast.assert_called_once()
        print("\n[OK] test_plumbing: Agent correctly handles ADD_ROOT_TASK directive.")

    @patch("llm_agent_x.agents.interactive_dag_agent.InteractiveDAGAgent._setup_agent_roles")
    async def test_handle_lifecycle_directives(self, mock_setup_agent_roles):
        """
Tests PAUSE and RESUME directives.
        """
        # --- Test-specific setup ---
        mock_broadcast = MagicMock()
        state_manager = InMemoryStateManager(broadcast_callback=mock_broadcast)
        self.agent = InteractiveDAGAgent(state_manager=state_manager)
        self.agent._listen_for_directives_target = MagicMock()
        # --- End setup ---

        add_directive = {"command": "ADD_ROOT_TASK", "payload": {"desc": "Lifecycle test"}}
        await self.agent._handle_directive(add_directive)
        task_id = list(self.agent.state_manager.get_all_tasks().keys())[0]

        pause_directive = {"task_id": task_id, "command": "PAUSE"}
        await self.agent._handle_directive(pause_directive)
        self.assertEqual(self.agent.state_manager.get_task(task_id).status, "paused_by_human")

        resume_directive = {"task_id": task_id, "command": "RESUME"}
        await self.agent._handle_directive(resume_directive)
        self.assertEqual(self.agent.state_manager.get_task(task_id).status, "pending")

        self.assertEqual(mock_broadcast.call_count, 3)
        print("\n[OK] test_plumbing: Agent correctly handles PAUSE and RESUME directives.")

    @patch("llm_agent_x.agents.interactive_dag_agent.threading.Thread")
    @patch("llm_agent_x.agents.interactive_dag_agent.pika.BlockingConnection")
    @patch("llm_agent_x.agents.interactive_dag_agent.InteractiveDAGAgent._run_taskflow", new_callable=AsyncMock)
    @patch("llm_agent_x.agents.interactive_dag_agent.InteractiveDAGAgent._setup_agent_roles")
    async def test_run_loop_schedules_pending_task(self, mock_setup_agent_roles, mock_run_taskflow, mock_pika_conn, mock_thread):
        """
Tests if the main `run` loop schedules a runnable task.
        """
        # --- Test-specific setup ---
        mock_broadcast = MagicMock()
        state_manager = InMemoryStateManager(broadcast_callback=mock_broadcast)
        self.agent = InteractiveDAGAgent(state_manager=state_manager)
        self.agent._listen_for_directives_target = MagicMock()
        # --- End setup ---

        mock_thread_instance = mock_thread.return_value
        mock_thread_instance.is_alive.return_value = True

        test_task = Task(id="runnable-1", desc="A runnable task", status="pending")
        self.agent.state_manager.upsert_task(test_task)

        with patch.object(self.agent._shutdown_event, 'is_set', side_effect=[False, True]):
            await self.agent.run()

        mock_run_taskflow.assert_awaited_once_with("runnable-1")
        print("\n[OK] test_plumbing: Agent `run` loop correctly schedules a pending task.")

if __name__ == '__main__':
    unittest.main()