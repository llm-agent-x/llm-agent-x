import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from llm_agent_x.entrypoints.worker import start_worker

class TestWorkerEntrypoint(unittest.TestCase):
    @patch("llm_agent_x.entrypoints.worker.InteractiveDAGAgent")
    def test_start_worker_initializes_and_runs_agent(self, MockInteractiveDAGAgent):
        """
Tests if the start_worker function correctly instantiates and
attempts to run the InteractiveDAGAgent.
        """
        # Arrange: Create a mock for the agent instance and its run method.
        # We make agent.run() raise a CancelledError to gracefully exit the
        # infinite loop in the real start_worker function for the test.
        mock_agent_instance = MockInteractiveDAGAgent.return_value
        mock_agent_instance.run = AsyncMock(side_effect=asyncio.CancelledError("Test Shutdown"))

        # Act: Run the worker entrypoint function.
        # We expect it to be cancelled by our mock's side_effect.
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(start_worker())

        # Assert: Verify that the agent was created and its run method was called.
        MockInteractiveDAGAgent.assert_called_once()
        mock_agent_instance.run.assert_awaited_once()

        print("\n[OK] test_entrypoints: Worker entrypoint correctly initializes and runs the agent.")

if __name__ == "__main__":
    unittest.main()