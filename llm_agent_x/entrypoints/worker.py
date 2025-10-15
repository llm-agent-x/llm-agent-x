import asyncio
import logging
import uuid
from dotenv import load_dotenv

# --- Import from your project structure ---
from llm_agent_x.agents.dag_agent import TaskRegistry, Task
from llm_agent_x.agents.interactive_dag_agent import InteractiveDAGAgent

# --- Basic Setup ---
load_dotenv(".env", override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AgentWorker")


def setup_initial_tasks() -> TaskRegistry:
    """Creates the initial task registry. It is now empty by default."""
    logger.info(
        "Setting up an empty initial task registry. Waiting for tasks from the gateway."
    )
    # The agent now starts clean. The operator will add the first task via the UI.
    return TaskRegistry()


async def start_worker():
    """Initializes and runs the interactive agent worker."""
    registry = setup_initial_tasks()

    agent = InteractiveDAGAgent(
        llm_model="gpt-4o-mini",
    )

    logger.info("Starting Interactive DAG Agent worker...")
    await agent.run()
    logger.info("Agent worker has been shut down.")


def main():
    """Poetry script entry point for the agent worker."""
    try:
        # For compatibility with some environments like Jupyter
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    try:
        asyncio.run(start_worker())
    except KeyboardInterrupt:
        logger.info("Agent worker stopped by user.")


if __name__ == "__main__":
    main()
