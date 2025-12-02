# llm_agent_x/worker/worker.py
import asyncio
import logging
from os import getenv

from dotenv import load_dotenv

# Import the registry and the context variable
from llm_agent_x.runtime.agent_registry import AGENT_REGISTRY, AGENT_TYPE_CONTEXT

# --- Basic Setup ---
load_dotenv(".env", override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("UnifiedAgentWorker")

# --- Configuration ---
# Instantiate the LLM model once to be shared by all agents
MODEL = getenv("DEFAULT_LLM", "openai/gpt-4o-mini")

# Comma-separated list of agent types to run.
# Defaults to "interactive_dag" for out-of-the-box functionality.
ENABLED_AGENTS_STR = getenv("ENABLED_AGENTS", "interactive_dag")


async def start_worker():
    """Initializes and runs all enabled agent workers concurrently."""

    if ENABLED_AGENTS_STR:
        enabled_agents = [agent.strip() for agent in ENABLED_AGENTS_STR.split(",")]
    else:
        # If the env var is empty, run nothing.
        enabled_agents = []

    if not enabled_agents:
        logger.warning("No agents enabled. Set the ENABLED_AGENTS environment variable. Exiting.")
        return

    logger.info(f"Starting unified worker for agent types: {enabled_agents}")

    tasks = []
    try:
        # Loop through the configured agents and set them up using the registry
        for agent_type in enabled_agents:
            if agent_type in AGENT_REGISTRY:
                setup_func = AGENT_REGISTRY[agent_type]

                # --- ContextVar Integration ---
                # 1. Set the context for the current agent type.
                # This ensures the setup function can read the correct queue_prefix.
                token = AGENT_TYPE_CONTEXT.set(agent_type)

                try:
                    # Call the setup function to get the fully configured agent instance
                    # The `setup_func` will now implicitly use the context we just set.
                    setup = setup_func(llm_model=MODEL)
                    agent_instance = setup["agent"]

                    # Create an asyncio task for the agent's main run loop
                    task = asyncio.create_task(agent_instance.run(), name=f"{agent_type}_AgentTask")
                    tasks.append(task)
                    logger.info(f"Scheduled main task for '{agent_type}' agent.")

                finally:
                    # 2. Reset the context to its previous state.
                    # This is critical for ensuring that if one setup fails,
                    # it doesn't affect the context of the next one in the loop.
                    AGENT_TYPE_CONTEXT.reset(token)

            else:
                logger.warning(f"Unknown agent type '{agent_type}' specified in ENABLED_AGENTS. Skipping.")

        if not tasks:
            logger.error("No valid agents were configured to run. Worker will exit.")
            return

        logger.info(f"Unified worker running with {len(tasks)} concurrent tasks.")
        # Run all scheduled agent tasks concurrently
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.critical(f"A critical error occurred in the unified worker: {e}", exc_info=True)
    finally:
        logger.info("Shutting down worker...")
        for task in tasks:
            if not task.done():
                task.cancel()
        logger.info("Worker shutdown complete.")


def main():
    """Poetry script entry point for the agent worker."""
    try:
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