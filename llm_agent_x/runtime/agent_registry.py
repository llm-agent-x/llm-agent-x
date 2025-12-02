import logging
from contextvars import ContextVar
from typing import Dict, Any

from llm_agent_x.agents.interactive_dag_agent import InteractiveDAGAgent
from llm_agent_x.runtime import (
    TaskProcessor,
    Scheduler,
)
from llm_agent_x.state_manager import InMemoryStateManager, AbstractStateManager
from llm_agent_x.tools.brave_web_search import brave_web_search

logger = logging.getLogger(__name__)

# --- ContextVar Definition ---
# 1. Define the ContextVar at the module level.
# This variable will hold the agent type name, which we'll use as the queue_prefix.
AGENT_TYPE_CONTEXT = ContextVar('agent_type_name', default=None)

# The AGENT_REGISTRY will hold setup functions for each agent type.
AGENT_REGISTRY: Dict[str, Any] = {}

def register_agent_type(agent_type_name: str):
    """A decorator to register an agent setup function."""
    def decorator(func):
        AGENT_REGISTRY[agent_type_name] = func
        return func
    return decorator


@register_agent_type("interactive_dag")
def setup_interactive_dag_agent(llm_model) -> Dict[str, Any]:
    """
    Sets up the InteractiveDAGAgent and its required components.
    It implicitly reads the queue_prefix from the AGENT_TYPE_CONTEXT.
    """
    # 2. Get the agent type name from the context.
    # The calling code (in the worker) is responsible for setting this.
    queue_prefix = AGENT_TYPE_CONTEXT.get()
    if not queue_prefix:
        raise ValueError(
            "Agent type name not found in context. "
            "The worker must set AGENT_TYPE_CONTEXT before calling this setup function."
        )

    logger.info(f"Setting up agent type: '{queue_prefix}' with model {llm_model.__class__.__name__}")

    # The agent's broadcast callback will be set after instantiation.
    state_manager = InMemoryStateManager()

    # 3. Use the retrieved 'queue_prefix' when creating the agent.
    # Also corrected 'llm_model' to 'base_llm_model' to match the agent's constructor.
    agent = InteractiveDAGAgent(
        state_manager=state_manager,
        base_llm_model=llm_model,
        queue_prefix=f"{queue_prefix}_", # Add a separator for readability
        tools=[brave_web_search]
    )

    # The broadcast callback must be set on the state_manager instance
    # that was passed to the agent.
    state_manager.set_broadcast_callback(agent._broadcast_state_update)

    task_processor = TaskProcessor(agent_base=agent, state_manager=state_manager)
    scheduler = Scheduler(state_manager, task_processor=task_processor)
    agent.set_scheduler(scheduler)

    return {"agent": agent, "scheduler": scheduler, "task_processor": task_processor}