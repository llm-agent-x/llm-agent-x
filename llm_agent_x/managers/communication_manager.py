# llm_agent_x/agents/communication_manager.py

import heapq
import logging
from typing import List, Dict, Any

from pydantic_ai import Agent, RunContext

from llm_agent_x.core.interrupts import AgentMessageInterrupt
from llm_agent_x.state_manager import AbstractStateManager
from llm_agent_x.core import TaskContext

logger = logging.getLogger(__name__)

# --- Constants for the Token Economy ---
BROADCAST_COST = 10
DIRECT_MESSAGE_COST = 2


class CommunicationManager:
    """
    Manages communication tools and instructions for an agent instance.
    This class is designed for future extension with new, unspecified features.
    """

    def __init__(self, agent: Agent, state_manager: AbstractStateManager):
        self.agent = agent
        self.state_manager = state_manager
        logger.info(f"CommunicationManager initialized for agent.")

    def setup_communication_handlers(self):
        """
        Attaches all communication-related tools and dynamic instructions
        to the agent instance provided during initialization.
        """
        self._create_communication_tools()
        self._add_dynamic_instructions()
        logger.info("Communication tools and instructions have been set up on the agent.")

    def _create_communication_tools(self):
        """
        Defines and registers the communication tools with the agent instance
        using the @agent.tool decorator.
        """

        @self.agent.tool
        def broadcast(ctx: RunContext[TaskContext], message: str, target_tags: List[str]) -> str:
            """
            Broadcasts a message to all active tasks that have one or more of the target tags.
            This is an expensive operation and should be used for important, widely relevant information.
            """
            task = ctx.deps.task
            if not task.comm_token_bucket.spend(BROADCAST_COST):
                return f"Broadcast failed: Not enough communication tokens. You have {task.comm_token_bucket.tokens:.1f}, but need {BROADCAST_COST}."

            all_tasks = self.state_manager.get_all_tasks().values()
            # Find tasks that are active and have at least one matching tag
            target_tasks = [
                t for t in all_tasks
                if t.id != task.id and t.status not in ["complete", "failed", "cancelled"]
                   and not set(target_tags).isdisjoint(t.tags)
            ]

            if not target_tasks:
                return "Broadcast sent, but no active tasks matched the target tags."

            interrupt = AgentMessageInterrupt(
                source_task_id=task.id,
                message=message,
                target_tags=target_tags
            )
            for target_task in target_tasks:
                heapq.heappush(target_task.interrupt_queue, interrupt)
                self.state_manager.upsert_task(target_task)

            return f"Broadcast successful. Message sent to {len(target_tasks)} tasks with tags {target_tags}."

        @self.agent.tool
        def send_direct_message(ctx: RunContext[TaskContext], message: str, target_task_id: str) -> str:
            """

            Sends a direct, private message to a specific task identified by its ID.
            This is a cheap operation, suitable for one-to-one coordination.
            """
            task = ctx.deps.task
            if not task.comm_token_bucket.spend(DIRECT_MESSAGE_COST):
                return f"Direct message failed: Not enough communication tokens. You have {task.comm_token_bucket.tokens:.1f}, but need {DIRECT_MESSAGE_COST}."

            target_task = self.state_manager.get_task(target_task_id)
            if not target_task or target_task.status in ["complete", "failed", "cancelled"]:
                return f"Direct message failed: Task '{target_task_id}' not found or is inactive."

            interrupt = AgentMessageInterrupt(source_task_id=task.id, message=message)
            heapq.heappush(target_task.interrupt_queue, interrupt)
            self.state_manager.upsert_task(target_task)

            return f"Direct message sent successfully to task '{target_task_id}'."

    def _add_dynamic_instructions(self):
        """
        Defines and registers the dynamic instruction functions that inject
        real-time context (like token balance) into the agent's prompt.
        """

        @self.agent.instructions
        def add_token_economy_status(ctx: RunContext[TaskContext]) -> str:
            """Provides the agent with its current communication token balance and action costs."""
            task = ctx.deps.task
            return (
                f"\n--- COMMUNICATION STATUS ---\n"
                f"Your current communication token balance is {task.comm_token_bucket.tokens:.1f}.\n"
                f"Action Costs: broadcast = {BROADCAST_COST} tokens, send_direct_message = {DIRECT_MESSAGE_COST} tokens.\n"
                f"You MUST have enough tokens to perform a communication action.\n"
                f"----------------------------"
            )