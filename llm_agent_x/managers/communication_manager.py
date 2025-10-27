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
    Manages and applies communication tools and dynamic instructions to an agent instance.
    """

    def __init__(self, state_manager: AbstractStateManager):
        self.state_manager = state_manager
        self.communication_tools: List[Callable] = []
        self._create_communication_tools()  # Pre-build the tool functions
        logger.info("CommunicationManager initialized with communication tools.")

    def apply_to_agent(self, agent: Agent):
        """
        Applies all communication-related tools and dynamic instructions
        to the provided agent instance.
        """
        # Attach the pre-built tools to the agent instance
        for tool_func in self.communication_tools:
            agent.tool(tool_func)

        # Attach the dynamic instructions to the agent instance
        self._add_dynamic_instructions(agent)
        logger.info(f"Communication handlers applied to agent instance.")

    def _create_communication_tools(self):
        """
        Defines the communication tools and stores them in self.communication_tools.
        Note: These are now regular methods, not decorated with @self.agent.tool.
        """

        def broadcast(ctx: RunContext[TaskContext], message: str, target_tags: List[str]) -> str:
            """
            Broadcasts a message to all active tasks that have one or more of the target tags.
            This is an expensive operation and should be used for important, widely relevant information.
            """
            task = ctx.deps.task
            if not task.comm_token_bucket.spend(BROADCAST_COST):
                return f"Broadcast failed: Not enough tokens. You have {task.comm_token_bucket.tokens:.1f}, need {BROADCAST_COST}."

            all_tasks = self.state_manager.get_all_tasks().values()
            target_tasks = [
                t for t in all_tasks
                if t.id != task.id and t.status not in ["complete", "failed", "cancelled"]
                   and not set(target_tags).isdisjoint(t.tags)
            ]

            if not target_tasks:
                return "Broadcast sent, but no active tasks matched the target tags."

            interrupt = AgentMessageInterrupt(source_task_id=task.id, message=message, target_tags=target_tags)
            for target_task in target_tasks:
                heapq.heappush(target_task.interrupt_queue, interrupt)
                self.state_manager.upsert_task(target_task)

            return f"Broadcast successful. Message sent to {len(target_tasks)} tasks with tags {target_tags}."

        def send_direct_message(ctx: RunContext[TaskContext], message: str, target_task_id: str) -> str:
            """
            Sends a direct, private message to a specific task identified by its ID.
            This is a cheap operation, suitable for one-to-one coordination.
            """
            task = ctx.deps.task
            if not task.comm_token_bucket.spend(DIRECT_MESSAGE_COST):
                return f"Direct message failed: Not enough tokens. You have {task.comm_token_bucket.tokens:.1f}, need {DIRECT_MESSAGE_COST}."

            target_task = self.state_manager.get_task(target_task_id)
            if not target_task or target_task.status in ["complete", "failed", "cancelled"]:
                return f"Direct message failed: Task '{target_task_id}' not found or is inactive."

            interrupt = AgentMessageInterrupt(source_task_id=task.id, message=message)
            heapq.heappush(target_task.interrupt_queue, interrupt)
            self.state_manager.upsert_task(target_task)

            return f"Direct message sent successfully to task '{target_task_id}'."

        self.communication_tools = [broadcast, send_direct_message]

    def _add_dynamic_instructions(self, agent: Agent):
        """
        Defines and registers the dynamic instruction functions with the provided agent.
        """

        @agent.instructions
        def add_token_economy_status(ctx: RunContext[TaskContext]) -> str:
            """Provides the agent with its current communication token balance and action costs."""
            task = ctx.deps.task
            return (
                f"\n--- COMMUNICATION STATUS ---\n"
                f"Your current communication token balance is {task.comm_token_bucket.tokens:.1f}.\n"
                f"Action Costs: broadcast = {BROADCAST_COST} tokens, send_direct_message = {DIRECT_MESSAGE_COST} tokens.\n"
                f"----------------------------"
            )

        @agent.instructions
        def add_available_tags_for_broadcast(ctx: RunContext[TaskContext]) -> str:
            """Provides the agent with a list of currently active tags in the swarm."""
            all_tags = set()
            active_tasks = [
                t for t in self.state_manager.get_all_tasks().values()
                if t.status not in ["complete", "failed", "cancelled"]
            ]
            for task in active_tasks:
                all_tags.update(task.tags)

            if not all_tags:
                return "\n--- AVAILABLE TAGS for broadcast ---\nNo other active tasks have tags.\n------------------------------------"

            tags_list_str = ", ".join(sorted(list(all_tags)))
            return (
                f"\n--- AVAILABLE TAGS for broadcast ---\n"
                f"You can broadcast to tasks with these tags: {tags_list_str}\n"
                f"------------------------------------"
            )