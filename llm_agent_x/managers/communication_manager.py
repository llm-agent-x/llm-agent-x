# llm_agent_x/agents/communication_manager.py

import heapq
import logging
from os import getenv
from typing import List, Dict, Any, Callable

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
        """

        # --- Keep the existing `broadcast` method as is ---
        def broadcast(ctx: RunContext[TaskContext], message: str, target_tags: List[str]) -> str:
            # ... (no changes to this method) ...
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

        # --- REPLACE the old `send_direct_message` with this new, smarter version ---
        async def send_direct_message(ctx: RunContext[TaskContext], message: str, target_task_id: str) -> str:
            """
            Sends a direct message to another task, active or completed.
            - If the task is ACTIVE, it sends an interrupt for it to process.
            - If the task is COMPLETED, it will answer a question based on its final state.
            """
            task = ctx.deps.task
            target_task = self.state_manager.get_task(target_task_id)

            if not target_task:
                return f"Error: Task '{target_task_id}' not found."

            # --- ROUTE 1: Target is ACTIVE ---
            if target_task.status not in ["complete", "failed", "cancelled", "pruned"]:
                if not task.comm_token_bucket.spend(DIRECT_MESSAGE_COST):
                    return f"Direct message failed: Not enough tokens. You have {task.comm_token_bucket.tokens:.1f}, need {DIRECT_MESSAGE_COST}."

                interrupt = AgentMessageInterrupt(source_task_id=task.id, message=message)
                heapq.heappush(target_task.interrupt_queue, interrupt)
                self.state_manager.upsert_task(target_task)
                return f"Direct message sent successfully to active task '{target_task_id}'."


            elif target_task.status == "complete":
                logger.info(f"Querying completed task '{target_task_id}' on behalf of task '{task.id}'.")
                prompt = (
                    f"An active agent is now asking you a question based on your work. "
                    f"Question: {message}"
                )

                spirit_agent = Agent(model=getenv("DEFAULT_LLM", "gpt-4o-mini"), output_type=str)
                response = await spirit_agent.run(user_prompt=prompt, message_history=target_task.last_llm_history)

                return f"Response from completed task '{target_task_id}': {response.output}"

            # --- ROUTE 3: Target is inactive but not 'complete' ---
            else:
                return f"Error: Cannot send message to task '{target_task_id}'. Its status is '{target_task.status}'."

        # Update the list of tools. No more `query_completed_task`.
        self.communication_tools = [broadcast, send_direct_message]

    def _add_dynamic_instructions(self, agent: Agent):
        """
        Defines and registers the dynamic instruction functions with the provided agent.
        """

        @agent.instructions
        def add_available_targets(ctx: RunContext[TaskContext]) -> str:
            """Provides the agent with a list of all available targets for communication."""

            all_tasks = self.state_manager.get_all_tasks().values()
            current_task_id = ctx.deps.task.id

            # --- START OF THE REFINED FIX ---

            # An "interruptible" task is one that is in an active reasoning state
            # or is explicitly waiting for external input.
            # `waiting_for_children` is explicitly excluded as it is a passive wait state.
            interruptible_statuses = {
                "running",
                "proposing",
                "planning",
                "paused_by_human",
                "waiting_for_user_response"  # <-- As you correctly identified.
            }

            # A "queryable" task is one that has a final, successful result.
            queryable_statuses = {"complete"}

            # The rest of the logic remains the same...

            active_interruptible_tasks = [
                t for t in all_tasks
                if t.id != current_task_id and t.status in interruptible_statuses
            ]

            active_queryable_tasks = [
                t for t in all_tasks
                if t.id != current_task_id and t.status in queryable_statuses
            ]

            # 1. Collect all unique tags ONLY from active, interruptible tasks.
            all_tags = set()
            for task in active_interruptible_tasks:
                all_tags.update(task.tags)

            prompt_parts = ["\n--- AVAILABLE COMMUNICATION TARGETS ---"]
            prompt_parts.append(
                "Use `send_direct_message(target_task_id, message)` to communicate with another task."
            )

            # 2. Format the tags section for broadcasting to active tasks.
            if all_tags:
                tags_list_str = ", ".join(f"'{tag}'" for tag in sorted(list(all_tags)))
                prompt_parts.append(f"\nYou can BROADCAST to groups of ACTIVE tasks using tags: {tags_list_str}")
            else:
                prompt_parts.append("\nNo active tags are available for broadcast.")

            # 3. Format the direct message targets section, separating active and completed tasks.
            if active_interruptible_tasks:
                targets_str = "\n".join(
                    f"- ID: '{t.id}' (Status: {t.status}), Description: '{t.desc[:60]}...'" for t in
                    active_interruptible_tasks)
                prompt_parts.append(
                    f"\n\nYou can send a DIRECT MESSAGE (interrupt) to these ACTIVE tasks:\n{targets_str}")

            if active_queryable_tasks:
                targets_str = "\n".join(
                    f"- ID: '{t.id}' (Status: {t.status}), Description: '{t.desc[:60]}...'" for t in
                    active_queryable_tasks)
                prompt_parts.append(
                    f"\n\nYou can send a DIRECT MESSAGE (query) to these COMPLETED tasks to ask questions about their results:\n{targets_str}")

            if not active_interruptible_tasks and not active_queryable_tasks:
                prompt_parts.append("\nNo other tasks are available for communication right now.")

            # --- END OF THE REFINED FIX ---

            prompt_parts.append("---------------------------------------")
            return "\n".join(prompt_parts)