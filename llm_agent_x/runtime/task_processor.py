import asyncio
import logging

from opentelemetry import trace, context as otel_context
from opentelemetry.trace import StatusCode

from typing import TYPE_CHECKING, Optional

from llm_agent_x.state_manager import AbstractStateManager, TaskContext

# Forward reference for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm_agent_x.agents import InteractiveDAGAgent

logger = logging.getLogger("TaskProcessor")

class TaskProcessor:
    """Handles the execution logic for a single task."""

    def __init__(self, state_manager: AbstractStateManager, agent_base: 'InteractiveDAGAgent'):
        self.state_manager = state_manager
        self.tracer = trace.get_tracer(__name__)
        # This is the dependency on the agent for its helper methods
        self.agent_base: Optional['InteractiveDAGAgent'] = agent_base or None

    def set_agent_base(self, agent_base: 'InteractiveDAGAgent'):
        """Injects the agent instance after construction."""
        if self.agent_base is not None:
            raise RuntimeError("TaskProcessor's agent_base is already set.")
        self.agent_base = agent_base
        logger.info("Agent base successfully injected into TaskProcessor.")

    async def process_task(self, task_id: str):
        """The entry point for processing a single task. This logic is unchanged."""

        if not self.agent_base:
            raise RuntimeError("TaskProcessor cannot run without an agent_base. Call set_agent_base() first.")

        task = self.state_manager.get_task(task_id)
        if not task:
            logger.error(f"TaskProcessor started for non-existent task ID: {task_id}")
            return

        ctx = TaskContext(task_id, self.state_manager)

        # The entire _run_taskflow logic from before fits perfectly here.
        # We just call methods on `self.agent_base` instead of `self`.
        with self.tracer.start_as_current_span(f"TaskProcessing: {task.desc[:50]}") as span:
            task.span = span
            span.set_attribute("dag.task.id", task.id)
            span.set_attribute("dag.task.status", task.status)

            if task.status in ["paused_by_human", "waiting_for_user_response"]:
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                def update_status(new_status):
                    if task.status != new_status:
                        task.status = new_status
                        self.state_manager.upsert_task(task)

                if task.status == "waiting_for_children":
                    update_status("running")
                    await self.agent_base._run_task_execution(ctx)
                elif task.needs_planning and not task.already_planned:
                    update_status("planning")
                    await self.agent_base._run_initial_planning(ctx)
                    if task.status not in ["paused_by_human", "waiting_for_user_response"]:
                        task.already_planned = True
                        update_status("waiting_for_children" if task.children else "complete")
                elif task.can_request_new_subtasks and task.status != "proposing":
                    update_status("proposing")
                    await self.agent_base._run_adaptive_decomposition(ctx)
                    if task.status not in ["paused_by_human", "waiting_for_user_response"]:
                        update_status("waiting_for_children")
                elif task.status in ["pending", "running"]:
                    update_status("running")
                    await self.agent_base._run_task_execution(ctx)

                if task.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    update_status("complete")
                    task.last_llm_history, task.agent_role_paused = None, None

            except asyncio.CancelledError:
                if task.status not in ["waiting_for_user_response"]: task.status = "paused_by_human"
                self.state_manager.upsert_task(task)
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                task.status = "failed"; task.result = f"Processing error: {e}"
                task.last_llm_history = task.agent_role_paused = None
                raise
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", task.status)
                self.state_manager.upsert_task(task)