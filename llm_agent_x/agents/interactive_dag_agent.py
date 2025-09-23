import asyncio
import json
import logging
import os
import threading
import uuid
from collections import deque
from typing import List, Tuple, Union, Dict, Any

import pika
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic_ai.agent import AgentRunResult

from llm_agent_x.agents.dag_agent import (
    DAGAgent, Task, TaskRegistry, TaskContext,
    ExecutionPlan, ProposedSubtask, verification, RetryDecision, UserQuestion, ProposalResolutionPlan
)

from dotenv import load_dotenv

from opentelemetry import trace, context as otel_context

load_dotenv(".env", override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InteractiveDAGAgent")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")


class InteractiveDAGAgent(DAGAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.directives_queue = asyncio.Queue()
        self.rabbit_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))

        self.publish_channel = self.rabbit_connection.channel()
        self.STATE_UPDATES_EXCHANGE = 'state_updates_exchange'
        self.publish_channel.exchange_declare(exchange=self.STATE_UPDATES_EXCHANGE, exchange_type='fanout')

        self.consume_channel = self.rabbit_connection.channel()
        self.DIRECTIVES_QUEUE = 'directives_queue'
        self.consume_channel.queue_declare(queue=self.DIRECTIVES_QUEUE, durable=True)
        self.consume_channel.basic_qos(prefetch_count=1)

        # Map agent role names to actual agent instances for dynamic lookup
        self._agent_role_map = {
            "initial_planner": self.initial_planner,
            "cycle_breaker": self.cycle_breaker,
            "adaptive_decomposer": self.adaptive_decomposer,
            "conflict_resolver": self.conflict_resolver,
            "executor": self.executor,
            "verifier": self.verifier,
            "retry_analyst": self.retry_analyst,
        }

    def _broadcast_state_update(self, task: Task):
        """Publishes the full state of a task to RabbitMQ."""
        # Exclude 'last_llm_history' and 'otel_context' as they might be too large or non-serializable for RabbitMQ,
        # or contain circular references if not handled carefully by model_dump_json's default behavior.
        message = task.model_dump_json(exclude={'span', 'otel_context', 'last_llm_history'})
        self.publish_channel.basic_publish(
            exchange=self.STATE_UPDATES_EXCHANGE,
            routing_key='',
            body=message
        )

    # --- Override _handle_agent_output for Interactive DAG Agent ---
    async def _handle_agent_output(self, ctx: TaskContext, agent_res: AgentRunResult, expected_output_type: Any,
                                   agent_role_name: str) -> Tuple[bool, Any]:
        """
        Processes an agent's output. If it's a UserQuestion, pauses the task.
        Stores the full message history of the agent run.
        Returns (is_paused, actual_output).
        """
        t = ctx.task
        self._add_llm_data_to_span(t.span, agent_res, t)

        # Store the full message history for context preservation,
        # regardless of whether a question was asked or not.
        t.last_llm_history = agent_res.all_messages()
        actual_output = agent_res.output

        if isinstance(actual_output, UserQuestion):
            logger.info(
                f"Task [{t.id}] asking human question (Priority: {actual_output.priority}): {actual_output.question[:80]}...")
            t.current_question = actual_output
            t.agent_role_paused = agent_role_name  # Store which agent role asked the question
            t.status = "waiting_for_user_response"
            self._broadcast_state_update(t)  # Broadcast that the task is now waiting for user response
            return True, None  # Indicate paused
        elif isinstance(actual_output, expected_output_type):
            return False, actual_output  # Not paused, here's the normal output
        else:
            logger.warning(
                f"Task [{t.id}] received unexpected output type from agent. Expected {expected_output_type.__name__} or UserQuestion, got {type(actual_output).__name__}.")
            return False, actual_output

    def _listen_for_directives(self):
        """Listens for messages from RabbitMQ and puts them in an async queue."""

        def callback(ch, method, properties, body):
            message = json.loads(body)
            logger.info(f"Received directive from RabbitMQ: {message}")
            # Ensure this is put into the loop's context
            if asyncio.get_event_loop().is_running():
                asyncio.get_event_loop().call_soon_threadsafe(self.directives_queue.put_nowait, message)
            else:
                # Fallback for synchronous environments, though less robust
                self.directives_queue.put_nowait(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        self.consume_channel.basic_consume(queue=self.DIRECTIVES_QUEUE, on_message_callback=callback)
        logger.info("Starting to consume directives...")
        try:
            self.consume_channel.start_consuming()
        except Exception as e:
            logger.error(f"Directive consumer stopped: {e}")

    async def _handle_directive(self, directive: dict):
        command = directive.get("command")

        if command == "ADD_ROOT_TASK":
            payload = directive.get("payload", {})
            desc = payload.get("desc")
            if not desc:
                logger.warning("ADD_ROOT_TASK directive received with no description.")
                return

            new_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=desc,
                needs_planning=payload.get("needs_planning", True),
                status="pending"
            )
            self.registry.add_task(new_task)
            logger.info(f"Added new root task from directive: {new_task.id} - {new_task.desc}")
            self._broadcast_state_update(new_task)
            return

        task_id = directive.get("task_id")
        payload = directive.get("payload")
        task = self.registry.tasks.get(task_id)
        if not task:
            logger.warning(f"Directive for unknown task {task_id} ignored.")
            return

        logger.info(f"Handling command '{command}' for task {task_id}")

        # If a task is currently in-flight, cancel its future before applying directives
        if task_id in self.task_futures and not self.task_futures[task_id].done():
            self.task_futures[task_id].cancel()
            logger.info(f"Cancelled in-flight asyncio.Task for {task_id} due to directive.")
            # We don't remove from self.inflight and self.task_futures here;
            # the main run loop's asyncio.wait will handle completed/cancelled futures.

        original_status = task.status
        if command == "PAUSE":
            task.status = "paused_by_human"
        elif command == "RESUME":
            task.status = "pending"  # Resuming means it's ready to be picked up
        elif command == "TERMINATE":
            task.status = "failed"
            task.result = f"Terminated by operator: {payload}"
            self._reset_task_and_dependents(task_id)
        elif command == "REDIRECT":
            task.status = "pending"  # Redirect means it should re-run
            task.result = None
            task.fix_attempts = 0
            task.grace_attempts = 0
            task.verification_scores = []
            task.human_directive = payload
            task.user_response = None  # Clear any pending user response
            task.current_question = None
            task.last_llm_history = None  # Clear LLM history to restart context
            task.agent_role_paused = None  # Clear paused role
            self._reset_task_and_dependents(task_id)
        elif command == "MANUAL_OVERRIDE":
            task.status = "complete"
            task.result = payload
            task.human_directive = None
            task.user_response = None
            task.current_question = None
            task.last_llm_history = None  # Clear LLM history on manual completion
            task.agent_role_paused = None  # Clear paused role
            self._reset_task_and_dependents(task_id)
        elif command == "ANSWER_QUESTION":
            if task.status == "waiting_for_user_response" and task.current_question:
                task.user_response = payload
                task.current_question = None  # Clear the question as it's been answered
                task.status = "pending"  # Set to pending so _run_taskflow picks it up for resumption
                logger.info(f"Task {task_id} received answer to question: {payload[:50]}...")
            else:
                logger.warning(f"Task {task_id} not waiting for a question, ignoring ANSWER_QUESTION directive.")

        if task.status != original_status or command == "MANUAL_OVERRIDE" or command == "ANSWER_QUESTION":
            self._broadcast_state_update(task)

    def _reset_task_and_dependents(self, task_id: str):
        """Implements cascading state invalidation."""
        q = deque([task_id])
        dependents = set()
        while q:
            current_id = q.popleft()
            for t in self.registry.tasks.values():
                if current_id in t.deps and t.id not in dependents:
                    dependents.add(t.id)
                    q.append(t.id)

        if not dependents: return
        logger.info(f"Cascading invalidation will reset tasks: {dependents}")
        for tid in dependents:
            task = self.registry.tasks[tid]
            if task.status in ["complete", "failed", "running", "paused_by_human", "waiting_for_user_response",
                               "planning", "proposing", "waiting_for_children"]:
                task.status = "pending"
                task.result = None
                task.fix_attempts = 0
                task.grace_attempts = 0
                task.verification_scores = []
                task.human_directive = None
                task.user_response = None
                task.current_question = None
                task.last_llm_history = None  # Clear LLM history on reset
                task.agent_role_paused = None  # Clear paused role on reset
                self._broadcast_state_update(task)

    async def run(self):
        consumer_thread = threading.Thread(target=self._listen_for_directives, daemon=True)
        consumer_thread.start()

        logger.info("Broadcasting initial state of all tasks (if any)...")
        for task in self.registry.tasks.values():
            self._broadcast_state_update(task)

        try:
            with self.tracer.start_as_current_span("InteractiveDAGAgent.run"):
                while True:
                    while not self.directives_queue.empty():
                        directive = await self.directives_queue.get()
                        await self._handle_directive(directive)

                    all_task_ids = set(self.registry.tasks.keys())
                    executable_statuses = {'pending', 'running', 'planning', 'proposing', 'waiting_for_children'}
                    non_executable_tasks = {t.id for t in self.registry.tasks.values() if
                                            t.status not in executable_statuses}

                    for tid in all_task_ids - non_executable_tasks:
                        task = self.registry.tasks[tid]
                        if any(self.registry.tasks.get(d, {}).status == "failed" for d in task.deps):
                            if task.status != "failed":
                                task.status, task.result = "failed", "Upstream dependency failed."
                                self._broadcast_state_update(task)
                                # Also clear history if task fails due to upstream
                                task.last_llm_history = None
                                task.agent_role_paused = None

                    pending_tasks_for_scheduling = {t.id for t in self.registry.tasks.values() if
                                                    t.status in executable_statuses}

                    ready_to_run_ids = set()
                    for tid in pending_tasks_for_scheduling:
                        task = self.registry.tasks[tid]
                        # Condition 1: Task is pending AND has a user response and history (ready to resume)
                        if task.status == "pending" and task.user_response is not None and task.last_llm_history is not None and task.agent_role_paused is not None:
                            ready_to_run_ids.add(tid)
                        # Condition 2: Task is pending AND all dependencies are complete (ready for initial run)
                        elif task.status == "pending" and \
                                all(self.registry.tasks.get(d, {}).status == "complete" for d in task.deps):
                            ready_to_run_ids.add(tid)
                        # Condition 3: Parent task waiting for children, and all children are complete/failed
                        elif task.status == 'waiting_for_children' and \
                                all(self.registry.tasks[c].status in ['complete', 'failed'] for c in task.children):
                            if any(self.registry.tasks[c].status == 'failed' for c in task.children):
                                task.status, task.result = "failed", "A child task failed, cannot synthesize."
                                self._broadcast_state_update(task)
                                task.last_llm_history = None  # Clear history on failure
                                task.agent_role_paused = None
                            else:
                                ready_to_run_ids.add(tid)
                                task.status = "pending"  # Transition to pending for execution/synthesis
                                self._broadcast_state_update(task)

                    ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

                    if not ready and not self.inflight and (all_task_ids - non_executable_tasks):
                        # This means there are tasks that *should* be runnable but aren't, indicating a stall.
                        logger.error("Stalled DAG! No ready tasks, nothing inflight, but pending tasks exist.");
                        self.registry.print_status_tree()
                        # Consider setting a global error status or raising here in a production system.
                        # For now, we'll just log and continue, though it might loop.

                    if not (all_task_ids - non_executable_tasks) and not self.inflight:
                        # All tasks are complete, failed, or paused, and nothing is running.
                        # If there are still 'waiting_for_user_response' tasks, we should not exit.
                        if any(t.status == 'waiting_for_user_response' for t in self.registry.tasks.values()):
                            logger.info(
                                "All executable tasks complete/failed/paused, but some are waiting for user response. Waiting...")
                        else:
                            logger.info(
                                "Interactive DAG execution complete (no pending, no inflight, no user interaction needed).")
                            break  # Exit the main loop if truly nothing left to do.

                    for tid in ready:
                        self.inflight.add(tid)
                        self.task_futures[tid] = asyncio.create_task(self._run_taskflow(tid))
                        self._broadcast_state_update(self.registry.tasks[tid])

                    if self.task_futures:
                        done, pending_futures = await asyncio.wait(list(self.task_futures.values()), timeout=0.1,
                                                                   return_when=asyncio.FIRST_COMPLETED)
                        for fut in done:
                            tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                            if tid:
                                self.inflight.discard(tid)
                                del self.task_futures[tid]
                                try:
                                    fut.result()
                                except asyncio.CancelledError:
                                    logger.info(f"Task [{tid}] was cancelled by operator or system.")
                                except Exception as e:
                                    logger.error(f"Task [{tid}] future failed: {e}")
                                    # Ensure task status is updated to failed here if it wasn't already
                                    # and broadcast the update.
                                    t = self.registry.tasks.get(tid)
                                    if t and t.status != "failed":
                                        t.status = "failed"
                                        self._broadcast_state_update(t)
                                        t.last_llm_history = None  # Clear history on failure
                                        t.agent_role_paused = None

                    if self.proposed_tasks_buffer:
                        global_resolver_ctx_id = "GLOBAL_RESOLVER"
                        if global_resolver_ctx_id not in self.registry.tasks:
                            self.registry.add_task(
                                Task(id=global_resolver_ctx_id, desc="Internal task for global conflict resolution.",
                                     status="pending"))
                        global_resolver_task = self.registry.tasks[global_resolver_ctx_id]

                        # Check if the global resolver itself is waiting for user response
                        if global_resolver_task.status == "waiting_for_user_response":
                            # If it was waiting and now has a response, trigger its resumption
                            if global_resolver_task.user_response is not None and global_resolver_task.last_llm_history is not None and global_resolver_task.agent_role_paused == "conflict_resolver":
                                await self._resume_agent_from_question(
                                    TaskContext(global_resolver_ctx_id, self.registry))
                                # After resumption, if it's now 'pending' or 'complete', _run_global_resolution might run.
                                if global_resolver_task.status == "pending" or global_resolver_task.status == "complete":
                                    await self._run_global_resolution()
                            else:
                                logger.debug(
                                    "Global resolver is waiting for user response, skipping buffer processing.")
                        else:
                            await self._run_global_resolution()

                    await asyncio.sleep(0.05)
        finally:
            logger.info("Agent run loop is shutting down. Closing RabbitMQ connection.")
            self.rabbit_connection.close()

    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)

            # If the task is explicitly paused by human, just return.
            if t.status == "paused_by_human":
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution for now.")
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                # Special handling for resuming an agent that asked a question
                if t.status == "pending" and t.user_response is not None and t.last_llm_history is not None and t.agent_role_paused is not None:
                    logger.info(f"[{t.id}] Resuming {t.agent_role_paused} after human response.")
                    await self._resume_agent_from_question(ctx)
                    # After resumption, the task status might have changed (e.g., complete, failed, or even asked another question)
                    if t.status in ["waiting_for_user_response", "failed", "paused_by_human", "complete"]:
                        return  # If it's paused or completed, we're done with this flow iteration.
                    # If it's "pending" and still has work, it will fall through to normal logic below.

                def update_status_and_broadcast(new_status):
                    if t.status != new_status:
                        t.status = new_status
                        self._broadcast_state_update(t)

                # Normal task processing flow
                if t.status == 'waiting_for_children':
                    update_status_and_broadcast('running')
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    update_status_and_broadcast('planning')
                    await self._run_initial_planning(ctx)
                    if t.status not in ["waiting_for_user_response",
                                        "paused_by_human"]:  # Ensure not paused by planning itself
                        t.already_planned = True
                        update_status_and_broadcast("waiting_for_children" if t.children else "complete")
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    update_status_and_broadcast('proposing')
                    await self._run_adaptive_decomposition(ctx)
                    if t.status not in ["waiting_for_user_response",
                                        "paused_by_human"]:  # Ensure not paused by proposing itself
                        update_status_and_broadcast("waiting_for_children")
                elif t.status == "pending" or t.status == "running":  # Default execution for leaf tasks or after resume
                    update_status_and_broadcast('running')
                    await self._run_task_execution(ctx)
                else:
                    logger.warning(f"[{t.id}] Task in unhandled status '{t.status}'. Skipping.")

                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    update_status_and_broadcast("complete")
                    t.last_llm_history = None  # Clear history on success
                    t.agent_role_paused = None
            except asyncio.CancelledError:
                logger.info(f"Task [{t.id}] asyncio future was cancelled.")
                t.status = "paused_by_human"  # Or 'terminated_by_system', depending on desired behavior
                self._broadcast_state_update(t)
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                self._broadcast_state_update(t)
                t.last_llm_history = None  # Clear history on failure
                t.agent_role_paused = None
                raise  # Re-raise to ensure the future's result indicates failure
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", t.status)
                self._broadcast_state_update(t)  # Final broadcast

    # --- New method to resume an agent that asked a question ---
    async def _resume_agent_from_question(self, ctx: TaskContext):
        t = ctx.task
        if not t.user_response or not t.last_llm_history or not t.agent_role_paused:
            logger.error(
                f"[{t.id}] Attempted to resume agent, but missing user_response, last_llm_history, or agent_role_paused.")
            t.status = "failed"
            self._broadcast_state_update(t)
            return

        agent_instance = self._agent_role_map.get(t.agent_role_paused)
        if not agent_instance:
            logger.error(f"[{t.id}] Unknown agent role '{t.agent_role_paused}' to resume.")
            t.status = "failed"
            self._broadcast_state_update(t)
            return

        logger.info(f"[{t.id}] Resuming '{t.agent_role_paused}' with human response: {t.user_response[:50]}...")

        # Re-run the agent with the user's response appended to the history
        # Pydantic-AI's .run() intelligently handles `message_history` if user_prompt is also provided,
        # treating `user_prompt` as the latest user message in the conversation.
        resume_res = await agent_instance.run(
            user_prompt=t.user_response,  # This is the *new* user message
            message_history=t.last_llm_history  # This is the *entire previous conversation* up to the agent's question
        )

        # Clear the question-related fields as they have been handled
        # The agent.run() result already has the new history, which will be stored by _handle_agent_output.
        # So we clear these here before _handle_agent_output gets called.
        t.user_response = None
        t.current_question = None
        # t.last_llm_history will be overwritten by _handle_agent_output
        # t.agent_role_paused will be overwritten by _handle_agent_output if a new question is asked.

        # Process the result of the resumed agent run based on the original role
        # Call the corresponding processing logic for each agent type.
        original_agent_role_paused = t.agent_role_paused  # Capture before _handle_agent_output might overwrite

        if original_agent_role_paused == "initial_planner":
            is_paused, initial_plan = await self._handle_agent_output(ctx, resume_res, ExecutionPlan,
                                                                      original_agent_role_paused)
            if not is_paused:
                await self._process_initial_planning_output(ctx, initial_plan)
                if ctx.task.status not in ["waiting_for_user_response",
                                           "paused_by_human"]:  # Only mark as planned if not paused again
                    ctx.task.already_planned = True
                    ctx.task.status = "waiting_for_children" if ctx.task.children else "complete"
                    self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "cycle_breaker":
            is_paused, fixed_plan = await self._handle_agent_output(ctx, resume_res, ExecutionPlan,
                                                                    original_agent_role_paused)
            if not is_paused:
                # After cycle breaking, it's still part of the initial planning flow.
                # So we continue processing the plan as if it just came from the cycle breaker.
                await self._process_initial_planning_output(ctx, fixed_plan)
                if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                    ctx.task.already_planned = True
                    ctx.task.status = "waiting_for_children" if ctx.task.children else "complete"
                    self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "adaptive_decomposer":
            is_paused, proposals = await self._handle_agent_output(ctx, resume_res, List[ProposedSubtask],
                                                                   original_agent_role_paused)
            if not is_paused:
                if proposals:
                    logger.info(f"Task [{t.id}] (resumed) proposing {len(proposals)} new sub-tasks.")
                    for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))
                if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                    ctx.task.status = "waiting_for_children"  # After proposals, it waits for global resolution
                    self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "conflict_resolver":
            # This is specifically for the GLOBAL_RESOLVER task
            is_paused, resolved_plan = await self._handle_agent_output(ctx, resume_res, ProposalResolutionPlan,
                                                                       original_agent_role_paused)
            if not is_paused and resolved_plan:
                logger.info(
                    f"Global resolver for {ctx.task.id} (resumed) approved {len(resolved_plan.approved_tasks)} sub-tasks.")
                # Set status to pending so _run_global_resolution can process the buffer in the next main loop iteration
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)
            elif not is_paused and resolved_plan is None:
                logger.warning(
                    f"Global resolver for {ctx.task.id} resumed but returned no plan. Check agent behavior. Marking as failed.")
                ctx.task.status = "failed"
                self._broadcast_state_update(ctx.task)
            # If it's still paused, _handle_agent_output already updated state.

        elif original_agent_role_paused == "executor":
            is_paused, result = await self._handle_agent_output(ctx, resume_res, str, original_agent_role_paused)
            if not is_paused:
                await self._process_executor_output_for_verification(ctx, result)  # Continue with verification

        elif original_agent_role_paused == "verifier":
            # The verifier usually runs within a loop in _run_task_execution.
            # We need to re-enter _run_task_execution to continue its flow.
            # _handle_agent_output already processes its result and updates history/status.
            # Setting status to pending will make _run_taskflow re-evaluate and re-call _run_task_execution.
            if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "retry_analyst":
            # Similar to verifier, this runs within _run_task_execution's loop.
            # _handle_agent_output will process the decision.
            # We ensure status is pending to re-enter _run_task_execution.
            if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)

        else:
            logger.warning(
                f"[{t.id}] Resumption for role '{original_agent_role_paused}' completed, but specific processing logic is missing. Setting to pending.")
            if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)

        # Clear remaining context fields after full resumption process
        # This prevents an infinite loop where the task keeps trying to resume the same agent
        t.agent_role_paused = None
        t.last_llm_history = None

    # --- Methods calling agent.run() in InteractiveDAGAgent ---

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning for: {t.desc}")

        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context_str = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt = f"Objective: {t.desc}\n\nAvailable completed data sources:\n{context_str}"

        # Inject human directive if present
        if t.human_directive:
            prompt += f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            t.human_directive = None
            self._broadcast_state_update(t)  # Broadcast that directive was consumed

        # No need for the t.user_response block here, it's handled by _resume_agent_from_question

        plan_res = await self.initial_planner.run(
            user_prompt=prompt,
            message_history=t.last_llm_history
        )
        is_paused, initial_plan = await self._handle_agent_output(
            ctx=ctx, agent_res=plan_res, expected_output_type=ExecutionPlan, agent_role_name="initial_planner"
        )
        if is_paused: return  # Task is paused for user input

        # After initial planning, if subtasks are identified, check for cycles
        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
            fixer_prompt = f"Analyze and fix cycles in this plan:\n\n{initial_plan.model_dump_json(indent=2)}"

            fixed_plan_res = await self.cycle_breaker.run(
                user_prompt=fixer_prompt,
                message_history=t.last_llm_history  # Pass the full message history
            )
            is_paused, plan = await self._handle_agent_output(
                ctx=ctx, agent_res=fixed_plan_res, expected_output_type=ExecutionPlan, agent_role_name="cycle_breaker"
            )
            if is_paused: return  # Task is paused for user input
        else:
            plan = initial_plan  # No subtasks or no issues

        if not plan.needs_subtasks:
            # If no subtasks, the parent task is complete
            t.status = "complete"
            t.already_planned = True
            self._broadcast_state_update(t)
            return

        # If subtasks are generated, process them
        await self._process_initial_planning_output(ctx, plan)  # This is a new helper method
        if t.status not in ["waiting_for_user_response",
                            "paused_by_human"]:  # Ensure not paused by subsequent _handle_agent_output from _process_initial_planning_output
            t.already_planned = True
            t.status = "waiting_for_children" if t.children else "complete"  # If no children, it means plan had no subtasks
            self._broadcast_state_update(t)

    async def _process_initial_planning_output(self, ctx: TaskContext, plan: ExecutionPlan):
        """Helper to encapsulate common logic after initial planning produces an ExecutionPlan."""
        t = ctx.task
        if not plan.needs_subtasks: return

        local_to_global_id_map = {}
        for sub in plan.subtasks:
            new_task = Task(id=str(uuid.uuid4())[:8], desc=sub.desc, parent=t.id,
                            can_request_new_subtasks=sub.can_request_new_subtasks)
            self.registry.add_task(new_task)
            t.children.append(new_task.id)
            self.registry.add_dependency(t.id, new_task.id)
            local_to_global_id_map[sub.local_id] = new_task.id
            self._broadcast_state_update(new_task)  # Broadcast new child task

        for sub in plan.subtasks:
            new_global_id = local_to_global_id_map.get(sub.local_id)
            if not new_global_id: continue
            for dep in sub.deps:
                dep_global_id = local_to_global_id_map.get(dep.local_id) or (
                    dep.local_id if dep.local_id in self.registry.tasks else None)
                if dep_global_id: self.registry.add_dependency(new_global_id, dep_global_id)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt = f"Task: {t.desc}\n\nResults from dependencies:\n{t.dep_results}"

        # Inject human directive if present
        if t.human_directive:
            prompt += f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            t.human_directive = None
            self._broadcast_state_update(t)

        # No need for t.user_response here, handled by _resume_agent_from_question

        proposals_res = await self.adaptive_decomposer.run(
            user_prompt=prompt,
            message_history=t.last_llm_history
        )
        is_paused, proposals = await self._handle_agent_output(
            ctx=ctx, agent_res=proposals_res, expected_output_type=List[ProposedSubtask],
            agent_role_name="adaptive_decomposer"
        )
        if is_paused: return  # Task is paused for user input

        if proposals:
            logger.info(f"Task [{t.id}] proposing {len(proposals)} new sub-tasks.")
            for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))

    async def _run_task_execution(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running task execution for: {t.desc}")

        child_results = {cid: self.registry.tasks[cid].result for cid in t.children if
                         self.registry.tasks[cid].status == 'complete'}
        dep_results = {did: self.registry.tasks[did].result for did in t.deps if
                       self.registry.tasks[did].status == 'complete' and did not in child_results}

        prompt = f"Your task is: {t.desc}\n"

        # Inject human directive if present (only applies to current attempt, not future retries implicitly)
        if t.human_directive:
            prompt = f"--- CRITICAL GUIDANCE FROM OPERATOR ---\n{t.human_directive}\n--------------------------------------\n\n" + prompt
            t.human_directive = None
            self._broadcast_state_update(t)

        if child_results:
            prompt += "\nSynthesize the results from your sub-tasks into a final answer:\n"
            for cid, res in child_results.items():
                prompt += f"- From sub-task '{self.registry.tasks[cid].desc}':\n{res}\n\n"
        elif dep_results:
            prompt += "\nUse data from dependencies to inform your answer:\n"
            for did, res in dep_results.items():
                prompt += f"- From dependency '{self.registry.tasks[did].desc}':\n{res}\n\n"

        # No need for t.user_response here, it's handled by _resume_agent_from_question

        while True:
            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})

            logger.info(f"[{t.id}] Attempt {current_attempt}: Calling executor LLM.")
            exec_res = await self.executor.run(
                user_prompt=prompt,
                message_history=t.last_llm_history
            )
            is_paused, result = await self._handle_agent_output(
                ctx=ctx, agent_res=exec_res, expected_output_type=str, agent_role_name="executor"
            )
            if is_paused: return

            await self._process_executor_output_for_verification(ctx, result)
            # _process_executor_output_for_verification can transition status to complete, failed, or waiting_for_user_response.
            # If it's waiting for user response, the task will return from _run_taskflow, and _resume_agent_from_question will handle it.
            # Otherwise, if it's complete/failed, this loop should break.
            if t.status in ["complete", "failed", "waiting_for_user_response", "paused_by_human"]:
                return

            # If it's still running/pending after _process_executor_output_for_verification (meaning retry is needed),
            # this loop will continue to the next iteration to re-run the executor with feedback.
            # The prompt will be updated within _process_executor_output_for_verification if a retry is needed.

    async def _process_executor_output_for_verification(self, ctx: TaskContext, result: str):
        """Helper to handle verification and retry logic after executor runs."""
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)
        is_successful = verify_task_result.get_successful()

        if is_successful:
            t.result = result
            logger.info(f"COMPLETED [{t.id}]");
            t.status = "complete"  # Explicitly set status to complete
            self._broadcast_state_update(t)
            t.last_llm_history = None  # Clear history on success
            t.agent_role_paused = None
            return

        # If not successful, proceed with retry logic
        t.fix_attempts += 1
        if t.fix_attempts >= t.max_fix_attempts and t.grace_attempts < self.max_grace_attempts:
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\n" \
                             f"Score > 5 is a success. Should we retry?"

            # No need for t.user_response here, handled by _resume_agent_from_question

            decision_res = await self.retry_analyst.run(
                user_prompt=analyst_prompt,
                message_history=t.last_llm_history
            )
            is_paused, decision = await self._handle_agent_output(
                ctx=ctx, agent_res=decision_res, expected_output_type=RetryDecision, agent_role_name="retry_analyst"
            )
            if is_paused: return  # Task is paused for user input

            if decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion  # Inject suggestion for next attempt
                logger.info(f"[{t.id}] Grace attempt granted. Next step: {decision.next_step_suggestion}")
                # Don't explicitly set status to "pending" here, the outer loop will continue.
                # The next execution will incorporate the human_directive.
                return  # Return to _run_task_execution to retry with updated prompt/directive

        # If max attempts (including grace) are exceeded
        if (t.fix_attempts + t.grace_attempts) >= (t.max_fix_attempts + self.max_grace_attempts):
            error_msg = f"Exceeded max attempts for task '{t.id}'"
            t.span.set_status(trace.Status(StatusCode.ERROR, error_msg));
            t.status = "failed"
            self._broadcast_state_update(t)
            t.last_llm_history = None  # Clear history on failure
            t.agent_role_paused = None
            raise Exception(error_msg)

        # If just a normal retry (within initial max_fix_attempts)
        # Add feedback to the prompt for the next executor run
        t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
        logger.info(f"[{t.id}] Retrying execution. Feedback: {verify_task_result.reason[:50]}...")
        # Status remains "running" or "pending" to trigger next iteration of _run_task_execution loop.
        # The executor will then receive this updated `human_directive`.

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."

        # No need for t.user_response here, handled by _resume_agent_from_question

        vres = await self.verifier.run(
            user_prompt=ver_prompt,
            message_history=t.last_llm_history
        )
        is_paused, vout = await self._handle_agent_output(
            ctx=TaskContext(t.id, self.registry), agent_res=vres, expected_output_type=verification,
            agent_role_name="verifier"
        )
        if is_paused:
            # If verifier asks a question, we must pause the *current* execution path.
            # The execution loop in _run_task_execution will handle this return.
            return verification(reason="Verifier paused to ask question.", message_for_user="", score=0)

        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout