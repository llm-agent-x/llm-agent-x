import asyncio
import json
import logging
import os
import threading
import uuid
from collections import deque
from typing import List, Tuple, Union, Dict, Any, Optional

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

        # Publisher connection for broadcasting state updates (used by main thread)
        self._publisher_connection: Optional[pika.BlockingConnection] = None
        self._publish_channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self.STATE_UPDATES_EXCHANGE = 'state_updates_exchange'

        # Consumer thread's dedicated connection (used only by the consumer thread)
        self._consumer_connection_for_thread: Optional[pika.BlockingConnection] = None
        self._consumer_channel_for_thread: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self.DIRECTIVES_QUEUE = 'directives_queue'  # The queue for inbound directives from gateway
        self._shutdown_event = threading.Event()  # Event to signal consumer thread to stop
        self._consumer_thread: Optional[threading.Thread] = None  # Reference to the consumer thread

        # Store the main event loop reference (set in run())
        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None

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

    def _get_publisher_channel(self) -> pika.adapters.blocking_connection.BlockingChannel:
        """Ensures a publisher channel is available and open."""
        if self._publisher_connection is None or self._publisher_connection.is_closed:
            logger.info("Establishing new publisher RabbitMQ connection and channel.")
            self._publisher_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
            self._publish_channel = self._publisher_connection.channel()
            self._publish_channel.exchange_declare(exchange=self.STATE_UPDATES_EXCHANGE, exchange_type='fanout')
        return self._publish_channel

    def _broadcast_state_update(self, task: Task):
        """Publishes the full state of a task to RabbitMQ."""
        try:
            channel = self._get_publisher_channel()
            message = task.model_dump_json(exclude={'span', 'otel_context', 'last_llm_history'})
            channel.basic_publish(
                exchange=self.STATE_UPDATES_EXCHANGE,
                routing_key='',
                body=message
            )
        except Exception as e:
            logger.error(f"Failed to broadcast state update for task {task.id}: {e}", exc_info=False)

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

        t.last_llm_history = agent_res.all_messages()
        actual_output = agent_res.output

        if isinstance(actual_output, UserQuestion):
            logger.info(
                f"Task [{t.id}] asking human question (Priority: {actual_output.priority}): {actual_output.question[:80]}...")
            t.current_question = actual_output
            t.agent_role_paused = agent_role_name
            t.status = "waiting_for_user_response"
            self._broadcast_state_update(t)
            return True, None
        elif isinstance(actual_output, expected_output_type):
            return False, actual_output
        else:
            logger.warning(
                f"Task [{t.id}] received unexpected output type from agent. Expected {expected_output_type.__name__} or UserQuestion, got {type(actual_output).__name__}.")
            return False, actual_output

    def _listen_for_directives_target(self):
        """
        Target function for the consumer thread. It runs in a separate thread.
        Manages its own RabbitMQ connection.
        """
        logger.info("Consumer thread starting: Connecting to RabbitMQ for directives...")
        try:
            # Establish an independent connection for this thread
            self._consumer_connection_for_thread = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
            self._consumer_channel_for_thread = self._consumer_connection_for_thread.channel()
            self._consumer_channel_for_thread.queue_declare(queue=self.DIRECTIVES_QUEUE, durable=True)
            self._consumer_channel_for_thread.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body):
                message = json.loads(body)
                logger.debug(f"Received directive from RabbitMQ: {message}")

                # Use the stored main event loop reference
                main_loop = self._main_event_loop
                if main_loop and main_loop.is_running():  # Check if it's available and running
                    main_loop.call_soon_threadsafe(self.directives_queue.put_nowait, message)
                else:
                    logger.warning(
                        "Main asyncio loop not running or not set, cannot put directive into queue. Directive dropped. Message: %s",
                        message)
                ch.basic_ack(delivery_tag=method.delivery_tag)

            self._consumer_channel_for_thread.basic_consume(queue=self.DIRECTIVES_QUEUE, on_message_callback=callback)
            logger.info("Consumer thread: Starting to consume directives...")

            # Keep consuming until instructed to shut down
            while not self._shutdown_event.is_set():
                # Process events for a short duration, allowing periodic checks for shutdown_event
                # This ensures the thread doesn't block indefinitely on consumption if no messages arrive.
                self._consumer_connection_for_thread.process_data_events(time_limit=0.1)

            logger.info("Consumer thread: Shutdown event received. Stopping consuming.")
            # Ensure consuming stops gracefully if it was active
            self._consumer_channel_for_thread.stop_consuming()

        except pika.exceptions.ConnectionClosedByBroker:
            logger.error("Consumer thread: RabbitMQ connection closed by broker.", exc_info=False)
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Consumer thread: AMQP Connection Error: {e}", exc_info=False)
        except Exception as e:
            logger.error(f"Consumer thread stopped unexpectedly: {e}", exc_info=True)
        finally:
            if self._consumer_connection_for_thread and self._consumer_connection_for_thread.is_open:
                logger.info("Consumer thread: Closing RabbitMQ connection.")
                self._consumer_connection_for_thread.close()
            self._consumer_connection_for_thread = None
            self._consumer_channel_for_thread = None
            logger.info("Consumer thread exited.")

    async def _handle_directive(self, directive: dict):
        command = directive.get("command")

        # The _handle_directive method *receives* directives from the queue.
        # It does NOT publish them. The publisher connection is managed by `run()` and `_broadcast_state_update`.
        # No publisher initialization is needed here.

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

        if task_id in self.task_futures and not self.task_futures[task_id].done():
            self.task_futures[task_id].cancel()
            logger.info(f"Cancelled in-flight asyncio.Task for {task_id} due to directive.")

        original_status = task.status
        if command == "PAUSE":
            task.status = "paused_by_human"
        elif command == "RESUME":
            task.status = "pending"
        elif command == "TERMINATE":
            task.status = "failed"
            task.result = f"Terminated by operator: {payload}"
            self._reset_task_and_dependents(task_id)
        elif command == "REDIRECT":
            task.status = "pending"
            task.result = None
            task.fix_attempts = 0
            task.grace_attempts = 0
            task.verification_scores = []
            task.human_directive = payload
            task.user_response = None
            task.current_question = None
            task.last_llm_history = None
            task.agent_role_paused = None
            self._reset_task_and_dependents(task_id)
        elif command == "MANUAL_OVERRIDE":
            task.status = "complete"
            task.result = payload
            task.human_directive = None
            task.user_response = None
            task.current_question = None
            task.last_llm_history = None
            task.agent_role_paused = None
            self._reset_task_and_dependents(task_id)
        elif command == "ANSWER_QUESTION":
            if task.status == "waiting_for_user_response" and task.current_question:
                task.user_response = payload
                task.current_question = None
                task.status = "pending"
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
                task.last_llm_history = None
                task.agent_role_paused = None
                self._broadcast_state_update(task)

    async def run(self):
        # Capture the main event loop reference *before* starting any new threads
        self._main_event_loop = asyncio.get_running_loop()

        # Start the consumer thread
        self._consumer_thread = threading.Thread(target=self._listen_for_directives_target, daemon=True)
        self._consumer_thread.start()

        # Initialize publisher connection for the main thread
        try:
            self._publisher_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
            self._publish_channel = self._publisher_connection.channel()
            self._publish_channel.exchange_declare(exchange=self.STATE_UPDATES_EXCHANGE, exchange_type='fanout')
            self._publish_channel.queue_declare(queue=self.DIRECTIVES_QUEUE, durable=True)
            logger.info("Publisher RabbitMQ connection and channel initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize publisher RabbitMQ connection, agent cannot communicate state: {e}",
                            exc_info=True)
            self._shutdown_event.set()  # Signal consumer thread to stop
            raise  # Propagate critical error to prevent agent from running silently.

        logger.info("Broadcasting initial state of all tasks (if any)...")
        for task in self.registry.tasks.values():
            self._broadcast_state_update(task)

        try:
            with self.tracer.start_as_current_span("InteractiveDAGAgent.run"):
                while True:
                    # Process any directives received from the consumer thread
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
                                task.last_llm_history = None
                                task.agent_role_paused = None
                            else:
                                ready_to_run_ids.add(tid)
                                task.status = "pending"  # Transition to pending for execution/synthesis
                                self._broadcast_state_update(task)

                    ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

                    # Check for graceful shutdown conditions.
                    # The loop continues if there's work for agents, or if the consumer thread is still active
                    # (meaning more directives could arrive).
                    has_pending_interaction = any(
                        t.status == 'waiting_for_user_response' for t in self.registry.tasks.values())
                    if not ready and \
                            not self.inflight and \
                            not has_pending_interaction and \
                            self.directives_queue.empty():  # Check internal queue for directives processed in this loop

                        # If no internal work or pending interaction, check if external input is still possible.
                        if self._shutdown_event.is_set() or not self._consumer_thread.is_alive():
                            # Consumer thread is either signaled to stop, or already dead.
                            # So no new directives will come in. It's safe to shut down.
                            logger.info(
                                "Interactive DAG execution complete (no pending, no inflight, no user interaction, no directives, consumer thread inactive/stopping).")
                            break
                        else:
                            # Agent is idle, but consumer thread is still alive and waiting for external directives.
                            # So, we should continue looping and waiting.
                            logger.debug("Agent idle, but consumer thread active. Waiting for directives...")

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
                                    t = self.registry.tasks.get(tid)
                                    if t and t.status != "failed":
                                        t.status = "failed"
                                        self._broadcast_state_update(t)
                                        t.last_llm_history = None
                                        t.agent_role_paused = None

                    if self.proposed_tasks_buffer:
                        global_resolver_ctx_id = "GLOBAL_RESOLVER"
                        if global_resolver_ctx_id not in self.registry.tasks:
                            self.registry.add_task(
                                Task(id=global_resolver_ctx_id, desc="Internal task for global conflict resolution.",
                                     status="pending"))
                        global_resolver_task = self.registry.tasks[global_resolver_ctx_id]

                        if global_resolver_task.status == "waiting_for_user_response":
                            if global_resolver_task.user_response is not None and global_resolver_task.last_llm_history is not None and global_resolver_task.agent_role_paused == "conflict_resolver":
                                await self._resume_agent_from_question(
                                    TaskContext(global_resolver_ctx_id, self.registry))
                                if global_resolver_task.status == "pending" or global_resolver_task.status == "complete":
                                    await self._run_global_resolution()
                            else:
                                logger.debug(
                                    "Global resolver is waiting for user response, skipping buffer processing.")
                        else:
                            await self._run_global_resolution()

                    await asyncio.sleep(0.05)
        finally:
            logger.info(
                "Agent run loop is shutting down. Signaling consumer thread to stop and closing publisher connection.")

            # Signal consumer thread to stop
            self._shutdown_event.set()
            if self._consumer_thread and self._consumer_thread.is_alive():
                self._consumer_thread.join(timeout=5)  # Wait for consumer to finish gracefully
                if self._consumer_thread.is_alive():
                    logger.warning("Consumer thread did not shut down gracefully after 5 seconds.")

            # Close publisher connection
            if self._publisher_connection and self._publisher_connection.is_open:
                logger.info("Closing publisher RabbitMQ connection.")
                self._publisher_connection.close()
            self._publisher_connection = None
            self._publish_channel = None

    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)

            if t.status == "paused_by_human":
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution for now.")
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                if t.status == "pending" and t.user_response is not None and t.last_llm_history is not None and t.agent_role_paused is not None:
                    logger.info(f"[{t.id}] Resuming {t.agent_role_paused} after human response.")
                    await self._resume_agent_from_question(ctx)
                    if t.status in ["waiting_for_user_response", "failed", "paused_by_human", "complete"]:
                        return

                def update_status_and_broadcast(new_status):
                    if t.status != new_status:
                        t.status = new_status
                        self._broadcast_state_update(t)

                if t.status == 'waiting_for_children':
                    update_status_and_broadcast('running')
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    update_status_and_broadcast('planning')
                    await self._run_initial_planning(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        t.already_planned = True
                        update_status_and_broadcast("waiting_for_children" if t.children else "complete")
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    update_status_and_broadcast('proposing')
                    await self._run_adaptive_decomposition(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        update_status_and_broadcast("waiting_for_children")
                elif t.status == "pending" or t.status == "running":
                    update_status_and_broadcast('running')
                    await self._run_task_execution(ctx)
                else:
                    logger.warning(f"[{t.id}] Task in unhandled status '{t.status}'. Skipping.")

                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    update_status_and_broadcast("complete")
                    t.last_llm_history = None
                    t.agent_role_paused = None
            except asyncio.CancelledError:
                logger.info(f"Task [{t.id}] asyncio future was cancelled.")
                t.status = "paused_by_human"
                self._broadcast_state_update(t)
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                self._broadcast_state_update(t)
                t.last_llm_history = None
                t.agent_role_paused = None
                raise
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", t.status)
                self._broadcast_state_update(t)

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

        resume_res = await agent_instance.run(
            user_prompt=t.user_response,
            message_history=t.last_llm_history
        )

        t.user_response = None
        t.current_question = None

        original_agent_role_paused = t.agent_role_paused

        if original_agent_role_paused == "initial_planner":
            is_paused, initial_plan = await self._handle_agent_output(ctx, resume_res, ExecutionPlan,
                                                                      original_agent_role_paused)
            if not is_paused:
                await self._process_initial_planning_output(ctx, initial_plan)
                if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                    ctx.task.already_planned = True
                    ctx.task.status = "waiting_for_children" if ctx.task.children else "complete"
                    if ctx.task.status == "complete":
                        ctx.task.last_llm_history = None
                        ctx.task.agent_role_paused = None
                    self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "cycle_breaker":
            is_paused, fixed_plan = await self._handle_agent_output(ctx, resume_res, ExecutionPlan,
                                                                    original_agent_role_paused)
            if not is_paused:
                await self._process_initial_planning_output(ctx, fixed_plan)
                if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                    ctx.task.already_planned = True
                    ctx.task.status = "waiting_for_children" if ctx.task.children else "complete"
                    if ctx.task.status == "complete":
                        ctx.task.last_llm_history = None
                        ctx.task.agent_role_paused = None
                    self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "adaptive_decomposer":
            is_paused, proposals = await self._handle_agent_output(ctx, resume_res, List[ProposedSubtask],
                                                                   original_agent_role_paused)
            if not is_paused:
                if proposals:
                    logger.info(f"Task [{t.id}] (resumed) proposing {len(proposals)} new sub-tasks.")
                    for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))
                if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                    ctx.task.status = "waiting_for_children"
                    self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "conflict_resolver":
            is_paused, resolved_plan = await self._handle_agent_output(ctx, resume_res, ProposalResolutionPlan,
                                                                       original_agent_role_paused)
            if not is_paused and resolved_plan:
                logger.info(
                    f"Global resolver for {ctx.task.id} (resumed) approved {len(resolved_plan.approved_tasks)} sub-tasks.")
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)
            elif not is_paused and resolved_plan is None:
                logger.warning(
                    f"Global resolver for {ctx.task.id} resumed but returned no plan. Check agent behavior. Marking as failed.")
                ctx.task.status = "failed"
                self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "executor":
            is_paused, result = await self._handle_agent_output(ctx, resume_res, str, original_agent_role_paused)
            if not is_paused:
                await self._process_executor_output_for_verification(ctx, result)

        elif original_agent_role_paused == "verifier":
            is_paused, vout = await self._handle_agent_output(ctx, resume_res, verification, original_agent_role_paused)
            if not is_paused:
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)

        elif original_agent_role_paused == "retry_analyst":
            is_paused, decision = await self._handle_agent_output(ctx, resume_res, RetryDecision,
                                                                  original_agent_role_paused)
            if not is_paused:
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)

        else:
            logger.warning(
                f"[{t.id}] Resumption for role '{original_agent_role_paused}' completed, but specific processing logic is missing. Setting to pending.")
            if ctx.task.status not in ["waiting_for_user_response", "paused_by_human"]:
                ctx.task.status = "pending"
                self._broadcast_state_update(ctx.task)

        if ctx.task.status != "waiting_for_user_response":
            t.agent_role_paused = None
            t.last_llm_history = None

    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning for: {t.desc}")

        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context_str = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt = f"Objective: {t.desc}\n\nAvailable completed data sources:\n{context_str}"

        if t.human_directive:
            prompt += f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            t.human_directive = None
            self._broadcast_state_update(t)

        plan_res = await self.initial_planner.run(
            user_prompt=prompt,
            message_history=t.last_llm_history
        )
        is_paused, initial_plan = await self._handle_agent_output(
            ctx=ctx, agent_res=plan_res, expected_output_type=ExecutionPlan, agent_role_name="initial_planner"
        )
        if is_paused: return

        if initial_plan and initial_plan.needs_subtasks and initial_plan.subtasks:
            fixer_prompt = f"Analyze and fix cycles in this plan:\n\n{initial_plan.model_dump_json(indent=2)}"

            fixed_plan_res = await self.cycle_breaker.run(
                user_prompt=fixer_prompt,
                message_history=t.last_llm_history
            )
            is_paused, plan = await self._handle_agent_output(
                ctx=ctx, agent_res=fixed_plan_res, expected_output_type=ExecutionPlan, agent_role_name="cycle_breaker"
            )
            if is_paused: return
        else:
            plan = initial_plan

        if not plan.needs_subtasks:
            t.status = "complete"
            t.already_planned = True
            self._broadcast_state_update(t)
            t.last_llm_history = None
            t.agent_role_paused = None
            return

        await self._process_initial_planning_output(ctx, plan)
        if t.status not in ["waiting_for_user_response", "paused_by_human"]:
            t.already_planned = True
            t.status = "waiting_for_children" if t.children else "complete"
            if t.status == "complete":
                t.last_llm_history = None
                t.agent_role_paused = None
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
            self._broadcast_state_update(new_task)

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

        if t.human_directive:
            prompt += f"\n\n--- HUMAN OPERATOR DIRECTIVE ---\n{t.human_directive}\n----------------------------\n"
            t.human_directive = None
            self._broadcast_state_update(t)

        proposals_res = await self.adaptive_decomposer.run(
            user_prompt=prompt,
            message_history=t.last_llm_history
        )
        is_paused, proposals = await self._handle_agent_output(
            ctx=ctx, agent_res=proposals_res, expected_output_type=List[ProposedSubtask],
            agent_role_name="adaptive_decomposer"
        )
        if is_paused: return

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

        prompt_base = f"Your task is: {t.desc}\n"
        if child_results:
            prompt_base += "\nSynthesize the results from your sub-tasks into a final answer:\n"
            for cid, res in child_results.items():
                prompt_base += f"- From sub-task '{self.registry.tasks[cid].desc}':\n{res}\n\n"
        elif dep_results:
            prompt_base += "\nUse data from dependencies to inform your answer:\n"
            for did, res in dep_results.items():
                prompt_base += f"- From dependency '{self.registry.tasks[did].desc}':\n{res}\n\n"

        while True:
            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})

            current_prompt = prompt_base
            if t.human_directive:
                current_prompt = f"--- CRITICAL GUIDANCE FROM OPERATOR ---\n{t.human_directive}\n--------------------------------------\n\n" + current_prompt
                t.human_directive = None
                self._broadcast_state_update(t)

            logger.info(f"[{t.id}] Attempt {current_attempt}: Calling executor LLM.")
            exec_res = await self.executor.run(
                user_prompt=current_prompt,
                message_history=t.last_llm_history
            )
            is_paused, result = await self._handle_agent_output(
                ctx=ctx, agent_res=exec_res, expected_output_type=str, agent_role_name="executor"
            )
            if is_paused: return

            await self._process_executor_output_for_verification(ctx, result)
            if t.status in ["complete", "failed", "waiting_for_user_response", "paused_by_human"]:
                return

    async def _process_executor_output_for_verification(self, ctx: TaskContext, result: str):
        """Helper to handle verification and retry logic after executor runs."""
        t = ctx.task
        verify_task_result = await self._verify_task(t, result)
        is_successful = verify_task_result.get_successful()

        if is_successful:
            t.result = result
            logger.info(f"COMPLETED [{t.id}]");
            t.status = "complete"
            self._broadcast_state_update(t)
            t.last_llm_history = None
            t.agent_role_paused = None
            return

        t.fix_attempts += 1
        if t.fix_attempts >= t.max_fix_attempts and t.grace_attempts < self.max_grace_attempts:
            analyst_prompt = f"Task: '{t.desc}'\nScores: {t.verification_scores}\n" \
                             f"Score > 5 is a success. Should we retry?"

            decision_res = await self.retry_analyst.run(
                user_prompt=analyst_prompt,
                message_history=t.last_llm_history
            )
            is_paused, decision = await self._handle_agent_output(
                ctx=ctx, agent_res=decision_res, expected_output_type=RetryDecision, agent_role_name="retry_analyst"
            )
            if is_paused: return

            if decision.should_retry:
                t.span.add_event("Grace attempt granted", {"reason": decision.reason})
                t.grace_attempts += 1
                t.human_directive = decision.next_step_suggestion
                logger.info(f"[{t.id}] Grace attempt granted. Next step: {decision.next_step_suggestion}")
                return

        if (t.fix_attempts + t.grace_attempts) >= (t.max_fix_attempts + self.max_grace_attempts):
            error_msg = f"Exceeded max attempts for task '{t.id}'"
            t.span.set_status(trace.Status(StatusCode.ERROR, error_msg));
            t.status = "failed"
            self._broadcast_state_update(t)
            t.last_llm_history = None
            t.agent_role_paused = None
            raise Exception(error_msg)

        t.human_directive = f"Your last answer was insufficient. Reason: {verify_task_result.reason}\nRe-evaluate and try again."
        logger.info(f"[{t.id}] Retrying execution. Feedback: {verify_task_result.reason[:50]}...")

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."

        vres = await self.verifier.run(
            user_prompt=ver_prompt,
            message_history=t.last_llm_history
        )
        is_paused, vout = await self._handle_agent_output(
            ctx=TaskContext(t.id, self.registry), agent_res=vres, expected_output_type=verification,
            agent_role_name="verifier"
        )
        if is_paused:
            return verification(reason="Verifier paused to ask question.", message_for_user="", score=0)

        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout