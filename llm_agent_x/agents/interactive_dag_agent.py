import asyncio
import json
import logging
import os
import threading
import uuid
from collections import deque
from typing import List, Tuple

import pika
from opentelemetry import trace
from opentelemetry.trace import StatusCode

# --- IMPORTANT: Import from your existing project structure ---
from llm_agent_x.agents.dag_agent import (
    DAGAgent, Task, TaskRegistry, TaskContext,
    ExecutionPlan, ProposedSubtask, verification, RetryDecision
)

# --- Basic Setup ---
logger = logging.getLogger("InteractiveDAGAgent")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")


class InteractiveDAGAgent(DAGAgent):
    """
    An enhanced DAGAgent that listens for real-time human directives
    and broadcasts state changes via RabbitMQ.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.directives_queue = asyncio.Queue()
        self.rabbit_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))

        # Channel for publishing state updates
        self.publish_channel = self.rabbit_connection.channel()
        self.STATE_UPDATES_EXCHANGE = 'state_updates_exchange'
        self.publish_channel.exchange_declare(exchange=self.STATE_UPDATES_EXCHANGE, exchange_type='fanout')

        # Channel for consuming directives
        self.consume_channel = self.rabbit_connection.channel()
        self.DIRECTIVES_QUEUE = 'directives_queue'
        self.consume_channel.queue_declare(queue=self.DIRECTIVES_QUEUE, durable=True)
        self.consume_channel.basic_qos(prefetch_count=1)

    # NEW - CORRECT
    def _broadcast_state_update(self, task: Task):
        """Publishes the full state of a task to RabbitMQ."""
        # .model_dump_json() correctly handles converting sets and other complex types to a JSON string.
        message = task.model_dump_json(exclude={'span', 'otel_context'})
        self.publish_channel.basic_publish(
            exchange=self.STATE_UPDATES_EXCHANGE,
            routing_key='',
            body=message  # The message is now already a properly formatted JSON string
        )

    def _listen_for_directives(self):
        """Listens for messages from RabbitMQ and puts them in an async queue."""

        def callback(ch, method, properties, body):
            message = json.loads(body)
            logger.info(f"Received directive from RabbitMQ: {message}")
            self.directives_queue.put_nowait(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        self.consume_channel.basic_consume(queue=self.DIRECTIVES_QUEUE, on_message_callback=callback)
        logger.info("Starting to consume directives...")
        try:
            self.consume_channel.start_consuming()
        except Exception as e:
            logger.error(f"Directive consumer stopped: {e}")

    # --- NEW: Directive Handling ---
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

        if task_id in self.task_futures and not self.task_futures[task_id].done():
            self.task_futures[task_id].cancel()
            logger.info(f"Cancelled in-flight asyncio.Task for {task_id}")

        original_status = task.status
        if command == "PAUSE":
            task.status = "paused_by_human"
        elif command == "RESUME":
            task.status = "pending"
        elif command == "TERMINATE":
            task.status = "failed"
            task.result = f"Terminated by operator: {payload}"
        elif command == "REDIRECT":
            task.status = "pending";
            task.result = None;
            task.fix_attempts = 0;
            task.grace_attempts = 0;
            task.verification_scores = [];
            task.human_directive = payload
        elif command == "MANUAL_OVERRIDE":
            task.status = "complete";
            task.result = payload
            self._reset_task_and_dependents(task_id)

        if task.status != original_status or command == "MANUAL_OVERRIDE":
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
            if task.status not in ["complete", "failed", "running"]: continue
            task.status = "pending";
            task.result = None;
            task.fix_attempts = 0;
            task.grace_attempts = 0;
            task.verification_scores = [];
            task.human_directive = None
            self._broadcast_state_update(task)

    # --- OVERRIDDEN: The Main `run` loop ---
    async def run(self):
        consumer_thread = threading.Thread(target=self._listen_for_directives, daemon=True)
        consumer_thread.start()

        logger.info("Broadcasting initial state of all tasks (if any)...")
        for task in self.registry.tasks.values():
            self._broadcast_state_update(task)

        try:
            with self.tracer.start_as_current_span("InteractiveDAGAgent.run"):
                while True:  # Loop forever
                    # 1. Handle interrupts first
                    while not self.directives_queue.empty():
                        directive = await self.directives_queue.get()
                        await self._handle_directive(directive)

                    # 2. Scheduling logic (unchanged from before)
                    all_task_ids = set(self.registry.tasks.keys())
                    completed_or_failed = {t.id for t in self.registry.tasks.values() if
                                           t.status in ('complete', 'failed')}

                    for tid in all_task_ids - completed_or_failed:
                        task = self.registry.tasks[tid]
                        if any(self.registry.tasks.get(d, {}).status == "failed" for d in task.deps):
                            if task.status != "failed":
                                task.status, task.result = "failed", "Upstream dependency failed."
                                self._broadcast_state_update(task)

                    pending_tasks = all_task_ids - completed_or_failed
                    ready_to_run_ids = {
                        tid for tid in pending_tasks if self.registry.tasks[tid].status == "pending" and
                                                        all(d in self.registry.tasks and self.registry.tasks[
                                                            d].status == "complete" for d in
                                                            self.registry.tasks[tid].deps)
                    }

                    ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

                    # --- REMOVED THE BREAK CONDITION ---
                    # The agent now idles if there's no work, it does not stop.

                    for tid in ready:
                        self.inflight.add(tid)
                        self.task_futures[tid] = asyncio.create_task(self._run_taskflow(tid))

                    # Process completed futures
                    if self.task_futures:
                        done, _ = await asyncio.wait(list(self.task_futures.values()), timeout=0.1,
                                                     return_when=asyncio.FIRST_COMPLETED)
                        for fut in done:
                            tid = next((tid for tid, f in self.task_futures.items() if f == fut), None)
                            if tid:
                                self.inflight.discard(tid)
                                del self.task_futures[tid]
                                try:
                                    fut.result()
                                except asyncio.CancelledError:
                                    logger.info(f"Task [{tid}] was cancelled by operator.")
                                except Exception as e:
                                    logger.error(f"Task [{tid}] future failed: {e}")

                    await asyncio.sleep(0.2)  # Yield control to prevent tight-looping when idle
        finally:
            logger.info("Agent run loop is shutting down. Closing RabbitMQ connection.")
            self.rabbit_connection.close()

    # --- UNCHANGED METHODS FROM BASE DAGAgent (INCLUDED FOR COMPLETENESS) ---
    async def _run_taskflow(self, tid: str):
        ctx = TaskContext(tid, self.registry)
        t = ctx.task
        with self.tracer.start_as_current_span(f"Task: {t.desc[:50]}") as span:
            t.span = span
            span.set_attribute("dag.task.id", t.id)
            span.set_attribute("dag.task.status", t.status)
            try:
                original_status = t.status

                def update_status(new_status):
                    if t.status != new_status:
                        t.status = new_status
                        self._broadcast_state_update(t)

                if t.status == 'waiting_for_children':
                    update_status('running')
                    await self._run_task_execution(ctx)
                elif t.needs_planning and not t.already_planned:
                    update_status('planning')
                    await self._run_initial_planning(ctx)
                    t.already_planned = True
                    update_status("waiting_for_children" if t.children else "complete")
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    update_status('proposing')
                    await self._run_adaptive_decomposition(ctx)
                    update_status("waiting_for_children")
                else:
                    update_status('running')
                    await self._run_task_execution(ctx)

                if t.status not in ["waiting_for_children", "failed"]:
                    update_status("complete")
            except Exception as e:
                span.record_exception(e);
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed";
                raise
            finally:
                span.set_attribute("dag.task.status", t.status)
                self._broadcast_state_update(t)

    async def _run_initial_planning(self, ctx: TaskContext):
        """
        Calls the LLM to break down a high-level task into a graph of sub-tasks.
        """
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning for: {t.desc}")

        # Gather context of existing, completed tasks (like documents)
        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context_str = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt = f"Objective: {t.desc}\n\nAvailable completed data sources:\n{context_str}"

        # Call the planning agent
        plan_res = await self.initial_planner.run(user_prompt=prompt)
        self._add_llm_data_to_span(t.span, plan_res, t)
        initial_plan = plan_res.output

        # Optional: Cycle breaking logic (can be added if needed)
        plan = initial_plan

        if not plan or not plan.needs_subtasks or not plan.subtasks:
            logger.warning(f"[{t.id}] Planner decided no sub-tasks are needed. Task will proceed to execution.")
            # Set needs_planning to False so we don't try to plan it again
            t.needs_planning = False
            return

        logger.info(f"[{t.id}] Planner created {len(plan.subtasks)} sub-tasks.")
        local_to_global_id_map = {}
        for sub in plan.subtasks:
            new_task = Task(
                id=str(uuid.uuid4())[:8],
                desc=sub.desc,
                parent=t.id,
                can_request_new_subtasks=sub.can_request_new_subtasks
            )
            self.registry.add_task(new_task)
            t.children.append(new_task.id)
            self.registry.add_dependency(t.id, new_task.id)  # Parent depends on child completion
            local_to_global_id_map[sub.local_id] = new_task.id
            self._broadcast_state_update(new_task)  # Broadcast new child task

        # Now link the dependencies between the new sub-tasks
        for sub in plan.subtasks:
            new_global_id = local_to_global_id_map.get(sub.local_id)
            if not new_global_id: continue
            for dep in sub.deps:
                # Dependency could be another new sub-task or a pre-existing completed one
                dep_global_id = local_to_global_id_map.get(dep.local_id) or (
                    dep.local_id if dep.local_id in self.registry.tasks else None)
                if dep_global_id:
                    self.registry.add_dependency(new_global_id, dep_global_id)

        # Broadcast the parent task now that its children are added
        self._broadcast_state_update(t)

    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt = f"Task: {t.desc}\n\nResults from dependencies:\n{t.dep_results}"
        proposals_res = await self.adaptive_decomposer.run(user_prompt=prompt)
        self._add_llm_data_to_span(t.span, proposals_res, t)
        if proposals := proposals_res.output:
            logger.info(f"Task [{t.id}] proposing {len(proposals)} new sub-tasks.")
            for sub in proposals: self.proposed_tasks_buffer.append((sub, t.id))

    async def _run_task_execution(self, ctx: TaskContext):
        """
        Calls the LLM to execute a single task, using results from dependencies.
        """
        t = ctx.task
        logger.info(f"[{t.id}] Running task execution for: {t.desc}")

        # Gather results from completed children (for synthesis tasks) or dependencies
        child_results = {cid: self.registry.tasks[cid].result for cid in t.children if
                         self.registry.tasks[cid].status == 'complete'}
        dep_results = {did: self.registry.tasks[did].result for did in t.deps if
                       self.registry.tasks[did].status == 'complete' and did not in child_results}

        prompt = f"Your task is: {t.desc}\n"

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

        # The execution/retry loop
        while True:
            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})

            logger.info(f"[{t.id}] Attempt {current_attempt}: Calling executor LLM.")
            exec_res = await self.executor.run(user_prompt=prompt)
            self._add_llm_data_to_span(t.span, exec_res, t)
            result = exec_res.output

            # Verification step
            verify_task_result = await self._verify_task(t, result)
            if verify_task_result.get_successful():
                t.result = result
                logger.info(f"COMPLETED [{t.id}] with result: {result[:100]}...")
                return

            t.fix_attempts += 1
            if t.fix_attempts >= t.max_fix_attempts:
                # Add your grace attempt / retry analyst logic here if needed
                logger.error(f"[{t.id}] Exceeded max attempts.")
                raise Exception(f"Exceeded max attempts for task '{t.id}'")

            prompt += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\nYour last answer was insufficient. Reason: {verify_task_result.reason}\nPlease re-evaluate and provide a better answer."

    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."
        vres = await self.verifier.run(user_prompt=ver_prompt)
        self._add_llm_data_to_span(t.span, vres, t);
        vout = vres.output
        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score);
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout