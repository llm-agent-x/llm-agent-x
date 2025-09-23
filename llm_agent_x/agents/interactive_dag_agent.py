import asyncio
import json
import logging
import os
import threading
import uuid
from collections import deque
from typing import List, Tuple, Union

import pika
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from llm_agent_x.agents.dag_agent import (
    DAGAgent, Task, TaskRegistry, TaskContext,
    ExecutionPlan, ProposedSubtask, verification, RetryDecision, UserQuestion
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


    def _broadcast_state_update(self, task: Task):
        """Publishes the full state of a task to RabbitMQ."""
        message = task.model_dump_json(exclude={'span', 'otel_context', 'last_llm_history'})
        self.publish_channel.basic_publish(
            exchange=self.STATE_UPDATES_EXCHANGE,
            routing_key='',
            body=message
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
                task.last_llm_history = task.last_llm_history
                task.agent_role_paused = task.agent_role_paused
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
            if task.status in ["complete", "failed", "running", "paused_by_human", "waiting_for_user_response", "planning", "proposing", "waiting_for_children"]:
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
                    non_executable_tasks = {t.id for t in self.registry.tasks.values() if t.status not in executable_statuses}

                    for tid in all_task_ids - non_executable_tasks:
                        task = self.registry.tasks[tid]
                        if any(self.registry.tasks.get(d, {}).status == "failed" for d in task.deps):
                            if task.status != "failed":
                                task.status, task.result = "failed", "Upstream dependency failed."
                                self._broadcast_state_update(task)

                    pending_tasks_for_scheduling = {t.id for t in self.registry.tasks.values() if t.status in executable_statuses}

                    ready_to_run_ids = set()
                    for tid in pending_tasks_for_scheduling:
                        task = self.registry.tasks[tid]
                        if task.status == "pending" and task.user_response and task.last_llm_history and task.agent_role_paused:
                            ready_to_run_ids.add(tid)
                        elif task.status == "pending" and \
                                all(self.registry.tasks.get(d, {}).status == "complete" for d in task.deps):
                            ready_to_run_ids.add(tid)
                        elif task.status == 'waiting_for_children' and \
                                all(self.registry.tasks[c].status in ['complete', 'failed'] for c in task.children):
                            if any(self.registry.tasks[c].status == 'failed' for c in task.children):
                                task.status, task.result = "failed", "A child task failed, cannot synthesize."
                                self._broadcast_state_update(task)
                            else:
                                ready_to_run_ids.add(tid)
                                task.status = "pending"
                                self._broadcast_state_update(task)


                    ready = [tid for tid in ready_to_run_ids if tid not in self.inflight]

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

                    if self.proposed_tasks_buffer:
                        global_resolver_ctx_id = "GLOBAL_RESOLVER"
                        if global_resolver_ctx_id not in self.registry.tasks:
                            self.registry.add_task(Task(id=global_resolver_ctx_id, desc="Internal task for global conflict resolution.", status="pending"))
                        global_resolver_task = self.registry.tasks[global_resolver_ctx_id]

                        if global_resolver_task.status != "waiting_for_user_response":
                            await self._run_global_resolution()
                        else:
                            logger.debug("Global resolver is waiting for user response, skipping buffer processing.")


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

            if t.status in ["paused_by_human", "waiting_for_user_response"]:
                logger.info(f"[{t.id}] Task is {t.status}, skipping execution for now.")
                return

            try:
                otel_ctx_token = otel_context.attach(trace.set_span_in_context(span))

                if t.user_response and t.last_llm_history and t.agent_role_paused:
                    await self._resume_agent_from_question(ctx)
                    if t.status in ["waiting_for_user_response", "failed", "paused_by_human"]:
                        return
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
                elif t.can_request_new_subtasks and t.status != 'proposing':
                    update_status_and_broadcast('proposing')
                    await self._run_adaptive_decomposition(ctx)
                    if t.status not in ["waiting_for_user_response", "paused_by_human"]:
                        update_status_and_broadcast("waiting_for_children")
                else:
                    update_status_and_broadcast('running')
                    await self._run_task_execution(ctx)

                if t.status not in ["waiting_for_children", "failed", "paused_by_human", "waiting_for_user_response"]:
                    update_status_and_broadcast("complete")
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(StatusCode.ERROR, str(e)))
                t.status = "failed"
                self._broadcast_state_update(t)
                raise
            finally:
                otel_context.detach(otel_ctx_token)
                span.set_attribute("dag.task.status", t.status)
                self._broadcast_state_update(t)


    async def _run_initial_planning(self, ctx: TaskContext):
        t = ctx.task
        logger.info(f"[{t.id}] Running initial planning for: {t.desc}")

        completed_tasks = [tk for tk in self.registry.tasks.values() if tk.status == "complete" and tk.id != t.id]
        context_str = "\n".join(f"- ID: {tk.id} Desc: {tk.desc}" for tk in completed_tasks)
        prompt = f"Objective: {t.desc}\n\nAvailable completed data sources:\n{context_str}"

        plan_res = await self.initial_planner.run(user_prompt=prompt)
        is_paused, initial_plan = await self._handle_agent_output(
            ctx=ctx, agent_res=plan_res, expected_output_type=ExecutionPlan, agent_role_name="initial_planner"
        )
        if is_paused: return

        await self._process_initial_planning_output(ctx, initial_plan)


    async def _run_adaptive_decomposition(self, ctx: TaskContext):
        t = ctx.task
        t.dep_results = {d: self.registry.tasks[d].result for d in t.deps}
        prompt = f"Task: {t.desc}\n\nResults from dependencies:\n{t.dep_results}"

        proposals_res = await self.adaptive_decomposer.run(user_prompt=prompt)
        is_paused, proposals = await self._handle_agent_output(
            ctx=ctx, agent_res=proposals_res, expected_output_type=List[ProposedSubtask], agent_role_name="adaptive_decomposer"
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

        while True:
            current_attempt = t.fix_attempts + t.grace_attempts + 1
            t.span.add_event(f"Execution Attempt", {"attempt": current_attempt})

            logger.info(f"[{t.id}] Attempt {current_attempt}: Calling executor LLM.")
            exec_res = await self.executor.run(user_prompt=prompt)
            is_paused, result = await self._handle_agent_output(
                ctx=ctx, agent_res=exec_res, expected_output_type=str, agent_role_name="executor"
            )
            if is_paused: return

            await self._process_executor_output_for_verification(ctx, result)
            if t.status in ["complete", "failed", "waiting_for_user_response", "paused_by_human"]:
                return
            break


    async def _verify_task(self, t: Task, candidate_result: str) -> verification:
        ver_prompt = f"Task: {t.desc}\nCandidate Result: {candidate_result}\n\nDoes the result fully and accurately complete the task? Be strict."

        vres = await self.verifier.run(user_prompt=ver_prompt)
        is_paused, vout = await self._handle_agent_output(
            ctx=TaskContext(t.id, self.registry), agent_res=vres, expected_output_type=verification, agent_role_name="verifier"
        )
        if is_paused:
            return verification(reason="Verifier paused to ask question.", message_for_user="", score=0)

        t.verification_scores.append(vout.score)
        t.span.set_attribute("verification.score", vout.score)
        t.span.set_attribute("verification.reason", vout.reason)
        logger.info(f"   > Verification for [{t.id}]: score={vout.score}, reason='{vout.reason[:40]}...'")
        return vout