import asyncio
import json # Added missing import
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Optional, List, Dict, Union
from black import Mode, format_str

# --- Pydantic-AI Imports ---
from pydantic_ai import Agent

# --- Original Imports (some will be replaced) ---
from llm_agent_x.backend.dot_tree import DotTree
from llm_agent_x.backend.exceptions import TaskFailedException
from opentelemetry import trace, context as otel_context
from pydantic import BaseModel, Field, validator, ValidationError
from llm_agent_x.backend.mergers.LLMMerger import MergeOptions, LLMMerger
from icecream import ic
from llm_agent_x.complexity_model import TaskEvaluation, evaluate_prompt
import logging
import tiktoken
from llm_agent_x.tools.summarize import summarize
from pydantic_ai.models import Model
from pydantic_ai.mcp import MCPServer

# Configure logging
logger = logging.getLogger(__name__)
# (Logger configuration remains unchanged)
handler = logging.FileHandler("llm_agent_x.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# --- All Pydantic Models, Configs, and helper functions remain unchanged ---
class TaskLimitConfig:
    @staticmethod
    def constant(max_tasks: int, max_depth: int) -> List[int]: return [max_tasks] * max_depth
    @staticmethod
    def array(task_limits: List[int]) -> List[int]: return task_limits
    @staticmethod
    def falloff(initial_tasks: int, max_depth: int, falloff_func: Callable[[int], int]) -> List[int]: return [falloff_func(i) for i in range(max_depth)]

class TaskLimit(BaseModel):
    limits: List[int]
    @validator("limits")
    def validate_limits(cls, v):
        if not all(isinstance(x, int) and x >= 0 for x in v): raise ValueError("All limits must be non-negative integers")
        return v
    @classmethod
    def from_constant(cls, max_tasks: int, max_depth: int): return cls(limits=TaskLimitConfig.constant(max_tasks, max_depth))
    @classmethod
    def from_array(cls, task_limits: List[int]): return cls(limits=TaskLimitConfig.array(task_limits))
    @classmethod
    def from_falloff(cls, initial_tasks: int, max_depth: int, falloff_func: Callable[[int], int]): return cls(limits=TaskLimitConfig.falloff(initial_tasks, max_depth, falloff_func))

class LLMTaskObject(BaseModel):
    task: str
    type: Literal["research", "search", "basic", "text/reasoning"]
    subtasks: int = 0
    allow_search: bool = True
    allow_tools: bool = True
    depends_on: List[str] = Field([], description="A list of task UUIDs or 1-based indices that this task depends on.")

class TaskObject(LLMTaskObject):
    uuid: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    @validator("uuid", pre=True)
    def validate_uuid(cls, v):
        if isinstance(v, int): return str(v)
        return v

class task(TaskObject): pass
class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best) evaluating the response's overall quality.")
    def get_successful(self): return self.score > 5

class SplitTask(BaseModel):
    needs_subtasks: bool
    subtasks: list[LLMTaskObject]
    evaluation: Optional[TaskEvaluation] = None
    def __bool__(self): return self.needs_subtasks

class TaskContext(BaseModel):
    task: str
    result: Optional[str] = None
    siblings: List["TaskContext"] = []
    parent_context: Optional["TaskContext"] = None
    dependency_results: Dict[str, str] = {}
    class Config: arbitrary_types_allowed = True

class RecursiveAgentOptions(BaseModel):
    task_limits: TaskLimit
    search_tool: Any = None
    pre_task_executed: Any = None
    on_task_executed: Any = None
    on_tool_call_executed: Any = None
    task_tree: list[Any] = []
    llm: Any = None
    tools: list = []
    mcp_servers: List[MCPServer]
    allow_search: bool = True
    allow_tools: bool = False
    tools_dict: dict = {}
    similarity_threshold: float = 0.8
    merger: Any = LLMMerger
    align_summaries: bool = True
    token_counter: Optional[Callable[[str], int]] = None
    summary_sentences_factor: int = 10
    task_registry: Dict[str, Any] = {}
    max_fix_attempts: int = 2
    class Config: arbitrary_types_allowed = True

def calculate_raw_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()

def _serialize_lc_messages_for_preview(messages: List[Dict[str, Any]], max_len: int = 500) -> str:
    if not messages: return "[]"
    content_parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content_str = str(msg.get("content", ""))
        content_parts.append(f"{role}: {content_str}")
    full_str = "\n".join(content_parts)
    if len(full_str) > max_len:
        return full_str[: max_len - 3] + "..."
    return full_str

def _build_history(system_prompt: str, human_prompt: str, conversation: List[Dict] = None) -> List[Dict[str, Any]]:
    history = [{"role": "system", "content": system_prompt}]
    if conversation:
        history.extend(conversation)
    history.append({"role": "user", "content": human_prompt})
    return history

class RecursiveAgent:
    def __init__(
        self,
        task: Any,
        u_inst: str,
        tracer: Optional[trace.Tracer] = None,
        tracer_span: Optional[trace.Span] = None,
        agent_options: Optional[RecursiveAgentOptions] = None,
        allow_subtasks: bool = True,
        current_layer: int = 0,
        parent: Optional["RecursiveAgent"] = None,
        context: Optional[TaskContext] = None,
        siblings: Optional[List["RecursiveAgent"]] = None,
        task_type_override: Optional[str] = None,
        max_fix_attempts: int = 2,
    ):
        if agent_options is None:
            # Note: self.logger is not available before it's defined.
            # Use the global logger for this initial message.
            logger.info("No agent_options provided, using default configuration.")
            agent_options = RecursiveAgentOptions(
                task_limits=TaskLimit.from_constant(max_tasks=3, max_depth=2)
            )
        self.options = agent_options
        if isinstance(task, TaskObject): self.task_obj = task
        else: self.task_obj = TaskObject(task=str(task), type=task_type_override or "research")
        self.task = self.task_obj.task
        self.uuid = self.task_obj.uuid
        self.task_type = task_type_override or self.task_obj.type
        self.logger = logging.getLogger(f"{__name__}.RecursiveAgent.{self.uuid}")
        self.logger.info(f"Initializing RecursiveAgent for task: '{self.task}' (Type: {self.task_type}) at layer {current_layer} with UUID: {self.uuid}")
        if self.options.task_registry is not None: self.options.task_registry[self.uuid] = self
        self.u_inst = u_inst
        self.tracer = tracer if tracer else trace.get_tracer(__name__)
        self.tracer_span = tracer_span
        self.allow_subtasks = allow_subtasks
        self.llm: DotTree = self.options.llm
        self.tools = self.options.tools
        self.current_layer = current_layer
        self.parent = parent
        self.siblings = siblings or []
        self.context = context or TaskContext(task=self.task)
        self.result: Optional[str] = None
        self.status: str = "pending"
        self.current_span: Optional[trace.Span] = None
        self.fix_attempt_count: int = 0
        self.max_fix_attempts = (max_fix_attempts or agent_options.max_fix_attempts if agent_options.max_fix_attempts is not None else 2)

    def _get_token_count(self, text: str) -> int:
        if self.options.token_counter:
            try: return self.options.token_counter(text)
            except Exception as e:
                self.logger.warning(f"Token counter failed for text: '{text[:50]}...': {e}", exc_info=False)
                return 0
        return 0

    def _build_context_information(self) -> dict:
        ancestor_chain_contexts_data = []
        current_ancestor_node = self.context.parent_context
        while current_ancestor_node:
            if current_ancestor_node.result is not None: ancestor_chain_contexts_data.append({"task": current_ancestor_node.task, "result": current_ancestor_node.result, "relation": "ancestor"})
            current_ancestor_node = current_ancestor_node.parent_context
        ancestor_chain_contexts_data.reverse()
        broader_family_contexts_data = []; tasks_to_exclude_from_broader = {ctx_data["task"] for ctx_data in ancestor_chain_contexts_data}; tasks_to_exclude_from_broader.add(self.context.task)
        ancestor_depth = 0; temp_node_for_sibling_scan = self.context
        while temp_node_for_sibling_scan:
            for sibling_of_temp_node in temp_node_for_sibling_scan.siblings:
                if sibling_of_temp_node.result is not None and sibling_of_temp_node.task not in tasks_to_exclude_from_broader:
                    relation = "direct_sibling" if ancestor_depth == 0 else f"ancestor_level_{ancestor_depth}_sibling"
                    broader_family_contexts_data.append({"task": sibling_of_temp_node.task, "result": sibling_of_temp_node.result, "relation": relation})
                    tasks_to_exclude_from_broader.add(sibling_of_temp_node.task)
            temp_node_for_sibling_scan = temp_node_for_sibling_scan.parent_context; ancestor_depth += 1
        dependency_contexts_data = []
        if self.context.dependency_results:
            for dep_key, dep_result in self.context.dependency_results.items():
                dep_agent = self.options.task_registry.get(dep_key)
                dep_task_desc = dep_agent.task if dep_agent else "Unknown Task"
                dependency_contexts_data.append({"task": dep_task_desc, "result": dep_result, "relation": "dependency", "uuid": dep_key})
        return {"ancestor_chain_contexts": ancestor_chain_contexts_data, "broader_family_contexts": broader_family_contexts_data, "dependency_contexts": dependency_contexts_data}

    def _format_history_parts(self, context_info: dict, purpose: str, subtask_results_map: Optional[Dict[str, str]] = None) -> List[str]:
        history = []
        if context_info.get("dependency_contexts"):
            history.append(f"Context from completed dependency tasks (for {purpose}):")
            for ctx in context_info["dependency_contexts"]: history.append(f"- Dependency Task (UUID: {ctx['uuid']}): {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...")
        if context_info.get("ancestor_chain_contexts"):
            history.append(f"\nContext from direct ancestor tasks (for {purpose}):")
            for ctx in context_info["ancestor_chain_contexts"]: history.append(f"- Ancestor Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...")
        if context_info.get("broader_family_contexts"):
            history.append(f"\nContext from other related tasks in the hierarchy (for {purpose}):")
            for ctx in context_info["broader_family_contexts"]: history.append(f"- {ctx['relation'].replace('_', ' ').capitalize()} Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...")
        if purpose == "verification" and subtask_results_map:
            history.append("\nThe current main task involved these subtasks and their results (for verification):")
            for sub_task, sub_result in subtask_results_map.items(): history.append(f"- Subtask (of current task): {sub_task}\n  - Result: {str(sub_result)[:200]}...")
        return history

    def _build_task_split_history(self) -> str:
        return "\n".join(self._format_history_parts(self._build_context_information(), "splitting"))
    def _build_task_verify_history(self, subtask_results_map: Optional[Dict[str, str]] = None) -> str:
        return "\n".join(self._format_history_parts(self._build_context_information(), "verification", subtask_results_map))

    async def run(self):
        self.status = "running"
        self.logger.info(f"Attempting to start run for task: '{self.task}' (UUID: {self.uuid}, Status: {self.status})")
        parent_otel_ctx = otel_context.get_current()
        if self.tracer_span: parent_otel_ctx = trace.set_span_in_context(self.tracer_span)
        with self.tracer.start_as_current_span(
            f"RecursiveAgent Task: {self.task[:50]}...", context=parent_otel_ctx,
            attributes={"agent.task.full": self.task, "agent.uuid": self.uuid, "agent.layer": self.current_layer, "agent.initial_status": self.status, "agent.allow_subtasks_flag": self.allow_subtasks}
        ) as span:
            self.current_span = span
            span.add_event("Agent Run Start", attributes={"task": self.task, "user_instructions_preview": str(self.u_inst)[:200], "current_layer": self.current_layer})
            try:
                result = await self._run()
                span.set_attribute("agent.final_status", self.status)
                span.add_event("Agent Run End", attributes={"result_preview": str(result)[:200], "final_status": self.status})
                self.logger.info(f"Run finished for task: '{self.task}'. Result: {str(result)[:100]}... Status: {self.status}")
                span.set_attribute("result", result);
                return result
            except Exception as e:
                self.logger.error(f"Critical error in agent run for task '{self.task}': {e}", exc_info=True)
                if span:
                    span.record_exception(e); self.status = "failed_critically"; span.set_attribute("agent.final_status", self.status)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
                raise TaskFailedException(f"Agent run for '{self.task}' failed critically: {e}") from e

    async def _run(self) -> str:
        span = self.current_span
        if not span: self.logger.warning("_run called without an active self.current_span. Tracing will be limited.")
        if span: span.add_event("Internal Execution Start", {"task": self.task})
        if self.options.pre_task_executed:
            if span: span.add_event("Pre-Task Callback Executing")
            self.options.pre_task_executed(task=self.task, uuid=self.uuid, parent_agent_uuid=(self.parent.uuid if self.parent else None))
        max_subtasks_for_this_layer = self._get_max_subtasks()
        if max_subtasks_for_this_layer == 0 or not self.allow_subtasks:
            if span: span.add_event("Executing as Single Task: Subtasks disabled.")
            self.result = await self._execute_and_verify_single_task()
            return self.result
        if self.parent:
            similarity = calculate_raw_similarity(self.task, self.parent.task)
            if span: span.add_event("Parent Similarity Check", {"similarity_score": similarity, "threshold": self.options.similarity_threshold})
            if similarity >= self.options.similarity_threshold:
                if span: span.add_event("Executing as Single Task: High Parent Similarity", {"similarity": similarity})
                self.result = await self._execute_and_verify_single_task()
                return self.result
        split_task_result = await self._split_task()
        ic(split_task_result)
        if span: span.add_event("Task Splitting Outcome", {"needs_subtasks": split_task_result.needs_subtasks, "count": len(split_task_result.subtasks)})
        if not split_task_result or not split_task_result.needs_subtasks:
            if span: span.add_event("Executing as Single Task: No subtasks needed.")
            self.result = await self._execute_and_verify_single_task()
            return self.result
        limited_subtasks = split_task_result.subtasks[:max_subtasks_for_this_layer]
        if span: span.add_event("Subtasks Limited", {"original": len(split_task_result.subtasks), "limited": len(limited_subtasks)})
        child_agents_in_order: List[RecursiveAgent] = []
        child_contexts: List[TaskContext] = []
        for llm_subtask_obj in limited_subtasks:
            subtask_obj = TaskObject(**llm_subtask_obj.model_dump())
            child_context = TaskContext(task=subtask_obj.task, parent_context=self.context)
            child_agent = RecursiveAgent(task=subtask_obj, u_inst=self.u_inst, tracer=self.tracer, tracer_span=span, agent_options=self.options, allow_subtasks=(self.current_layer + 1 < len(self.options.task_limits.limits)), current_layer=self.current_layer + 1, parent=self, context=child_context)
            child_agents_in_order.append(child_agent); child_contexts.append(child_context)
        index_to_uuid_map = {str(i + 1): agent.uuid for i, agent in enumerate(child_agents_in_order)}
        for agent in child_agents_in_order:
            if agent.task_obj.depends_on: agent.task_obj.depends_on = [index_to_uuid_map.get(dep, dep) for dep in agent.task_obj.depends_on]
        child_agents: Dict[str, RecursiveAgent] = {agent.uuid: agent for agent in child_agents_in_order}
        for agent in child_agents.values(): agent.context.siblings = [ctx for ctx in child_contexts if ctx.task != agent.task]
        completed_tasks: Dict[str, str] = {}; pending_agents: Dict[str, RecursiveAgent] = child_agents.copy()
        loop_guard = 0; max_loops = len(pending_agents) + 2
        while pending_agents and loop_guard < max_loops:
            loop_guard += 1; runnable_agents: List[RecursiveAgent] = []
            for agent_uuid, agent in pending_agents.items():
                if all(dep_uuid in completed_tasks for dep_uuid in agent.task_obj.depends_on): runnable_agents.append(agent)
            if not runnable_agents:
                for agent_uuid, agent in pending_agents.items():
                    dependencies = agent.task_obj.depends_on; non_sibling_deps_met = True
                    for dep_uuid in dependencies:
                        if dep_uuid not in child_agents:
                            dep_agent = self.options.task_registry.get(dep_uuid)
                            if not dep_agent or dep_agent.status != "succeeded": non_sibling_deps_met = False; break
                            else:
                                if dep_uuid not in completed_tasks: completed_tasks[dep_uuid] = dep_agent.result
                    if non_sibling_deps_met and all(dep_uuid in completed_tasks for dep_uuid in dependencies): runnable_agents.append(agent)
            if not runnable_agents and pending_agents:
                error_msg = f"Circular or unresolved dependency. Pending: {[a.task for a in pending_agents.values()]}"
                self.logger.error(error_msg);
                if span: span.add_event("Dependency Error", {"details": error_msg})
                raise TaskFailedException(error_msg)
            for agent in runnable_agents:
                if agent.uuid not in pending_agents: continue
                if span: span.add_event(f"Dependency Met: Running Task", {"child_task": agent.task, "uuid": agent.uuid})
                agent.context.dependency_results = {dep_uuid: completed_tasks[dep_uuid] for dep_uuid in agent.task_obj.depends_on if dep_uuid in completed_tasks}
                result = await agent.run()
                completed_tasks[agent.uuid] = result if result is not None else "No result."
                del pending_agents[agent.uuid]
                if span: span.add_event(f"Task Completed", {"child_task": agent.task, "uuid": agent.uuid, "result_preview": str(result)[:100]})
        if pending_agents: self.logger.warning(f"Loop finished with pending agents: {list(pending_agents.keys())}")
        subtask_tasks = [child_agents[uuid].task for uuid in completed_tasks if uuid in child_agents]
        subtask_results = [result for uuid, result in completed_tasks.items() if uuid in child_agents]
        self.result = await self._summarize_subtask_results(subtask_tasks, subtask_results)
        if span: span.set_attribute("result", self.result)
        self.context.result = self.result
        subtask_results_map = {child_agents[uuid].task: result for uuid, result in completed_tasks.items() if uuid in child_agents}
        try:
            await self.verify_result(subtask_results_map)
        except TaskFailedException:
            if span: span.add_event("Verification Failed, Attempting Fix")
            await self._fix(subtask_results_map)
        if self.options.on_task_executed: self.options.on_task_executed(self.task, self.uuid, self.result, self.parent.uuid if self.parent else None)
        return self.result

    async def _execute_and_verify_single_task(self) -> str:
        self.result = await self._run_single_task()
        self.context.result = self.result
        try:
            await self.verify_result(None)
        except TaskFailedException:
            if self.current_span: self.current_span.add_event("Single Task Verification Failed, Attempting Fix")
            await self._fix(None)
        if self.options.on_task_executed: self.options.on_task_executed(self.task, self.uuid, self.result, self.parent.uuid if self.parent else None)
        self.current_span.set_attribute("result", self.result)
        return self.result

    async def _run_single_task(self) -> str:
        """
        FIXED: This method was simplified to perform a single, complete agent run.
        The original implementation contained a redundant `while` loop with an internal
        verification step that interfered with the main verification-and-fix mechanism.
        The `pydantic-ai` Agent's `.run()` method handles the entire multi-step tool
        interaction internally, so a single call is sufficient.
        """
        agent_span = self.current_span
        parent_context = trace.set_span_in_context(agent_span) if agent_span else otel_context.get_current()
        with self.tracer.start_as_current_span("Run Single Task Operation", context=parent_context) as single_task_span:
            context_info = self._build_context_information()
            full_context_str = "\n".join(self._format_history_parts(context_info, "single task execution"))
            current_task_type = getattr(self, "task_type", "research")
            if current_task_type in ["basic", "task"]:
                system_prompt_content = (f"You are a helpful assistant. Directly execute or answer the following. Provide a direct, concise answer or the output of any tools used. Avoid narrative. Current Task: {self.task}\n\nRelevant history:\n{full_context_str}")
            else:
                system_prompt_content = (f"Your task is to answer the following question, using tools. Make sure to include citations [1] and a citations section at the end.\n\nRelevant history:\n{full_context_str}")
            
            human_message_content = self.task
            if self.u_inst: 
                human_message_content += f"\n\nFollow these specific instructions: {self.u_inst}"
            human_message_content += "\n\nApply the distributive property to any tool calls (e.g., make 3 separate search calls for 3 topics)."
            
            tool_agent = Agent(model=self.llm, system_prompt=system_prompt_content, output_type=str, tools=self.tools, mcp_servers=self.options.mcp_servers)
            
            single_task_span.add_event("Executing Pydantic-AI agent")
            ic(self.tools)
            ic(human_message_content)

            # A single call to .run() is sufficient. The pydantic-ai library handles
            # the internal loop of calling tools and re-prompting the LLM.
            async with tool_agent.run_mcp_servers():
                response = await tool_agent.run(user_prompt=human_message_content)

            print(format_str(str(response.all_messages()), mode = Mode()))
            
            ic(response.output)
            final_result_content = response.output or "No result."
            ic(final_result_content)
            
            return str(final_result_content)

    async def _split_task(self) -> SplitTask:
        agent_span = self.current_span; parent_context = trace.set_span_in_context(agent_span) if agent_span else otel_context.get_current()
        with self.tracer.start_as_current_span("Split Task Operation", context=parent_context) as split_span:
            task_history = self._build_task_split_history()
            max_subtasks = self._get_max_subtasks()
            ancestor_uuids = set(); current_agent_for_traversal = self
            while current_agent_for_traversal: ancestor_uuids.add(current_agent_for_traversal.uuid); current_agent_for_traversal = current_agent_for_traversal.parent
            valid_deps = [agent for uuid, agent in self.options.task_registry.items() if uuid not in ancestor_uuids and agent.status == "succeeded"]
            existing_tasks_str = "\n".join([f'- Task: "{a.task}"\n  UUID: {a.uuid}' for a in sorted(valid_deps, key=lambda a: a.task)]) or "No other tasks can be depended on."
            import inspect
            tools_docs = "\n".join([f"{t.name}: {getattr(t, 'description', '') or inspect.cleandoc(t.__doc__)}" for t in self.options.tools if t.__doc__ or getattr(t, 'description', '')])
            system_msg_content = (f"Split the task into up to {max_subtasks} subtasks if complex. Use `depends_on` with UUIDs for existing tasks or 1-based indices for new tasks.\n\n=== COMPLETED TASKS (for dependency):\n{existing_tasks_str}\n\n=== HISTORY:\n{task_history}\n\n=== TASK TO SPLIT:\n'{self.task}'\n\n=== AVAILABLE TOOLS:\n{tools_docs}\n\nStrictly output JSON matching the schema. If simple, set `needs_subtasks` to false.")
            if self.u_inst: system_msg_content += f"\nUser instructions:\n{self.u_inst}"
            evaluation = evaluate_prompt(f"Prompt: {self.task}")
            if evaluation.prompt_complexity_score[0] < 0.1 and evaluation.domain_knowledge[0] > 0.8: return SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)
            split_agent = Agent(model=self.llm, system_prompt=system_msg_content, output_type=SplitTask)
            try:
                response = await split_agent.run(user_prompt=self.task)
                split_task_result = response.output
                split_task_result.evaluation = evaluation
            except (ValidationError, Exception) as e:
                self.logger.error(f"Error parsing LLM JSON for splitting: {e}.", exc_info=True)
                split_span.record_exception(e)
                split_task_result = SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)
            if split_task_result.subtasks:
                split_task_result.subtasks = split_task_result.subtasks[:max_subtasks]
                split_task_result.needs_subtasks = bool(split_task_result.subtasks)
            return split_task_result

    async def _verify_result_internal(self, subtask_results_map: Optional[Dict[str, str]] = None) -> bool:
        agent_span = self.current_span
        parent_context = trace.set_span_in_context(agent_span) if agent_span else otel_context.get_current()
        with self.tracer.start_as_current_span("Verify Result Operation", context=parent_context) as verify_span:
            if self.result is None or not self.result.strip():
                verify_span.add_event("Verification Failed: No result provided.")
                return False

            task_history = self._build_task_verify_history(subtask_results_map)

            system_msg = (
                "You are a strict quality assurance verifier. Your job is to check if the provided 'Result' accurately and completely answers the 'Task', considering the history and user instructions. "
                "Score the result based on how well it answers the task, taking into account accuracy, completeness, relevance, adherence to instructions, and clarity. "
                f"Output JSON matching the '{verification.__name__}' schema."
            )
            ic("-"*100)
            ic(self.task)
            ic(self.result)
            ic(self.u_inst)
            ic("-"*85)
            human_msg = (
                f"Task:\n'''{self.task}'''\n\n"
                f"Result:\n'''{self.result}'''\n\n"
                f"User Instructions:\n'''{self.u_inst or 'None'}'''\n\n"
                f"History:\n'''{task_history or 'None'}'''\n\n"
                "Based on these criteria, was the task successfully completed?"
            )

            ic(human_msg)
            verify_agent = Agent(model=self.llm, system_prompt=system_msg, output_type=verification)
            try:
                response = await verify_agent.run(user_prompt=human_msg)
                verification_output = response.output
                ic(verification_output)
                ic("-" * 50)
                if not verification_output.get_successful():
                    self.logger.warning(f"Verification failed for task '{self.task}'. Reason: {verification_output.reason}")
                return verification_output.get_successful()
            except (ValidationError, Exception) as e:
                self.logger.error(f"Error parsing verification JSON for task '{self.task}': {e}", exc_info=True)
                verify_span.record_exception(e)
                return False

    async def verify_result(self, subtask_results_map: Optional[Dict[str, str]] = None):
        """
        FIXED: The logic for handling a failed verification was corrected.
        It now correctly sets the agent status to 'failed_verification' and, crucially,
        raises a `TaskFailedException`. This exception is essential for triggering the
        `_fix` method in the calling `_execute_and_verify_single_task` function.
        """
        successful = await self._verify_result_internal(subtask_results_map)
        if successful:
            self.status = "succeeded"
            if self.current_span: self.current_span.add_event("Verification Passed")
        else:
            self.status = "failed_verification"
            ic("Verification failed")
            if self.current_span: self.current_span.add_event("Verification Failed")
            # This exception is critical to trigger the fix/retry mechanism.
            raise TaskFailedException(f"Task '{self.task}' failed verification.")

    async def _fix(self, failed_subtask_results_map: Optional[Dict[str, str]]):
        agent_span = self.current_span
        parent_context_for_fix = trace.set_span_in_context(agent_span) if agent_span else otel_context.get_current()

        with self.tracer.start_as_current_span("Fix Task Operation", context=parent_context_for_fix) as fix_span:
            self.fix_attempt_count += 1
            self.logger.info(f"Attempting fix {self.fix_attempt_count}/{self.max_fix_attempts} for task: '{self.task}'")

            if self.fix_attempt_count > self.max_fix_attempts:
                self.status = "failed"
                error_msg = f"Fix attempt for '{self.task}' failed: max attempts ({self.max_fix_attempts}) reached."
                self.logger.error(error_msg)
                fix_span.add_event("Fix Attempt Skipped: Max attempts reached.", {"max_attempts": self.max_fix_attempts})
                raise TaskFailedException(error_msg)

            fix_instructions = (
                "A previous attempt to complete this task was unsatisfactory and failed verification. "
                "Your objective is to retry and provide a corrected, complete, and high-quality solution to the original task. "
                "Focus strictly on the original goal and ignore the previous failed attempt."
            )

            original_u_inst = self.u_inst
            self.u_inst = f"{fix_instructions}\n\nOriginal User Instructions: {original_u_inst or 'None'}"

            fix_span.add_event("Retrying task execution with fix instructions.")

            try:
                # Re-run the single task execution and verification within the same agent instance
                # This call will attempt execution and then verification. If the new verification
                # also fails, it will raise an exception that is caught below.
                self.result = await self._execute_and_verify_single_task()
                # If it gets here without raising an exception, it succeeded.
                self.status = "succeeded"
                fix_span.add_event("Fix Attempt Succeeded and Verified", {"final_status": self.status})
            except TaskFailedException as e_fix_verify:
                self.status = "failed"
                fix_span.record_exception(e_fix_verify, attributes={"reason": "Re-verification failed"})
                raise TaskFailedException(f"Fix attempt for '{self.task}' ultimately failed after re-verification.") from e_fix_verify
            except Exception as e_fix_run:
                self.status = "failed"
                fix_span.record_exception(e_fix_run, attributes={"reason": "Fixer agent run error"})
                raise TaskFailedException(f"Fixer agent for '{self.task}' run failed.") from e_fix_run
            finally:
                # Restore original instructions to prevent contamination on subsequent calls
                self.u_inst = original_u_inst


    def _get_max_subtasks(self) -> int:
        if self.current_layer >= len(self.options.task_limits.limits): return 0
        return self.options.task_limits.limits[self.current_layer]

    async def _summarize_subtask_results(self, tasks: List[str], subtask_results: List[str]) -> str:
        current_task_type = getattr(self, "task_type", "research")
        if current_task_type in ["basic", "task"]:
            if not subtask_results: return "No subtask results to report."
            status_update_parts = [f"Status update for task: {self.task}"]
            if not tasks and len(subtask_results) == 1:
                status_update_parts.append("Result:"); status_update_parts.append(str(subtask_results[0]))
            else:
                for i, (task_item, result_item) in enumerate(zip(tasks, subtask_results)):
                    status_update_parts.append(f"Sub-action {i+1}: {task_item}"); status_update_parts.append(f"  Result: {str(result_item)}")
            return "\n".join(status_update_parts)

        agent_span = self.current_span
        parent_context = trace.set_span_in_context(agent_span) if agent_span else otel_context.get_current()
        with self.tracer.start_as_current_span("Summarize Subtasks Operation", context=parent_context) as summary_span:
            if not subtask_results: return "No subtask results to summarize."
            documents_to_merge = [f"SUBTASK QUESTION: {q}\n\nSUBTASK ANSWER:\n{a}" for q, a in zip(tasks, subtask_results) if a]
            if not documents_to_merge: return "All subtasks yielded empty results."

            llm_for_merge_str = self.llm
            llm_for_merge = self.llm

            merged_content_str = ""
            if not llm_for_merge:
                self.logger.warning("LLM for merging not available. Using simple join.")
                merged_content_str = "\n\n---\n\n".join(documents_to_merge)
            else:
                try:
                    merge_options = MergeOptions(llm=llm_for_merge, context_window=15000)
                    merger = self.options.merger(merge_options)
                    # Run the synchronous merge_documents in a separate thread
                    merged_content_str = await asyncio.to_thread(merger.merge_documents, documents_to_merge)
                except Exception as e_merge:
                    self.logger.warning(f"LLMMerger failed: {e_merge}. Using simple join.", exc_info=True)
                    merged_content_str = "\n\n---\n\n".join(documents_to_merge)

            final_summary = merged_content_str
            if self.options.align_summaries:
                alignment_prompt = (f"Information from subtasks:\n\n{merged_content_str[:10000]}\n\nCompile a comprehensive report answering: '{self.task}'.\nUser instructions: {self.u_inst or 'None'}")
                align_agent = Agent(model=self.llm, system_prompt="You are a report-writing assistant.", output_type=str)
                response = await align_agent.run(user_prompt=alignment_prompt)
                final_summary = response.output
            return final_summary