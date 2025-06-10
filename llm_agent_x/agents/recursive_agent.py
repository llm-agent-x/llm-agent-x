import asyncio
import json
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Optional, List, Dict, Union
from llm_agent_x.backend.dot_tree import DotTree
from llm_agent_x.backend.exceptions import TaskFailedException
from opentelemetry import trace, context as otel_context
from pydantic import BaseModel, Field, validator, ValidationError
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables.base import RunnableConfig
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import (
    OutputParserException,
)
from llm_agent_x.backend.mergers.LLMMerger import MergeOptions, LLMMerger
from icecream import ic
from llm_agent_x.complexity_model import TaskEvaluation, evaluate_prompt
import logging
import tiktoken
from llm_agent_x.tools.summarize import summarize
from langchain_core.runnables.base import Runnable
from langchain_core.tools.structured import StructuredTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set logger to write to file instead of to stdout
handler = logging.FileHandler("llm_agent_x.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class TaskLimitConfig:
    @staticmethod
    def constant(max_tasks: int, max_depth: int) -> List[int]:
        return [max_tasks] * max_depth

    @staticmethod
    def array(task_limits: List[int]) -> List[int]:
        return task_limits

    @staticmethod
    def falloff(
        initial_tasks: int, max_depth: int, falloff_func: Callable[[int], int]
    ) -> List[int]:
        return [falloff_func(i) for i in range(max_depth)]


class TaskLimit(BaseModel):
    limits: List[int]

    @validator("limits")
    def validate_limits(cls, v):
        if not all(isinstance(x, int) and x >= 0 for x in v):
            raise ValueError("All limits must be non-negative integers")
        return v

    @classmethod
    def from_constant(cls, max_tasks: int, max_depth: int):
        return cls(limits=TaskLimitConfig.constant(max_tasks, max_depth))

    @classmethod
    def from_array(cls, task_limits: List[int]):
        return cls(limits=TaskLimitConfig.array(task_limits))

    @classmethod
    def from_falloff(
        cls, initial_tasks: int, max_depth: int, falloff_func: Callable[[int], int]
    ):
        return cls(
            limits=TaskLimitConfig.falloff(initial_tasks, max_depth, falloff_func)
        )


# --- MODIFICATION: Updated TaskObject and introduced LLMTaskObject ---
class LLMTaskObject(BaseModel):
    """Pydantic model for the task object structure expected from the LLM."""

    task: str
    type: Literal["research", "search", "basic", "text/reasoning"]
    subtasks: int = 0
    allow_search: bool = True
    allow_tools: bool = True
    depends_on: List[str] = Field(
        [], description="A list of task UUIDs or 1-based indices that this task depends on."
    )


class TaskObject(LLMTaskObject):
    """The internal representation of a task, with a system-assigned UUID."""

    uuid: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))

    @validator("uuid", pre=True)
    def validate_uuid(cls, v):
        if isinstance(v, int):
            return str(v)
        return v


class task(TaskObject):  # pylint: disable=invalid-name
    """Alias for TaskObject for convenience."""

    pass


class verification(BaseModel):  # pylint: disable=invalid-name
    successful: bool


class SplitTask(BaseModel):
    needs_subtasks: bool
    # --- MODIFICATION: The list now contains LLMTaskObjects from the LLM's perspective ---
    subtasks: list[LLMTaskObject]
    evaluation: Optional[TaskEvaluation] = None

    def __bool__(self):
        return self.needs_subtasks


class TaskContext(BaseModel):
    task: str
    result: Optional[str] = None
    siblings: List["TaskContext"] = []
    parent_context: Optional["TaskContext"] = None
    # --- MODIFICATION: Added to hold results from dependent tasks ---
    dependency_results: Dict[str, str] = {}

    class Config:
        arbitrary_types_allowed = True


class RecursiveAgentOptions(BaseModel):
    task_limits: TaskLimit
    search_tool: Any = None
    pre_task_executed: Any = None
    on_task_executed: Any = None
    on_tool_call_executed: Any = None
    task_tree: list[Any] = []
    llm: Any = None
    tools: list = []
    allow_search: bool = True
    allow_tools: bool = False
    tools_dict: dict = {}
    similarity_threshold: float = 0.8
    merger: Any = LLMMerger
    align_summaries: bool = True
    token_counter: Optional[Callable[[str], int]] = None
    summary_sentences_factor: int = 10  # Added for _summarize_subtask_results
    # --- MODIFICATION: Added central task registry ---
    task_registry: Dict[str, Any] = {}
    max_fix_attempts: int = 2

    class Config:
        arbitrary_types_allowed = True


def calculate_raw_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()


def _serialize_lc_messages_for_preview(
    messages: List[BaseMessage], max_len: int = 500
) -> str:
    # (No changes in this helper function)
    if not messages:
        return "[]"
    content_parts = []
    for msg in messages:
        role = msg.type.upper()
        content_str = str(msg.content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            content_str += f" (Tool Calls: {len(msg.tool_calls)})"
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            content_str += f" (Tool Call ID: {msg.tool_call_id})"
        content_parts.append(f"{role}: {content_str}")
    full_str = "\n".join(content_parts)
    if len(full_str) > max_len:
        return full_str[: max_len - 3] + "..."
    return full_str


class RecursiveAgent:
    def __init__(
        self,
        # --- MODIFICATION: Task can be TaskObject, which now includes UUID and depends_on ---
        task: Any,
        u_inst: str,
        tracer: Optional[trace.Tracer] = None,
        tracer_span: Optional[trace.Span] = None,
        # uuid is now derived from the task object
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
            self.logger.info("No agent_options provided, using default configuration.")
            agent_options = RecursiveAgentOptions(
                task_limits=TaskLimit.from_constant(max_tasks=3, max_depth=2)
            )
        self.options = agent_options

        # --- MODIFICATION: Handle TaskObject and register agent ---
        if isinstance(task, TaskObject):
            self.task_obj = task
        else:
            # If a string or other type is passed, create a default TaskObject
            self.task_obj = TaskObject(
                task=str(task), type=task_type_override or "research"
            )

        self.task = self.task_obj.task
        self.uuid = self.task_obj.uuid
        self.task_type = task_type_override or self.task_obj.type

        self.logger = logging.getLogger(f"{__name__}.RecursiveAgent.{self.uuid}")
        self.logger.info(
            f"Initializing RecursiveAgent for task: '{self.task}' (Type: {self.task_type}) at layer {current_layer} with UUID: {self.uuid}"
        )

        # Register this agent instance in the central registry
        if self.options.task_registry is not None:
            self.options.task_registry[self.uuid] = self

        self.u_inst = u_inst
        self.tracer = tracer if tracer else trace.get_tracer(__name__)
        self.tracer_span = tracer_span
        self.allow_subtasks = allow_subtasks
        self.llm: DotTree = self.options.llm
        self.tools = self.options.tools
        self.task_split_parser = JsonOutputParser(pydantic_object=SplitTask)
        self.task_verification_parser = JsonOutputParser(pydantic_object=verification)
        self.current_layer = current_layer
        self.parent = parent
        self.siblings = siblings or []
        self.context = context or TaskContext(task=self.task)
        self.result: Optional[str] = None
        self.status: str = (
            "pending"  # status can be: pending, running, succeeded, failed_verification, failed
        )
        self.current_span: Optional[trace.Span] = None
        self.fix_attempt_count: int = 0
        self.max_fix_attempts = (
            max_fix_attempts or agent_options.max_fix_attempts
            if agent_options.max_fix_attempts is not None
            else 2
        )

    def _get_token_count(self, text: str) -> int:
        if self.options.token_counter:
            try:
                return self.options.token_counter(text)
            except Exception as e:
                self.logger.warning(
                    f"Token counter failed for text: '{text[:50]}...': {e}",
                    exc_info=False,
                )
                return 0
        return 0

    def _build_context_information(self) -> dict:
        ancestor_chain_contexts_data = []
        current_ancestor_node = self.context.parent_context
        while current_ancestor_node:
            if current_ancestor_node.result is not None:
                ancestor_chain_contexts_data.append(
                    {
                        "task": current_ancestor_node.task,
                        "result": current_ancestor_node.result,
                        "relation": "ancestor",
                    }
                )
            current_ancestor_node = current_ancestor_node.parent_context
        ancestor_chain_contexts_data.reverse()

        broader_family_contexts_data = []
        tasks_to_exclude_from_broader = {
            ctx_data["task"] for ctx_data in ancestor_chain_contexts_data
        }
        tasks_to_exclude_from_broader.add(self.context.task)

        ancestor_depth = 0
        temp_node_for_sibling_scan = self.context
        while temp_node_for_sibling_scan:
            for sibling_of_temp_node in temp_node_for_sibling_scan.siblings:
                if (
                    sibling_of_temp_node.result is not None
                    and sibling_of_temp_node.task not in tasks_to_exclude_from_broader
                ):
                    relation = ""
                    if ancestor_depth == 0:
                        relation = "direct_sibling"
                    else:
                        relation = f"ancestor_level_{ancestor_depth}_sibling"
                    broader_family_contexts_data.append(
                        {
                            "task": sibling_of_temp_node.task,
                            "result": sibling_of_temp_node.result,
                            "relation": relation,
                        }
                    )
                    tasks_to_exclude_from_broader.add(sibling_of_temp_node.task)
            temp_node_for_sibling_scan = temp_node_for_sibling_scan.parent_context
            ancestor_depth += 1

        dependency_contexts_data = []
        if self.context.dependency_results:
            for dep_key, dep_result in self.context.dependency_results.items():
                # This part of the code now correctly handles UUIDs post-translation
                dep_agent = self.options.task_registry.get(dep_key)
                dep_task_desc = dep_agent.task if dep_agent else "Unknown Task"
                dependency_contexts_data.append(
                    {
                        "task": dep_task_desc,
                        "result": dep_result,
                        "relation": "dependency",
                        "uuid": dep_key,
                    }
                )

        return {
            "ancestor_chain_contexts": ancestor_chain_contexts_data,
            "broader_family_contexts": broader_family_contexts_data,
            "dependency_contexts": dependency_contexts_data,
        }

    def _format_history_parts(
        self,
        context_info: dict,
        purpose: str,
        subtask_results_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        history = []

        # --- MODIFICATION: Add formatting for dependency context ---
        if context_info.get("dependency_contexts"):
            history.append(f"Context from completed dependency tasks (for {purpose}):")
            for ctx in context_info["dependency_contexts"]:
                history.append(
                    f"- Dependency Task (UUID: {ctx['uuid']}): {ctx['task']}\n  Result: {str(ctx['result'])[:150]}..."
                )

        if context_info.get("ancestor_chain_contexts"):
            history.append(f"\nContext from direct ancestor tasks (for {purpose}):")
            for ctx in context_info["ancestor_chain_contexts"]:
                history.append(
                    f"- Ancestor Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}..."
                )

        if context_info.get("broader_family_contexts"):
            history.append(
                f"\nContext from other related tasks in the hierarchy (for {purpose}):"
            )
            for ctx in context_info["broader_family_contexts"]:
                relation_desc = ctx["relation"].replace("_", " ").capitalize()
                history.append(
                    f"- {relation_desc} Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}..."
                )

        if purpose == "verification" and subtask_results_map:
            history.append(
                "\nThe current main task involved these subtasks and their results (for verification):"
            )
            for sub_task, sub_result in subtask_results_map.items():
                history.append(
                    f"- Subtask (of current task): {sub_task}\n  - Result: {str(sub_result)[:200]}..."
                )
        return history

    def _build_task_split_history(self) -> str:
        context_info = self._build_context_information()
        return "\n".join(self._format_history_parts(context_info, "splitting"))

    def _build_task_verify_history(
        self, subtask_results_map: Optional[Dict[str, str]] = None
    ) -> str:
        context_info = self._build_context_information()
        return "\n".join(
            self._format_history_parts(
                context_info, "verification", subtask_results_map
            )
        )

    def run(self):
        self.status = "running"
        # (Rest of the run wrapper method is unchanged)
        self.logger.info(
            f"Attempting to start run for task: '{self.task}' (UUID: {self.uuid}, Status: {self.status})"
        )
        parent_otel_ctx = otel_context.get_current()
        if self.tracer_span:
            parent_otel_ctx = trace.set_span_in_context(self.tracer_span)
        with self.tracer.start_as_current_span(
            f"RecursiveAgent Task: {self.task[:50]}...",
            context=parent_otel_ctx,
            attributes={
                "agent.task.full": self.task,
                "agent.uuid": self.uuid,
                "agent.layer": self.current_layer,
                "agent.initial_status": self.status,
                "agent.allow_subtasks_flag": self.allow_subtasks,
            },
        ) as span:
            self.current_span = span
            span.add_event(
                "Agent Run Start",
                attributes={
                    "task": self.task,
                    "user_instructions_preview": str(self.u_inst)[:200],
                    "current_layer": self.current_layer,
                },
            )
            try:
                result = self._run()
                span.set_attribute("agent.final_status", self.status)
                span.add_event(
                    "Agent Run End",
                    attributes={
                        "result_preview": str(result)[:200],
                        "final_status": self.status,
                    },
                )
                self.logger.info(
                    f"Run finished for task: '{self.task}'. Result: {str(result)[:100]}... Status: {self.status}"
                )
                return result
            except Exception as e:
                self.logger.error(
                    f"Critical error in agent run for task '{self.task}': {e}",
                    exc_info=True,
                )
                if span:
                    span.record_exception(e)
                    self.status = "failed_critically"
                    span.set_attribute("agent.final_status", self.status)
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=str(e))
                    )
                raise TaskFailedException(
                    f"Agent run for '{self.task}' failed critically: {e}"
                ) from e

    # --- MAJOR REFACTOR: _run method with dependency resolution ---
    def _run(self) -> str:
        span = self.current_span
        if not span:
            self.logger.warning(
                "_run called without an active self.current_span. Tracing will be limited for this operation."
            )
        self.logger.info(
            f"Starting _run for task: '{self.task}' at layer {self.current_layer}"
        )
        if span:
            span.add_event("Internal Execution Start", {"task": self.task})

        if self.options.pre_task_executed:
            if span:
                span.add_event("Pre-Task Callback Executing")
            self.options.pre_task_executed(
                task=self.task,
                uuid=self.uuid,
                parent_agent_uuid=(self.parent.uuid if self.parent else None),
            )

        max_subtasks_for_this_layer = self._get_max_subtasks()
        if max_subtasks_for_this_layer == 0 or not self.allow_subtasks:
            if span:
                span.add_event(
                    "Executing as Single Task: Subtasks disabled for this layer."
                )
            return self._execute_and_verify_single_task()

        if self.parent:
            similarity = calculate_raw_similarity(self.task, self.parent.task)
            if span:
                span.add_event(
                    "Parent Similarity Check",
                    {
                        "similarity_score": similarity,
                        "threshold": self.options.similarity_threshold,
                    },
                )
            if similarity >= self.options.similarity_threshold:
                if span:
                    span.add_event(
                        "Executing as Single Task: High Parent Similarity",
                        {"similarity": similarity},
                    )
                return self._execute_and_verify_single_task()

        split_task_result = self._split_task()
        if span:
            span.add_event(
                "Task Splitting Outcome",
                {
                    "needs_subtasks": split_task_result.needs_subtasks,
                    "generated_subtasks_count": len(split_task_result.subtasks),
                },
            )

        if not split_task_result or not split_task_result.needs_subtasks:
            if span:
                span.add_event(
                    "Executing as Single Task: Splitting indicated no subtasks needed."
                )
            return self._execute_and_verify_single_task()

        # --- NEW: Dependency-aware execution logic ---
        limited_subtasks = split_task_result.subtasks[:max_subtasks_for_this_layer]
        if span:
            span.add_event(
                "Subtasks Limited",
                {
                    "original_count": len(split_task_result.subtasks),
                    "limited_count": len(limited_subtasks),
                },
            )

        # 1. Create all child agent instances first, preserving order for dependency mapping
        child_agents_in_order: List[RecursiveAgent] = []
        child_contexts: List[TaskContext] = []
        for llm_subtask_obj in limited_subtasks:
            # Convert LLMTaskObject to the internal TaskObject, assigning a UUID
            subtask_obj = TaskObject(**llm_subtask_obj.model_dump())

            child_context = TaskContext(
                task=subtask_obj.task, parent_context=self.context
            )
            child_agent = RecursiveAgent(
                task=subtask_obj,  # Pass the full object
                u_inst=self.u_inst,
                tracer=self.tracer,
                tracer_span=span,
                agent_options=self.options,
                allow_subtasks=(
                    self.current_layer + 1 < len(self.options.task_limits.limits)
                ),
                current_layer=self.current_layer + 1,
                parent=self,
                context=child_context,
            )
            child_agents_in_order.append(child_agent)
            child_contexts.append(child_context)

        # 1a. Create a mapping from 1-based index to UUID for new sibling tasks
        index_to_uuid_map = {
            str(i + 1): agent.uuid for i, agent in enumerate(child_agents_in_order)
        }

        # 1b. Translate index-based dependencies to UUID-based dependencies for all new agents
        for agent in child_agents_in_order:
            if agent.task_obj.depends_on:
                # Use .get(dep, dep) to keep existing UUIDs and translate only indices
                translated_deps = [
                    index_to_uuid_map.get(dep, dep) for dep in agent.task_obj.depends_on
                ]
                agent.task_obj.depends_on = translated_deps

        # 1c. Create the final dictionary of child agents and set siblings
        child_agents: Dict[str, RecursiveAgent] = {
            agent.uuid: agent for agent in child_agents_in_order
        }

        # Set siblings for all created children
        for agent in child_agents.values():
            agent.context.siblings = [
                ctx for ctx in child_contexts if ctx.task != agent.task
            ]

        # 2. Execute tasks based on dependencies
        completed_tasks: Dict[str, str] = {}
        pending_agents: Dict[str, RecursiveAgent] = child_agents.copy()

        loop_guard = 0
        max_loops = len(pending_agents) + 2  # Safety break for dependency resolution

        while pending_agents and loop_guard < max_loops:
            loop_guard += 1
            runnable_agents: List[RecursiveAgent] = []

            # Find agents whose dependencies are met
            for agent_uuid, agent in pending_agents.items():
                dependencies = agent.task_obj.depends_on
                if all(dep_uuid in completed_tasks for dep_uuid in dependencies):
                    runnable_agents.append(agent)

            if not runnable_agents:
                # Before failing, check if the dependency exists in the global registry (was already completed)
                # This handles cases where a subtask depends on a task outside its immediate sibling group
                for agent_uuid, agent in pending_agents.items():
                    dependencies = agent.task_obj.depends_on
                    non_sibling_deps_met = True
                    for dep_uuid in dependencies:
                        # If a dependency is NOT a sibling and NOT completed, we must wait
                        if dep_uuid not in child_agents:
                            dep_agent = self.options.task_registry.get(dep_uuid)
                            if not dep_agent or dep_agent.status != "succeeded":
                                non_sibling_deps_met = False
                                break  # This dependency is not met
                            else:  # The dependency is met, add its result to our completed list
                                if dep_uuid not in completed_tasks:
                                    completed_tasks[dep_uuid] = dep_agent.result

                    if non_sibling_deps_met and all(
                        dep_uuid in completed_tasks for dep_uuid in dependencies
                    ):
                        runnable_agents.append(agent)

            if not runnable_agents and pending_agents:
                failed_tasks_info = {
                    uuid: agent.task_obj.depends_on
                    for uuid, agent in pending_agents.items()
                }
                error_msg = f"Circular or unresolved dependency detected. Cannot proceed. Pending tasks: {failed_tasks_info}"
                self.logger.error(error_msg)
                if span:
                    span.add_event("Dependency Error", {"details": error_msg})
                raise TaskFailedException(error_msg)

            # Run the agents that are ready
            for agent in runnable_agents:
                if agent.uuid not in pending_agents:
                    continue  # Already processed in this loop

                if span:
                    span.add_event(
                        f"Dependency Met: Running Task",
                        {"child_task": agent.task, "child_uuid": agent.uuid},
                    )

                # Inject dependency results into context
                agent.context.dependency_results = {
                    dep_uuid: completed_tasks[dep_uuid]
                    for dep_uuid in agent.task_obj.depends_on
                    if dep_uuid in completed_tasks
                }

                result = agent.run()
                completed_tasks[agent.uuid] = (
                    result if result is not None else "No result from subtask."
                )
                del pending_agents[agent.uuid]  # Move from pending
                if span:
                    span.add_event(
                        f"Task Completed",
                        {
                            "child_task": agent.task,
                            "child_uuid": agent.uuid,
                            "result_preview": str(result)[:100],
                        },
                    )

        if pending_agents:
            self.logger.warning(
                f"Dependency resolution loop finished with pending agents, something went wrong. {list(pending_agents.keys())}"
            )

        # 3. Summarize results
        subtask_tasks_for_summary = [
            child_agents[uuid].task
            for uuid in completed_tasks.keys()
            if uuid in child_agents
        ]
        subtask_results_for_summary = [
            result for uuid, result in completed_tasks.items() if uuid in child_agents
        ]

        self.result = self._summarize_subtask_results(
            subtask_tasks_for_summary, subtask_results_for_summary
        )
        self.context.result = self.result

        subtask_results_map = {
            child_agents[uuid].task: result
            for uuid, result in completed_tasks.items()
            if uuid in child_agents
        }
        try:
            self.verify_result(subtask_results_map)
        except TaskFailedException:
            if span:
                span.add_event(
                    "Subtask Combined Result Verification Failed, Attempting Fix"
                )
            self._fix(subtask_results_map)

        if self.options.on_task_executed:
            self.options.on_task_executed(
                self.task,
                self.uuid,
                self.result,
                self.parent.uuid if self.parent else None,
            )

        return self.result

    def _execute_and_verify_single_task(self) -> str:
        """Helper to run a single task, verify, and potentially fix it."""
        self.result = self._run_single_task()
        self.context.result = self.result
        try:
            self.verify_result(None)
        except TaskFailedException:
            if self.current_span:
                self.current_span.add_event(
                    "Single Task Verification Failed, Attempting Fix"
                )
            self._fix(None)

        if self.options.on_task_executed:
            self.options.on_task_executed(
                self.task,
                self.uuid,
                self.result,
                self.parent.uuid if self.parent else None,
            )
        return self.result

    def _run_single_task(self) -> str:
        agent_span = self.current_span
        parent_context_for_single_task = (
            trace.set_span_in_context(agent_span)
            if agent_span
            else otel_context.get_current()
        )

        with self.tracer.start_as_current_span(
            "Run Single Task Operation", context=parent_context_for_single_task
        ) as single_task_span:
            self.logger.info(f"Running single task: '{self.task}'")
            context_info = self._build_context_information()
            full_context_str = "\n".join(
                self._format_history_parts(context_info, "single task execution")
            )

            single_task_span.add_event(
                "Single Task Execution Start",
                {
                    "task": self.task,
                    "user_instructions": self.u_inst,
                    "context_preview": full_context_str[:300],
                },
            )

            current_task_type = getattr(self, "task_type", "research")

            if current_task_type in ["basic", "task"]:
                self.logger.info(
                    f"Adjusting system prompt for '{current_task_type}' task type."
                )
                system_prompt_content = (
                    f"You are a helpful assistant. Your current task is to directly execute or answer the following. "
                    f"Provide a direct, concise answer or the direct output of any tools used. Avoid narrative summaries unless the output itself is narrative. "
                    f"If you use tools, present their output clearly. For code execution, provide the results of the execution (e.g. stdout, stderr, or the final value of a variable). "
                    f"For search, provide the information found. "
                    f"Focus on actionable results or direct answers.\n\n"
                    f"Current Task: {self.task}\n\n"
                    f"Relevant contextual history from other tasks (if any):\n{full_context_str}"
                )
            else:
                self.logger.info(
                    f"Using standard system prompt for '{current_task_type}' task type."
                )
                system_prompt_content = (
                    f"Your task is to answer the following question, using any tools that you deem necessary. "
                    f"Make sure to phrase your search phrase in a way that it could be understood easily without context. "
                    f"If you use the web search tool, make sure you include citations (just use a pair of square "
                    f"brackets and a number in text, and at the end, include a citations section).\n\n"
                    f"Relevant contextual history from other tasks (if any):\n{full_context_str}"
                )

            human_message_content = self.task
            if self.u_inst:
                human_message_content += (
                    f"\n\nFollow these specific instructions: {self.u_inst}"
                )
            human_message_content += "\n\nApply the distributive property to any tool calls. For instance, if you need to search for 3 related things, make 3 separate calls to the search tool, because that will yield better results."

            history = [
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=human_message_content),
            ]

            loop_count = 0
            max_loops = 10
            final_result_content = (
                "Max tool loop iterations reached without a final answer."
            )

            while loop_count < max_loops:
                loop_count += 1
                single_task_span.add_event(
                    f"Tool Interaction Loop Iteration {loop_count}"
                )

                full_prompt_str_for_tokens = "\n".join(
                    [str(m.content) for m in history]
                )
                prompt_preview_str = _serialize_lc_messages_for_preview(
                    history, max_len=1000
                )
                prompt_tokens = self._get_token_count(full_prompt_str_for_tokens)

                with self.tracer.start_as_current_span(
                    "LLM Invocation (Tool LLM)",
                    context=trace.set_span_in_context(single_task_span),
                ) as llm_span:
                    llm_span.set_attributes(
                        {
                            "llm.type": "tool_llm",
                            "iteration": loop_count,
                            "prompt.messages_count": len(history),
                            "prompt.preview": prompt_preview_str,
                            "estimated_prompt_tokens": prompt_tokens,
                        }
                    )
                    llm_span.add_event("LLM Invocation Start")
                    llm = self.llm.resolve("llm.tools").value
                    current_llm_response = llm.invoke(history)
                    response_content_str = str(current_llm_response.content)
                    completion_tokens = self._get_token_count(response_content_str)
                    llm_span.add_event("LLM Invocation End")

                    llm_span.set_attributes(
                        {
                            "response.content_preview": response_content_str[:200],
                            "response.has_tool_calls": bool(
                                current_llm_response.tool_calls
                            ),
                            "response.tool_calls_count": len(
                                current_llm_response.tool_calls or []
                            ),
                            "estimated_completion_tokens": completion_tokens,
                            "estimated_total_tokens": prompt_tokens + completion_tokens,
                        }
                    )
                history.append(current_llm_response)

                if not current_llm_response.tool_calls:
                    final_result_content = str(current_llm_response.content)
                    single_task_span.add_event(
                        "LLM Responded Without Tool Calls - Final Answer",
                        {"final_answer_preview": final_result_content[:200]},
                    )
                    break

                with self.tracer.start_as_current_span(
                    "Processing Tool Calls Batch",
                    context=trace.set_span_in_context(single_task_span),
                ) as batch_tool_span:
                    tool_calls_in_batch = current_llm_response.tool_calls or []
                    batch_tool_span.set_attribute(
                        "tool_calls.count_in_batch", len(tool_calls_in_batch)
                    )

                    tool_messages_for_this_turn = []
                    guidance_for_llm_reprompt = []
                    any_tool_requires_llm_replan = False

                    for tool_call_idx, tool_call in enumerate(tool_calls_in_batch):
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

                        with self.tracer.start_as_current_span(
                            f"Process Tool Call: {tool_name}",
                            context=trace.set_span_in_context(batch_tool_span),
                        ) as individual_tool_span:
                            individual_tool_span.set_attributes(
                                {
                                    "tool.name": tool_name,
                                    "tool.args": json.dumps(tool_args, default=str),
                                    "tool.call_id": tool_call_id,
                                    "tool.index_in_batch": tool_call_idx,
                                }
                            )

                            tool_executed_successfully_this_iteration = False
                            this_tool_call_needs_llm_replan = False
                            human_message_for_this_tool_failure = None
                            current_tool_output_payload = None

                            individual_tool_span.add_event(
                                "Tool execution attempt initiated"
                            )

                            try:
                                if tool_name not in self.options.tools_dict:
                                    raise KeyError(f"Tool '{tool_name}' not found.")
                                tool_to_execute = self.options.tools_dict[tool_name]

                                if tool_name == "search" and "query" in tool_args:
                                    query = tool_args["query"]
                                    sim = calculate_raw_similarity(query, self.task)
                                    individual_tool_span.add_event(
                                        "Search Tool Query Similarity Check",
                                        {
                                            "query": query,
                                            "similarity_score": sim,
                                            "threshold": self.options.similarity_threshold
                                            / 5,
                                        },
                                    )
                                    if sim < (self.options.similarity_threshold / 5):
                                        error_detail = f"Search query '{query}' too dissimilar (score: {sim:.2f}) to main task. Revise query."
                                        current_tool_output_payload = {
                                            "error": error_detail,
                                            "guidance": "The search query was too different from the main task. Please rephrase your search to be more relevant or break down the problem differently.",
                                        }
                                        human_message_for_this_tool_failure = f"Search query '{query}' for tool '{tool_name}' was too dissimilar to the main task. Please adjust your plan or query."
                                        this_tool_call_needs_llm_replan = True
                                        individual_tool_span.add_event(
                                            "Search Tool Query Dissimilar",
                                            {"error_detail": error_detail},
                                        )
                                        individual_tool_span.set_status(
                                            trace.Status(
                                                trace.StatusCode.ERROR,
                                                "Search query dissimilar",
                                            )
                                        )
                                    else:
                                        individual_tool_span.add_event(
                                            "Executing search tool with accepted query"
                                        )
                                        current_tool_output_payload = tool_to_execute(
                                            **tool_args
                                        )
                                        tool_executed_successfully_this_iteration = True
                                else:
                                    if hasattr(tool_to_execute, "invoke"):
                                        individual_tool_span.add_event(
                                            f"Executing non-search tool (using .invoke()): {tool_name}"
                                        )
                                        current_tool_output_payload = asyncio.run(self.run_structured_tool(tool_to_execute, tool_args))
                                    else:
                                        individual_tool_span.add_event(
                                            f"Executing non-search tool: {tool_name}"
                                        )
                                        current_tool_output_payload = tool_to_execute(**tool_args)
                                    tool_executed_successfully_this_iteration = True

                                individual_tool_span.add_event(
                                    "Tool execution completed",
                                    {
                                        "output_preview": str(
                                            current_tool_output_payload
                                        )[:200]
                                    },
                                )
                                individual_tool_span.set_attribute(
                                    "output", str(current_tool_output_payload)
                                )
                                if tool_executed_successfully_this_iteration:
                                    individual_tool_span.set_status(
                                        trace.Status(trace.StatusCode.OK)
                                    )

                            except Exception as e:
                                error_msg_str = f"Error with tool {tool_name} (ID: {tool_call_id}): {e}"
                                self.logger.error(error_msg_str, exc_info=True)
                                current_tool_output_payload = {
                                    "error": "Tool execution failed",
                                    "details": error_msg_str,
                                }
                                human_message_for_this_tool_failure = f"Error with tool '{tool_name}': {e}. Adjust your plan."
                                this_tool_call_needs_llm_replan = True

                                individual_tool_span.record_exception(e)
                                individual_tool_span.set_status(
                                    trace.Status(
                                        trace.StatusCode.ERROR,
                                        f"Tool {tool_name} execution failed: {str(e)}",
                                    )
                                )

                            tool_message_content_final_str = json.dumps(
                                current_tool_output_payload, default=str
                            )
                            tool_messages_for_this_turn.append(
                                ToolMessage(
                                    content=tool_message_content_final_str,
                                    tool_call_id=tool_call_id,
                                )
                            )
                            individual_tool_span.add_event(
                                "ToolMessage prepared",
                                {
                                    "content_preview": tool_message_content_final_str[
                                        :200
                                    ]
                                },
                            )

                            individual_tool_span.set_attributes(
                                {
                                    "tool.execution_successful": tool_executed_successfully_this_iteration,
                                    "tool.requires_llm_replan": this_tool_call_needs_llm_replan,
                                    "tool.response_preview": tool_message_content_final_str[
                                        :200
                                    ],
                                }
                            )

                            if self.options.on_tool_call_executed:
                                individual_tool_span.add_event(
                                    "Invoking on_tool_call_executed callback"
                                )
                                try:
                                    self.options.on_tool_call_executed(
                                        self.task,
                                        self.uuid,
                                        tool_name,
                                        tool_args,
                                        current_tool_output_payload,
                                        tool_executed_successfully_this_iteration,
                                        tool_call_id,
                                    )
                                    individual_tool_span.add_event(
                                        "on_tool_call_executed callback finished successfully"
                                    )
                                except Exception as cb_ex:
                                    individual_tool_span.record_exception(
                                        cb_ex,
                                        {"callback_name": "on_tool_call_executed"},
                                    )
                                    self.logger.error(
                                        f"Error in on_tool_call_executed callback: {cb_ex}",
                                        exc_info=True,
                                    )

                            if this_tool_call_needs_llm_replan:
                                any_tool_requires_llm_replan = True
                                if human_message_for_this_tool_failure:
                                    guidance_for_llm_reprompt.append(
                                        human_message_for_this_tool_failure
                                    )

                    batch_tool_span.set_attribute(
                        "any_tool_requires_llm_replan_after_batch",
                        any_tool_requires_llm_replan,
                    )
                    if guidance_for_llm_reprompt:
                        batch_tool_span.set_attribute(
                            "llm_reprompt_guidance_messages_count",
                            len(guidance_for_llm_reprompt),
                        )
                        batch_tool_span.add_event(
                            "LLM reprompt guidance prepared",
                            {
                                "guidance_preview": "\n".join(
                                    guidance_for_llm_reprompt
                                )[:300]
                            },
                        )

                    if not any_tool_requires_llm_replan:
                        batch_tool_span.set_status(trace.Status(trace.StatusCode.OK))
                    else:
                        batch_tool_span.set_status(
                            trace.Status(
                                trace.StatusCode.ERROR,
                                "One or more tools require LLM replan",
                            )
                        )

                history.extend(tool_messages_for_this_turn)

                if any_tool_requires_llm_replan:
                    replan_message_content = (
                        "One or more tool actions resulted in errors or require a change of plan. "
                        "Review the previous tool outputs and your reasoning. Adjust your plan and continue towards the main goal.\n"
                        "Specific issues encountered:\n"
                        + "\n".join([f"- {g}" for g in guidance_for_llm_reprompt])
                        if guidance_for_llm_reprompt
                        else "One or more tool calls had issues. Please review the tool responses and adjust your plan. Re-evaluate your approach to the main task."
                    )
                    history.append(HumanMessage(content=replan_message_content))
                    single_task_span.add_event(
                        "LLM Re-Plan Requested After Tool Batch",
                        {"replan_reason_preview": replan_message_content[:300]},
                    )
                    continue

            if loop_count >= max_loops:
                single_task_span.add_event(
                    "Max Tool Loop Iterations Reached", {"max_loops": max_loops}
                )
                single_task_span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR, "Max tool loop iterations reached"
                    )
                )
                if history and isinstance(history[-1], AIMessage):
                    final_result_content = str(history[-1].content)
                elif history and isinstance(history[-1], ToolMessage):
                    final_result_content = "Max tool loop iterations reached. Last action was a tool call. No final AI response generated."
            else:
                single_task_span.set_status(trace.Status(trace.StatusCode.OK))

            single_task_span.add_event(
                "Single Task Execution End",
                {"final_result_preview": final_result_content[:200]},
            )
            return final_result_content

    def _split_task(self) -> SplitTask:
        agent_span = self.current_span
        parent_context_for_split = (
            trace.set_span_in_context(agent_span)
            if agent_span
            else otel_context.get_current()
        )

        with self.tracer.start_as_current_span(
            "Split Task Operation", context=parent_context_for_split
        ) as split_span:
            self.logger.info(f"Splitting task: '{self.task}'")
            task_history_for_splitting = self._build_task_split_history()
            max_subtasks = self._get_max_subtasks()

            ancestor_uuids = set()
            current_agent_for_traversal = self
            while current_agent_for_traversal:
                ancestor_uuids.add(current_agent_for_traversal.uuid)
                current_agent_for_traversal = current_agent_for_traversal.parent

            existing_tasks_summary = []
            if self.options.task_registry:
                valid_dependency_candidates = []
                for task_uuid, agent_instance in self.options.task_registry.items():
                    if task_uuid in ancestor_uuids:
                        continue
                    if agent_instance.status != "succeeded":
                        continue

                    valid_dependency_candidates.append(agent_instance)

                if valid_dependency_candidates:
                    existing_tasks_summary.append(
                        "You can create dependencies on the following tasks that have ALREADY COMPLETED SUCCESSFULLY. Use their UUIDs in the `depends_on` field."
                    )
                    for agent_instance in sorted(
                        valid_dependency_candidates, key=lambda a: a.task
                    ):
                        status_info = agent_instance.status
                        if agent_instance.result:
                            status_info += (
                                f" (Result: {str(agent_instance.result)[:80]}...)"
                            )

                        existing_tasks_summary.append(
                            f'- Task: "{agent_instance.task}"\n  UUID: {agent_instance.uuid}\n  Status: {status_info}'
                        )
            
            existing_tasks_str = "\n".join(existing_tasks_summary)
            if not existing_tasks_summary:
                existing_tasks_str = (
                    "No other tasks have been created yet that can be depended on."
                )

            split_span.add_event(
                "Task Splitting Start",
                {"task": self.task, "max_subtasks_allowed": max_subtasks},
            )

            tools_to_use = list(self.options.tools_dict.values())
            tools_to_use = sorted(tools_to_use, key=lambda t: t.__name__ if hasattr(t, '__name__') else t.name)

            tools_dict = {}
            for tool in tools_to_use:
                tools_dict[tool.name if hasattr(tool, 'name') else tool.__name__] = tool
            tools_to_use = sorted(tools_dict.values(), key=lambda t: t.name if hasattr(t, 'name') else t.__name__)

            import inspect

            tools_with_docstrings = "\n".join(
                [
                    f"{tool.name if not hasattr(tool, '__name__') else tool.__name__}: {getattr(tool, 'description', '') or inspect.cleandoc(tool.__doc__)}"
                    for tool in tools_to_use
                    if tool.__doc__ or getattr(tool, 'description', '')
                ]
            )
            # --- MODIFICATION START: Updated instructions for dependency creation ---
            system_msg_content = (
                f"Split this task into smaller subtasks only if it's complex. You can create up to {max_subtasks} subtasks. "
                "You can create dependencies on previously completed tasks (using their UUIDs) or on new tasks you are creating now (using their index).\n\n"
                "=== AVAILABLE TASKS FOR DEPENDENCIES (use their UUID) ===\n"
                f"{existing_tasks_str}\n\n"
                "**Do not create dependencies on tasks that are not yet completed or don't exist.**\n\n"
                "=== CONTEXTUAL HISTORY (ancestors, siblings) ===\n"
                f"{task_history_for_splitting}\n\n"
                f"=== CURRENT TASK TO SPLIT ===\n'{self.task}'\n\n"
                "INSTRUCTIONS:\n"
                "1. Analyze the CURRENT TASK. If it's simple, set `needs_subtasks` to `false`.\n"
                "2. If it's complex, break it into a logical sequence of smaller, independent subtasks.\n"
                "3. If a subtask requires the result of a PREVIOUSLY COMPLETED task, add its UUID to the `depends_on` list.\n"
                "4. To create a dependency on another subtask you are creating in this same list, use its 1-based index number (as a string) in the `depends_on` field. For example, for the third subtask to depend on the first, its `depends_on` would be `['1']`.\n"
                f"5. Do not exceed {max_subtasks} subtasks in total for this split.\n\n"
                f"The tasks you create can also use these tools: \n{tools_with_docstrings}"
            )
            # --- MODIFICATION END ---
            logger.debug(ic.format(system_msg_content))
            if self.u_inst:
                system_msg_content += (
                    f"\nUser-provided instructions for the main task:\n{self.u_inst}"
                )

            split_msgs_hist = [
                SystemMessage(content=system_msg_content),
                HumanMessage(self.task),
            ]

            evaluation = evaluate_prompt(f"Prompt: {self.task}")
            split_span.add_event(
                "Prompt Complexity Evaluation",
                {
                    "complexity_score": evaluation.prompt_complexity_score[0],
                    "domain_knowledge_score": evaluation.domain_knowledge[0],
                },
            )

            if (
                evaluation.prompt_complexity_score[0] < 0.1
                and evaluation.domain_knowledge[0] > 0.8
            ):
                split_span.add_event(
                    "Skipping LLM Split: Low Complexity / High Domain Knowledge"
                )
                return SplitTask(
                    needs_subtasks=False, subtasks=[], evaluation=evaluation
                )

            primed_hist_1 = split_msgs_hist + [AIMessage(content="1. ")]
            prompt_str_1 = _serialize_lc_messages_for_preview(primed_hist_1)
            full_prompt_str_1_tokens = "\n".join(
                [str(m.content) for m in primed_hist_1]
            )
            prompt_tokens_1 = self._get_token_count(full_prompt_str_1_tokens)
            split_span.add_event(
                "LLM Invocation Start (Splitting - Initial List)",
                {
                    "llm_type": "main_llm",
                    "prompt_preview": prompt_str_1,
                    "estimated_prompt_tokens": prompt_tokens_1,
                },
            )
            response1 = self.llm.resolve("llm").value.invoke(primed_hist_1)
            response_content_1 = "1. " + str(response1.content)
            completion_tokens_1 = self._get_token_count(str(response1.content))
            split_span.add_event(
                "LLM Invocation End (Splitting - Initial List)",
                {
                    "llm_type": "main_llm",
                    "response_preview": response_content_1[:200],
                    "estimated_completion_tokens": completion_tokens_1,
                },
            )
            split_msgs_hist.append(AIMessage(content=response_content_1))

            refine_human_msg = HumanMessage(
                content="Can you make these more specific? Remember, each of these is sent off to another agent, with no context, asynchronously. All they know is what you put in this list."
            )
            hist_for_refine = split_msgs_hist + [refine_human_msg]
            prompt_str_2 = _serialize_lc_messages_for_preview(hist_for_refine)
            full_prompt_str_2_tokens = "\n".join(
                [str(m.content) for m in hist_for_refine]
            )
            prompt_tokens_2 = self._get_token_count(full_prompt_str_2_tokens)
            split_span.add_event(
                "LLM Invocation Start (Splitting - Refine List)",
                {
                    "llm_type": "main_llm",
                    "prompt_preview": prompt_str_2,
                    "estimated_prompt_tokens": prompt_tokens_2,
                },
            )
            response2 = self.llm.resolve("llm").value.invoke(hist_for_refine)
            completion_tokens_2 = self._get_token_count(str(response2.content))
            split_span.add_event(
                "LLM Invocation End (Splitting - Refine List)",
                {
                    "llm_type": "main_llm",
                    "response_preview": str(response2.content)[:200],
                    "estimated_completion_tokens": completion_tokens_2,
                },
            )
            split_msgs_hist.append(response2)

            json_format_msg = self._construct_subtask_to_json_prompt()
            hist_for_json = split_msgs_hist + [json_format_msg]
            prompt_str_3 = _serialize_lc_messages_for_preview(hist_for_json)
            full_prompt_str_3_tokens = "\n".join(
                [str(m.content) for m in hist_for_json]
            )
            prompt_tokens_3 = self._get_token_count(full_prompt_str_3_tokens)
            split_span.add_event(
                "LLM Invocation Start (Splitting - JSON Format)",
                {
                    "llm_type": "main_llm",
                    "prompt_preview": prompt_str_3,
                    "estimated_prompt_tokens": prompt_tokens_3,
                },
            )
            structured_response_msg = self.llm.resolve("llm").value.invoke(
                hist_for_json
            )
            completion_tokens_3 = self._get_token_count(
                str(structured_response_msg.content)
            )
            split_span.add_event(
                "LLM Invocation End (Splitting - JSON Format)",
                {
                    "llm_type": "main_llm",
                    "response_preview": str(structured_response_msg.content)[:200],
                    "estimated_completion_tokens": completion_tokens_3,
                },
            )

            split_task_result: SplitTask
            try:
                parsed_output_from_llm = self.task_split_parser.invoke(
                    structured_response_msg
                )
                if isinstance(parsed_output_from_llm, dict):
                    parsed_output_from_llm["evaluation"] = evaluation
                    split_task_result = SplitTask(**parsed_output_from_llm)
                elif isinstance(parsed_output_from_llm, SplitTask):
                    split_task_result = parsed_output_from_llm
                    split_task_result.evaluation = evaluation
                else:
                    raise TypeError(
                        f"Unexpected type from parser: {type(parsed_output_from_llm)}"
                    )

                split_span.set_attribute(
                    "subtasks", [task.task for task in split_task_result.subtasks]
                )
                split_span.add_event(
                    "Task Splitting JSON Parsed",
                    {
                        "needs_subtasks": split_task_result.needs_subtasks,
                        "count": len(split_task_result.subtasks),
                    },
                )

            except (ValidationError, OutputParserException, TypeError) as e:
                self.logger.error(
                    f"Error parsing LLM JSON for task splitting: {e}. Content: {str(structured_response_msg.content)[:500]}",
                    exc_info=True,
                )
                split_span.record_exception(
                    e,
                    attributes={
                        "llm_content_preview": str(structured_response_msg.content)[
                            :200
                        ]
                    },
                )
                split_task_result = SplitTask(
                    needs_subtasks=False, subtasks=[], evaluation=evaluation
                )

            if split_task_result.subtasks:
                original_subtask_count = len(split_task_result.subtasks)
                split_task_result.subtasks = split_task_result.subtasks[:max_subtasks]
                split_task_result.needs_subtasks = bool(split_task_result.subtasks)

            return split_task_result

    def _verify_result_internal(
        self, subtask_results_map: Optional[Dict[str, str]] = None
    ) -> bool:
        agent_span = self.current_span
        parent_context_for_verify = (
            trace.set_span_in_context(agent_span)
            if agent_span
            else otel_context.get_current()
        )

        with self.tracer.start_as_current_span(
            "Verify Result Operation", context=parent_context_for_verify
        ) as verify_span:
            self.logger.info(f"Verifying result for task: '{self.task}'")
            if self.result is None:
                verify_span.add_event("Verification Skipped: Result is None")
                return False

            task_history_for_verification = self._build_task_verify_history(
                subtask_results_map
            )

            verify_span.add_event(
                "Verification Process Start",
                {
                    "task": self.task,
                    "current_result_preview": str(self.result)[:200],
                    "user_instructions_preview": str(self.u_inst)[:200],
                    "verification_context_preview": task_history_for_verification[:300],
                },
            )

            system_msg_content = (
                "You are an AI assistant tasked with verifying the successful completion of a task. "
                "You will be given the original task, the result produced, and contextual history "
                "(ancestor tasks, other related tasks, and any subtasks that contributed to this result).\n\n"
                f"Contextual History:\n{task_history_for_verification}\n\n"
                "Based on ALL information (original task, produced result, and full context), critically evaluate if the produced result "
                "comprehensively, accurately, and directly addresses the original task. "
                "Do not verify external information sources, but focus on the quality and relevance of the result to the task. "
                "Output a JSON object with a 'successful' boolean field."
            )
            human_msg_content = (
                f"Original Task Statement:\n'''\n{self.task}\n'''\n\n"
                f"Produced Result for the Original Task:\n'''\n{self.result}\n'''\n\n"
                "User Instructions (if any) for the Original Task:\n'''\n"
                f"{self.u_inst if self.u_inst else 'No specific user instructions were provided.'}\n'''\n\n"
                "Considering all the above and the contextual history, was the original task successfully completed by the produced result?"
            )
            verify_msgs_hist_for_llm = [
                SystemMessage(content=system_msg_content),
                HumanMessage(content=human_msg_content),
            ]

            prompt_str = _serialize_lc_messages_for_preview(verify_msgs_hist_for_llm)
            full_prompt_str_tokens = "\n".join(
                [str(m.content) for m in verify_msgs_hist_for_llm]
            )
            prompt_tokens = self._get_token_count(full_prompt_str_tokens)
            verify_span.add_event(
                "LLM Invocation Start (Verification)",
                {
                    "llm_type": "main_llm",
                    "prompt_preview": prompt_str,
                    "estimated_prompt_tokens": prompt_tokens,
                },
            )
            structured_response_msg = self.llm.resolve("llm").value.invoke(
                verify_msgs_hist_for_llm
            )

            llm_response_content_for_parser = str(structured_response_msg.content)
            completion_tokens = self._get_token_count(llm_response_content_for_parser)
            verify_span.add_event(
                "LLM Invocation End (Verification)",
                {
                    "llm_type": "main_llm",
                    "response_preview": llm_response_content_for_parser[:200],
                    "estimated_completion_tokens": completion_tokens,
                },
            )

            verification_obj: Optional[verification] = None
            try:
                full_json_ai_message = AIMessage(
                    content=llm_response_content_for_parser
                )
                parsed_output = self.task_verification_parser.invoke(
                    full_json_ai_message
                )

                if isinstance(parsed_output, dict):
                    verification_obj = verification(**parsed_output)
                elif isinstance(parsed_output, verification):
                    verification_obj = parsed_output
                else:
                    raise TypeError(
                        f"Unexpected type from verification parser: {type(parsed_output)}"
                    )

                if verification_obj is None:
                    raise ValueError(
                        "Parsed output could not be converted to verification object."
                    )

                verify_span.add_event(
                    "Verification JSON Parsed",
                    {"task_successful": verification_obj.successful},
                )
                return verification_obj.successful
            except (ValidationError, OutputParserException, TypeError, ValueError) as e:
                self.logger.error(
                    f"Error parsing LLM JSON for verification: {e}. LLM content: {llm_response_content_for_parser[:500]}",
                    exc_info=True,
                )
                verify_span.record_exception(
                    e,
                    attributes={
                        "llm_content_preview": llm_response_content_for_parser[:200]
                    },
                )
                return False
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during verification finalization: {e}. LLM content: {llm_response_content_for_parser[:500]}",
                    exc_info=True,
                )
                verify_span.record_exception(
                    e,
                    attributes={
                        "llm_content_preview": llm_response_content_for_parser[:200]
                    },
                )
                return False

    def verify_result(self, subtask_results_map: Optional[Dict[str, str]] = None):
        agent_span = self.current_span
        successful = self._verify_result_internal(subtask_results_map)

        if successful:
            self.status = "succeeded"
            if agent_span:
                agent_span.add_event(
                    "Task Verification Passed",
                    {"new_status": self.status, "task": self.task},
                )
        else:
            self.status = "failed_verification"
            if agent_span:
                agent_span.add_event(
                    "Task Verification Failed",
                    {"new_status": self.status, "task": self.task},
                )
            raise TaskFailedException(
                f"Task '{self.task}' (UUID: {self.uuid}) was not completed successfully according to verification."
            )

    def _fix(self, failed_subtask_results_map: Optional[Dict[str, str]]):
        agent_span = self.current_span
        parent_context_for_fix = (
            trace.set_span_in_context(agent_span)
            if agent_span
            else otel_context.get_current()
        )

        with self.tracer.start_as_current_span(
            "Fix Task Operation", context=parent_context_for_fix
        ) as fix_span:
            self.logger.info(
                f"Attempting to fix task: '{self.task}' (UUID: {self.uuid}) which failed verification."
            )
            if self.fix_attempt_count >= self.max_fix_attempts:
                self.fix_attempt_count += 1
                original_task_str = self.task
                failed_result_str = str(
                    self.result
                    if self.result is not None
                    else "No result was produced."
                )

                fix_span.add_event(
                    "Fix Attempt Initiated",
                    {
                        "original_task": original_task_str,
                        "failed_result_preview": failed_result_str[:200],
                        "original_user_instructions_preview": str(self.u_inst)[:200],
                        "failed_subtasks_map_preview": (
                            json.dumps(failed_subtask_results_map, default=str)[:300]
                            if failed_subtask_results_map
                            else "None"
                        ),
                    },
                )

                fix_instructions_parts = [
                    f"The original task was: '{original_task_str}'.",
                    f"A previous attempt to solve it resulted in (or failed to produce a result): '{failed_result_str[:700]}...'. This outcome was deemed unsatisfactory/incomplete by an automated verification step.",
                ]
                if failed_subtask_results_map:
                    fix_instructions_parts.append(
                        "The failed attempt may have involved these subtasks and their results:"
                    )
                    for sub_task, sub_result in failed_subtask_results_map.items():
                        fix_instructions_parts.append(
                            f"  - Subtask: {sub_task}\n    - Result: {str(sub_result)[:200]}..."
                        )

                context_for_fix = self._build_context_information()
                formatted_context_for_fix = "\n".join(
                    self._format_history_parts(context_for_fix, "fixing a failed task")
                )
                if formatted_context_for_fix:
                    fix_instructions_parts.append(
                        f"\nRelevant contextual history from other tasks:\n{formatted_context_for_fix}"
                    )

                if self.u_inst:
                    fix_instructions_parts.append(
                        f"\nOriginal user instructions for the task were: {self.u_inst}"
                    )
                fix_instructions_parts.append(
                    "\nYour current objective is to FIX this failure. Analyze the original task, the failed outcome, any subtask information, original user instructions, and all provided context. "
                    "Then, provide a corrected and complete solution to the original task: '{original_task_str}'. "
                    "You can break this fix attempt into a small number of sub-steps if that helps achieve a high-quality corrected solution. "
                    "Focus on addressing the deficiencies of the previous attempt."
                )
                full_fix_instructions = "\n".join(fix_instructions_parts)

                current_agent_max_subtasks_at_this_layer = self._get_max_subtasks()
                fixer_max_subtasks_for_its_level = 0
                if current_agent_max_subtasks_at_this_layer > 0:
                    fixer_max_subtasks_for_its_level = max(
                        1, int(current_agent_max_subtasks_at_this_layer / 2)
                    )

                original_max_depth = len(self.options.task_limits.limits)
                fixer_limits_config_array = [0] * original_max_depth
                if self.current_layer < original_max_depth:
                    fixer_limits_config_array[self.current_layer] = (
                        fixer_max_subtasks_for_its_level
                    )

                fixer_task_limits = TaskLimit.from_array(fixer_limits_config_array)
                fixer_options = self.options.model_copy()
                fixer_options.task_limits = fixer_task_limits

                fixer_agent_uuid = str(uuid.uuid4())
                fix_span.add_event(
                    "Fixer Agent Configuration",
                    {
                        "fixer_agent_uuid": fixer_agent_uuid,
                        "fixer_max_subtasks": fixer_max_subtasks_for_its_level,
                        "fixer_instructions_preview": full_fix_instructions[:300],
                    },
                )

                fixer_agent_context = TaskContext(
                    task=original_task_str,
                    parent_context=self.context.parent_context,
                    siblings=self.context.siblings,
                )
                fixer_agent = RecursiveAgent(
                    task=original_task_str,
                    u_inst=full_fix_instructions,
                    tracer=self.tracer,
                    tracer_span=fix_span,
                    uuid=fixer_agent_uuid,
                    agent_options=fixer_options,
                    allow_subtasks=True,
                    current_layer=self.current_layer,
                    parent=self.parent,
                    context=fixer_agent_context,
                    siblings=self.siblings,
                    task_type_override=self.task_type,
                    max_fix_attempts=max(self.max_fix_attempts - 1, 0),
                )

                try:
                    fix_span.add_event(
                        "Fixer Agent Run Start", {"fixer_agent_uuid": fixer_agent_uuid}
                    )
                    fixer_result = fixer_agent.run()
                    fix_span.add_event(
                        "Fixer Agent Run Completed",
                        {
                            "fixer_agent_uuid": fixer_agent_uuid,
                            "fixer_result_preview": str(fixer_result)[:200],
                            "fixer_agent_final_status": fixer_agent.status,
                        },
                    )

                    self.result = fixer_result
                    self.context.result = self.result

                    fix_span.add_event("Re-verifying Fixed Result")
                    self.verify_result(None)

                    self.status = "fixed_and_verified"
                    fix_span.add_event(
                        "Fix Attempt Succeeded",
                        {
                            "final_status": self.status,
                            "new_result_preview": str(self.result)[:200],
                        },
                    )

                except TaskFailedException as e_fix_verify:
                    self.logger.error(
                        f"Verification of fixer agent's result FAILED for task '{self.task}': {e_fix_verify}. Marking task as terminally failed."
                    )
                    self.status = "failed"
                    fix_span.record_exception(
                        e_fix_verify,
                        attributes={"reason": "Re-verification of fixed result failed"},
                    )
                    fix_span.add_event(
                        "Fix Attempt Failed: Re-verification",
                        {"final_status": self.status, "error": str(e_fix_verify)},
                    )
                    raise TaskFailedException(
                        f"Fix attempt for '{self.task}' ultimately failed after re-verification: {e_fix_verify}"
                    ) from e_fix_verify

                except Exception as e_fix_run:
                    self.logger.error(
                        f"Fixer agent (UUID: {fixer_agent_uuid}) encountered an UNHANDLED ERROR: {e_fix_run}",
                        exc_info=True,
                    )
                    self.status = "failed"
                    if self.result is None:
                        self.result = f"Fix attempt failed with error: {e_fix_run}"
                    fix_span.record_exception(
                        e_fix_run, attributes={"reason": "Fixer agent run error"}
                    )
                    fix_span.add_event(
                        "Fix Attempt Failed: Fixer Agent Error",
                        {"final_status": self.status, "error": str(e_fix_run)},
                    )
                    raise TaskFailedException(
                        f"Fixer agent for '{self.task}' run failed: {e_fix_run}"
                    ) from e_fix_run

    def _get_max_subtasks(self) -> int:
        if self.current_layer >= len(self.options.task_limits.limits):
            return 0
        return self.options.task_limits.limits[self.current_layer]

    def _summarize_subtask_results(
        self, tasks: List[str], subtask_results: List[str]
    ) -> str:
        # ADD THIS BLOCK AT THE BEGINNING OF THE METHOD
        # Ensure json is imported at the top of the file: import json
        current_task_type = getattr(
            self, "task_type", "research"
        )  # Default to 'research' if not set
        if current_task_type in ["basic", "task"]:
            self.logger.info(
                f"Task type is '{current_task_type}'. Providing status update instead of full summary for task: '{self.task}'."
            )
            agent_span = self.current_span  # Ensure agent_span is defined for the event
            if agent_span:
                summary_span = self.tracer.start_span(
                    "Summarize Subtasks Operation (Status Update)",
                    context=trace.set_span_in_context(agent_span),
                )
            else:  # Fallback if no agent_span
                summary_span = self.tracer.start_span(
                    "Summarize Subtasks Operation (Status Update)"
                )

            if not subtask_results:
                summary_span.add_event(
                    "Summarization End: No Results to report for basic/task type."
                )
                summary_span.end()
                return "No subtask results to report."

            status_update_parts = [f"Status update for task: {self.task}"]
            if (
                not tasks and len(subtask_results) == 1
            ):  # Only a single result, no specific task questions
                status_update_parts.append("Result:")
                status_update_parts.append(str(subtask_results[0]))
            elif (
                not tasks and len(subtask_results) > 1
            ):  # Multiple results, no specific task questions
                status_update_parts.append("Results:")
                # Try to format as JSON if possible, else just join
                try:
                    status_update_parts.append(json.dumps(subtask_results, indent=2))
                except TypeError:
                    status_update_parts.append("\n".join(map(str, subtask_results)))
            else:  # Results are tied to specific subtask descriptions
                for i, (task_item, result_item) in enumerate(
                    zip(tasks, subtask_results)
                ):
                    status_update_parts.append(f"Sub-action {i+1}: {task_item}")
                    status_update_parts.append(f"  Result: {str(result_item)}")

            final_status_update = "\n".join(status_update_parts)
            summary_span.add_event(
                "Summarization End: Provided status update for basic/task type.",
                {"status_update_preview": final_status_update[:200]},
            )
            summary_span.end()
            return final_status_update
        else:
            self.logger.info(
                f"Task type is '{current_task_type}'. Proceeding with standard summarization for task: '{self.task}'."
            )
        # END OF ADDED BLOCK - Original method continues from here

        # Original method content starts here...
        agent_span = self.current_span
        parent_context_for_summary = (
            trace.set_span_in_context(agent_span)
            if agent_span
            else otel_context.get_current()
        )

        with self.tracer.start_as_current_span(
            "Summarize Subtasks Operation", context=parent_context_for_summary
        ) as summary_span:
            self.logger.info(
                f"Summarizing {len(subtask_results)} subtask results for task: '{self.task}'"
            )

            summary_span.add_event(
                "Summarization Start",
                {
                    "main_task_for_summary": self.task,
                    "subtask_count": len(tasks),
                    "input_tasks_preview": json.dumps(tasks, default=str)[:300],
                    "input_results_preview": json.dumps(
                        [
                            str(r)[:100] + "..." if r else "None"
                            for r in subtask_results
                        ],
                        default=str,
                    )[:300],
                    "align_summaries_enabled": self.options.align_summaries,
                },
            )

            if not subtask_results:
                summary_span.add_event(
                    "Summarization End: No Results to Summarize (Input List Empty)"
                )
                return "No subtask results were generated to summarize."

            documents_to_merge = [
                f"SUBTASK QUESTION: {q}\n\nSUBTASK ANSWER:\n{a}"
                for q, a in zip(tasks, subtask_results)
                if a is not None
            ]

            if not documents_to_merge:
                summary_span.add_event(
                    "Summarization End: All Subtasks Yielded Empty/No Results"
                )
                return "All subtasks yielded empty or no results."

            merged_content_list = []
            for document in documents_to_merge:
                summary_sentences_factor = self.options.summary_sentences_factor
                if self._get_token_count(document) > 5000:
                    num_sentences = max(
                        1, int(len(document.split()) / 5000 * summary_sentences_factor)
                    )
                    summary_span.add_event(
                        "Summarizing Long Document",
                        {
                            "document_length_chars": len(document),
                            "estimated_num_sentences_for_summary": num_sentences,
                        },
                    )
                    try:
                        merged_content_list.append(summarize(document, num_sentences))
                    except Exception as e_summarize:
                        self.logger.warning(
                            f"Summarize tool failed for a document part: {e_summarize}. Using original.",
                            exc_info=True,
                        )
                        merged_content_list.append(document)
                else:
                    merged_content_list.append(document)

            llm_for_merge = self.llm.resolve("llm.small.tiny").value
            merged_content_str = ""
            if not llm_for_merge:
                self.logger.warning(
                    "LLM for merging documents (llm.small.tiny) is not available. Using simple join."
                )
                merged_content_str = "\n\n---\n\n".join(merged_content_list)
            else:
                merge_options = MergeOptions(llm=llm_for_merge, context_window=15000)
                merger = self.options.merger(merge_options)
                try:
                    merged_content_str = merger.merge_documents(merged_content_list)
                except Exception as e_merge:
                    self.logger.warning(
                        f"LLMMerger failed: {e_merge}. Using simple join of content.",
                        exc_info=True,
                    )
                    merged_content_str = "\n\n---\n\n".join(merged_content_list)

            summary_span.add_event(
                "Documents Merged (Pre-Alignment)",
                {
                    "merged_content_length": len(merged_content_str),
                    "merged_content_preview": merged_content_str[:200],
                },
            )

            final_summary = merged_content_str
            if self.options.align_summaries:
                max_merged_content_len = 10000
                merged_content_for_alignment = merged_content_str
                if len(merged_content_str) > max_merged_content_len:
                    merged_content_for_alignment = (
                        merged_content_str[:max_merged_content_len]
                        + "\n... [Content Truncated]"
                    )
                    summary_span.add_event("Merged Content Truncated for Alignment LLM")

                alignment_prompt_messages = [
                    HumanMessage(
                        f"The following information has been gathered from subtasks:\n\n{merged_content_for_alignment}\n\n"
                        f"Based on this information, compile a comprehensive and well-structured report that directly answers this main question: '{self.task}'.\n"
                        f"User instructions for the main question (if any): {self.u_inst if self.u_inst else 'None'}\n\n"
                        "Report Requirements:\n"
                        "- Go into detail where relevant information is provided.\n"
                        "- Disregard irrelevant information from subtasks.\n"
                        "- Ensure clear structure and directness in addressing the main question.\n"
                        "- Preserve or synthesize citations (e.g., [1], [2]) if present in subtask answers."
                    )
                ]

                prompt_str = _serialize_lc_messages_for_preview(
                    alignment_prompt_messages
                )
                full_prompt_str_tokens = "\n".join(
                    [str(m.content) for m in alignment_prompt_messages]
                )
                prompt_tokens = self._get_token_count(full_prompt_str_tokens)
                summary_span.add_event(
                    "LLM Invocation Start (Alignment Summary)",
                    {
                        "llm_type": "main_llm",
                        "prompt_preview": prompt_str,
                        "estimated_prompt_tokens": prompt_tokens,
                    },
                )

                llm_for_alignment = self.llm.resolve("llm").value
                if not llm_for_alignment:
                    self.logger.error(
                        "LLM for alignment summary (main llm) is None. Cannot proceed with alignment."
                    )
                    summary_span.add_event(
                        "LLM Invocation Error (Alignment Summary)",
                        {"error": "LLM (main) is None"},
                    )
                else:
                    try:
                        aligned_response = llm_for_alignment.invoke(
                            alignment_prompt_messages
                        )
                        final_summary = str(aligned_response.content)
                        completion_tokens = self._get_token_count(final_summary)
                        summary_span.add_event(
                            "LLM Invocation End (Alignment Summary)",
                            {
                                "llm_type": "main_llm",
                                "response_preview": final_summary[:200],
                                "estimated_completion_tokens": completion_tokens,
                            },
                        )
                    except Exception as e_align:
                        self.logger.error(
                            f"LLM alignment failed: {e_align}. Using pre-alignment summary.",
                            exc_info=True,
                        )
                        summary_span.record_exception(
                            e_align, {"during": "alignment_summary_llm_invocation"}
                        )

            summary_span.add_event(
                "Summarization End",
                {
                    "final_summary_preview": final_summary[:200],
                    "aligned": self.options.align_summaries,
                },
            )
            return final_summary

    def _construct_subtask_to_json_prompt(
        self,
    ):
        json_schema_str = SplitTask.model_json_schema()
        try:
            schema_dict = json.loads(json.dumps(json_schema_str))
            if "required" in schema_dict and "evaluation" in schema_dict["required"]:
                schema_dict["required"].remove("evaluation")
            if (
                "properties" in schema_dict
                and "evaluation" in schema_dict["properties"]
            ):
                del schema_dict["properties"]["evaluation"]
            if "$defs" in schema_dict and "TaskEvaluation" in schema_dict["$defs"]:
                del schema_dict["$defs"]["TaskEvaluation"]
            simplified_schema_str = json.dumps(schema_dict)
        except Exception:
            simplified_schema_str = json.dumps(
                SplitTask.model_json_schema(exclude={"evaluation"})
            )

        prompt_content = (
            f"Now, format the subtask list strictly according to the following JSON schema. "
            f"The 'uuid' field is system-assigned; do not include it. Use the `depends_on` field to list UUIDs of tasks that must be completed before this new subtask can start.\n\n"
            f"Schema:\n```json\n{simplified_schema_str}\n```\n\n"
            f"Example (with dependency):\n"
            f"```json\n{{\n"
            f'  "needs_subtasks": true,\n'
            f'  "subtasks": [\n'
            f'    {{ "task": "First, research topic A.", "type": "research", "subtasks": 0, "allow_search": true, "allow_tools": true, "depends_on": [] }},\n'
            f'    {{ "task": "Then, using the findings from the research on topic A, write a summary.", "type": "text/reasoning", "subtasks": 0, "allow_search": false, "allow_tools": false, "depends_on": ["uuid-of-task-A"] }}\n'
            f'    {{ "task": "Finally, after the summary is written, analyze it for sentiment.", "type": "text/reasoning", "subtasks": 0, "allow_search": false, "allow_tools": false, "depends_on": ["2"] }}\n'
            f"  ]\n"
            f"}}\n```\n\n"
            f"Example (NO subtasks):\n"
            f"```json\n{{\n"
            f'  "needs_subtasks": false,\n'
            f'  "subtasks": []\n'
            f"}}\n```\n"
            "Provide ONLY the JSON object as your response, in a single line without any other text or explanations."
        )
        return HumanMessage(prompt_content)

    def _construct_verify_answer_prompt(self):
        json_schema_str = verification.model_json_schema()
        prompt_content = (
            f"Based on your evaluation, provide the outcome as a JSON object. Use the `successful` boolean field. "
            f"Schema:\n```json\n{json_schema_str}\n```\n"
            "Example:\n"
            f"```json\n{{\n"
            f'  "successful": true\n'
            f"}}\n```\n"
            "Provide ONLY the JSON object as your response."
        )
        return HumanMessage(prompt_content)
    
    async def run_structured_tool(self, tool: StructuredTool, args: Dict):
        ic(tool)
        return await tool.ainvoke(input=args)