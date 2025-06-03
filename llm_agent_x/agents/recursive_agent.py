import json
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Optional, List, Dict
from llm_agent_x.backend.exceptions import TaskFailedException
from opentelemetry import trace, context as otel_context
from pydantic import BaseModel, validator, ValidationError
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
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


class TaskObject(BaseModel):
    task: str
    type: Literal["research", "search", "basic", "text/reasoning"]
    subtasks: int
    allow_search: bool
    allow_tools: bool


class task(TaskObject):  # pylint: disable=invalid-name
    uuid: str


class verification(BaseModel):  # pylint: disable=invalid-name
    successful: bool


class SplitTask(BaseModel):
    needs_subtasks: bool
    subtasks: list[TaskObject]
    evaluation: Optional[TaskEvaluation] = None

    def __bool__(self):
        return self.needs_subtasks


class TaskContext(BaseModel):
    task: str
    result: Optional[str] = None
    siblings: List["TaskContext"] = []
    parent_context: Optional["TaskContext"] = None

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
    tool_llm: Any = None
    tools: list = []
    allow_search: bool = True
    allow_tools: bool = False
    tools_dict: dict = {}
    similarity_threshold: float = 0.8
    merger: Any = LLMMerger
    align_summaries: bool = True
    token_counter: Optional[Callable[[str], int]] = None  # For token counting

    class Config:
        arbitrary_types_allowed = True


def calculate_raw_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()


# Helper to serialize Langchain messages for tracing previews and token counting
def _serialize_lc_messages_for_preview(
    messages: List[BaseMessage], max_len: int = 500
) -> str:
    if not messages:
        return "[]"

    content_parts = []
    for msg in messages:
        role = msg.type.upper()  # e.g., HUMAN, AI, SYSTEM, TOOL
        content_str = str(msg.content)

        # Add tool call info if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            content_str += f" (Tool Calls: {len(msg.tool_calls)})"
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:  # For ToolMessage
            content_str += f" (Tool Call ID: {msg.tool_call_id})"

        content_parts.append(f"{role}: {content_str}")

    full_str = "\n".join(content_parts)
    if len(full_str) > max_len:
        return full_str[: max_len - 3] + "..."
    return full_str


class RecursiveAgent:
    def __init__(
        self,
        task: str,
        u_inst: str,
        tracer: Optional[trace.Tracer] = None, 
        tracer_span: Optional[trace.Span] = None, 
        uuid: str = str(uuid.uuid4()),
        agent_options: Optional[RecursiveAgentOptions] = None,
        allow_subtasks: bool = True,
        current_layer: int = 0,
        parent: Optional["RecursiveAgent"] = None,
        context: Optional[TaskContext] = None,
        siblings: Optional[List["RecursiveAgent"]] = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.RecursiveAgent.{uuid}")
        self.logger.info(
            f"Initializing RecursiveAgent for task: '{task}' at layer {current_layer} with UUID: {uuid}"
        )

        if agent_options is None:
            self.logger.info("No agent_options provided, using default configuration.")
            agent_options = RecursiveAgentOptions(
                task_limits=TaskLimit.from_constant(max_tasks=3, max_depth=2)
            )

        self.task = task
        self.u_inst = u_inst
        self.tracer = tracer if tracer else trace.get_tracer(__name__) 
        self.tracer_span = tracer_span # This is the PARENT span for the current agent's operations
        self.options = agent_options
        self.allow_subtasks = allow_subtasks
        self.llm = self.options.llm
        self.tool_llm = self.options.tool_llm
        self.tools = self.options.tools
        self.task_split_parser = JsonOutputParser(pydantic_object=SplitTask)
        self.task_verification_parser = JsonOutputParser(pydantic_object=verification)
        self.uuid = uuid
        self.current_layer = current_layer
        self.parent = parent
        self.siblings = siblings or []
        self.context = context or TaskContext(task=task)
        self.result: Optional[str] = None
        self.status: str = "pending"
        self.current_span: Optional[trace.Span] = None # This will be this agent's OWN main span

        # self.logger.debug(f"Agent initialized with options: {self.options.model_dump_json(exclude={'token_counter', 'pre_task_executed', 'on_task_executed', 'on_tool_call_executed'}, indent=2)}")
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
        parent_contexts = []
        current_p_context = self.context.parent_context
        while current_p_context:
            if current_p_context.result:
                parent_contexts.append(
                    {"task": current_p_context.task, "result": current_p_context.result}
                )
            current_p_context = current_p_context.parent_context
        parent_contexts.reverse()

        sibling_contexts = []
        for sibling_agent in self.siblings:
            if sibling_agent.result and sibling_agent != self:
                sibling_contexts.append(
                    {"task": sibling_agent.task, "result": sibling_agent.result}
                )
        return {
            "parent_contexts": parent_contexts,
            "sibling_contexts": sibling_contexts,
        }

    def _format_history_parts(
        self,
        context_info: dict,
        purpose: str,
        subtask_results_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        history = []
        if context_info["parent_contexts"]:
            history.append(f"Previous parent tasks and their results (for {purpose}):")
            for ctx in context_info["parent_contexts"]:
                history.append(
                    f"- Parent Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}..."
                )
        if context_info["sibling_contexts"]:
            history.append(
                f"\nParallel sibling tasks and their results (for {purpose}):"
            )
            for ctx in context_info["sibling_contexts"]:
                history.append(
                    f"- Sibling Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}..."
                )
        if purpose == "verification" and subtask_results_map:
            history.append(
                "\nThe current main task involved these subtasks and their results:"
            )
            for sub_task, sub_result in subtask_results_map.items():
                history.append(
                    f"- Subtask: {sub_task}\n  - Result: {str(sub_result)[:200]}..."
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
                if span:  # Ensure span exists before using it
                    span.record_exception(e)
                    self.status = "failed_critically"
                    span.set_attribute("agent.final_status", self.status)
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=str(e))
                    )
                raise TaskFailedException(
                    f"Agent run for '{self.task}' failed critically: {e}"
                ) from e

    def _run(self) -> str:
        span = self.current_span
        if not span:
            self.logger.warning(
                "_run called without an active self.current_span. Tracing will be limited for this operation."
            )
            # Fallback: Create a temporary span if absolutely necessary, though it might orphan this part of the trace
            # For robustness, one might add:
            # with self.tracer.start_as_current_span(f"Orphaned _run: {self.task[:50]}") as temp_span:
            #     return self.__execute_run_logic(temp_span)
            # For now, we proceed, but this indicates a potential setup issue if span is None.
            # Let's assume span exists for the main logic flow.
            # If not, the following add_event calls would fail.
            # A safer approach is to ensure self.current_span is always valid or handle its absence gracefully.
            # For this refactor, we assume `run` sets `self.current_span`.

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
            if span:
                span.add_event("Pre-Task Callback Executed")

        max_subtasks_for_this_layer = self._get_max_subtasks()
        if max_subtasks_for_this_layer == 0:
            if span:
                span.add_event("Executing as Single Task: Max Subtasks at Layer is 0")
            self.result = self._run_single_task()
            self.context.result = self.result
            try:
                self.verify_result(None)
            except TaskFailedException:
                if span:
                    span.add_event("Single Task Verification Failed, Attempting Fix")
                self._fix(None)

            if self.options.on_task_executed:
                if span:
                    span.add_event("On-Task-Executed Callback Executing")
                self.options.on_task_executed(
                    self.task,
                    self.uuid,
                    self.result,
                    self.parent.uuid if self.parent else None,
                )
                if span:
                    span.add_event("On-Task-Executed Callback Executed")
            return self.result

        while self.allow_subtasks:
            if span:
                span.add_event("Subtask Processing Loop Iteration Start")
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
                    self.result = self._run_single_task()
                    self.context.result = self.result
                    try:
                        self.verify_result(None)
                    except TaskFailedException:
                        if span:
                            span.add_event(
                                "High Similarity Single Task Verification Failed, Attempting Fix"
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

            split_task_result = self._split_task()

            if span:
                span.add_event(
                    "Task Splitting Outcome",
                    {
                        "needs_subtasks": split_task_result.needs_subtasks,
                        "generated_subtasks_count": len(split_task_result.subtasks),
                        "evaluation_score": (
                            split_task_result.evaluation.prompt_complexity_score[0]
                            if split_task_result.evaluation
                            else "N/A"
                        ),
                    },
                )

            if not split_task_result or not split_task_result.needs_subtasks:
                if span:
                    span.add_event(
                        "Executing as Single Task: Splitting Indicated No Subtasks"
                    )
                break

            limited_subtasks = split_task_result.subtasks[:max_subtasks_for_this_layer]
            if span:
                span.add_event(
                    "Subtasks Limited",
                    {
                        "original_count": len(split_task_result.subtasks),
                        "limited_count": len(limited_subtasks),
                        "max_allowed": max_subtasks_for_this_layer,
                    },
                )

            child_agents: List[RecursiveAgent] = []
            child_contexts_for_siblings: List[TaskContext] = []

            for subtask_obj in limited_subtasks:
                child_task_uuid = str(uuid.uuid4())
                child_context = TaskContext(
                    task=subtask_obj.task, parent_context=self.context
                )
                child_agent = RecursiveAgent(
                    task=subtask_obj.task,
                    u_inst=self.u_inst,
                    tracer=self.tracer,
                    tracer_span=span,
                    uuid=child_task_uuid,
                    agent_options=self.options,
                    allow_subtasks=True,
                    current_layer=self.current_layer + 1,
                    parent=self,
                    context=child_context,
                    siblings=child_agents[:],
                )
                child_agents.append(child_agent)
                child_contexts_for_siblings.append(child_context)

            for i, child_agent_to_update in enumerate(child_agents):
                child_agent_to_update.context.siblings = [
                    ctx for j, ctx in enumerate(child_contexts_for_siblings) if i != j
                ]

            subtask_results = []
            subtask_tasks_for_summary = []
            if span:
                span.add_event("Executing Child Agents", {"count": len(child_agents)})
            for i, child_agent in enumerate(child_agents):
                if span:
                    span.add_event(
                        f"Child Agent {i+1} Run Start",
                        {
                            "child_task": child_agent.task,
                            "child_uuid": child_agent.uuid,
                        },
                    )
                child_result = child_agent.run()
                subtask_results.append(
                    child_result
                    if child_result is not None
                    else "Error or no result from subtask."
                )
                subtask_tasks_for_summary.append(child_agent.task)
                if span:
                    span.add_event(
                        f"Child Agent {i+1} Run End",
                        {
                            "child_task": child_agent.task,
                            "child_result_preview": str(child_result)[:100],
                            "child_status": child_agent.status,
                        },
                    )

            if span:
                span.add_event("All Child Agents Completed")
            self.result = self._summarize_subtask_results(
                subtask_tasks_for_summary, subtask_results
            )
            self.context.result = self.result

            subtask_results_map = dict(zip(subtask_tasks_for_summary, subtask_results))
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

        if span:
            span.add_event("Executing as Single Task: Fallback or Subtask Loop Exit")
        self.result = self._run_single_task()
        self.context.result = self.result
        try:
            self.verify_result(None)
        except TaskFailedException:
            if span:
                span.add_event(
                    "Fallback Single Task Verification Failed, Attempting Fix"
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
        # Ensure agent_span is not None before attempting to use it for context
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

            system_prompt_content = (
                f"Your task is to answer the following question, using any tools that you deem necessary. "
                f"Make sure to phrase your search phrase in a way that it could be understood easily without context. "
                f"If you use the web search tool, make sure you include citations (just use a pair of square "
                f"brackets and a number in text, and at the end, include a citations section).{full_context_str}"
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

                # Use full history for token counting, but preview for logging
                full_prompt_str_for_tokens = "\n".join(
                    [str(m.content) for m in history]
                )
                prompt_preview_str = _serialize_lc_messages_for_preview(
                    history, max_len=1000
                )
                prompt_tokens = self._get_token_count(full_prompt_str_for_tokens)

                single_task_span.add_event(
                    "LLM Invocation Start (Tool LLM)",
                    {
                        "llm_type": "tool_llm",
                        "iteration": loop_count,
                        "prompt_messages_count": len(history),
                        "prompt_preview": prompt_preview_str,
                        "estimated_prompt_tokens": prompt_tokens,
                    },
                )

                current_llm_response = self.tool_llm.invoke(history)

                response_content_str = str(current_llm_response.content)
                completion_tokens = self._get_token_count(response_content_str)

                single_task_span.add_event(
                    "LLM Invocation End (Tool LLM)",
                    {
                        "llm_type": "tool_llm",
                        "iteration": loop_count,
                        "response_content_preview": response_content_str[:200],
                        "response_has_tool_calls": bool(
                            current_llm_response.tool_calls
                        ),
                        "tool_calls_count": len(current_llm_response.tool_calls or []),
                        "estimated_completion_tokens": completion_tokens,
                        "estimated_total_tokens": prompt_tokens + completion_tokens,
                    },
                )
                history.append(current_llm_response)

                if not current_llm_response.tool_calls:
                    final_result_content = str(current_llm_response.content)
                    single_task_span.add_event(
                        "LLM Responded Without Tool Calls - Final Answer",
                        {"final_answer_preview": final_result_content[:200]},
                    )
                    break

                tool_messages_for_this_turn = []
                guidance_for_llm_reprompt = []
                any_tool_requires_llm_replan = False

                for tool_call in current_llm_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    tool_executed_successfully_this_iteration = False
                    this_tool_call_needs_llm_replan = False
                    human_message_for_this_tool_failure = None
                    current_tool_output_payload = None

                    single_task_span.add_event(
                        "Tool Call Processing Start",
                        {
                            "tool_name": tool_name,
                            "tool_args": json.dumps(tool_args, default=str),
                            "tool_call_id": tool_call_id,
                        },
                    )

                    try:
                        if tool_name not in self.options.tools_dict:
                            raise KeyError(f"Tool '{tool_name}' not found.")
                        tool_to_execute = self.options.tools_dict[tool_name]

                        if tool_name == "search" and "query" in tool_args:
                            query = tool_args["query"]
                            sim = calculate_raw_similarity(query, self.task)
                            single_task_span.add_event(
                                "Search Tool Query Similarity Check",
                                {"query": query, "similarity_score": sim},
                            )
                            if sim < (self.options.similarity_threshold / 5):
                                error_detail = f"Search query '{query}' too dissimilar (score: {sim:.2f}). Revise."
                                current_tool_output_payload = {"error": error_detail}
                                human_message_for_this_tool_failure = error_detail
                                this_tool_call_needs_llm_replan = True
                                single_task_span.add_event(
                                    "Search Tool Query Dissimilar",
                                    {"error_detail": error_detail},
                                )
                            else:
                                current_tool_output_payload = tool_to_execute(
                                    **tool_args
                                )
                                tool_executed_successfully_this_iteration = True
                        else:
                            current_tool_output_payload = tool_to_execute(**tool_args)
                            tool_executed_successfully_this_iteration = True
                    except Exception as e:
                        error_msg_str = (
                            f"Error with tool {tool_name} (ID: {tool_call_id}): {e}"
                        )
                        self.logger.error(error_msg_str, exc_info=True)
                        current_tool_output_payload = {
                            "error": "Tool execution failed",
                            "details": error_msg_str,
                        }
                        human_message_for_this_tool_failure = (
                            f"Error with '{tool_name}': {e}. Adjust plan."
                        )
                        this_tool_call_needs_llm_replan = True
                        single_task_span.record_exception(
                            e,
                            attributes={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                            },
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

                    single_task_span.add_event(
                        "Tool Call Processing End",
                        {
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "tool_response_preview": tool_message_content_final_str[
                                :200
                            ],
                            "tool_success": tool_executed_successfully_this_iteration,
                            "requires_llm_replan": this_tool_call_needs_llm_replan,
                        },
                    )

                    if self.options.on_tool_call_executed:
                        self.options.on_tool_call_executed(
                            self.task,
                            self.uuid,
                            tool_name,
                            tool_args,
                            current_tool_output_payload,
                            tool_executed_successfully_this_iteration,
                            tool_call_id,
                        )

                    if this_tool_call_needs_llm_replan:
                        any_tool_requires_llm_replan = True
                        if human_message_for_this_tool_failure:
                            guidance_for_llm_reprompt.append(
                                human_message_for_this_tool_failure
                            )

                history.extend(tool_messages_for_this_turn)

                if any_tool_requires_llm_replan:
                    replan_message_content = (
                        "Review tool issues and adjust plan:\n"
                        + "\n".join(guidance_for_llm_reprompt)
                        if guidance_for_llm_reprompt
                        else "One or more tool calls had issues. Please review the tool responses and adjust your plan."
                    )
                    history.append(HumanMessage(content=replan_message_content))
                    single_task_span.add_event(
                        "LLM Re-Plan Requested",
                        {"reason": replan_message_content[:200]},
                    )
                    continue

            if loop_count >= max_loops:
                single_task_span.add_event(
                    "Max Tool Loop Iterations Reached", {"max_loops": max_loops}
                )
                if history and isinstance(
                    history[-1], AIMessage
                ):  # Check if history is not empty and last is AIMessage
                    final_result_content = str(history[-1].content)

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

            split_span.add_event(
                "Task Splitting Start",
                {
                    "task": self.task,
                    "max_subtasks_allowed": max_subtasks,
                    "context_for_splitting_preview": task_history_for_splitting[:300],
                },
            )

            system_msg_content = (
                f"Split this task into smaller, parallelizable subtasks only if it's complex and benefits from it. You can create up to {max_subtasks} subtasks. "
                "Do not attempt to answer the main task yourself.\n\n"
                "Consider this contextual history (parent tasks, sibling tasks already in progress):\n"
                f"{task_history_for_splitting}\n\n"
                "When deciding on subtasks for the current task ('{self.task}'), ensure they:\n"
                "1. Avoid redundant overlap with tasks in the history.\n"
                "2. Logically break down the current task if it's too broad for a single LLM response.\n"
                "3. Are distinct and can be worked on independently if possible.\n"
                f"4. Do not exceed {max_subtasks} subtasks in total for this split.\n\n"
            )
            if self.u_inst:
                system_msg_content += f"User-provided instructions for the main task (consider these when splitting):\n{self.u_inst}\n"

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

            # LLM Call 1: Initial subtask generation
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
            response1 = self.llm.invoke(primed_hist_1)  # Pass primed history
            response_content_1 = "1. " + str(
                response1.content
            )  # Prepend the prime for full response
            completion_tokens_1 = self._get_token_count(
                str(response1.content)
            )  # Count only new content
            split_span.add_event(
                "LLM Invocation End (Splitting - Initial List)",
                {
                    "llm_type": "main_llm",
                    "response_preview": response_content_1[:200],
                    "estimated_completion_tokens": completion_tokens_1,
                },
            )
            split_msgs_hist.append(
                AIMessage(content=response_content_1)
            )  # Add the full AI response including prime

            # LLM Call 2: Refine subtasks
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
            response2 = self.llm.invoke(
                hist_for_refine
            )  # Use the history including the refine human message
            completion_tokens_2 = self._get_token_count(str(response2.content))
            split_span.add_event(
                "LLM Invocation End (Splitting - Refine List)",
                {
                    "llm_type": "main_llm",
                    "response_preview": str(response2.content)[:200],
                    "estimated_completion_tokens": completion_tokens_2,
                },
            )
            split_msgs_hist.append(response2)  # Add refined AI response

            # LLM Call 3: Format to JSON
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
            structured_response_msg = self.llm.invoke(hist_for_json)
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
                    self.logger.error(
                        f"Unexpected type from task_split_parser: {type(parsed_output_from_llm)}."
                    )
                    raise TypeError(
                        f"Unexpected type from parser: {type(parsed_output_from_llm)}"
                    )

                split_span.add_event(
                    "Task Splitting JSON Parsed",
                    {
                        "parsed_needs_subtasks": split_task_result.needs_subtasks,
                        "parsed_subtasks_count": len(split_task_result.subtasks),
                    },
                )

            except (ValidationError, OutputParserException, TypeError) as e:
                self.logger.error(
                    f"Error parsing LLM JSON for task splitting: {e}. LLM content: {str(structured_response_msg.content)[:500]}",
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
                if len(split_task_result.subtasks) < original_subtask_count:
                    split_span.add_event(
                        "Subtasks Trimmed to Max Allowed",
                        {
                            "original_count": original_subtask_count,
                            "trimmed_count": len(split_task_result.subtasks),
                        },
                    )
                split_task_result.needs_subtasks = bool(split_task_result.subtasks)

            split_span.add_event(
                "Task Splitting Finished",
                {
                    "final_needs_subtasks": split_task_result.needs_subtasks,
                    "final_subtasks_count": len(split_task_result.subtasks),
                    "subtasks_preview": json.dumps(
                        [
                            st.model_dump(exclude={"evaluation"})
                            for st in split_task_result.subtasks
                        ],
                        default=str,
                    )[:500],
                },
            )
            return split_task_result

    def _verify_result(
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
                "(parent tasks, sibling tasks, and any subtasks that contributed to this result).\n\n"
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
            verify_msgs_hist_initial = [
                SystemMessage(content=system_msg_content),
                HumanMessage(content=human_msg_content),
            ]

            # primed_ai_msg_content = f'```json\n{{\n  "successful": '
            verify_msgs_hist_for_llm = verify_msgs_hist_initial + [
                # AIMessage(content=primed_ai_msg_content)
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
            structured_response_msg = self.llm.invoke(
                verify_msgs_hist_for_llm
            )  # LLM completes the primed JSON

            llm_completion_part = str(structured_response_msg.content)
            full_llm_response_content_for_parser = (
                llm_completion_part
            )
            completion_tokens = self._get_token_count(llm_completion_part)
            verify_span.add_event(
                "LLM Invocation End (Verification)",
                {
                    "llm_type": "main_llm",
                    "response_preview": full_llm_response_content_for_parser[:200],
                    "estimated_completion_tokens": completion_tokens,
                },
            )

            verification_obj: Optional[verification] = None
            try:
                # Pass the full content, assuming parser can handle the primed start + completion
                # Or, if parser expects only the completion part that makes it valid JSON:
                # parsed_output = self.task_verification_parser.parse(f'{{"successful": {llm_completion_part}')
                # For Langchain's JsonOutputParser, passing the AIMessage containing the JSON is standard.
                # The AIMessage from LLM already contains the completed JSON.
                # However, since we primed, the `structured_response_msg` might only be the boolean value.
                # Let's reconstruct the full message as Langchain JsonOutputParser expects a message
                # or a string that is valid JSON.

                # Create a new AIMessage with the fully formed JSON string for the parser
                full_json_ai_message = AIMessage(
                    content=full_llm_response_content_for_parser
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
                    f"Error parsing LLM JSON for verification: {e}. LLM content: {full_llm_response_content_for_parser[:500]}",
                    exc_info=True,
                )
                verify_span.record_exception(
                    e,
                    attributes={
                        "llm_content_preview": full_llm_response_content_for_parser[
                            :200
                        ]
                    },
                )
                return False
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during verification finalization: {e}. LLM content: {full_llm_response_content_for_parser[:500]}",
                    exc_info=True,
                )
                verify_span.record_exception(
                    e,
                    attributes={
                        "llm_content_preview": full_llm_response_content_for_parser[
                            :200
                        ]
                    },
                )
                return False

    def verify_result(self, subtask_results_map: Optional[Dict[str, str]] = None):
        agent_span = self.current_span  # This is the main agent span
        successful = self._verify_result(
            subtask_results_map
        )  # _verify_result creates its own child span

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
            original_task_str = self.task
            failed_result_str = str(
                self.result if self.result is not None else "No result was produced."
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
            if self.u_inst:
                fix_instructions_parts.append(
                    f"\nOriginal user instructions for the task were: {self.u_inst}"
                )
            fix_instructions_parts.append(
                "\nYour current objective is to FIX this failure. Analyze the original task, the failed outcome, any subtask information, and original user instructions. "
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
                task=original_task_str, parent_context=self.context.parent_context
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
                parent=self,
                context=fixer_agent_context,
                siblings=[],
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

    def _get_max_subtasks(self) -> int:
        if self.current_layer >= len(self.options.task_limits.limits):
            return 0
        return self.options.task_limits.limits[self.current_layer]

    def _summarize_subtask_results(
        self, tasks: List[str], subtask_results: List[str]
    ) -> str:
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

            if not subtask_results:  # Check if list itself is empty
                summary_span.add_event(
                    "Summarization End: No Results to Summarize (Input List Empty)"
                )
                return "No subtask results were generated to summarize."

            documents_to_merge = [
                f"SUBTASK QUESTION: {q}\n\nSUBTASK ANSWER:\n{a}"
                for q, a in zip(tasks, subtask_results)
                if a is not None
            ]  # Filter out None answers

            if not documents_to_merge:  # Check if, after filtering None, list is empty
                summary_span.add_event(
                    "Summarization End: All Subtasks Yielded Empty/No Results"
                )
                return "All subtasks yielded empty or no results."

            merged_content = []
            for document in documents_to_merge:
                if self._get_token_count(document) > 5000:
                    num_sentences = int(
                        len(document.split()) / 5000 * self.options.summary_sentences_factor
                    )
                    summary_span.add_event(
                        "Summarizing Long Document",
                        {
                            "document_length": len(document),
                            "num_sentences": num_sentences,
                        },
                    )
                    merged_content.append(summarize(document, num_sentences))
                else:
                    merged_content.append(document)

            merge_options = MergeOptions(
                llm=self.llm, context_window=15000
            )  # Assuming self.llm is not None
            merger = self.options.merger(merge_options)
            merged_content_str = "\n".join(merged_content)
            merged_content_str = merger.merge_documents([merged_content_str])
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
                # Ensure self.llm is not None before invoking
                if not self.llm:
                    self.logger.error(
                        "LLM for alignment summary is None. Cannot proceed with alignment."
                    )
                    summary_span.add_event(
                        "LLM Invocation Error (Alignment Summary)",
                        {"error": "LLM is None"},
                    )
                    # Keep final_summary as merged_content if alignment LLM is missing
                else:
                    aligned_response = self.llm.invoke(alignment_prompt_messages)
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
            simplified_schema_str = json.dumps(json_schema_str)

        prompt_content = (
            f"Now, format the subtask list (or indicate no subtasks are needed) strictly according to the following JSON schema. "
            f"The 'evaluation' field is optional and you should omit it (it will be added programmatically).\n"
            f"Schema:\n```json\n{simplified_schema_str}\n```\n"
            f"Example (needs subtasks):\n"
            f"```json\n{{\n"
            f'  "needs_subtasks": true,\n'
            f'  "subtasks": [\n'
            f'    {{ "task": "Subtask 1...", "type": "research", "subtasks": 0, "allow_search": true, "allow_tools": true }},\n'
            f'    {{ "task": "Subtask 2...", "type": "basic", "subtasks": 0, "allow_search": false, "allow_tools": false }}\n'
            f"  ]\n"
            f"}}\n```\n"
            f"Example (NO subtasks):\n"
            f"```json\n{{\n"
            f'  "needs_subtasks": false,\n'
            f'  "subtasks": []\n'
            f"}}\n```\n"
            "Provide ONLY the JSON object as your response, without any other text or explanations. Also, can you do it in a single line?"
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
