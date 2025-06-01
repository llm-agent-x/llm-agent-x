import json
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Optional, List
from llm_agent_x.backend.exceptions import TaskFailedException
from opentelemetry import trace, context as otel_context  # Modified import
from pydantic import BaseModel, validator, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import (
    OutputParserException,
)  # Import for specific exception handling
from llm_agent_x.backend.mergers.LLMMerger import MergeOptions, LLMMerger
from icecream import ic
from llm_agent_x.complexity_model import TaskEvaluation, evaluate_prompt
import logging

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
    """Configuration for task limits at each layer"""

    @staticmethod
    def constant(max_tasks: int, max_depth: int) -> List[int]:
        """Create a constant limit configuration"""
        return [max_tasks] * max_depth

    @staticmethod
    def array(task_limits: List[int]) -> List[int]:
        """Use an explicit array of limits"""
        return task_limits

    @staticmethod
    def falloff(
        initial_tasks: int, max_depth: int, falloff_func: Callable[[int], int]
    ) -> List[int]:
        """Create limits using a falloff function"""
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


class SplitTask(BaseModel):
    needs_subtasks: bool
    subtasks: list[TaskObject]
    evaluation: Optional[TaskEvaluation] = None  # Set default to None

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
    on_tool_call_executed: Any = (
        None  # Expects: (task, uuid, tool_name, tool_args, tool_response, success, tool_call_id)
    )
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

    class Config:
        arbitrary_types_allowed = True


def calculate_raw_similarity(text1: str, text2: str) -> float:
    """
    Calculate the similarity ratio between two texts using SequenceMatcher.
    Returns a float between 0 and 1, where 1 means the texts are identical.
    """
    return SequenceMatcher(None, text1, text2).ratio()


class RecursiveAgent:
    def __init__(
        self,
        task: str,
        u_inst: str,
        tracer: trace.Tracer = None,
        tracer_span: trace.Span = None,
        uuid: str = str(uuid.uuid4()),  # pylint: disable=redefined-outer-name
        agent_options: RecursiveAgentOptions = None,
        allow_subtasks: bool = True,
        current_layer: int = 0,
        parent: Optional["RecursiveAgent"] = None,
        context: Optional[TaskContext] = None,
        siblings: List["RecursiveAgent"] = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.RecursiveAgent.{uuid}")
        self.logger.info(
            f"Initializing RecursiveAgent for task: '{task}' at layer {current_layer} with UUID: {uuid}"
        )

        if agent_options is None:
            self.logger.info("No agent_options provided, using default configuration.")
            # Default configuration: 3 tasks per layer, max 2 layers
            agent_options = RecursiveAgentOptions(
                task_limits=TaskLimit.from_constant(max_tasks=3, max_depth=2)
            )

        self.task = task
        self.u_inst = u_inst
        self.tracer = tracer
        self.tracer_span = tracer_span
        self.options = agent_options
        self.allow_subtasks = allow_subtasks
        self.llm = self.options.llm
        self.tool_llm = self.options.tool_llm
        self.tools = self.options.tools
        self.task_split_parser = JsonOutputParser(pydantic_object=SplitTask)
        self.uuid = uuid
        self.current_layer = current_layer
        self.parent = parent
        self.siblings = siblings or []
        self.context = context or TaskContext(task=task)
        self.result = None
        self.current_span = None  # Initialize current_span

        self.logger.debug(f"Agent initialized with options: {agent_options}")

    def _build_context_information(self) -> dict:
        """
        Build context information including parent chain and sibling tasks
        """
        self.logger.debug("Building context information.")
        # Get parent context chain
        parent_contexts = []
        current_context = self.context.parent_context
        while current_context:
            if current_context.result:
                parent_contexts.append(
                    {"task": current_context.task, "result": current_context.result}
                )
            current_context = current_context.parent_context

        # Get sibling contexts
        sibling_contexts = []
        for sibling in self.siblings:
            if sibling.result and sibling != self:  # type: ignore
                sibling_contexts.append(
                    {"task": sibling.task, "result": sibling.result}  # type: ignore
                )

        context_info = {
            "parent_contexts": parent_contexts,
            "sibling_contexts": sibling_contexts,
        }
        self.logger.debug(f"Context information built: {context_info}")
        return context_info

    def _build_task_split_history(self) -> str:
        """
        Build a string representation of the task history for context
        """
        self.logger.debug("Building task history prompt.")
        context_info = self._build_context_information()
        history = []

        # Add parent tasks
        if context_info["parent_contexts"]:
            history.append("Previous parent tasks:")
            for ctx in context_info["parent_contexts"]:
                history.append(f"- {ctx['task']}")

        # Add sibling tasks
        if context_info["sibling_contexts"]:
            history.append("\nParallel tasks already being worked on:")
            for ctx in context_info["sibling_contexts"]:
                history.append(f"- {ctx['task']}")

        task_history_str = "\n".join(history)
        self.logger.debug(f"Task history string built: {task_history_str}")
        return task_history_str
    def _build_task_verify_history(self) -> str:
        """
        Build a string representation of the prompt to verify the result of the agent before closing
        """
        self.logger.debug("Building task verification prompt.")
        context_info = self._build_context_information()
        history = []

        # Add parent tasks
        if context_info["parent_contexts"]:
            history.append("Previous parent tasks:")
            for ctx in context_info["parent_contexts"]:
                history.append(f"- {ctx['task']}")

        # Add sibling tasks
        if context_info["sibling_contexts"]:
            history.append("\nParallel tasks already being worked on:")
            for ctx in context_info["sibling_contexts"]:
                history.append(f"- {ctx['task']}")

        task_history_str = "\n".join(history)
        self.logger.debug(f"Task history string built: {task_history_str}")
        return task_history_str

    # In class RecursiveAgent:
    def run(self):
        self.logger.info(f"Starting run for task: '{self.task}'")
        if self.tracer and self.tracer_span:
            self.logger.debug("Tracer and tracer_span available, creating new span.")
            parent_otel_ctx = trace.set_span_in_context(self.tracer_span)
            with self.tracer.start_as_current_span(
                f"Execute Task: {self.task}",
                context=parent_otel_ctx,
            ) as span:
                self.current_span = span
                self.logger.debug(f"Span created: {span.get_span_context().span_id}")
                span.add_event(
                    "Start", {"layer": self.current_layer, "task": self.task}
                )
                result = self._run(span=span)
                self.logger.info(
                    f"Run finished for task: '{self.task}'. Result: {str(result)[:100]}..."
                )
                return result
        else:
            self.logger.debug("No tracer or tracer_span, running without new span.")
            result = self._run()
            self.logger.info(
                f"Run finished for task: '{self.task}'. Result: {str(result)[:100]}..."
            )
            return result

    def _run(self, span=None) -> str:
        self.logger.info(
            f"Starting _run for task: '{self.task}' at layer {self.current_layer}"
        )
        if span:
            span.add_event(
                "run",
                {
                    "task": self.task,
                },
            )
        if self.options.pre_task_executed:
            self.logger.debug("Executing pre_task_executed callback.")
            self.options.pre_task_executed(
                task=self.task,
                uuid=self.uuid,
                parent_agent_uuid=(self.parent.uuid if self.parent else None),
            )

        max_subtasks = self._get_max_subtasks()
        if max_subtasks == 0:
            self.logger.info(
                f"Max subtasks is 0 for layer {self.current_layer}. Executing as single task."
            )
            self.result = self._run_single_task(span=span)
            self.context.result = self.result
            if self.options.on_task_executed:
                self.logger.debug(
                    "Executing on_task_executed callback for single task."
                )
                self.options.on_task_executed(
                    self.task,
                    self.uuid,
                    self.result,
                    self.parent.uuid if self.parent else None,
                )
            self.logger.info(
                f"_run finished for single task '{self.task}'. Result: {str(self.result)[:100]}..."
            )
            return self.result

        while self.allow_subtasks:
            self.logger.debug("Subtasks allowed, entering subtask loop.")
            if self.parent:
                similarity = calculate_raw_similarity(self.task, self.parent.task)
                self.logger.debug(
                    f"Similarity with parent task '{self.parent.task}': {similarity:.2f}"
                )
                if similarity >= self.options.similarity_threshold:
                    self.logger.info(
                        f"Task similarity with parent is high ({similarity:.2f}), executing as single task."
                    )
                    if self.current_span:
                        self.current_span.add_event(
                            f"Task similarity with parent is high ({similarity:.2f}), executing as single task."
                        )
                    self.result = self._run_single_task(span=span)
                    self.context.result = self.result
                    self.logger.info(
                        f"_run finished for high-similarity task '{self.task}'. Result: {str(self.result)[:100]}..."
                    )
                    return self.result

            self.logger.debug("Splitting task.")
            split_task_result = self._split_task()
            if span:
                span.set_attribute("task_split_result", ic.format(split_task_result))

            if not split_task_result or not split_task_result.needs_subtasks:
                self.logger.info(
                    "Task does not need subtasks or splitting failed. Breaking subtask loop."
                )
                break

            limited_subtasks = split_task_result.subtasks[:max_subtasks]
            self.logger.info(
                f"Limited subtasks to {len(limited_subtasks)} based on max_subtasks={max_subtasks}."
            )

            child_agents: List[RecursiveAgent] = []
            for subtask_obj in limited_subtasks:
                self.logger.info(
                    f"Creating child agent for subtask: '{subtask_obj.task}'"
                )
                child_context = TaskContext(
                    task=subtask_obj.task, parent_context=self.context
                )
                # Determine subtask_uuid
                subtask_uuid_val = getattr(subtask_obj, "uuid", str(uuid.uuid4()))

                child_agent = RecursiveAgent(
                    task=subtask_obj.task,
                    u_inst=self.u_inst,
                    tracer=self.tracer,
                    tracer_span=span,  # Use current task's span as parent for child's span
                    uuid=subtask_uuid_val,
                    agent_options=self.options,
                    allow_subtasks=True,  # Child tasks can be further split by default
                    current_layer=self.current_layer + 1,
                    parent=self,
                    context=child_context,
                    siblings=child_agents,  # Pass currently created children as siblings to next child
                )
                child_agents.append(child_agent)

            subtask_results = []
            subtask_tasks_for_summary = []
            self.logger.info(f"Running {len(child_agents)} child agents.")
            for child_agent in child_agents:
                self.logger.debug(f"Running child agent for task: '{child_agent.task}'")
                result = (
                    child_agent.run()
                )  # This will create its own span as a child of 'span'
                child_agent.result = result
                child_agent.context.result = result
                subtask_results.append(result)
                subtask_tasks_for_summary.append(child_agent.task)
                self.logger.debug(
                    f"Child agent for task '{child_agent.task}' finished. Result: {str(result)[:100]}..."
                )
            
            # TODO: Check if task is complete, and if not, fix/complete with more ai

            self.logger.info(f"Checking if we have enough to complete the task")

            try:
                self.fix_task(dict(zip(subtask_tasks_for_summary, subtask_results)))
                self.status = "succeded"
            except TaskFailedException:
                self.status = "failed"
            
            self.logger.info("Summarizing subtask results.")
            self.result = self._summarize_subtask_results(
                subtask_tasks_for_summary, subtask_results
            )
            self.context.result = self.result

            if self.options.on_task_executed:
                self.logger.debug("Executing on_task_executed callback after subtasks.")
                self.options.on_task_executed(
                    self.task,
                    self.uuid,
                    self.result,
                    self.parent.uuid if self.parent else None,
                )
            self.logger.info(
                f"_run finished after subtasks for '{self.task}'. Result: {str(self.result)[:100]}..."
            )
            return self.result

        self.logger.info(
            "Exited subtask loop or subtasks not allowed. Executing as single task."
        )
        self.result = self._run_single_task(span=span)
        self.context.result = self.result
        if self.options.on_task_executed:
            self.logger.debug(
                "Executing on_task_executed callback for final single task execution."
            )
            self.options.on_task_executed(
                self.task,
                self.uuid,
                self.result,
                self.parent.uuid if self.parent else None,
            )
        self.logger.info(
            f"_run finished for final single task '{self.task}'. Result: {str(self.result)[:100]}..."
        )
        return self.result

    def _run_single_task(self, span: Optional[trace.Span] = None) -> str:
        self.logger.info(f"Running single task: '{self.task}'")
        context_info = self._build_context_information()

        context_str = ""
        if context_info["parent_contexts"]:
            context_str += "\n\nParent task history:\n" + "\n".join(
                f"Parent task: {ctx['task']}\nResult: {ctx['result']}\n"
                for ctx in context_info["parent_contexts"]
            )

        if context_info["sibling_contexts"]:
            context_str += "\n\nParallel tasks in progress:\n" + "\n".join(
                f"Task: {ctx['task']}\nResult: {ctx['result']}\n"
                for ctx in context_info["sibling_contexts"]
            )
        self.logger.debug(f"Context string for single task: {context_str}")

        history = [
            SystemMessage(
                f"Your task is to answer the following question, using any tools that you deem necessary. "
                f"Make sure to phrase your search phrase in a way that it could be understood easily without context. "
                f"If you use the web search tool, make sure you include citations (just use a pair of square "
                f"brackets and a number in text, and at the end, include a citations section).{context_str}"
            ),
            HumanMessage(
                self.task
                + "\n\nApply the distributive property to any tool calls. for instance if you need to search for 3 related things, make 3 separate calls to the search tool, because that will yield better results."
                + (f"Use this info to help you: {self.u_inst}" if self.u_inst else "")
            ),
        ]
        self.logger.debug(f"Initial history for single task LLM: {history}")

        # Ensure self.tracer is valid before using it
        tracer_to_use = self.tracer if self.tracer else trace.get_tracer(__name__)

        # Use the passed span as parent context for the single task execution span
        parent_otel_ctx_for_single_task = (
            trace.set_span_in_context(span) if span else otel_context.get_current()
        )

        with tracer_to_use.start_as_current_span(
            "Single Task LLM Interaction", context=parent_otel_ctx_for_single_task
        ) as interaction_span:
            interaction_span.set_attribute("task", self.task)

            while True:
                self.logger.info("Invoking tool_llm for single task.")
                current_llm_response = self.tool_llm.invoke(history)
                self.logger.debug(
                    f"LLM response content: {current_llm_response.content}"
                )
                self.logger.debug(
                    f"LLM response tool calls: {current_llm_response.tool_calls}"
                )

                interaction_span.add_event(
                    "LLM Invoked",
                    {
                        "history_length": len(history),
                        "response_content_preview": str(current_llm_response.content)[
                            :100
                        ],
                        "tool_calls_count": len(current_llm_response.tool_calls or []),
                    },
                )

                history.append(current_llm_response)

                if not current_llm_response.tool_calls:
                    self.logger.info(
                        "No tool calls in LLM response. Returning content."
                    )
                    interaction_span.set_attribute(
                        "final_result_preview", str(current_llm_response.content)[:100]
                    )
                    return current_llm_response.content

                tool_messages_for_this_turn = []
                guidance_for_llm_reprompt = []
                any_tool_requires_llm_replan = False

                self.logger.debug(
                    f"Processing {len(current_llm_response.tool_calls)} tool calls."
                )
                for tool_call in current_llm_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    self.logger.info(
                        f"Processing tool call: {tool_name} with args: {tool_args}, ID: {tool_call_id}"
                    )

                    current_tool_output_payload: Any = None
                    tool_executed_successfully_this_iteration = False
                    this_tool_call_needs_llm_replan = False
                    human_message_for_this_tool_failure: Optional[str] = None

                    with tracer_to_use.start_as_current_span(
                        f"Tool Call: {tool_name}",
                        context=trace.set_span_in_context(interaction_span),
                    ) as tool_span:
                        tool_span.set_attributes(
                            {
                                "tool.name": tool_name,
                                "tool.args": json.dumps(tool_args),
                                "tool.call_id": tool_call_id,
                            }
                        )

                        try:
                            if tool_name not in self.options.tools_dict:
                                raise KeyError(f"Tool '{tool_name}' not found.")

                            tool_to_execute = self.options.tools_dict[tool_name]

                            if tool_name in [
                                "search",
                                "web_search",
                                "brave_web_search",
                            ]:
                                if "query" not in tool_args:
                                    error_detail = f"Search tool '{tool_name}' (ID: {tool_call_id}) called without a 'query' argument. Please provide a query."
                                    self.logger.error(error_detail)
                                    current_tool_output_payload = {
                                        "error": "Missing query argument",
                                        "details": error_detail,
                                    }
                                    human_message_for_this_tool_failure = error_detail
                                    this_tool_call_needs_llm_replan = True
                                    tool_span.set_attribute(
                                        "tool.error", "Missing query argument"
                                    )
                                else:
                                    query = tool_args["query"]
                                    similarity = calculate_raw_similarity(
                                        query, self.task
                                    )
                                    tool_span.set_attributes(
                                        {
                                            "search.query": query,
                                            "search.similarity_to_task": similarity,
                                        }
                                    )
                                    self.logger.debug(
                                        f"Search query: '{query}', similarity to task '{self.task}': {similarity:.2f} for tool_call_id {tool_call_id}"
                                    )

                                    if similarity < (
                                        self.options.similarity_threshold / 5
                                    ):
                                        error_detail = (
                                            f"The search query '{query}' for tool '{tool_name}' (ID: {tool_call_id}) is not sufficiently related to the overall task: '{self.task}'. "
                                            f"Similarity score: {similarity:.2f}. "
                                            f"Please regenerate this specific query to be more semantically similar to the task, "
                                            f"or explain why this query is relevant. Other tool calls in this batch (if any) are being processed independently."
                                        )
                                        self.logger.warning(
                                            f"Search query needs revision for ID {tool_call_id}: {error_detail}"
                                        )
                                        current_tool_output_payload = {
                                            "error": "Query needs revision due to low similarity.",
                                            "tool_call_id": tool_call_id,
                                            "query_provided": query,
                                            "task_context": self.task,
                                            "similarity_score": f"{similarity:.2f}",
                                            "required_action": "Revise this specific query or justify its relevance.",
                                        }
                                        human_message_for_this_tool_failure = (
                                            error_detail
                                        )
                                        this_tool_call_needs_llm_replan = True
                                        tool_span.set_attribute(
                                            "tool.error", "Query needs revision"
                                        )
                                    else:
                                        self.logger.debug(
                                            f"Search query '{query}' (ID: {tool_call_id}) is good. Executing tool."
                                        )
                                        current_tool_output_payload = tool_to_execute(
                                            **tool_args
                                        )
                                        tool_executed_successfully_this_iteration = True
                            else:
                                self.logger.debug(
                                    f"Executing non-search tool: '{tool_name}' (ID: {tool_call_id})"
                                )
                                current_tool_output_payload = tool_to_execute(
                                    **tool_args
                                )
                                tool_executed_successfully_this_iteration = True

                        except KeyError as e:
                            error_msg_str = f"Tool '{tool_name}' (ID: {tool_call_id}) not found. Available: {list(self.options.tools_dict.keys())}. Error: {str(e)}"
                            self.logger.error(error_msg_str, exc_info=True)
                            current_tool_output_payload = {
                                "error": "Tool not found",
                                "details": error_msg_str,
                            }
                            human_message_for_this_tool_failure = (
                                f"You attempted to use a tool named '{tool_name}' (ID: {tool_call_id}) which is not available. "
                                f"Please choose from: {list(self.options.tools_dict.keys())} or adjust your plan for this call."
                            )
                            this_tool_call_needs_llm_replan = True
                            tool_span.record_exception(e)
                            tool_span.set_attribute("tool.error", "Tool not found")
                        except Exception as e:
                            error_msg_str = f"Error executing tool {tool_name} (ID: {tool_call_id}) with {tool_args}: {str(e)}"
                            self.logger.error(error_msg_str, exc_info=True)
                            current_tool_output_payload = {
                                "error": "Tool execution failed",
                                "details": error_msg_str,
                            }
                            human_message_for_this_tool_failure = (
                                f"Error executing '{tool_name}' (ID: {tool_call_id}): {str(e)}. Args: {tool_args}. "
                                f"Please check args, try a different approach for this call, or clarify."
                            )
                            this_tool_call_needs_llm_replan = True
                            tool_span.record_exception(e)
                            tool_span.set_attribute("tool.error", "Execution failed")

                        tool_message_content_final_str: str
                        try:
                            tool_message_content_final_str = json.dumps(
                                current_tool_output_payload
                            )
                        except TypeError as serialization_error:
                            self.logger.error(
                                f"Error serializing output for '{tool_name}' (ID: {tool_call_id}): {serialization_error}",
                                exc_info=True,
                            )
                            tool_span.record_exception(serialization_error)
                            error_detail_serialization = f"Tool '{tool_name}' (ID: {tool_call_id}) output unserializable: {serialization_error}. Output (partial): {str(current_tool_output_payload)[:200]}"
                            current_tool_output_payload = {
                                "error": "Tool output serialization error",
                                "details": error_detail_serialization,
                            }
                            tool_message_content_final_str = json.dumps(
                                current_tool_output_payload
                            )
                            tool_executed_successfully_this_iteration = False
                            if not this_tool_call_needs_llm_replan:
                                ser_guidance = f"Tool '{tool_name}' (ID: {tool_call_id}) data unprocessable (serialization issue). Check format or retry this call."
                                human_message_for_this_tool_failure = (
                                    (
                                        human_message_for_this_tool_failure
                                        + "\nAdditionally: "
                                        + ser_guidance
                                    )
                                    if human_message_for_this_tool_failure
                                    else ser_guidance
                                )
                                this_tool_call_needs_llm_replan = True
                            tool_span.set_attribute("tool.error", "Serialization error")

                        tool_span.set_attribute(
                            "tool.executed_successfully",
                            tool_executed_successfully_this_iteration,
                        )
                        tool_span.set_attribute(
                            "tool.output_preview",
                            str(tool_message_content_final_str)[:200],
                        )

                        if self.options.on_tool_call_executed:
                            try:
                                self.logger.debug(
                                    f"Executing on_tool_call_executed for '{tool_name}' (ID: {tool_call_id})."
                                )
                                self.options.on_tool_call_executed(
                                    task=self.task,
                                    uuid=self.uuid,
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tool_response=current_tool_output_payload,
                                    success=tool_executed_successfully_this_iteration,
                                    tool_call_id=tool_call_id,
                                )
                            except Exception as cb_ex:
                                self.logger.error(
                                    f"Error in on_tool_call_executed for {tool_name} (ID: {tool_call_id}): {cb_ex}",
                                    exc_info=True,
                                )
                                interaction_span.record_exception(
                                    cb_ex,
                                    {"callback_error": True, "tool_name": tool_name},
                                )

                        tool_messages_for_this_turn.append(
                            ToolMessage(
                                content=tool_message_content_final_str,
                                tool_call_id=tool_call_id,
                            )
                        )
                        self.logger.debug(
                            f"Appended ToolMessage for tool_call_id: {tool_call_id}"
                        )

                        if this_tool_call_needs_llm_replan:
                            any_tool_requires_llm_replan = True
                            if human_message_for_this_tool_failure:
                                guidance_for_llm_reprompt.append(
                                    human_message_for_this_tool_failure
                                )

                history.extend(tool_messages_for_this_turn)
                self.logger.debug(
                    f"Extended history with {len(tool_messages_for_this_turn)} tool messages."
                )
                interaction_span.add_event(
                    "Tool Processing Complete",
                    {
                        "num_tool_messages": len(tool_messages_for_this_turn),
                        "replan_needed": any_tool_requires_llm_replan,
                    },
                )

                if any_tool_requires_llm_replan:
                    self.logger.info(
                        "One or more tool calls require LLM replanning. Reprompting."
                    )
                    if guidance_for_llm_reprompt:
                        if len(guidance_for_llm_reprompt) > 1:
                            combined_guidance = (
                                "Multiple tool calls require attention. Please review the following issues and adjust your plan accordingly:\n\n"
                                + "\n\n".join(guidance_for_llm_reprompt)
                            )
                        else:
                            combined_guidance = (
                                "A tool call requires attention. Please review the following issue and adjust your plan accordingly:\n\n"
                                + guidance_for_llm_reprompt[0]
                            )

                        history.append(HumanMessage(content=combined_guidance))
                        self.logger.debug(
                            f"Added human guidance message for reprompt: {combined_guidance}"
                        )
                        interaction_span.add_event(
                            "LLM Replan Initiated", {"guidance": combined_guidance}
                        )
                    else:
                        fallback_msg = "An issue occurred with one or more tool calls. Please review tool responses and adjust your plan."
                        history.append(HumanMessage(content=fallback_msg))
                        self.logger.warning(
                            f"Added fallback human guidance for reprompt: {fallback_msg}"
                        )
                        interaction_span.add_event(
                            "LLM Replan Initiated (Fallback)",
                            {"guidance": fallback_msg},
                        )
                    continue

                self.logger.debug(
                    "All tool calls processed without requiring immediate LLM replan. Continuing LLM interaction."
                )

    def _split_task(self) -> SplitTask:
        self.logger.info(f"Splitting task: '{self.task}'")
        task_history = self._build_task_split_history()
        max_subtasks = self._get_max_subtasks()
        self.logger.debug(f"Max subtasks for splitting: {max_subtasks}")

        system_msg_content = (
            f"Split this task into smaller ones only if necessary. You can create up to {max_subtasks} subtasks. "
            "Do not attempt to answer it yourself.\n\n"
            "Consider these tasks that are already being worked on or have been completed:\n"
            f"{task_history}\n\n"
            "When splitting the current task, make sure to:\n"
            "1. Avoid overlap with existing tasks\n"
            "2. Build upon completed parent tasks\n"
            "3. Complement parallel tasks\n"
            "4. Split only if the task is too complex for a single response\n"
            f"5. Create no more than {max_subtasks} subtasks\n\n"
        )

        if self.u_inst:
            system_msg_content += (
                "Also, use this user-provided info to help you, and include any details in your output:\n"
                f"{self.u_inst}\n"
            )
        self.logger.debug(
            f"System message for task splitting LLM: {system_msg_content}"
        )

        system_msg = SystemMessage(content=system_msg_content)
        split_msgs_hist = [system_msg, HumanMessage(self.task)]

        self.logger.debug("Evaluating prompt complexity for task splitting.")
        evaluation = evaluate_prompt(
            f"Prompt: {self.task}"
        )
        self.logger.debug(f"Prompt evaluation result: {evaluation}")

        current_task_span = self.current_span  # Use the main task span if available
        if current_task_span:
            current_task_span.set_attribute(
                "prompt_complexity.score", evaluation.prompt_complexity_score[0]
            )
            current_task_span.set_attribute(
                "prompt_complexity.domain_knowledge", evaluation.domain_knowledge[0]
            )
            current_task_span.set_attribute(
                "prompt_complexity.contextual_knowledge",
                evaluation.contextual_knowledge[0],
            )
            current_task_span.set_attribute(
                "prompt_complexity.task_type", evaluation.task_type_1[0]
            )

        if (
            evaluation.prompt_complexity_score[0] < 0.5
            and evaluation.domain_knowledge[0] > 0.8
        ):  # Adjusted logic based on example
            self.logger.info(
                "Task complexity/domain knowledge suggests no subtasks needed based on evaluation."
            )
            if current_task_span:
                current_task_span.add_event(
                    "Skipping split due to low complexity/high domain knowledge"
                )
            return SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)

        if current_task_span:
            current_task_span.add_event(
                "Attempting Task Split",
                {"task": self.task, "max_subtasks": max_subtasks},
            )

        self.logger.info("Invoking LLM for task splitting (initial textual response).")
        # Priming with "1. " to encourage a list format
        response = self.llm.invoke(split_msgs_hist + [AIMessage(content="1. ")])
        self.logger.debug(f"LLM initial split response content: {response.content}")
        # Add the LLM's generated list (including our "1. " prefix) to history for the JSON formatting step
        split_msgs_hist.append(AIMessage(content="1. " + response.content))

        split_msgs_hist.append(self._construct_subtask_to_json_prompt())
        self.logger.info("Invoking LLM for task splitting (JSON formatting).")
        structured_response_msg = self.llm.invoke(
            split_msgs_hist
        )  # This is an AIMessage
        self.logger.debug(
            f"LLM structured split response content: {structured_response_msg.content}"
        )

        split_task_result: SplitTask
        try:
            # JsonOutputParser expects a str or BaseMessage.
            # structured_response_msg is an AIMessage.
            parsed_output_from_llm = self.task_split_parser.invoke(
                structured_response_msg
            )

            if isinstance(parsed_output_from_llm, dict):
                # Ensure our locally computed 'evaluation' takes precedence or is added.
                parsed_output_from_llm["evaluation"] = evaluation
                split_task_result = SplitTask(**parsed_output_from_llm)
            elif isinstance(parsed_output_from_llm, SplitTask):
                split_task_result = parsed_output_from_llm
                split_task_result.evaluation = (
                    evaluation  # Override or set the evaluation
                )
            else:
                self.logger.error(
                    f"Unexpected type from task_split_parser: {type(parsed_output_from_llm)}. Content: {str(parsed_output_from_llm)[:500]}"
                )
                split_task_result = SplitTask(
                    needs_subtasks=False, subtasks=[], evaluation=evaluation
                )

            if current_task_span:
                current_task_span.add_event(
                    "Split Task Parsed",
                    {
                        "parsed_ok": True,
                        "needs_subtasks": split_task_result.needs_subtasks,
                        "num_subtasks_parsed": len(split_task_result.subtasks),
                    },
                )

        except (
            ValidationError,
            OutputParserException,
            TypeError,
        ) as e:  # More specific exceptions
            self.logger.error(
                f"Error parsing LLM JSON response for task splitting or instantiating SplitTask: {e}. LLM content: {str(structured_response_msg.content)[:500]}",
                exc_info=True,
            )
            if current_task_span:
                current_task_span.record_exception(e)
                current_task_span.add_event("Split Task Parsing Failed")
            split_task_result = SplitTask(
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )
        except Exception as e:  # Catch-all for unexpected issues
            self.logger.error(
                f"Unexpected error during task splitting finalization: {e}. LLM content: {str(structured_response_msg.content)[:500]}",
                exc_info=True,
            )
            if current_task_span:
                current_task_span.record_exception(e)
                current_task_span.add_event("Split Task Finalization Failed")
            split_task_result = SplitTask(
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )

        self.logger.debug(f"Parsed split task result: {split_task_result}")

        if split_task_result.subtasks:
            original_subtask_count = len(split_task_result.subtasks)
            split_task_result.subtasks = split_task_result.subtasks[:max_subtasks]
            current_num_subtasks = len(split_task_result.subtasks)
            split_task_result.needs_subtasks = current_num_subtasks > 0
            if current_num_subtasks < original_subtask_count:
                self.logger.info(
                    f"Trimmed subtasks from {original_subtask_count} to {current_num_subtasks} due to max_subtasks limit of {max_subtasks}."
                )
                if current_task_span:
                    current_task_span.add_event(
                        "Subtasks Trimmed",
                        {
                            "original_count": original_subtask_count,
                            "new_count": current_num_subtasks,
                            "max_allowed": max_subtasks,
                        },
                    )

        else:  # No subtasks parsed or an error occurred leading to empty subtasks
            split_task_result.needs_subtasks = False
            split_task_result.subtasks = []

        # Assign UUIDs to subtasks if they are TaskObject and don't have one (they shouldn't)
        # The 'task' class (lowercase, subclass of TaskObject) has uuid, but SplitTask uses List[TaskObject]
        for i, subtask_obj in enumerate(split_task_result.subtasks):
            # We'll create 'task' instances (which include UUID) for child agents later.
            # Here, we're just dealing with TaskObject.
            # If TaskObject itself needed a UUID directly, it would be:
            # if not hasattr(subtask_obj, "uuid"):
            #    subtask_obj.uuid = str(uuid.uuid4()) # This would modify TaskObject schema or require it to allow extra fields.
            # For now, UUID generation for subtasks is implicitly handled when creating child RecursiveAgents.
            self.logger.debug(f"Subtask {i+1} for splitting: {subtask_obj.task}")

        self.logger.info(
            f"Task splitting finished. Needs subtasks: {split_task_result.needs_subtasks}. Number of subtasks generated: {len(split_task_result.subtasks)}"
        )
        return split_task_result

    def fix_task(self, tar: List) -> SplitTask:
        self.logger.info(f"Splitting task: '{self.task}'")
        task_history = self._build_task_split_history(tar)
        max_subtasks = int(self._get_max_subtasks()/2)
        self.logger.debug(f"Max subtasks for splitting: {max_subtasks}")

        system_msg_content = (
            f"Split this task into smaller ones only if necessary. You can create up to {max_subtasks} subtasks. "
            "Do not attempt to answer it yourself.\n\n"
            "Consider these tasks that are already being worked on or have been completed:\n"
            f"{task_history}\n\n"
            "When splitting the current task, make sure to:\n"
            "1. Avoid overlap with existing tasks\n"
            "2. Build upon completed parent tasks\n"
            "3. Complement parallel tasks\n"
            "4. Split only if the task is too complex for a single response\n"
            f"5. Create no more than {max_subtasks} subtasks\n\n"
        )

        if self.u_inst:
            system_msg_content += (
                "Also, use this user-provided info to help you, and include any details in your output:\n"
                f"{self.u_inst}\n"
            )
        self.logger.debug(
            f"System message for task splitting LLM: {system_msg_content}"
        )

        system_msg = SystemMessage(content=system_msg_content)
        split_msgs_hist = [system_msg, HumanMessage(self.task)]

        self.logger.debug("Evaluating prompt complexity for task splitting.")
        evaluation = evaluate_prompt(
            f"Prompt: {self.task}"
        )
        self.logger.debug(f"Prompt evaluation result: {evaluation}")

        current_task_span = self.current_span  # Use the main task span if available
        if current_task_span:
            current_task_span.set_attribute(
                "prompt_complexity.score", evaluation.prompt_complexity_score[0]
            )
            current_task_span.set_attribute(
                "prompt_complexity.domain_knowledge", evaluation.domain_knowledge[0]
            )
            current_task_span.set_attribute(
                "prompt_complexity.contextual_knowledge",
                evaluation.contextual_knowledge[0],
            )
            current_task_span.set_attribute(
                "prompt_complexity.task_type", evaluation.task_type_1[0]
            )

        if (
            evaluation.prompt_complexity_score[0] < 0.5
            and evaluation.domain_knowledge[0] > 0.8
        ):  # Adjusted logic based on example
            self.logger.info(
                "Task complexity/domain knowledge suggests no subtasks needed based on evaluation."
            )
            if current_task_span:
                current_task_span.add_event(
                    "Skipping split due to low complexity/high domain knowledge"
                )
            return SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)

        if current_task_span:
            current_task_span.add_event(
                "Attempting Task Split",
                {"task": self.task, "max_subtasks": max_subtasks},
            )

        self.logger.info("Invoking LLM for task splitting (initial textual response).")
        # Priming with "1. " to encourage a list format
        response = self.llm.invoke(split_msgs_hist + [AIMessage(content="Here is a list of smaller tasks contributing to the task you gave:\n\n1. ")])
        self.logger.debug(f"LLM initial split response content: {response.content}")
        # Add the LLM's generated list (including our "1. " prefix) to history for the JSON formatting step
        split_msgs_hist.append(AIMessage(content="1. " + response.content))

        split_msgs_hist.append(self._construct_subtask_to_json_prompt())
        self.logger.info("Invoking LLM for task splitting (JSON formatting).")
        structured_response_msg = self.llm.invoke(
            split_msgs_hist
        )  # This is an AIMessage
        self.logger.debug(
            f"LLM structured split response content: {structured_response_msg.content}"
        )

        split_task_result: SplitTask
        try:
            # JsonOutputParser expects a str or BaseMessage.
            # structured_response_msg is an AIMessage.
            parsed_output_from_llm = self.task_split_parser.invoke(
                structured_response_msg
            )

            if isinstance(parsed_output_from_llm, dict):
                # Ensure our locally computed 'evaluation' takes precedence or is added.
                parsed_output_from_llm["evaluation"] = evaluation
                split_task_result = SplitTask(**parsed_output_from_llm)
            elif isinstance(parsed_output_from_llm, SplitTask):
                split_task_result = parsed_output_from_llm
                split_task_result.evaluation = (
                    evaluation  # Override or set the evaluation
                )
            else:
                self.logger.error(
                    f"Unexpected type from task_split_parser: {type(parsed_output_from_llm)}. Content: {str(parsed_output_from_llm)[:500]}"
                )
                split_task_result = SplitTask(
                    needs_subtasks=False, subtasks=[], evaluation=evaluation
                )

            if current_task_span:
                current_task_span.add_event(
                    "Split Task Parsed",
                    {
                        "parsed_ok": True,
                        "needs_subtasks": split_task_result.needs_subtasks,
                        "num_subtasks_parsed": len(split_task_result.subtasks),
                    },
                )

        except (
            ValidationError,
            OutputParserException,
            TypeError,
        ) as e:  # More specific exceptions
            self.logger.error(
                f"Error parsing LLM JSON response for task splitting or instantiating SplitTask: {e}. LLM content: {str(structured_response_msg.content)[:500]}",
                exc_info=True,
            )
            if current_task_span:
                current_task_span.record_exception(e)
                current_task_span.add_event("Split Task Parsing Failed")
            split_task_result = SplitTask(
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )
        except Exception as e:  # Catch-all for unexpected issues
            self.logger.error(
                f"Unexpected error during task splitting finalization: {e}. LLM content: {str(structured_response_msg.content)[:500]}",
                exc_info=True,
            )
            if current_task_span:
                current_task_span.record_exception(e)
                current_task_span.add_event("Split Task Finalization Failed")
            split_task_result = SplitTask(
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )

        self.logger.debug(f"Parsed split task result: {split_task_result}")

        if split_task_result.subtasks:
            original_subtask_count = len(split_task_result.subtasks)
            split_task_result.subtasks = split_task_result.subtasks[:max_subtasks]
            current_num_subtasks = len(split_task_result.subtasks)
            split_task_result.needs_subtasks = current_num_subtasks > 0
            if current_num_subtasks < original_subtask_count:
                self.logger.info(
                    f"Trimmed subtasks from {original_subtask_count} to {current_num_subtasks} due to max_subtasks limit of {max_subtasks}."
                )
                if current_task_span:
                    current_task_span.add_event(
                        "Subtasks Trimmed",
                        {
                            "original_count": original_subtask_count,
                            "new_count": current_num_subtasks,
                            "max_allowed": max_subtasks,
                        },
                    )

        else:  # No subtasks parsed or an error occurred leading to empty subtasks
            split_task_result.needs_subtasks = False
            split_task_result.subtasks = []

        # Assign UUIDs to subtasks if they are TaskObject and don't have one (they shouldn't)
        # The 'task' class (lowercase, subclass of TaskObject) has uuid, but SplitTask uses List[TaskObject]
        for i, subtask_obj in enumerate(split_task_result.subtasks):
            # We'll create 'task' instances (which include UUID) for child agents later.
            # Here, we're just dealing with TaskObject.
            # If TaskObject itself needed a UUID directly, it would be:
            # if not hasattr(subtask_obj, "uuid"):
            #    subtask_obj.uuid = str(uuid.uuid4()) # This would modify TaskObject schema or require it to allow extra fields.
            # For now, UUID generation for subtasks is implicitly handled when creating child RecursiveAgents.
            self.logger.debug(f"Subtask {i+1} for splitting: {subtask_obj.task}")

        self.logger.info(
            f"Task splitting finished. Needs subtasks: {split_task_result.needs_subtasks}. Number of subtasks generated: {len(split_task_result.subtasks)}"
        )
        return split_task_result

    def _get_max_subtasks(self) -> int:
        """Get the maximum number of subtasks allowed at the current layer"""
        if self.current_layer >= len(self.options.task_limits.limits):
            self.logger.warning(
                f"Current layer {self.current_layer} exceeds max depth {len(self.options.task_limits.limits)}. No subtasks allowed."
            )
            return 0
        max_s = self.options.task_limits.limits[self.current_layer]
        self.logger.debug(
            f"Max subtasks for current layer {self.current_layer}: {max_s}"
        )
        return max_s

    def _summarize_subtask_results(
        self, tasks: List[str], subtask_results: List[str]
    ) -> str:
        self.logger.info(
            f"Summarizing {len(subtask_results)} subtask results for task: '{self.task}'"
        )
        if not subtask_results:  # Handle case with no results to summarize
            self.logger.warning(
                "No subtask results to summarize. Returning empty string or a note."
            )
            return "No subtask results were generated to summarize for this task."

        merge_options = MergeOptions(
            llm=self.llm, context_window=15000  # Increased context window for merger
        )
        merger = self.options.merger(merge_options)
        self.logger.debug(f"Merger initialized with options: {merge_options}")

        documents_to_merge = [
            f"QUESTION: {question}\n\nANSWER\n{answer}"
            for (question, answer) in zip(tasks, subtask_results)
            if answer  # Ensure answer is not None
        ]
        if not documents_to_merge:
            self.logger.warning("All subtask results were empty. Returning a note.")
            return "All subtasks yielded empty results."

        self.logger.debug(
            f"Documents to merge ({len(documents_to_merge)}): {str(documents_to_merge)[:500]}..."
        )

        merged_content = merger.merge_documents(documents_to_merge)
        self.logger.debug(f"Merged content (raw): {str(merged_content)[:200]}...")

        if self.options.align_summaries:
            self.logger.info("Aligning summaries.")
            # Ensure merged_content is not overly long for the alignment prompt
            max_merged_content_len = (
                10000  # Example limit, adjust based on LLM capacity
            )
            if len(merged_content) > max_merged_content_len:
                self.logger.warning(
                    f"Merged content length ({len(merged_content)}) exceeds limit ({max_merged_content_len}). Truncating for alignment."
                )
                merged_content_for_alignment = (
                    merged_content[:max_merged_content_len]
                    + "\n... [Content Truncated]"
                )
            else:
                merged_content_for_alignment = merged_content

            alignment_prompt_messages = [
                HumanMessage(
                    f"The following information has been gathered from subtasks:\n\n{merged_content_for_alignment}\n\n"
                    f"Based on this information, compile a comprehensive and well-structured report that directly answers this main question: '{self.task}'.\n\n"
                    "Custom instructions for the report:\n"
                    "- Go into extreme detail where relevant information is provided.\n"
                    "- Disregard any clearly irrelevant information from the subtasks.\n"
                    "- Ensure the final report has a good structure, clear organization, and directly addresses the main question.\n"
                    "- If citations were provided in the subtask answers (e.g., [1], [2]), try to preserve them or synthesize them appropriately in the final report."
                )
            ]
            self.logger.debug(f"Alignment prompt messages: {alignment_prompt_messages}")
            aligned_response = self.llm.invoke(alignment_prompt_messages)
            final_summary = aligned_response.content
            self.logger.info(
                f"Summarization complete (aligned). Result: {str(final_summary)[:100]}..."
            )
            return final_summary
        else:
            self.logger.info(
                f"Summarization complete (not aligned). Result: {str(merged_content)[:100]}..."
            )
            return merged_content  # Return the direct merged content

    def _construct_subtask_to_json_prompt(self):
        # Providing an example might help the LLM more than just the schema.
        example_task_object = TaskObject(
            task="Example subtask", subtasks=0, allow_search=True, allow_tools=True
        )
        example_split_task = SplitTask(
            needs_subtasks=True, subtasks=[example_task_object], evaluation=None
        )  # Evaluation is optional

        json_schema_str = SplitTask.model_json_schema()
        # Remove 'evaluation' from the required list in the schema string for the prompt, as it's often better if the LLM doesn't try to generate it.
        # We set it programmatically.
        try:
            schema_dict = json.loads(json.dumps(json_schema_str))
            if "required" in schema_dict and "evaluation" in schema_dict["required"]:
                schema_dict["required"].remove("evaluation")
            # Also remove 'evaluation' from properties to simplify for LLM
            if (
                "properties" in schema_dict
                and "evaluation" in schema_dict["properties"]
            ):
                del schema_dict["properties"]["evaluation"]
            # Remove $defs if TaskEvaluation is complex and not needed for LLM to fill
            if "$defs" in schema_dict and "TaskEvaluation" in schema_dict["$defs"]:
                del schema_dict["$defs"]["TaskEvaluation"]

            simplified_schema_str = json.dumps(schema_dict)
        except json.JSONDecodeError:
            self.logger.warning(
                "Could not parse/simplify SplitTask JSON schema for prompt. Using full schema."
            )
            simplified_schema_str = json_schema_str

        prompt_content = (
            f"Now, format the subtask list (or indicate no subtasks are needed) strictly according to the following JSON schema. "
            f"The 'evaluation' field is optional and you can omit it.\n"
            f"Schema:\n```json\n{simplified_schema_str}\n```\n"
            f"Example of desired JSON format for a task that needs subtasks:\n"
            f"```json\n{{\n"
            f'  "needs_subtasks": true,\n'
            f'  "subtasks": [\n'
            f'    {{ "task": "Subtask 1 description", "subtasks": 0, "allow_search": true, "allow_tools": true }},\n'
            f'    {{ "task": "Subtask 2 description", "subtasks": 0, "allow_search": true, "allow_tools": false }}\n'
            f"  ]\n"
            f"}}\n```\n"
            f"Example for a task that does NOT need subtasks:\n"
            f"```json\n{{\n"
            f'  "needs_subtasks": false,\n'
            f'  "subtasks": []\n'
            f"}}\n```\n"
            "Provide only the JSON object as your response."
        )
        self.logger.debug(f"Constructed subtask to JSON prompt: {prompt_content}")
        return HumanMessage(prompt_content)
