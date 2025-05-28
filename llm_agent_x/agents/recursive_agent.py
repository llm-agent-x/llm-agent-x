import json
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Optional, List
from opentelemetry import trace, context as otel_context  # Modified import
from pydantic import BaseModel, validator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
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
    subtasks: int
    allow_search: bool
    allow_tools: bool


class task(TaskObject):
    uuid: str


class SplitTask(BaseModel):
    needs_subtasks: bool
    subtasks: list[TaskObject]
    evaluation: Optional[TaskEvaluation]

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
        uuid: str = str(uuid.uuid4()),
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
            if sibling.result and sibling != self:
                sibling_contexts.append(
                    {"task": sibling.task, "result": sibling.result}
                )

        context_info = {
            "parent_contexts": parent_contexts,
            "sibling_contexts": sibling_contexts,
        }
        self.logger.debug(f"Context information built: {context_info}")
        return context_info

    def _build_task_history(self) -> str:
        """
        Build a string representation of the task history for context
        """
        self.logger.debug("Building task history string.")
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
            # Create an OpenTelemetry Context object where self.tracer_span is the active span.
            # This makes self.tracer_span the parent of the new span created by start_as_current_span.
            parent_otel_ctx = trace.set_span_in_context(self.tracer_span)
            with self.tracer.start_as_current_span(
                f"Execute Task: {self.task}",
                context=parent_otel_ctx,  # Pass the OTEL Context object here
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
            # If no subtasks allowed at this layer, execute as single task
            self.result = self._run_single_task()
            self.context.result = self.result  # Store result in context
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
                    self.result = self._run_single_task()
                    self.context.result = self.result  # Store result in context
                    self.logger.info(
                        f"_run finished for high-similarity task '{self.task}'. Result: {str(self.result)[:100]}..."
                    )
                    return self.result

            self.logger.debug("Splitting task.")
            split_task_result = (
                self._split_task()
            )  # Renamed from self.tasks to avoid confusion
            if span:
                span.set_attribute(
                    "task_split_result", ic.format(split_task_result)
                )  # Changed from self.tasks

            if not split_task_result or not split_task_result.needs_subtasks:
                self.logger.info(
                    "Task does not need subtasks or splitting failed. Breaking subtask loop."
                )
                break

            # Limit number of subtasks based on configuration
            limited_subtasks = split_task_result.subtasks[:max_subtasks]
            self.logger.info(
                f"Limited subtasks to {len(limited_subtasks)} based on max_subtasks={max_subtasks}."
            )

            # Create child agents with shared context
            child_agents = []
            for (
                subtask_obj
            ) in limited_subtasks:  # Iterate over subtask objects from SplitTask
                self.logger.info(
                    f"Creating child agent for subtask: '{subtask_obj.task}'"
                )
                child_context = TaskContext(
                    task=subtask_obj.task, parent_context=self.context
                )

                child_agent = RecursiveAgent(
                    task=subtask_obj.task,  # Use task from subtask_obj
                    u_inst=self.u_inst,
                    tracer=self.tracer,
                    tracer_span=span,  # Use current span as parent for child's span
                    uuid=str(
                        uuid.uuid4()
                    ),  # Generate new UUID for subtask or use subtask_obj.uuid if it exists
                    agent_options=self.options,
                    allow_subtasks=True,  # Or determine based on subtask_obj if applicable
                    current_layer=self.current_layer + 1,
                    parent=self,
                    context=child_context,
                    siblings=child_agents,  # Pass currently created child_agents as siblings
                )
                child_agents.append(child_agent)

            # Run all child agents and collect results
            subtask_results = []
            subtask_tasks_for_summary = []
            self.logger.info(f"Running {len(child_agents)} child agents.")
            for child_agent in child_agents:
                self.logger.debug(f"Running child agent for task: '{child_agent.task}'")
                result = child_agent.run()
                child_agent.result = result
                child_agent.context.result = (
                    result  # Ensure child context also gets the result
                )
                subtask_results.append(result)
                subtask_tasks_for_summary.append(child_agent.task)
                self.logger.debug(
                    f"Child agent for task '{child_agent.task}' finished. Result: {str(result)[:100]}..."
                )

            self.logger.info("Summarizing subtask results.")
            self.result = self._summarize_subtask_results(
                subtask_tasks_for_summary, subtask_results  # Use collected tasks
            )
            self.context.result = self.result  # Store result in context

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
        self.result = self._run_single_task()
        self.context.result = self.result  # Store result in context
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

    def _run_single_task(self) -> str:
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

        with tracer_to_use.start_as_current_span(
            "Single Task Execution"
        ) as span:  # Ensure span is always created
            span.set_attribute("task", self.task)

            while True:  # Loop for multi-turn interaction
                self.logger.info("Invoking tool_llm for single task.")
                current_llm_response = self.tool_llm.invoke(history)
                self.logger.debug(
                    f"LLM response content: {current_llm_response.content}"
                )
                self.logger.debug(
                    f"LLM response tool calls: {current_llm_response.tool_calls}"
                )

                span.set_attribute("response_content", current_llm_response.content)
                span.set_attribute(
                    "response_tool_calls",
                    (
                        json.dumps(current_llm_response.tool_calls)
                        if current_llm_response.tool_calls
                        else "None"
                    ),
                )

                history.append(current_llm_response)

                if not current_llm_response.tool_calls:
                    self.logger.info(
                        "No tool calls in LLM response. Returning content."
                    )
                    return current_llm_response.content

                needs_llm_reprompt_after_this_batch = False
                first_error_guiding_human_message_content = None
                tool_messages_for_this_turn = []

                self.logger.debug(
                    f"Processing {len(current_llm_response.tool_calls)} tool calls."
                )
                for tool_call in current_llm_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    self.logger.info(
                        f"Processing tool call: {tool_name} with args: {tool_args}"
                    )

                    current_tool_output_payload: Any = None
                    current_tool_human_message_guidance: Optional[str] = None
                    tool_call_failed_for_reprompt_this_iteration = False
                    tool_executed_successfully_this_iteration = False

                    if (
                        needs_llm_reprompt_after_this_batch
                    ):  # A previous tool in this batch already failed
                        skipped_error_message = (
                            f"Skipped execution of tool '{tool_name}' because a preceding "
                            f"tool call in this batch resulted in an error requiring a plan revision."
                        )
                        self.logger.warning(skipped_error_message)
                        current_tool_output_payload = {
                            "error": "Skipped due to prior error in batch.",
                            "details": skipped_error_message,
                        }
                    else:
                        try:
                            if tool_name in [
                                "search",
                                "web_search",
                                "brave_web_search",
                            ]:
                                self.logger.debug(
                                    f"Tool '{tool_name}' is a search tool."
                                )
                                if "query" in tool_args:
                                    query = tool_args["query"]
                                    similarity = calculate_raw_similarity(
                                        query, self.task
                                    )
                                    self.logger.debug(
                                        f"Search query: '{query}', similarity to task '{self.task}': {similarity:.2f}"
                                    )
                                    with tracer_to_use.start_as_current_span(
                                        f"{tool_name} (Query Check)"
                                    ) as search_span:
                                        search_span.set_attribute("query", query)
                                        search_span.set_attribute(
                                            "similarity_to_task", similarity
                                        )
                                        search_span.set_attribute(
                                            "task_for_similarity", self.task
                                        )

                                    if (
                                        similarity
                                        < self.options.similarity_threshold / 1.4
                                    ):
                                        error_detail = (
                                            f"The search query '{query}' is not sufficiently related to the overall task: '{self.task}'. "
                                            f"Similarity score: {similarity:.2f}. "
                                            f"Please regenerate a query that is more directly related to the task, "
                                            f"or explain why this query is relevant if you believe it is."
                                        )
                                        self.logger.warning(
                                            f"Search query not related: {error_detail}"
                                        )
                                        current_tool_output_payload = {
                                            "error": "Query not related to task.",
                                            "details": error_detail,
                                        }
                                        current_tool_human_message_guidance = (
                                            error_detail
                                        )
                                        tool_call_failed_for_reprompt_this_iteration = (
                                            True
                                        )
                                    else:  # Query is good
                                        self.logger.debug(
                                            f"Search query '{query}' is good. Executing tool."
                                        )
                                        tool_to_execute = self.options.tools_dict[
                                            tool_name
                                        ]
                                        current_tool_output_payload = tool_to_execute(
                                            **tool_args
                                        )
                                        tool_executed_successfully_this_iteration = True
                                        self.logger.debug(
                                            f"Tool '{tool_name}' executed successfully. Output: {str(current_tool_output_payload)[:100]}..."
                                        )
                                else:  # "query" not in tool_args for search tool
                                    error_detail = f"Search tool '{tool_name}' called without a 'query' argument. Please provide a query."
                                    self.logger.error(error_detail)
                                    current_tool_output_payload = {
                                        "error": "Missing query argument for search tool.",
                                        "details": error_detail,
                                    }
                                    current_tool_human_message_guidance = error_detail
                                    tool_call_failed_for_reprompt_this_iteration = True
                            else:  # Not a search tool
                                self.logger.debug(
                                    f"Executing non-search tool: '{tool_name}'"
                                )
                                tool_to_execute = self.options.tools_dict[tool_name]
                                current_tool_output_payload = tool_to_execute(
                                    **tool_args
                                )
                                tool_executed_successfully_this_iteration = True
                                self.logger.debug(
                                    f"Tool '{tool_name}' executed successfully. Output: {str(current_tool_output_payload)[:100]}..."
                                )
                        except KeyError:
                            error_msg_str = f"Tool '{tool_name}' not found. Available tools are: {list(self.options.tools_dict.keys())}"
                            self.logger.error(error_msg_str, exc_info=True)
                            current_tool_output_payload = {
                                "error": "Tool not found",
                                "details": error_msg_str,
                            }
                            current_tool_human_message_guidance = (
                                f"You attempted to use a tool named '{tool_name}' which is not available. "
                                f"Please choose from the available tools: {list(self.options.tools_dict.keys())} or re-evaluate your approach."
                            )
                            tool_call_failed_for_reprompt_this_iteration = True
                        except Exception as e:
                            error_msg_str = f"Error executing tool {tool_name} with args {tool_args}: {str(e)}"
                            self.logger.error(error_msg_str, exc_info=True)
                            current_tool_output_payload = {
                                "error": "Tool execution failed",
                                "details": error_msg_str,
                            }
                            span.record_exception(e)
                            current_tool_human_message_guidance = (
                                f"An error occurred while executing the tool '{tool_name}': {str(e)}. "
                                f"The arguments were: {tool_args}. "
                                f"Please check the arguments, try a different approach, or ask for clarification if needed."
                            )
                            tool_call_failed_for_reprompt_this_iteration = True

                    # Attempt to serialize the payload for the ToolMessage
                    tool_message_content_final_str: str
                    try:
                        tool_message_content_final_str = json.dumps(
                            current_tool_output_payload
                        )
                    except TypeError as serialization_error:
                        self.logger.error(
                            f"Error serializing tool output for '{tool_name}': {serialization_error}",
                            exc_info=True,
                        )
                        span.record_exception(serialization_error)
                        error_detail_serialization = f"Tool '{tool_name}' output could not be serialized to JSON: {serialization_error}. Original output was: {str(current_tool_output_payload)[:200]}"
                        current_tool_output_payload = {  # Overwrite payload with error
                            "error": "Tool output serialization error",
                            "details": error_detail_serialization,
                        }
                        tool_message_content_final_str = json.dumps(
                            current_tool_output_payload
                        )  # Should be safe now

                        tool_executed_successfully_this_iteration = False  # Serialization failure means tool call effectively failed
                        if (
                            not tool_call_failed_for_reprompt_this_iteration
                        ):  # If not already failing for a primary reason
                            current_tool_human_message_guidance = f"Tool '{tool_name}' produced data that could not be processed due to a serialization issue. Please check the tool's output format or try a different approach."
                            tool_call_failed_for_reprompt_this_iteration = True

                    # Call on_tool_call_executed once with the final status and payload
                    if self.options.on_tool_call_executed:
                        self.logger.debug(
                            f"Executing on_tool_call_executed callback for tool '{tool_name}'."
                        )
                        self.options.on_tool_call_executed(
                            task=self.task,
                            uuid=self.uuid,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            tool_response=current_tool_output_payload,  # Pass the (possibly error) payload
                            success=tool_executed_successfully_this_iteration,
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

                    if (
                        tool_call_failed_for_reprompt_this_iteration
                        and not needs_llm_reprompt_after_this_batch
                    ):
                        self.logger.warning(
                            f"Tool call '{tool_name}' failed, setting reprompt flag. Guidance: {current_tool_human_message_guidance}"
                        )
                        needs_llm_reprompt_after_this_batch = True
                        first_error_guiding_human_message_content = (
                            current_tool_human_message_guidance
                        )

                history.extend(tool_messages_for_this_turn)
                self.logger.debug(
                    f"Extended history with {len(tool_messages_for_this_turn)} tool messages."
                )

                if needs_llm_reprompt_after_this_batch:
                    self.logger.info("Reprompting LLM due to tool call error(s).")
                    if first_error_guiding_human_message_content:
                        history.append(
                            HumanMessage(
                                content=first_error_guiding_human_message_content
                            )
                        )
                        self.logger.debug(
                            f"Added human guidance message for reprompt: {first_error_guiding_human_message_content}"
                        )
                    else:
                        # Fallback if somehow a reprompt was triggered without specific guidance
                        fallback_msg = "An error occurred with one or more tool calls, or their output was invalid. Please review the tool responses and adjust your plan."
                        history.append(HumanMessage(content=fallback_msg))
                        self.logger.warning(
                            f"Added fallback human guidance message for reprompt: {fallback_msg}"
                        )
                    continue  # Go to next iteration of while loop to re-invoke LLM

                # If no reprompt needed, it means all tool calls were successful (or skipped after a success that didn't need reprompt, which is not current logic)
                # and the LLM didn't return content, so we continue the loop expecting more tool calls or final content.
                # This 'continue' should effectively not be hit if there are no tool_calls because the first 'if not current_llm_response.tool_calls:' would catch it.
                # If there *were* tool calls, and they all succeeded, and we are here, it means the LLM expects more turns after tool execution.
                self.logger.debug(
                    "Tool calls processed, continuing LLM interaction loop."
                )

    def _split_task(self) -> SplitTask:  # Return type hint updated
        self.logger.info(f"Splitting task: '{self.task}'")
        task_history = self._build_task_history()
        max_subtasks = self._get_max_subtasks()
        self.logger.debug(f"Max subtasks for splitting: {max_subtasks}")

        system_msg = (
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
            system_msg += (
                "Also, use this user-provided info to help you, and include any details in your output:\n"
                f"{self.u_inst}\n"
            )
        self.logger.debug(f"System message for task splitting LLM: {system_msg}")

        split_msgs_hist = [SystemMessage(system_msg), HumanMessage(self.task)]

        self.logger.debug("Evaluating prompt complexity for task splitting.")
        evaluation = evaluate_prompt(f"Prompt: {self.task}")
        self.logger.debug(f"Prompt evaluation result: {evaluation}")
        if self.current_span:  # Check if current_span is initialized
            self.current_span.set_attribute(
                "prompt_complexity", evaluation.prompt_complexity_score[0]
            )
            self.current_span.set_attribute(
                "domain_knowledge", evaluation.domain_knowledge[0]
            )
            self.current_span.set_attribute(
                "contextual_knowledge", evaluation.contextual_knowledge[0]
            )
            self.current_span.set_attribute("task_type", evaluation.task_type_1[0])

        if (
            evaluation.prompt_complexity_score[0] > 0.5
            and evaluation.domain_knowledge[0] < 0.8
        ):
            self.logger.info(
                "Task complexity/domain knowledge suggests no subtasks needed based on evaluation."
            )
            split_task_result = SplitTask(  # Renamed variable
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )
            return split_task_result

        if self.current_span:
            self.current_span.add_event("Split Task", {"task": self.task})

        self.logger.info("Invoking LLM for task splitting (initial response).")
        response = self.llm.invoke(split_msgs_hist + [AIMessage("1. ")])
        self.logger.debug(f"LLM initial split response: {response.content}")
        split_msgs_hist.append(AIMessage(content="1. " + response.content))

        split_msgs_hist.append(self._construct_subtask_to_json_prompt())
        self.logger.info("Invoking LLM for task splitting (JSON formatting).")
        structured_response = self.llm.invoke(split_msgs_hist)
        self.logger.debug(
            f"LLM structured split response: {structured_response.content}"
        )

        try:
            parsed_split_task = self.task_split_parser.invoke(
                structured_response
            )  # Renamed variable
            # Check if parsed_split_task is a dictionary as expected from JsonOutputParser
            if isinstance(parsed_split_task, dict):
                split_task_result = SplitTask(
                    **parsed_split_task, evaluation=evaluation
                )  # Create SplitTask instance
            elif isinstance(
                parsed_split_task, SplitTask
            ):  # If parser already returns SplitTask
                split_task_result = parsed_split_task
                split_task_result.evaluation = evaluation
            else:
                self.logger.error(
                    f"Unexpected type from task_split_parser: {type(parsed_split_task)}. Content: {parsed_split_task}"
                )
                split_task_result = SplitTask(
                    needs_subtasks=False, subtasks=[], evaluation=evaluation
                )

        except Exception as e:
            self.logger.error(
                f"Error parsing LLM response for task splitting: {e}", exc_info=True
            )
            split_task_result = SplitTask(
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )

        self.logger.debug(f"Parsed split task: {split_task_result}")

        # Ensure we don't exceed the maximum number of subtasks
        # This block assumes split_task_result is a SplitTask object
        if split_task_result.subtasks:
            original_subtask_count = len(split_task_result.subtasks)
            split_task_result.subtasks = split_task_result.subtasks[:max_subtasks]
            split_task_result.needs_subtasks = len(split_task_result.subtasks) > 0
            if len(split_task_result.subtasks) < original_subtask_count:
                self.logger.info(
                    f"Trimmed subtasks from {original_subtask_count} to {len(split_task_result.subtasks)} due to max_subtasks limit."
                )
        else:
            split_task_result.needs_subtasks = False
            split_task_result.subtasks = []

        for (
            subtask_obj
        ) in split_task_result.subtasks:  # Iterate over TaskObject in subtasks list
            # Assuming TaskObject needs a UUID, though it's not in its definition.
            # The 'task' class (lowercase) has uuid. Let's add it here if it's intended.
            if not hasattr(subtask_obj, "uuid"):
                # This is a bit of a hack, ideally TaskObject would have uuid or we'd cast to 'task'
                subtask_obj_dict = subtask_obj.model_dump()
                subtask_obj_dict["uuid"] = str(uuid.uuid4())
                # This part is tricky because TaskObject doesn't have uuid.
                # For now, let's log and not modify the object structure if it doesn't fit.
                self.logger.debug(
                    f"Generated UUID for subtask '{subtask_obj.task}': {subtask_obj_dict['uuid']}"
                )

        self.logger.info(
            f"Task splitting finished. Needs subtasks: {split_task_result.needs_subtasks}. Number of subtasks: {len(split_task_result.subtasks)}"
        )
        return split_task_result

    def _get_max_subtasks(self) -> int:
        """Get the maximum number of subtasks allowed at the current layer"""
        if self.current_layer >= len(self.options.task_limits.limits):
            self.logger.warning(
                f"Current layer {self.current_layer} exceeds max depth {len(self.options.task_limits.limits)}. No subtasks allowed."
            )
            return 0  # No more subtasks allowed beyond configured depth
        max_s = self.options.task_limits.limits[self.current_layer]
        self.logger.debug(
            f"Max subtasks for current layer {self.current_layer}: {max_s}"
        )
        return max_s

    def _summarize_subtask_results(self, tasks, subtask_results) -> str:
        self.logger.info(
            f"Summarizing {len(subtask_results)} subtask results for task: '{self.task}'"
        )
        merge_options = MergeOptions(
            llm=self.llm, context_window=15
        )  # context_window seems low, but keeping as is
        merger = self.options.merger(merge_options)
        self.logger.debug(f"Merger initialized with options: {merge_options}")

        documents_to_merge = [
            f"QUESTION: {question}\n\nANSWER\n{answer}"
            for (question, answer) in zip(tasks, subtask_results)
        ]
        self.logger.debug(f"Documents to merge: {documents_to_merge}")

        merged_content = merger.merge_documents(documents_to_merge)
        self.logger.debug(f"Merged content (raw): {str(merged_content)[:200]}...")

        if self.options.align_summaries:
            self.logger.info("Aligning summaries.")
            alignment_prompt = [
                HumanMessage(
                    f"{merged_content}\n\nCompile a comprehensive report to answer this question:\n{self.task}\n\nCustom instructions:\ngo into extreme detail. disregard irrelevant information, but include anything relevant. ensure that your report has a good structure and organization."
                )
            ]
            self.logger.debug(f"Alignment prompt: {alignment_prompt}")
            aligned_response = self.llm.invoke(alignment_prompt)
            final_summary = aligned_response.content
            self.logger.info(
                f"Summarization complete (aligned). Result: {str(final_summary)[:100]}..."
            )
            return final_summary
        else:
            self.logger.info(
                f"Summarization complete (not aligned). Result: {str(merged_content)[:100]}..."
            )
            return merged_content

    def _construct_subtask_to_json_prompt(self):
        prompt_content = f"Now format it in JSON: {SplitTask.model_json_schema()}"
        self.logger.debug(f"Constructed subtask to JSON prompt: {prompt_content}")
        return HumanMessage(prompt_content)
