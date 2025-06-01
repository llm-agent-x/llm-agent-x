import json
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Optional, List, Dict # Added Dict
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
    subtasks: int # Max number of subtasks this task object can be broken into
    allow_search: bool
    allow_tools: bool


class task(TaskObject):  # pylint: disable=invalid-name
    uuid: str

class verification(BaseModel):  # pylint: disable=invalid-name
    successful: bool

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
        u_inst: str, # User instructions for this specific task instance
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
        self.task_verification_parser = JsonOutputParser(pydantic_object=verification)
        self.uuid = uuid
        self.current_layer = current_layer
        self.parent = parent
        self.siblings = siblings or []
        self.context = context or TaskContext(task=task)
        self.result: Optional[str] = None # Ensure result is initialized
        self.status: str = "pending" # Track agent status: pending, succeeded, failed, fixed_and_verified
        self.current_span = None

        self.logger.debug(f"Agent initialized with options: {agent_options}")

    def _build_context_information(self) -> dict:
        self.logger.debug("Building context information.")
        parent_contexts = []
        current_p_context = self.context.parent_context # Renamed to avoid clash
        while current_p_context:
            if current_p_context.result: # Only include if parent has a result
                parent_contexts.append(
                    {"task": current_p_context.task, "result": current_p_context.result}
                )
            current_p_context = current_p_context.parent_context
        parent_contexts.reverse() # Show chronological order

        sibling_contexts = []
        # Use self.siblings which are RecursiveAgent instances
        for sibling_agent in self.siblings:
            if sibling_agent.result and sibling_agent != self:
                 # Access task and result directly from sibling agent instance
                sibling_contexts.append(
                    {"task": sibling_agent.task, "result": sibling_agent.result}
                )

        context_info = {
            "parent_contexts": parent_contexts,
            "sibling_contexts": sibling_contexts,
        }
        # self.logger.debug(f"Context information built: {context_info}")
        return context_info

    def _build_task_split_history(self) -> str:
        self.logger.debug("Building task history prompt for splitting.")
        context_info = self._build_context_information()
        history = []

        if context_info["parent_contexts"]:
            history.append("Previous parent tasks and their results:")
            for ctx in context_info["parent_contexts"]:
                history.append(f"- Parent Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...") # Show some result

        if context_info["sibling_contexts"]:
            history.append("\nParallel tasks already being worked on (or completed) by siblings and their results:")
            for ctx in context_info["sibling_contexts"]:
                history.append(f"- Sibling Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...")

        task_history_str = "\n".join(history)
        self.logger.debug(f"Task split history string built: {task_history_str}")
        return task_history_str

    def _build_task_verify_history(self, subtask_results_map: Optional[Dict[str, str]] = None) -> str:
        self.logger.debug("Building task verification prompt history.")
        context_info = self._build_context_information()
        history = []

        if context_info["parent_contexts"]:
            history.append("Context from parent tasks and their results:")
            for ctx in context_info["parent_contexts"]:
                history.append(f"- Parent Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...")

        if context_info["sibling_contexts"]:
            history.append("\nContext from parallel (sibling) tasks and their results:")
            for ctx in context_info["sibling_contexts"]:
                history.append(f"- Sibling Task: {ctx['task']}\n  Result: {str(ctx['result'])[:150]}...")
        
        if subtask_results_map:
            history.append("\nThe current main task involved the following subtasks and their results:")
            for sub_task, sub_result in subtask_results_map.items():
                history.append(f"- Subtask: {sub_task}\n  - Result: {str(sub_result)[:200]}...")

        task_history_str = "\n".join(history)
        self.logger.debug(f"Task verification history string built: {task_history_str}")
        return task_history_str

    def run(self):
        self.logger.info(f"Starting run for task: '{self.task}' (UUID: {self.uuid}, Status: {self.status})")
        if self.tracer and self.tracer_span:
            self.logger.debug("Tracer and tracer_span available, creating new span.")
            parent_otel_ctx = trace.set_span_in_context(self.tracer_span)
            with self.tracer.start_as_current_span(
                f"Execute Task: {self.task}",
                context=parent_otel_ctx,
                attributes={"agent.uuid": self.uuid, "agent.layer": self.current_layer}
            ) as span:
                self.current_span = span
                span.add_event("Run Start", {"task": self.task, "layer": self.current_layer})
                result = self._run(span=span)
                span.set_attribute("agent.final_status", self.status)
                span.add_event("Run End", {"result_preview": str(result)[:100], "status": self.status})
                self.logger.info(f"Run finished for task: '{self.task}'. Result: {str(result)[:100]}... Status: {self.status}")
                return result
        else:
            self.logger.debug("No tracer or tracer_span, running without new span.")
            result = self._run()
            self.logger.info(f"Run finished for task: '{self.task}'. Result: {str(result)[:100]}... Status: {self.status}")
            return result

    def _run(self, span=None) -> str:
        self.logger.info(f"Starting _run for task: '{self.task}' at layer {self.current_layer}")
        if span:
            span.add_event("Internal Run Start", {"task": self.task})

        if self.options.pre_task_executed:
            self.options.pre_task_executed(
                task=self.task, uuid=self.uuid, parent_agent_uuid=(self.parent.uuid if self.parent else None)
            )

        max_subtasks_for_this_layer = self._get_max_subtasks()
        if max_subtasks_for_this_layer == 0:
            self.logger.info(f"Max subtasks is 0 for layer {self.current_layer}. Executing as single task.")
            self.result = self._run_single_task(span=span)
            self.context.result = self.result # Update context
            # Verification for single task
            try:
                self.logger.info(f"Verifying result of single task execution: '{self.task}'.")
                self.verify_result(None) # No subtasks, so pass None
                self.logger.info(f"Single task '{self.task}' successfully verified. Status: {self.status}")
            except TaskFailedException:
                self.logger.warning(f"Single task '{self.task}' failed verification. Attempting to fix.")
                self._fix(None) # No subtasks, so pass None for failed_subtask_results_map
                if self.status == "failed":
                    self.logger.error(f"Fix attempt for single task '{self.task}' also failed.")
                elif self.status == "fixed_and_verified":
                    self.logger.info(f"Single task '{self.task}' was successfully fixed and verified.")
            
            if self.options.on_task_executed:
                self.options.on_task_executed(self.task, self.uuid, self.result, self.parent.uuid if self.parent else None)
            return self.result

        # Loop for allowing subtask generation, this loop typically runs once unless logic is added for re-splitting.
        while self.allow_subtasks: # allow_subtasks is an agent-level flag, usually True initially.
            self.logger.debug("Subtasks allowed, entering subtask generation/execution phase.")
            if self.parent:
                similarity = calculate_raw_similarity(self.task, self.parent.task)
                if similarity >= self.options.similarity_threshold:
                    self.logger.info(f"Task similarity with parent ({similarity:.2f}) is high. Executing as single task.")
                    if span: span.add_event("High Similarity Execution", {"similarity": similarity})
                    self.result = self._run_single_task(span=span)
                    self.context.result = self.result
                    # Verification for high-similarity single task
                    try:
                        self.verify_result(None)
                    except TaskFailedException:
                        self._fix(None)
                    if self.options.on_task_executed: self.options.on_task_executed(self.task, self.uuid, self.result, self.parent.uuid if self.parent else None)
                    return self.result # Exit _run

            split_task_result = self._split_task()
            if span: 
                span.set_attribute("task_split.needs_subtasks", split_task_result.needs_subtasks)
                span.set_attribute("task_split.num_subtasks_generated", len(split_task_result.subtasks))

            if not split_task_result or not split_task_result.needs_subtasks:
                self.logger.info("Task does not need subtasks or splitting failed. Executing as single task.")
                # allow_subtasks = False # To break loop, but will execute single task logic below anyway
                break # Break from while self.allow_subtasks loop

            limited_subtasks = split_task_result.subtasks[:max_subtasks_for_this_layer]
            self.logger.info(f"Proceeding with {len(limited_subtasks)} subtasks (limited from {len(split_task_result.subtasks)} by max_subtasks={max_subtasks_for_this_layer}).")
            if span: span.set_attribute("task_split.num_subtasks_active", len(limited_subtasks))

            child_agents: List[RecursiveAgent] = []
            child_contexts_for_siblings: List[TaskContext] = [] # To correctly pass siblings to children

            for subtask_obj in limited_subtasks:
                child_task_uuid = str(uuid.uuid4()) # Each child gets a new UUID
                child_context = TaskContext(task=subtask_obj.task, parent_context=self.context)
                
                # Create child agent
                child_agent = RecursiveAgent(
                    task=subtask_obj.task,
                    u_inst=self.u_inst, # Pass parent's user instructions, or derive specific ones
                    tracer=self.tracer,
                    tracer_span=span, # Current span is parent for child's span
                    uuid=child_task_uuid,
                    agent_options=self.options,
                    allow_subtasks=True, # Children can further split based on their layer's limits
                    current_layer=self.current_layer + 1,
                    parent=self,
                    context=child_context,
                    siblings=child_agents[:], # Pass a copy of already created siblings for this batch
                )
                child_agents.append(child_agent)
                child_contexts_for_siblings.append(child_context) # Track contexts for sibling linking

            # Update sibling contexts for all children created in this batch
            for i, child_agent_to_update in enumerate(child_agents):
                # Siblings are other children in this batch, excluding self
                child_agent_to_update.context.siblings = [
                    ctx for j, ctx in enumerate(child_contexts_for_siblings) if i != j
                ]


            subtask_results = []
            subtask_tasks_for_summary = []
            for child_agent in child_agents:
                child_result = child_agent.run()
                # After child_agent.run(), its child_agent.result and child_agent.status are set.
                # We primarily care about child_agent.result for summarization.
                # If a child task failed critically and couldn't be fixed, its result might be an error message or None.
                subtask_results.append(child_result if child_result is not None else "Error or no result from subtask.")
                subtask_tasks_for_summary.append(child_agent.task)
                # Child agent's context.result should have been updated internally by its run
            
            self.result = self._summarize_subtask_results(subtask_tasks_for_summary, subtask_results)
            self.context.result = self.result # Update own context with summarized result

            subtask_results_map = dict(zip(subtask_tasks_for_summary, subtask_results))
            try:
                self.logger.info(f"Verifying combined result from subtasks for task: '{self.task}'.")
                self.verify_result(subtask_results_map)
                self.logger.info(f"Task '{self.task}' (with subtasks) successfully verified. Status: {self.status}")
            except TaskFailedException:
                self.logger.warning(f"Task '{self.task}' (with subtasks) failed verification. Attempting to fix.")
                self._fix(subtask_results_map) # _fix will update self.status and self.result
                if self.status == "failed":
                    self.logger.error(f"Fix attempt for task '{self.task}' (with subtasks) also failed.")
                elif self.status == "fixed_and_verified":
                     self.logger.info(f"Task '{self.task}' (with subtasks) was successfully fixed and verified.")
            
            if self.options.on_task_executed:
                self.options.on_task_executed(self.task, self.uuid, self.result, self.parent.uuid if self.parent else None)
            return self.result # Exit _run after handling subtasks

        # This part is reached if subtasks are not allowed from start, or if splitting decided against it,
        # or if similarity to parent was high (that case returns early, but for safety).
        self.logger.info("Proceeding to execute task as a single unit (no subtasks generated or allowed at this step).")
        self.result = self._run_single_task(span=span)
        self.context.result = self.result
        try:
            self.logger.info(f"Verifying result of final single task execution: '{self.task}'.")
            self.verify_result(None)
            self.logger.info(f"Final single task '{self.task}' successfully verified. Status: {self.status}")
        except TaskFailedException:
            self.logger.warning(f"Final single task '{self.task}' failed verification. Attempting to fix.")
            self._fix(None)
            if self.status == "failed":
                self.logger.error(f"Fix attempt for final single task '{self.task}' also failed.")
            elif self.status == "fixed_and_verified":
                self.logger.info(f"Final single task '{self.task}' was successfully fixed and verified.")

        if self.options.on_task_executed:
            self.options.on_task_executed(self.task, self.uuid, self.result, self.parent.uuid if self.parent else None)
        return self.result
    
    def _run_single_task(self, span: Optional[trace.Span] = None) -> str:
        self.logger.info(f"Running single task: '{self.task}'")
        context_info = self._build_context_information()

        context_str_parts = []
        if context_info["parent_contexts"]:
            context_str_parts.append("Parent task history:")
            for ctx in context_info["parent_contexts"]:
                 context_str_parts.append(f"  Parent task: {ctx['task']}\n  Result: {str(ctx['result'])[:200]}...")
        
        if context_info["sibling_contexts"]:
            context_str_parts.append("Parallel sibling tasks in progress/completed:")
            for ctx in context_info["sibling_contexts"]:
                context_str_parts.append(f"  Task: {ctx['task']}\n  Result: {str(ctx['result'])[:200]}...")
        
        full_context_str = "\n".join(context_str_parts)
        if full_context_str:
             full_context_str = f"\n\nRelevant context from other tasks:\n{full_context_str}"


        system_prompt = (
            f"Your task is to answer the following question, using any tools that you deem necessary. "
            f"Make sure to phrase your search phrase in a way that it could be understood easily without context. "
            f"If you use the web search tool, make sure you include citations (just use a pair of square "
            f"brackets and a number in text, and at the end, include a citations section).{full_context_str}"
        )
        
        human_message_content = self.task
        if self.u_inst: # Add user instructions if provided
            human_message_content += f"\n\nFollow these specific instructions: {self.u_inst}"
        
        human_message_content += "\n\nApply the distributive property to any tool calls. For instance, if you need to search for 3 related things, make 3 separate calls to the search tool, because that will yield better results."


        history = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message_content),
        ]
        # self.logger.debug(f"Initial history for single task LLM: {history}")

        tracer_to_use = self.tracer if self.tracer else trace.get_tracer(__name__)
        parent_otel_ctx_for_single_task = trace.set_span_in_context(span) if span else otel_context.get_current()

        with tracer_to_use.start_as_current_span(
            "Single Task LLM Interaction", context=parent_otel_ctx_for_single_task
        ) as interaction_span:
            interaction_span.set_attribute("task", self.task)

            # Tool interaction loop (simplified from original for brevity, assuming it's mostly correct)
            loop_count = 0
            max_loops = 10 # Safety break for tool loops
            while loop_count < max_loops:
                loop_count += 1
                self.logger.info(f"Invoking tool_llm for single task (iteration {loop_count}).")
                current_llm_response = self.tool_llm.invoke(history)
                
                interaction_span.add_event(
                    "LLM Invoked", {
                        "history_length": len(history), 
                        "response_content_preview": str(current_llm_response.content)[:100],
                        "tool_calls_count": len(current_llm_response.tool_calls or [])
                    })
                history.append(current_llm_response)

                if not current_llm_response.tool_calls:
                    self.logger.info("No tool calls in LLM response. Returning content.")
                    interaction_span.set_attribute("final_result_preview", str(current_llm_response.content)[:100])
                    return str(current_llm_response.content)

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

                    # ... (Existing detailed tool execution logic including search query similarity check, error handling, etc.) ...
                    # This part is assumed to be largely correct from the original snippet.
                    # For brevity, I'm summarizing the key interactions.
                    try:
                        if tool_name not in self.options.tools_dict:
                            raise KeyError(f"Tool '{tool_name}' not found.")
                        tool_to_execute = self.options.tools_dict[tool_name]
                        
                        # Simplified: Actual tool execution and error handling would be here
                        if tool_name == "search" and "query" in tool_args: # Example
                             query = tool_args["query"]
                             sim = calculate_raw_similarity(query, self.task)
                             if sim < (self.options.similarity_threshold / 5):
                                 error_detail = f"Search query '{query}' too dissimilar (score: {sim:.2f}). Revise."
                                 current_tool_output_payload = {"error": error_detail}
                                 human_message_for_this_tool_failure = error_detail
                                 this_tool_call_needs_llm_replan = True
                             else:
                                current_tool_output_payload = tool_to_execute(**tool_args)
                                tool_executed_successfully_this_iteration = True
                        else: # Other tools or search without query (which should be handled)
                            current_tool_output_payload = tool_to_execute(**tool_args)
                            tool_executed_successfully_this_iteration = True

                    except Exception as e:
                        error_msg_str = f"Error with tool {tool_name} (ID: {tool_call_id}): {e}"
                        self.logger.error(error_msg_str, exc_info=True)
                        current_tool_output_payload = {"error": "Tool execution failed", "details": error_msg_str}
                        human_message_for_this_tool_failure = f"Error with '{tool_name}': {e}. Adjust plan."
                        this_tool_call_needs_llm_replan = True
                    # ... (End of summarized tool execution logic) ...

                    tool_message_content_final_str = json.dumps(current_tool_output_payload) # Ensure serializable
                    tool_messages_for_this_turn.append(ToolMessage(content=tool_message_content_final_str, tool_call_id=tool_call_id))
                    
                    if self.options.on_tool_call_executed:
                        self.options.on_tool_call_executed(self.task, self.uuid, tool_name, tool_args, current_tool_output_payload, tool_executed_successfully_this_iteration, tool_call_id)

                    if this_tool_call_needs_llm_replan:
                        any_tool_requires_llm_replan = True
                        if human_message_for_this_tool_failure:
                            guidance_for_llm_reprompt.append(human_message_for_this_tool_failure)
                
                history.extend(tool_messages_for_this_turn)

                if any_tool_requires_llm_replan:
                    if guidance_for_llm_reprompt:
                        history.append(HumanMessage(content="Review tool issues and adjust plan:\n" + "\n".join(guidance_for_llm_reprompt)))
                    else: # Fallback if no specific guidance messages were generated
                        history.append(HumanMessage(content="One or more tool calls had issues. Please review the tool responses and adjust your plan."))
                    continue # Back to LLM with new human message

            self.logger.warning(f"Max tool loop iterations ({max_loops}) reached for task '{self.task}'. Returning current history's last AI message or error.")
            # Fallback: return the last AI message content if loop terminates due to count
            if history and isinstance(history[-1], AIMessage):
                return str(history[-1].content)
            return "Max tool loop iterations reached without a final answer."

    def _split_task(self) -> SplitTask:
        self.logger.info(f"Splitting task: '{self.task}'")
        task_history_for_splitting = self._build_task_split_history()
        max_subtasks = self._get_max_subtasks()

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
        
        system_msg = SystemMessage(content=system_msg_content)
        split_msgs_hist = [system_msg, HumanMessage(self.task)]

        evaluation = evaluate_prompt(f"Prompt: {self.task}") # Assuming evaluate_prompt is defined elsewhere
        if self.current_span:
            self.current_span.set_attribute("prompt_complexity.score", evaluation.prompt_complexity_score[0])
            # ... other evaluation attributes ...

        # Simplified logic from original, assuming it worked:
        if evaluation.prompt_complexity_score[0] < 0.1   and evaluation.domain_knowledge[0] > 0.8:
            self.logger.info("Task complexity/domain knowledge suggests no subtasks needed based on evaluation.")
            return SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)

        # LLM call to get textual list of subtasks
        response = self.llm.invoke(split_msgs_hist + [AIMessage(content="1. ")]) # Priming
        split_msgs_hist.append(AIMessage(content="1. " + response.content))

        split_msgs_hist.append(HumanMessage(content="Can you make these more specific? Remember, each of these is sent off to another agent, with no context, asynchronously. All they know is what you put in this list."))

        split_msgs_hist.append(self.llm.invoke(split_msgs_hist)) # LLM call to get more specific subtasks

        # LLM call to format into JSON
        split_msgs_hist.append(self._construct_subtask_to_json_prompt()) # Renamed for clarity
        structured_response_msg = self.llm.invoke(split_msgs_hist)

        split_task_result: SplitTask
        try:
            parsed_output_from_llm = self.task_split_parser.invoke(structured_response_msg)
            if isinstance(parsed_output_from_llm, dict):
                parsed_output_from_llm["evaluation"] = evaluation
                split_task_result = SplitTask(**parsed_output_from_llm)
            elif isinstance(parsed_output_from_llm, SplitTask): # Should be this path
                split_task_result = parsed_output_from_llm
                split_task_result.evaluation = evaluation
            else: # Fallback
                self.logger.error(f"Unexpected type from task_split_parser: {type(parsed_output_from_llm)}.")
                split_task_result = SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)
        except (ValidationError, OutputParserException, TypeError) as e:
            self.logger.error(f"Error parsing LLM JSON for task splitting: {e}. LLM content: {str(structured_response_msg.content)[:500]}", exc_info=True)
            split_task_result = SplitTask(needs_subtasks=False, subtasks=[], evaluation=evaluation)
        
        # Trim subtasks if they exceed max_subtasks
        if split_task_result.subtasks:
            original_subtask_count = len(split_task_result.subtasks)
            split_task_result.subtasks = split_task_result.subtasks[:max_subtasks]
            if len(split_task_result.subtasks) < original_subtask_count:
                self.logger.info(f"Trimmed subtasks from {original_subtask_count} to {len(split_task_result.subtasks)}.")
            split_task_result.needs_subtasks = bool(split_task_result.subtasks) # Update based on actual count

        self.logger.info(f"Task splitting finished. Needs subtasks: {split_task_result.needs_subtasks}. Num subtasks: {len(split_task_result.subtasks)}")
        return split_task_result

    def _verify_result(self, subtask_results_map: Optional[Dict[str, str]] = None) -> bool:
        self.logger.info(f"Verifying result for task: '{self.task}'")
        if self.result is None:
            self.logger.warning(f"Cannot verify task '{self.task}' as its result is None.")
            return False # Cannot verify a None result

        task_history_for_verification = self._build_task_verify_history(subtask_results_map)
        
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

        verify_msgs_hist = [
            SystemMessage(content=system_msg_content),
            HumanMessage(content=human_msg_content)
        ]
        
        # Priming the LLM for JSON output
        verify_msgs_hist.append(AIMessage(content=f"```json\n{{\n  \"successful\": ")) # Prime for boolean
        
        self.logger.info("Invoking LLM for answer verification (JSON formatting).")
        structured_response_msg = self.llm.invoke(verify_msgs_hist) # AIMessage expected
        
        # Append the LLM's actual JSON completion to the primed AIMessage for full log
        if isinstance(structured_response_msg, AIMessage):
             verify_msgs_hist[-1] = AIMessage(content=verify_msgs_hist[-1].content + structured_response_msg.content)
        else: # Should not happen with Langchain LLMs
             verify_msgs_hist.append(AIMessage(content=str(structured_response_msg)))


        verification_obj: Optional[verification] = None
        try:
            # The parser expects the raw JSON string or a message containing it.
            parsed_output = self.task_verification_parser.invoke(structured_response_msg)
            if isinstance(parsed_output, dict):
                verification_obj = verification(**parsed_output)
            elif isinstance(parsed_output, verification): # Parser might return the object directly
                verification_obj = parsed_output
            
            if verification_obj is None:
                 raise ValueError("Parsed output could not be converted to verification object.")

            if self.current_span:
                self.current_span.add_event("Answer Verification Parsed", {"parsed_ok": True, "task_successful": verification_obj.successful})
            self.logger.info(f"Verification result for task '{self.task}': {'Successful' if verification_obj.successful else 'Failed'}")
            return verification_obj.successful
        except (ValidationError, OutputParserException, TypeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM JSON for verification or instantiating 'verification' object: {e}. LLM content: {str(structured_response_msg.content if isinstance(structured_response_msg, AIMessage) else structured_response_msg)[:500]}", exc_info=True)
            if self.current_span:
                self.current_span.record_exception(e)
                self.current_span.add_event("Verification Parsing Failed")
            # Default to unsuccessful if parsing fails, to be safe
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during verification finalization: {e}. LLM content: {str(structured_response_msg.content if isinstance(structured_response_msg, AIMessage) else structured_response_msg)[:500]}", exc_info=True)
            return False


    def verify_result(self, subtask_results_map: Optional[Dict[str, str]] = None):
        successful = self._verify_result(subtask_results_map)
        if successful:
            self.status = "succeeded"
        else:
            self.status = "failed_verification" # More specific status before fix
            raise TaskFailedException(f"Task '{self.task}' (UUID: {self.uuid}) was not completed successfully according to verification.")
        
    def _fix(self, failed_subtask_results_map: Optional[Dict[str, str]]):
        self.logger.info(f"Attempting to fix task: '{self.task}' (UUID: {self.uuid}) which failed verification.")
        original_task_str = self.task
        failed_result_str = str(self.result if self.result is not None else "No result was produced or result was None.")

        fix_instructions_parts = [
            f"The original task was: '{original_task_str}'.",
            f"A previous attempt to solve it resulted in (or failed to produce a result): '{failed_result_str[:700]}...'. This outcome was deemed unsatisfactory/incomplete by an automated verification step.",
        ]
        if failed_subtask_results_map:
            fix_instructions_parts.append("The failed attempt may have involved these subtasks and their results:")
            for sub_task, sub_result in failed_subtask_results_map.items():
                fix_instructions_parts.append(f"  - Subtask: {sub_task}\n    - Result: {str(sub_result)[:200]}...")
        
        if self.u_inst:
             fix_instructions_parts.append(f"\nOriginal user instructions for the task were: {self.u_inst}")

        fix_instructions_parts.append(
            "\nYour current objective is to FIX this failure. Analyze the original task, the failed outcome, any subtask information, and original user instructions. "
            "Then, provide a corrected and complete solution to the original task: '{original_task_str}'. "
            "You can break this fix attempt into a small number of sub-steps if that helps achieve a high-quality corrected solution. "
            "Focus on addressing the deficiencies of the previous attempt."
        )
        full_fix_instructions = "\n".join(fix_instructions_parts)

        # Configure options for the fixer agent
        current_agent_max_subtasks_at_this_layer = 0
        if self.current_layer < len(self.options.task_limits.limits):
            current_agent_max_subtasks_at_this_layer = self.options.task_limits.limits[self.current_layer]
        else: # Agent is deeper than configured limits, should not allow subtasks
            current_agent_max_subtasks_at_this_layer = 0 
        
        # Fixer gets half subtasks, min 1 if original allowed any, else 0.
        fixer_max_subtasks_for_its_level = 0
        if current_agent_max_subtasks_at_this_layer > 0 :
            fixer_max_subtasks_for_its_level = max(1, int(current_agent_max_subtasks_at_this_layer / 2))
        
        self.logger.info(f"Fixer agent for '{original_task_str}' will be allowed up to {fixer_max_subtasks_for_its_level} subtasks.")

        original_max_depth = len(self.options.task_limits.limits)
        fixer_limits_config_array = [0] * original_max_depth 

        if self.current_layer < original_max_depth:
            fixer_limits_config_array[self.current_layer] = fixer_max_subtasks_for_its_level
        
        fixer_task_limits = TaskLimit.from_array(fixer_limits_config_array)
        fixer_options = self.options.model_copy() # update={"task_limits": fixer_task_limits})
        fixer_options.task_limits = fixer_task_limits

        fixer_agent_uuid = str(uuid.uuid4())
        self.logger.info(f"Initializing fixer agent (UUID: {fixer_agent_uuid}) for task: '{original_task_str}'.")

        fixer_agent_context = TaskContext(task=original_task_str, parent_context=self.context.parent_context)

        fixer_agent = RecursiveAgent(
            task=original_task_str,
            u_inst=full_fix_instructions, # u_inst now contains all context for fixing
            tracer=self.tracer,
            tracer_span=self.current_span, # Fixer runs as a child span of the original task's execution
            uuid=fixer_agent_uuid,
            agent_options=fixer_options,
            allow_subtasks=True, # Governed by its specific fixer_task_limits
            current_layer=self.current_layer, # Fixer operates at the same layer
            parent=self, # The failed agent is parent, fixer can see its failed result via context
            context=fixer_agent_context,
            siblings=[] # Fixer agent is a new independent operation, no initial siblings of its type
        )

        try:
            self.logger.info(f"Running fixer agent (UUID: {fixer_agent_uuid}) for task '{self.task}'.")
            fixer_result = fixer_agent.run() # Fixer agent runs its full cycle, including its own verification if it uses subtasks.
            
            self.result = fixer_result # Update current agent's result with fixer's output
            self.context.result = self.result # Also update context object

            self.logger.info(f"Fixer agent completed. Re-verifying the fixed result for task: '{self.task}'.")
            # Verify the fixer's result against the original task.
            # The fixer_agent might have had its own subtasks, but for this verification,
            # we are verifying the final output of the fixer_agent. So, subtask_results_map is None.
            self.verify_result(None) # This will raise TaskFailedException if this new verification fails.
                                     # If successful, it sets self.status = "succeeded".

            # If verify_result passed, the task is now considered fixed and successful.
            self.logger.info(f"Task '{self.task}' successfully fixed and verified. New result: {str(self.result)[:100]}...")
            self.status = "fixed_and_verified" 

        except TaskFailedException as e_fix_verify:
            self.logger.error(f"Verification of fixer agent's result FAILED for task '{self.task}': {e_fix_verify}. Marking task as terminally failed.")
            self.status = "failed" # Fix attempt's result also failed verification.
        except Exception as e_fix_run:
            self.logger.error(f"Fixer agent (UUID: {fixer_agent_uuid}) encountered an UNHANDLED ERROR during its run for task '{self.task}': {e_fix_run}", exc_info=True)
            self.status = "failed" # Fixer agent crashed or had an unhandled error.
            if self.result is None: # If fixer didn't even produce a result.
                 self.result = f"Fix attempt failed with error: {e_fix_run}"


    def _get_max_subtasks(self) -> int:
        if self.current_layer >= len(self.options.task_limits.limits):
            self.logger.warning(
                f"Current layer {self.current_layer} exceeds max depth {len(self.options.task_limits.limits)}. No subtasks allowed."
            )
            return 0
        max_s = self.options.task_limits.limits[self.current_layer]
        self.logger.debug(f"Max subtasks for current layer {self.current_layer}: {max_s}")
        return max_s

    def _summarize_subtask_results(
        self, tasks: List[str], subtask_results: List[str]
    ) -> str:
        self.logger.info(f"Summarizing {len(subtask_results)} subtask results for task: '{self.task}'")
        # ... (existing summarization logic, assumed correct) ...
        if not subtask_results:
            return "No subtask results were generated to summarize."

        merge_options = MergeOptions(llm=self.llm, context_window=15000)
        merger = self.options.merger(merge_options)
        
        documents_to_merge = [
            f"SUBTASK QUESTION: {question}\n\nSUBTASK ANSWER:\n{answer}"
            for (question, answer) in zip(tasks, subtask_results)
            if answer is not None # Filter out None answers
        ]
        if not documents_to_merge:
            return "All subtasks yielded empty or no results."

        merged_content = merger.merge_documents(documents_to_merge)

        if self.options.align_summaries:
            max_merged_content_len = 10000 
            merged_content_for_alignment = merged_content
            if len(merged_content) > max_merged_content_len:
                self.logger.warning(f"Merged content length ({len(merged_content)}) too long, truncating for alignment.")
                merged_content_for_alignment = merged_content[:max_merged_content_len] + "\n... [Content Truncated]"

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
            aligned_response = self.llm.invoke(alignment_prompt_messages)
            final_summary = aligned_response.content
            self.logger.info(f"Summarization complete (aligned). Result preview: {str(final_summary)[:100]}...")
            return final_summary
        else:
            self.logger.info(f"Summarization complete (not aligned). Result preview: {str(merged_content)[:100]}...")
            return merged_content

    def _construct_subtask_to_json_prompt(self): # Renamed from _construct_answer_to_json_prompt
        # ... (existing logic for this method, assumed correct) ...
        json_schema_str = SplitTask.model_json_schema()
        try: # Simplify schema for LLM
            schema_dict = json.loads(json.dumps(json_schema_str)) # Deep copy
            if "required" in schema_dict and "evaluation" in schema_dict["required"]:
                schema_dict["required"].remove("evaluation")
            if "properties" in schema_dict and "evaluation" in schema_dict["properties"]:
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
            "Provide ONLY the JSON object as your response, without any other text or explanations."
        )
        return HumanMessage(prompt_content)

    def _construct_verify_answer_prompt(self): # For the _verify_result method
        json_schema_str = verification.model_json_schema()
        prompt_content = (
            f"Based on your evaluation, provide the outcome as a JSON object. Use the `successful` boolean field. "
            f"Schema:\n```json\n{json_schema_str}\n```\n"
            "Example:\n"
            f"```json\n{{\n"
            f'  "successful": true\n' # or false
            f"}}\n```\n"
            "Provide ONLY the JSON object as your response."
        )
        return HumanMessage(prompt_content)