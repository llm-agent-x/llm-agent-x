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

        if agent_options is None:
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

    def _build_context_information(self) -> dict:
        """
        Build context information including parent chain and sibling tasks
        """
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

        return {
            "parent_contexts": parent_contexts,
            "sibling_contexts": sibling_contexts,
        }

    def _build_task_history(self) -> str:
        """
        Build a string representation of the task history for context
        """
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

        return "\n".join(history)

    # In class RecursiveAgent:
    def run(self):
        if self.tracer and self.tracer_span:
            # Create an OpenTelemetry Context object where self.tracer_span is the active span.
            # This makes self.tracer_span the parent of the new span created by start_as_current_span.
            parent_otel_ctx = trace.set_span_in_context(self.tracer_span)
            with self.tracer.start_as_current_span(
                f"Execute Task: {self.task}",
                context=parent_otel_ctx,  # Pass the OTEL Context object here
            ) as span:
                self.current_span = span
                span.add_event(
                    "Start", {"layer": self.current_layer, "task": self.task}
                )
                return self._run(span=span)
        else:
            return self._run()

    def _run(self, span=None) -> str:
        if span:
            span.add_event(
                "run",
                {
                    "task": self.task,
                },
            )
        if self.options.pre_task_executed:
            self.options.pre_task_executed(
                task=self.task,
                uuid=self.uuid,
                parent_agent_uuid=(self.parent.uuid if self.parent else None),
            )

        max_subtasks = self._get_max_subtasks()
        if max_subtasks == 0:
            # If no subtasks allowed at this layer, execute as single task
            self.result = self._run_single_task()
            if self.options.on_task_executed:
                self.options.on_task_executed(
                    self.task,
                    self.uuid,
                    self.result,
                    self.parent.uuid if self.parent else None,
                )

            return self.result

        while self.allow_subtasks:
            if self.parent:
                similarity = calculate_raw_similarity(self.task, self.parent.task)
                if similarity >= self.options.similarity_threshold:
                    self.current_span.add_event(
                        f"Task similarity with parent is high ({similarity:.2f}), executing as single task."
                    )
                    self.result = self._run_single_task()
                    return self.result

            self.tasks = self._split_task()
            span.set_attribute("task", ic.format(self.tasks))
            if not self.tasks or not self.tasks["needs_subtasks"]:
                break

            # Limit number of subtasks based on configuration
            limited_subtasks = self.tasks["subtasks"][:max_subtasks]

            # Create child agents with shared context
            child_agents = []
            for subtask in limited_subtasks:
                child_context = TaskContext(
                    task=subtask["task"], parent_context=self.context
                )

                child_agent = RecursiveAgent(
                    task=subtask["task"],
                    u_inst=self.u_inst,
                    tracer=self.tracer,
                    tracer_span=span,
                    uuid=subtask["uuid"],
                    agent_options=self.options,
                    allow_subtasks=True,
                    current_layer=self.current_layer + 1,
                    parent=self,
                    context=child_context,
                    siblings=child_agents,
                )
                child_agents.append(child_agent)

            # Run all child agents and collect results
            subtask_results = []
            for child_agent in child_agents:
                result = child_agent.run()
                child_agent.result = result
                child_agent.context.result = result
                subtask_results.append(result)

            self.result = self._summarize_subtask_results(
                [child_agent.task for child_agent in child_agents], subtask_results
            )

            if self.options.on_task_executed:
                self.options.on_task_executed(
                    self.task,
                    self.uuid,
                    self.result,
                    self.parent.uuid if self.parent else None,
                )

            return self.result

        self.result = self._run_single_task()
        if self.options.on_task_executed:
            self.options.on_task_executed(
                self.task,
                self.uuid,
                self.result,
                self.parent.uuid if self.parent else None,
            )
        return self.result

    def _run_single_task(self) -> str:
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
                + f"Use this info to help you: {self.u_inst}"
                if self.u_inst
                else ""
            ),
        ]

        self.current_span.set_attribute(
            "single_task_context", json.dumps(history, indent=0).replace("\n", " ")
        )

        response = self.tool_llm.invoke(history)
        history.append(response)

        for tool_call in response.tool_calls:
            try:
                tool = self.options.tools_dict[tool_call["name"]]
                tool_response = tool(**tool_call["args"])
                if self.options.on_tool_call_executed:
                    self.options.on_tool_call_executed(
                        task=self.task,
                        uuid=self.uuid,
                        tool_name=tool_call["name"],
                        tool_args=tool_call["args"],
                        tool_response=tool_response,
                    )
                history.append(
                    ToolMessage(json.dumps(tool_response), tool_call_id=tool_call["id"])
                )
            except KeyError:
                history.append(
                    ToolMessage("Tool not found", tool_call_id=tool_call["id"])
                )
                if self.options.on_tool_call_executed:
                    self.options.on_tool_call_executed(
                        task=self.task,
                        uuid=self.uuid,
                        tool_name=tool_call["name"],
                        tool_args=tool_call["args"],
                        tool_response=tool_response,
                        success=False,
                    )
        return self.tool_llm.invoke(history).content

    def _split_task(self) -> SplitTask:
        task_history = self._build_task_history()
        max_subtasks = self._get_max_subtasks()

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

        split_msgs_hist = [SystemMessage(system_msg), HumanMessage(self.task)]

        evaluation = evaluate_prompt(f"Prompt: {self.task}")
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
        # ic(evaluation)
        if (
            evaluation.prompt_complexity_score[0] > 0.5
            and evaluation.domain_knowledge[0] < 0.8
        ):
            split_task = SplitTask(
                needs_subtasks=False, subtasks=[], evaluation=evaluation
            )
            return split_task
        response = self.llm.invoke(split_msgs_hist + [AIMessage("1. ")])
        split_msgs_hist.append(AIMessage(content="1. " + response.content))
        split_msgs_hist.append(self._construct_subtask_to_json_prompt())
        structured_response = self.llm.invoke(split_msgs_hist)
        split_task = self.task_split_parser.invoke(structured_response)
        split_task["evaluation"] = evaluation
        # ic(type(split_task))
        # Ensure we don't exceed the maximum number of subtasks
        try:
            split_task["subtasks"] = split_task["subtasks"][:max_subtasks]
            split_task["needs_subtasks"] = len(split_task["subtasks"]) > 0
        except KeyError:
            split_task["needs_subtasks"] = False
            split_task["subtasks"] = []

        for subtask in split_task["subtasks"]:
            subtask["uuid"] = str(uuid.uuid4())

        return split_task

    def _get_max_subtasks(self) -> int:
        """Get the maximum number of subtasks allowed at the current layer"""
        if self.current_layer >= len(self.options.task_limits.limits):
            return 0  # No more subtasks allowed beyond configured depth
        return self.options.task_limits.limits[self.current_layer]

    def _summarize_subtask_results(self, tasks, subtask_results) -> str:
        merge_options = MergeOptions(llm=self.llm, context_window=15)
        merger = self.options.merger(merge_options)

        x = [
            f"QUESTION: {question}\n\nANSWER\n{answer}"
            for (question, answer) in zip(tasks, subtask_results)
        ]
        merged = merger.merge_documents(x)
        if self.options.align_summaries:
            aligned = self.llm.invoke(
                [
                    HumanMessage(
                        f"{merged}\n\nCompile a comprehensive report to answer this question:\n{self.task}\n\nCustom instructions:\ngo into extreme detail. disregard irrelevant information, but include anything relevant. ensure that your report has a good structure and organization."
                    )
                ]
            )
            return aligned.content
        else:
            return merged

    def _construct_subtask_to_json_prompt(self):
        return HumanMessage(f"Now format it in JSON: {SplitTask.model_json_schema()}")
