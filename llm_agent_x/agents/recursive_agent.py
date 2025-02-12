import json
import uuid
from difflib import SequenceMatcher
from typing import Any, Optional, List

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.output_parsers import JsonOutputParser

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
    def __bool__(self):
        return self.needs_subtasks

class TaskContext(BaseModel):
    task: str
    result: Optional[str] = None
    siblings: List['TaskContext'] = []
    parent_context: Optional['TaskContext'] = None

    class Config:
        arbitrary_types_allowed = True

class RecursiveAgentOptions(BaseModel):
    max_layers: int=2
    search_tool: Any = None
    pre_task_executed: Any = None
    on_task_executed: Any = None
    on_tool_call_executed: Any = None
    task_tree: list[task] = []
    llm: Any = None
    tool_llm: Any = None
    tools: list = []
    allow_search: bool = True
    allow_tools: bool = False
    tools_dict: dict = {}
    similarity_threshold: float = 0.8

    class Config:
        arbitrary_types_allowed = True

def calculate_raw_similarity(text1: str, text2: str) -> float:
    """
    Calculate the similarity ratio between two texts using SequenceMatcher.
    Returns a float between 0 and 1, where 1 means the texts are identical.
    """
    return SequenceMatcher(None, text1, text2).ratio()

class RecursiveAgent():
    def __init__(self, 
                 task: str, 
                 uuid: str = str(uuid.uuid4()), 
                 agent_options: RecursiveAgentOptions = RecursiveAgentOptions(), 
                 allow_subtasks: bool = True, 
                 current_layer: int = 0, 
                 parent: Optional['RecursiveAgent'] = None,
                 context: Optional[TaskContext] = None,
                 siblings: List['RecursiveAgent'] = None):
        self.task = task
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
                parent_contexts.append({
                    "task": current_context.task,
                    "result": current_context.result
                })
            current_context = current_context.parent_context

        # Get sibling contexts
        sibling_contexts = []
        for sibling in self.siblings:
            if sibling.result and sibling != self:
                sibling_contexts.append({
                    "task": sibling.task,
                    "result": sibling.result
                })

        return {
            "parent_contexts": parent_contexts,
            "sibling_contexts": sibling_contexts
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

    def run(self) -> str:
        if self.options.pre_task_executed:
            self.options.pre_task_executed(task=self.task, uuid=self.uuid, parent_agent_uuid = (self.parent.uuid) if self.parent is not None else None)

        while self.allow_subtasks and self.current_layer < self.options.max_layers:
            if self.parent:
                similarity = calculate_raw_similarity(self.task, self.parent.task)
                if similarity >= self.options.similarity_threshold:
                    print(f"Task similarity with parent is high ({similarity:.2f}), executing as single task.")
                    self.result = self._run_single_task()
                    return self.result

            self.tasks = self._split_task()
            if not self.tasks or not self.tasks["needs_subtasks"]:
                break

            # Create child agents with shared context
            child_agents = []
            for subtask in self.tasks["subtasks"]:
                child_context = TaskContext(
                    task=subtask["task"],
                    parent_context=self.context
                )
                
                child_agent = RecursiveAgent(
                    task=subtask["task"],
                    uuid=subtask["uuid"],
                    agent_options=self.options,
                    allow_subtasks=True,
                    current_layer=self.current_layer + 1,
                    parent=self,
                    context=child_context,
                    siblings=child_agents
                )
                child_agents.append(child_agent)

            # Run all child agents and collect results
            subtask_results = []
            for child_agent in child_agents:
                result = child_agent.run()
                child_agent.result = result
                child_agent.context.result = result
                subtask_results.append(result)
            
            self.result = self._summarize_subtask_results(subtask_results)
            return self.result
        
        self.result = self._run_single_task()
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
            SystemMessage(f"Your task is to answer the following question, using any tools that you deem necessary. "
                        f"If you use the web search tool, make sure you include citations (just use a pair of square "
                        f"brackets and a number in text, and at the end, include a citations section).{context_str}"),
            HumanMessage(self.task)
        ]
        
        response = self.tool_llm.invoke(history)
        history.append(response)
        
        for tool_call in response.tool_calls:
            tool = self.options.tools_dict[tool_call["name"]]
            tool_response = tool(**tool_call["args"])
            if self.options.on_tool_call_executed:
                self.options.on_tool_call_executed(task=self.task, uuid=self.uuid, tool_name=tool_call["name"], tool_args=tool_call["args"], tool_response=tool_response)
            history.append(ToolMessage(json.dumps(tool_response), tool_call_id=tool_call["id"]))
        
        return self.tool_llm.invoke(history).content

    def _split_task(self) -> SplitTask:
        # Get task history for context
        task_history = self._build_task_history()
        
        # Create system message with context
        system_msg = (
            "Split this task into smaller ones only if necessary. Do not attempt to answer it yourself.\n\n"
            "Consider these tasks that are already being worked on or have been completed:\n"
            f"{task_history}\n\n"
            "When splitting the current task, make sure to:\n"
            "1. Avoid overlap with existing tasks\n"
            "2. Build upon completed parent tasks\n"
            "3. Complement parallel tasks\n"
            "4. Split only if the task is too complex for a single response"
        )
        
        split_msgs_hist = [
            SystemMessage(system_msg),
            HumanMessage(self.task)
        ]
        
        response = self.llm.invoke(split_msgs_hist + [AIMessage("1. ")])
        split_msgs_hist.append(AIMessage(content="1. " + response.content))
        split_msgs_hist.append(self._construct_subtask_to_json_prompt())
        structured_response = self.llm.invoke(split_msgs_hist)
        split_task = self.task_split_parser.invoke(structured_response)

        for subtask in split_task["subtasks"]:
            subtask["uuid"] = str(uuid.uuid4())

        return split_task

    def _summarize_subtask_results(self, subtask_results) -> str:
        summary_history = [
            SystemMessage("Summarize the results into a single response with citations."),
            HumanMessage(json.dumps(subtask_results, indent=2))
        ]
        return self.llm.invoke(summary_history).content

    def _construct_subtask_to_json_prompt(self):
        return HumanMessage(f"Now format it in JSON: {SplitTask.model_json_schema()}")