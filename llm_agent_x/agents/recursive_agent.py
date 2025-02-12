import json
import uuid

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.output_parsers import JsonOutputParser

from typing import Any
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def cosine_similarity(text1, text2):

    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return cosine_sim


class RecursiveAgent():
    def __init__(self, task, uuid= str(uuid.uuid4()), agent_options: RecursiveAgentOptions = RecursiveAgentOptions(), allow_subtasks = True, current_layer = 0, parent: 'RecursiveAgent' = None):
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
    
    def run(self) -> str:
        if self.options.pre_task_executed:
            self.options.pre_task_executed(task=self.task, uuid=self.uuid, parent_agent_uuid = (self.parent.uuid) if self.parent is not None else None)

        while self.allow_subtasks and self.current_layer < self.options.max_layers:
            if self.parent:
                similarity = cosine_similarity(self.task, self.parent.task)
                if similarity >= self.options.similarity_threshold:
                    print(f"Task similarity with parent is high ({similarity:.2f}), executing as single task.")
                    return self._run_single_task()

            self.tasks = self._split_task()
            if not self.tasks or not self.tasks["needs_subtasks"]:
                break

            agent_responses = [RecursiveAgent(
                task=subtask["task"],
                uuid=subtask["uuid"],
                agent_options=self.options,
                allow_subtasks=True,
                current_layer=self.current_layer + 1,
                parent=self
            ).run() for subtask in self.tasks["subtasks"]]
            
            return self._summarize_subtask_results(agent_responses)
        
        return self._run_single_task()

    def _run_single_task(self) -> str:
        history = [
            SystemMessage("Your task is to answer the following question, using any tools that you deem necessary. If you use the web search tool, make sure you include citations (just use a pair of square brackets and a number in text, and at the end, include a citations section):"),
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
        split_msgs_hist = [self._construct_subtask_sys_msg(), HumanMessage(self.task)]
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

    def _construct_subtask_sys_msg(self):
        return SystemMessage("Split this task into smaller ones only if necessary. Do not attempt to answer it yourself.")
    
    def _construct_subtask_to_json_prompt(self):
        return HumanMessage(f"Now format it in JSON: {SplitTask.model_json_schema()}")