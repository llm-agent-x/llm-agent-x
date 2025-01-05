import json
import math
import uuid
from icecream import ic
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall
)
from langchain_core.output_parsers import (
    JsonOutputParser,
)
import logging
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class Task(BaseModel):
    task: str
    allow_search: bool = True
    max_subtasks: int

class SplitTask(BaseModel):
    needs_multiple_parts: bool
    tasks: List[Task] = Field(optional=True)

class Agent:
    def __init__(self, search_tool, task: str, max_layers: int = 3, max_agents: int = 5, tools={}, llm=None, search_llm=None, allow_search=False, task_thread=None):
        self.task = task
        self.max_layers = max_layers
        self.max_agents = max_agents
        self.context = {}
        self.current_layer = 0
        self.llm = llm
        self.task_split_llm = self.llm.with_structured_output(SplitTask)
        self.search_llm = self.llm.bind_tools([search_tool]) if search_llm is None else search_llm
        self.task_split_parser = JsonOutputParser(pydantic_object=SplitTask)
        self.allow_search = allow_search
        self.tools = tools
        self.tools.update({"web_search": search_tool} if allow_search else {})
        self.task_thread = task_thread if task_thread is not None else []
    def add_context(self, context: Dict[str, Any]):
        self.context.update(context)
    def run(self) -> Dict[str, Any]:
        self.task_thread.append(self.task)
        if self.max_agents > 1 and self.max_layers > 0:
            self.msgs = [
                
            ]
            sys_msg_1 = f"Your assignment is to divide this task into smaller parts. Please provide a list of parts. Each part should be simple and clear, and shouldn't depend on any other parts. They should, in other words, be self contained, and not depend on any other context. For instance, if the user gives you a set of simple questions, assign one to each task, but you are limited to {self.max_agents} tasks. If you feel this is simple enough, just say so. (You are an agent in a system of agents, tasked with recursively splitting a task into smaller ones and executing them. Your history is as follows: [{"->".join(self.task_thread)}]){"\nBefore you proceed, retrieve the tool with the retrieve_context tool." if self.context else ""}"
            

            ic(self.task_thread)
            ic(self.max_layers)

            ctx_id = str(uuid.uuid4())
            self.msgs.extend([
                SystemMessage(content=sys_msg_1),
                HumanMessage(content=self.task),
            ])
            if self.context is not None:
                self.msgs.extend([
                    AIMessage(content="", tool_calls=[ToolCall(name="retrieve_context", id=ctx_id, args={})]),
                    ToolMessage(content=json.dumps(self.context), tool_call_id=ctx_id),
                ])
            response = self.llm.invoke(self.msgs)

            self.msgs.append(response)
            self.msgs.append(HumanMessage(content=(
                f"USE THIS SCHEMA:\n```json\n{json.dumps(SplitTask.model_json_schema())}\n```\nOk. Now put those tasks into a list, or if you don't need to subdivide the task, set the \"needs_multiple_parts\" property to false, and don't add any other keys (use this for low level tasks, like searching for factual information). Each task should be a JSON dictionary with a \"task\" key, a \"allow_search\" key, and a \"max_subtasks\" key. The \"task\" key should contain the task text, the \"allow_search\" key indicates whether each agent can use their search tool, and the \"max_subtasks\" key should contain the maximum number of subtasks that can be generated from this task. For the \"max_subtasks\" key, just think about how complex each request is, and how many subtasks you can generate from it."
            )))
            response = self.llm.invoke(self.msgs)
            self.msgs.pop(-1)
            
            self.msgs.append(response)
            self.msgs.append(HumanMessage(content=(
                "Ok. Now use the execute_tasks tool to execute the tasks. Pass the list of tasks as the \"tasks\" argument. Once you get the response, answer the initial questions using the results of the subtasks."
            )))
            uuid_x = str(uuid.uuid4())
            self.parts = self.task_split_parser.parse(response.content)
            self.msgs.append(AIMessage(content = "", tool_calls=[ToolCall(name="execute_tasks", id=uuid_x, args={"tasks":self.parts})]))
            
            ic(self.parts)
            if type(self.parts) != list:
                ic(self.parts.get("needs_multiple_parts", False))

                if not self.parts.get("needs_multiple_parts", False):
                    return {"result":self.llm.invoke(self.parts["tasks"][0]["task"]).content,
                            "task":self.task}
            ic(response.content)
            
            if False: # input("Approve parts? (y/n) ").startswith("n"):
                raise ValueError("Parts not approved.")
            
            self.agent_results = []

            for i, part in enumerate(self.parts["tasks"] if type(self.parts) == dict else self.parts):
                self.c = {}

                new_task_thread = self.task_thread.copy()
                ic(new_task_thread)
                ic(part)
                current_agent = Agent(task=part["task"], max_layers=self.max_layers - 1, max_agents=min(self.max_agents - 1, part["max_subtasks"]), llm=self.llm, search_llm=self.search_llm, allow_search=part["allow_search"], task_thread=new_task_thread)
                current_agent.add_context({"previous_results": json.dumps(self.agent_results)}) # I'm working on making model remember what it was doing in previous steps (in the same layer)
                self.agent_results.append(current_agent.run())
            
            
            r = json.dumps([
                {
                    "x": ic(result),
                    "task": (result["task"]) if type(result) == dict else result,
                    "result": (result["result"]) if type(result) == dict else result,
                } for result in ic(self.agent_results)
            ])

            ic(r)
            self.msgs.append(ToolMessage(content=r, tool_call_id=uuid_x))
            response = self.llm.invoke(self.msgs)
            return response.content
        else:
            response = None
            if not self.allow_search:
                response = self.llm.invoke([
                    SystemMessage(content=f"Your assignment is to answer the following question or complete the following task. (You are an agent in a system of agents, tasked with recursively splitting a task into smaller ones and executing them. Your history is as follows: {"->".join(self.task_thread)}):"),
                    HumanMessage(content=self.task)
                ])

            else:
                history = [
                    SystemMessage(content="Your assignment is to answer the following question using the search tool:"),
                    HumanMessage(content=self.task)
                ]
                llm_r1 = self.search_llm.invoke(history)

                history.append(llm_r1)

                for tool_call in llm_r1.tool_calls:
                    name = tool_call["name"]
                    args = tool_call["args"]
                    id = tool_call["id"]
                    tool = self.tools[name]

                    tool_response = tool(**args)

                    history.append(ToolMessage(content=tool_response, tool_call_id=id))
                response = self.llm.invoke(history)
            return {
                "task": self.task,
                "result": response.content
            }