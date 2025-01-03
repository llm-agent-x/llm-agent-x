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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Task(BaseModel):
    task: str
    allow_search: bool = True
    max_subtasks: int

class SplitTask(BaseModel):
    needs_multiple_parts: bool
    tasks: List[Task]


class Agent:
    def __init__(self, task: str, max_layers: int = 3, max_agents: int = 5, temperature=0, llm=None):
        self.task = task
        self.max_layers = max_layers
        self.max_agents = max_agents
        self.context = {}
        self.current_layer = 0
        self.llm = llm
        self.task_split_llm = self.llm.with_structured_output(SplitTask)
        self.task_split_parser = JsonOutputParser(pydantic_object = SplitTask)

    def run(self) -> Dict[str, Any]:
        if self.max_agents > 1:
            self.msgs = [
                SystemMessage(content=f"Your assignment is to divide this task into smaller parts. Please provide a list of parts. Each part should be simple and clear, and shouldn't depend on any other parts. For instance, if the user gives you a set of simple questions, assign one to each task, but you are limited to {self.max_agents}. If you feel this is simple enough, just say so."),
                HumanMessage(content=self.task)
            ]
            response = self.llm.invoke(self.msgs)
            ic(response)
            self.msgs.append(response)
            self.msgs.append(HumanMessage(content=(
                "Ok. Now put those tasks into a list, or if you don't need to subdivide the task, set the \"needs_multiple_parts\" property to false. Each task should be a JSON dictionary with a \"task\" key, a \"allow_search\" key, and a \"max_subtasks\" key. The \"task\" key should contain the task text, the \"allow_search\" key indicates whether each agent can use their search tool, and the \"max_subtasks\" key should contain the maximum number of subtasks that can be generated from this task."
            )))
            response = self.llm.invoke(self.msgs)
            self.msgs.append(response)
            uuid_x = str(uuid.uuid4())
            self.parts = ic(self.task_split_parser.parse(response.content))
            self.msgs.append(AIMessage(content = "", tool_calls=[ToolCall(name="execute_tasks", id=uuid_x, args={"tasks":self.parts})]))
            ic(self.parts)
            
            if input("Approve parts? (y/n) ").startswith("n"):
                raise ValueError("Parts not approved.")
            
            self.agent_results = []

            for part in self.parts:
                self.c = {}
                self.c["part"] = part["task"]
                self.c["layer"] = self.current_layer + 1

                self.c["agent"] = Agent(task=part["task"], max_layers=self.max_layers - 1, max_agents=min(self.max_agents, part["max_subtasks"]), llm=self.llm)
                
                self.agent_results.append(self.c["agent"].run())
            r = json.dumps([
                {
                    "task": result["task"],
                    "result": result["result"]
                } for result in self.agent_results
            ])
            ic(r)
            self.msgs.append(ToolMessage(content=r, tool_call_id=uuid_x))
            response = self.llm.invoke(self.msgs)
            return response.content
        else:
            response = self.llm.invoke([
                SystemMessage(content="Your assignment is to answer the following question:"),
                HumanMessage(content=self.task)
            ])
            return {
                "task": self.task,
                "result": response.content
            }