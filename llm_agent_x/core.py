import json
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
from langchain_core.output_parsers import JsonOutputParser
import logging
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class Task(BaseModel):
    task: str
    allow_search: bool = True
    max_subtasks: int

class SplitTask(BaseModel):
    needs_multiple_parts: bool
    tasks: List[Task] = Field(default_factory=list)

class Agent:
    def __init__(
        self,
        task: str,
        max_layers: int = 3,
        max_agents: int = 5,
        tools: Dict[str, Any] = {},
        llm=None,
        search_llm=None,
        allow_search=False,
        task_thread=None,
        search_tool = None,
    ):
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
        self.search_tool = search_tool
        if allow_search:
            self.tools.update({"web_search": search_tool})
        self.task_thread = task_thread if task_thread is not None else []

    def add_context(self, context: Dict[str, Any]):
        self.context.update(context)

    def run(self) -> Dict[str, Any]:
        self.task_thread.append(self.task)
        if self.max_agents > 1 and self.max_layers > 0:
            return self._run_with_subtasks()
        else:
            return self._run_single_task()

    def _run_with_subtasks(self) -> Dict[str, Any]:
        ctx_id = str(uuid.uuid4())
        msgs = [
            SystemMessage(content=self._generate_system_message()),
            HumanMessage(content=self.task),
        ]

        if self.context:
            msgs.extend([
                AIMessage(content="", tool_calls=[ToolCall(name="retrieve_context", id=ctx_id, args={})]),
                ToolMessage(content=json.dumps(self.context), tool_call_id=ctx_id),
            ])

        response = self.llm.invoke(msgs)
        msgs.append(response)
        msgs.append(HumanMessage(content=self._generate_schema_message()))

        response = self.llm.invoke(msgs)
        msgs.pop(-1)
        msgs.append(response)

        msgs.append(HumanMessage(content=self._generate_execute_tasks_message()))
        parts = self.task_split_parser.parse(response.content)
        msgs.append(AIMessage(content="", tool_calls=[ToolCall(name="execute_tasks", id=str(uuid.uuid4()), args={"tasks": parts})]))
        if(type(parts) == list):
            pass
        elif not parts["needs_multiple_parts"]:
            return {"result": self.llm.invoke(parts.tasks[0].task).content, "task": self.task}

        self.agent_results = []
        for part in (parts["tasks"]) if type(parts) != list else parts:
            new_task_thread = self.task_thread.copy()
            current_agent = Agent(
                task=part["task"],
                max_layers=self.max_layers - 1,
                max_agents=min(self.max_agents - 1, part["max_subtasks"]),
                llm=self.llm,
                search_llm=self.search_llm,
                allow_search=part["allow_search"],
                task_thread=new_task_thread,
                search_tool=self.search_tool
            )
            current_agent.add_context({"previous_results": json.dumps(self.agent_results)})
            self.agent_results.append(current_agent.run())

        results = json.dumps([
            {"task": result.get("task", result), "result": result.get("result", result)}
            for result in self.agent_results
        ])

        msgs.append(ToolMessage(content=results, tool_call_id=str(uuid.uuid4())))
        response = self.llm.invoke(msgs)
        return {"task": self.task, "result": response.content}

    def _run_single_task(self) -> Dict[str, Any]:
        if not self.allow_search:
            response = self.llm.invoke([
                SystemMessage(content=self._generate_single_task_message()),
                HumanMessage(content=self.task)
            ])
        else:
            history = [
                SystemMessage(content=self._generate_search_message()),
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

        return {"task": self.task, "result": response.content}

    def _generate_system_message(self) -> str:
        task_thread = "->".join(self.task_thread)
        return (
            f"Your assignment is to divide this task into smaller parts. "
            f"Please provide a list of parts. Each part should be simple and clear, "
            f"and shouldn't depend on any other parts. They should, in other words, be self-contained, "
            f"and not depend on any other context. For instance, if the user gives you a set of simple questions, "
            f"assign one to each task, but you are limited to {self.max_agents} tasks. "
            f"If you feel this is simple enough, just say so. "
            f"(You are an agent in a system of agents, tasked with recursively splitting a task into smaller ones and executing them. "
            f"Your history is as follows: [{task_thread}])"
            f"\nBefore you proceed, retrieve the tool with the retrieve_context tool." if self.context else ""
        )

    def _generate_schema_message(self) -> str:
        schema = json.dumps(SplitTask.model_json_schema())
        return (
            f"USE THIS SCHEMA:\n```json\n{schema}\n```\n"
            f"Ok. Now put those tasks into a list, or if you don't need to subdivide the task, "
            f"set the \"needs_multiple_parts\" property to false, and don't add any other keys "
            f"(use this for low-level tasks, like searching for factual information). "
            f"Each task should be a JSON dictionary with a \"task\" key, a \"allow_search\" key, "
            f"and a \"max_subtasks\" key. The \"task\" key should contain the task text, the \"allow_search\" key "
            f"indicates whether each agent can use their search tool, and the \"max_subtasks\" key should contain "
            f"the maximum number of subtasks that can be generated from this task. "
            f"For the \"max_subtasks\" key, just think about how complex each request is, "
            f"and how many subtasks you can generate from it."
        )

    def _generate_execute_tasks_message(self) -> str:
        return (
            "Ok. Now use the execute_tasks tool to execute the tasks. "
            "Pass the list of tasks as the \"tasks\" argument. "
            "Once you get the response, answer the initial questions using the results of the subtasks."
        )

    def _generate_single_task_message(self) -> str:
        task_thread = "->".join(self.task_thread)
        return (
            f"Your assignment is to answer the following question or complete the following task. "
            f"(You are an agent in a system of agents, tasked with recursively splitting a task into smaller ones and executing them. "
            f"Your history is as follows: {task_thread}):"
        )

    def _generate_search_message(self) -> str:
        return (
            "Your assignment is to answer the following question using the search tool:"
        )