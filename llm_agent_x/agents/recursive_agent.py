import json
import uuid

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.output_parsers import JsonOutputParser

from typing import Any

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
    max_agents: list|int = [5, 3, 2]
    search_tool: Any = None
    pre_task_executed: Any = None
    on_task_executed: Any = None
    on_tool_call_executed: Any = None
    task_tree: list[task] = []
    llm: Any = None
    search_llm: Any = None
    tools: list = []
    allow_search: bool = True
    allow_tools: bool = False
    tools_dict: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def propagate(self):
        return RecursiveAgentOptions(
            max_layers=self.max_layers,
            max_agents=(lambda x: x if isinstance(x, int) else x[1:])(self.max_agents),
            search_tool=self.search_tool,
            pre_task_executed=self.pre_task_executed,
            on_task_executed=self.on_task_executed,
            on_tool_call_executed=self.on_tool_call_executed,
            task_tree=self.task_tree,
            llm=self.llm,
            search_llm=self.search_llm,
            tools=self.tools,
            allow_search=self.allow_search,
            allow_tools=self.allow_tools,
            tools_dict=self.tools_dict,
        )


class RecursiveAgent():
    def __init__(self, task, uuid= str(uuid.uuid4()), agent_options: RecursiveAgentOptions = RecursiveAgentOptions(), allow_subtasks = True, current_layer = 0, complexity = 3, parent: 'RecursiveAgent' = None):
        self.task = task
        self.options = agent_options
        self.allow_subtasks = allow_subtasks
        self.complexity = complexity

        self.llm = self.options.llm
        self.search_llm = self.options.search_llm
        self.tools = self.options.tools

        self.tools_llm = (self.search_llm.bind_tools(self.tools) if self.options.allow_search else self.llm.bind_tools(self.tools))
        self.task_split_parser = JsonOutputParser(pydantic_object=SplitTask)
        self.uuid = uuid
        self.current_layer = current_layer
        self.parent = parent
    def run(self) -> str:
        # Determine whether/not to split task (based on allow_subtasks and the layer)
        if self.options.pre_task_executed:
            self.options.pre_task_executed(task=self.task, uuid=self.uuid, parent_agent_uuid = (self.parent.uuid) if self.parent is not None else None)

        if self.allow_subtasks and (self.current_layer < self.options.max_layers) and ((self.options.max_agents[0] if len(self.options.max_agents) > 0 else 0)) > 0:
            # Split task
            self.tasks = self._split_task()

            if self.tasks:
                result = self._run_subtasks()
                if result is None:
                    # Skip to single task
                    result = self._run_single_task()
                    if self.options.on_task_executed:
                        self.options.on_task_executed(task=self.task, uuid=self.uuid, response=result, parent_agent_uuid = (self.parent.uuid) if self.parent is not None else None)
                    return result

                if self.options.on_task_executed:
                    self.options.on_task_executed(task=self.task, uuid=self.uuid, response=result, parent_agent_uuid = (self.parent.uuid) if self.parent is not None else None)
                
                return result
        # If no subtasks or not allowed to split, run single task
        result = self._run_single_task()
        if self.options.on_task_executed:
            self.options.on_task_executed(task=self.task, uuid=self.uuid, response=result, parent_agent_uuid = (self.parent.uuid) if self.parent is not None else None)

        return result


    def _run_subtasks(self) -> str:
        # We can assume that the tasks are valid, as they were parsed by the SplitTask parser
        if self.tasks["needs_subtasks"] == False:
            return None
        agent_responses = []
        for task_x in self.tasks["subtasks"]:
            # Recursively run subtasks
            task_x = task(**task_x)
            agent = RecursiveAgent(
                task=task_x.task,
                uuid=task_x.uuid,
                agent_options=self.options.propagate(),
                allow_subtasks=False,
                current_layer=self.current_layer + 1,
                complexity=min(task_x.subtasks, ((self.options.max_agents[0] if len(self.options.max_agents) > 0 else 0))),
                parent=self
            )
            response = agent.run()
            agent_responses.append(response)
            
        content = self._summarize_subtask_results(agent_responses)
        return content
    
    def _run_single_task(self) -> str:
        # Run the task
        history = [
            SystemMessage("Your task is to answer the following question, using any tools that you deem necessary. If you use the web search tool, make sure you include citations (just use a pair of square brackets and a number in text, and at the end, include a citations section):"),
            HumanMessage(self.task)
            ]
        response = self.search_llm.invoke(history)
        
        history.append(response)

        for tool_call in response.tool_calls:
            name = tool_call['name']
            id = tool_call['id']
            args = tool_call['args']
            tool = self.options.tools_dict[name]

            tool_response = tool(**args)
            if self.options.on_tool_call_executed:
                self.options.on_tool_call_executed(task=self.task, uuid=self.uuid, tool_name=name, tool_args=args, tool_response=tool_response)

            history.append(
                ToolMessage(json.dumps(tool_response), tool_call_id = id)
            )
        summary = self.tools_llm.invoke(history)
        return summary.content
    def _split_task(self)->SplitTask:
        """Split the agent's task int subtasks (or not)

        Returns:
            SplitTask: an object specifying the split up tasks
        """
        split_msgs_hist = [
            self._construct_subtask_sys_msg(),
            HumanMessage(self.task),
            
        ]
        response = self.llm.invoke(split_msgs_hist + [AIMessage("1. "),])

        split_msgs_hist.append(AIMessage(content = "1. " + response.content))
        split_msgs_hist.append(self._construct_subtask_to_json_prompt())

        structured_response = self.llm.invoke(split_msgs_hist)

        split_task = self.task_split_parser.invoke(structured_response)

        # Assign UUIDs to subtasks
        for subtask in split_task["subtasks"]:
            subtask["uuid"] = str(uuid.uuid4())

        return split_task
    def _summarize_subtask_results(self, subtask_results) -> str:
        summary_history = [
            SystemMessage("Your task is to summarize this list of tasks and results into a single response to the provided task:"),
            HumanMessage(
                f"\"\"\"\n{json.dumps(subtask_results, indent=2)}\n\"\"\"\n\n"
                f"And here is the task:\n\n\"\"\n{self.task}\n\"\"\n"
                f"Also, make sure to cite everything. An in-text citation will just be a pair of square brackets, "
                f"with a number. Then, include a citations section at the end, showing all your sources. "
                f"Include citations from the sources that your are summarizing. "
                ),
        ]
        response = self.llm.invoke(summary_history)
        return response.content
    def _construct_subtask_sys_msg(self):
        return SystemMessage(f"Your task is to split this task into {self.complexity} subtasks. "
                             f"Please list the subtasks below. DO NOT attempt to answer the question yourself. "
                             f"Simply list the subtasks necessary to find the answer. "
                             f"If you don't need to split the task, respond with \"1. No subtasks needed.\" "
                             f"Keep in mind, each task will be run individually, so they won't be able to see each previous "
                             f"task's results. Once each task is executed, you will summarize all the results to complete your task,"
                             f"so **DO NOT** include any summarization tasks. For researching tasks, **think about the task you are given, and simply split it up into "
                             f"simpler ones.**"
                             )
    def _construct_subtask_to_json_prompt(self):
        return HumanMessage(
            f"Great. Now, please put this into JSON format following this schema: "
            f"{SplitTask.model_json_schema()}\n\n"
            f"If you don't need subtasks, still use the schema, but set 'needs_subtasks' to False and tasks to an empty list."
            )