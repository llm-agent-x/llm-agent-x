import json
import uuid

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.output_parsers import JsonOutputParser

class task(BaseModel):
    task: str
    uuid: str

class TaskObject(BaseModel):
    task: str
    subtasks: int
    allow_search: bool
    allow_tools: bool

class SplitTask(BaseModel):
    needs_subtasks: bool
    subtasks: list[TaskObject]
    def __bool__(self):
        return self.needs_subtasks

class AgentOptions(BaseModel):
    max_layers: int=2
    max_agents: list|int = [5, 3, 2]
    search_tool: callable
    on_subtasks_created: callable
    on_task_executed: callable
    on_tool_call_executed: callable
    task_tree: list[task]
    llm: any
    search_llm: any
    tools: list
    allow_search: bool
    allow_tools: bool
    tools_dict: dict

    def propagate(self):
        return AgentOptions(
            max_layers=self.max_layers,
            max_agents=(lambda x: x if isinstance(x, int) else x[1:])(self.max_agents),
            search_tool=self.search_tool,
            on_subtasks_created=self.on_subtasks_created,
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


class Agent1():
    def __init__(self, task, uuid= str(uuid.uuid4()), agent_options: AgentOptions = AgentOptions(), allow_subtasks = True, current_layer = 0, complexity = 3):
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
    def run(self) -> str:
        try:
            # Determine whether/not to split task (based on allow_subtasks and the layer)
            if self.allow_subtasks and (self.current_layer < self.options.max_layers):
                # Split task
                self.tasks = self._split_task()

                if self.tasks:
                    self.options.on_subtasks_created(task=self.task, uuid=self.uuid, subtasks=[task.task for task in self.tasks.subtasks])
                    return self._run_subtasks()
            # If no subtasks or not allowed to split, run single task
            return self._run_single_task()
        except Exception as e:
            return f"An error occurred: {str(e)}"


    def _run_subtasks(self) -> str:
        # We can assume that the tasks are valid, as they were parsed by the SplitTask parser

        agent_responses = []
        for task in self.tasks.subtasks:
            # Recursively run subtasks
            agent = Agent1(
                task=task.task,
                agent_options=self.options.propagate(),
                allow_subtasks=False,
                current_layer=self.current_layer + 1,
                complexity=task.subtasks
            )
            response = agent.run()
            agent_responses.append(response)
            self.options.on_task_executed(task=task.task, uuid=agent.uuid, response=response)
            
        content = self._summarize_subtask_results(agent_responses)
        return content
    
    def _run_single_task(self) -> str:
        # Run the task
        history = [
            SystemMessage("Your task is to answer the following question, using any tools that you deem necessary:"),
            HumanMessage(self.task)
            ]
        response = self.tools_llm.invoke(history)
        
        history.append(response)

        for tool_call in response.tool_calls:
            name = tool_call['name']
            id = tool_call['id']
            args = tool_call['args']
            tool = self.options.tools_dict[name]

            tool_response = tool(**args)
            self.options.on_tool_call_executed(task=self.task, uuid=self.uuid, tool_name=name, tool_args=args, tool_response=tool_response)

            history.append(
                ToolMessage(json.dumps(tool_response), tool_call_id = id)
            )
        summary = self.tools_llm.invoke(history)
        return summary.content
    def _split_task(self)->SplitTask:
        """Split the agent's task int subtasks (or not)

        Returns:
            SplitTask: _description_
        """
        split_msgs_hist = [
            self._construct_subtask_sys_msg(),
            AIMessage("1. "),
        ]
        response = self.llm.invoke(split_msgs_hist)
        split_msgs_hist.append(response)
        split_msgs_hist.append(self._construct_subtask_to_json_prompt())

        structured_response = self.llm.invoke(split_msgs_hist)
        split_task = self.task_split_parser.parse(structured_response)

        return split_task
    def _summarize_subtask_results(self, subtask_results) -> str:
        summary_history = [
            SystemMessage("Your task is to summarize this list of tasks and results into a single response to the provided task:"),
            HumanMessage(
                f"\"\"\"\n{json.dumps(subtask_results, indent=2)}\n\"\"\"\n\n"
                f"And here is the task:\n\n\"\"\n{self.task}\n\"\""
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
                             f"task's results. Once each task is executed, you will summarize all the results to complete your task, so don't include any summarization tasks."
                             )
    def _construct_subtask_to_json_prompt(self):
        return HumanMessage(
            f"Great. Now, please put this into JSON format following this schema: "
            f"{SplitTask.model_json_schema()}\n\n"
            f"If you don't need subtasks, still use the schema, but set 'needs_subtasks' to False and tasks to an empty list."
            )