import argparse
import json
import time
from os import getenv
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.tree import Tree
from rich.live import Live
from rich.text import Text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
import llm_agent_x
from llm_agent_x import int_to_base26

# Load environment variables
load_dotenv(".env", override=True)

# Initialize Console and Live Display
console = Console()
live = None  # Global live display manager
task_tree = Tree("Agent Execution")  # Root of the real-time task tree
task_nodes = {}  # Store references to tree nodes

# Initialize LLM and Search
llm = ChatOpenAI(
    base_url=getenv("OPENAI_BASE_URL"),
    api_key=getenv("OPENAI_API_KEY"),
    model="qwen2.5-coder-long:latest",
    temperature=0,
)
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))

# Flowchart Tracking
flowchart = ["flowchart TD"]
task_ids: dict[str, str] = {}


def get_or_set_task_id(id: str) -> str | None:
    if id not in task_ids:
        result = int_to_base26(len(task_ids))
        task_ids[id] = result
        return result
    else:
        return task_ids.get(id)


def add_to_flowchart(line: str):
    flowchart.append(f"    {line}")


def render_flowchart():
    return "\n".join(flowchart)


def web_search(query: str, num_results: int) -> str:
    try:
        results = search.results(query, num_results=num_results)
        return json.dumps(results)
    except Exception as error:
        print(error)
        return json.dumps([])

def exec_python(code, globals=None, locals=None):
    """
    Execute the given code with empty globals and locals.

    This function executes the provided code string in an isolated
    environment where both the global and local namespaces are empty,
    preventing the code from accessing or modifying external variables.

    Parameters:
    code (str): The code to be executed.
    globals (dict, optional): A dictionary of global variables. Defaults to None.
    locals (dict, optional): A dictionary of local variables. Defaults to None.

    Returns:
    None
    """
    exec(code, {}, {})


def pre_tasks_executed(task, uuid, parent_agent_uuid):
    global live

    id = get_or_set_task_id(uuid)
    parent_id = get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None

    # Flowchart Update
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{parent_id} -->|Subtask| {id}[{task}]")
    else:
        add_to_flowchart(f"{id}[{task}]")

    # Real-time Hierarchy Update
    task_text = Text(task, style="bold yellow")
    if parent_agent_uuid is None:
        task_nodes[uuid] = task_tree.add(task_text)  # Top-level task
    else:
        task_nodes[uuid] = task_nodes[parent_agent_uuid].add(task_text)  # Subtask

    if live:
        live.update(task_tree)


def on_task_executed(task, uuid, response, parent_agent_uuid):
    global live

    id = get_or_set_task_id(uuid)
    parent_id = get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None

    # Flowchart Update
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{id} -->|Completed| {parent_id}")
    add_to_flowchart(f"{id} --> |Result| (\"`{response}`\")")
    # Real-time Hierarchy Update
    if uuid in task_nodes:
        task_nodes[uuid].label = Text(f"{task} ✅", style="green")

    if live:
        live.update(task_tree)


def on_tool_call_executed(task, uuid, tool_name, tool_args, tool_response, success=True):
    global live

    tool_task_id = f"{uuid}: {tool_name}"
    add_to_flowchart(f"{get_or_set_task_id(uuid)} -->|Tool call| {get_or_set_task_id(tool_task_id)}[{tool_name}]")
    add_to_flowchart(f"{get_or_set_task_id(tool_task_id)} --> {get_or_set_task_id(uuid)}")

    # Real-time Hierarchy Update
    tool_text = Text(f"{tool_name} 🔧 {json.dumps(tool_args, indent=0).replace("\n", '')}", style="blue")
    if not success:
        tool_text.stylize("bold red")
        task_nodes[tool_task_id] = task_nodes[uuid].add(tool_text)
        task_nodes[tool_task_id].label = Text(f"{tool_name} ❌", style="red")
    else:
        task_nodes[tool_task_id] = task_nodes[uuid].add(tool_text)

    if live:
        live.update(task_tree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM agent.")
    parser.add_argument("task", type=str, help="The task to execute.")
    parser.add_argument("--max_layers", type=int, default=3, help="The maximum number of layers.")
    parser.add_argument("--output", type=str, default="output.md", help="The output file path")
    parser.add_argument("--model", type=str, default=getenv("DEFAULT_LLM"), help="The name of the LLM to use")
    args = parser.parse_args()

    tool_llm = llm.bind_tools([web_search]) #, exec_python])
    print(f"Using {llm.name}")
    # Create the agent
    agent = llm_agent_x.RecursiveAgent(
        task=args.task,
        agent_options=llm_agent_x.RecursiveAgentOptions(
            max_layers=args.max_layers,
            search_tool=web_search,
            pre_task_executed=pre_tasks_executed,
            on_task_executed=on_task_executed,
            on_tool_call_executed=on_tool_call_executed,
            llm=llm,
            tool_llm=tool_llm,
            tools=[],
            allow_search=True,
            allow_tools=False,
            tools_dict={"web_search": web_search}, # "exec_python": exec_python, "exec": exec_python},
            task_limits=llm_agent_x.TaskLimit.from_array([2,3, 2, 0])
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Start Live Display
    with Live(task_tree, console=console, auto_refresh=True) as live_display:
        live = live_display  # Assign to global variable
        response = agent.run()  # Execute the agent

    # Save Flowchart
    with (output_dir / "flowchart.mmd").open("w") as flowchart_o:
        flowchart_o.write(render_flowchart())

    # Save Response
    with (output_dir / args.output).open("w") as output:
        output.write(response)
