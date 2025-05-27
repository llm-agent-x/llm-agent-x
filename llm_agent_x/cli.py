import argparse
import json
import time
from os import getenv, environ
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.tree import Tree
from rich.live import Live
from rich.text import Text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper

from . import (
    int_to_base26,
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
)  # Adjusted import
from .backend import AppendMerger, LLMMerger

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Setup (only needed once)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))

# Load environment variables
load_dotenv(
    ".env", override=True
)  # This might need adjustment if .env is not in the right place relative to cli.py
# Initialize Console and Live Display
console = Console()
live = None  #  display manager
task_tree = Tree("Agent Execution")  # Root of the real-time task tree
task_nodes = {}  # Store references to tree nodes

# Initialize LLM and Search

llm = ChatOpenAI(
    base_url=getenv("OPENAI_BASE_URL"),
    api_key=getenv("OPENAI_API_KEY"),
    model=getenv("DEFAULT_LLM", "gpt-4-0613"),
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
    """
    Perform a web search with the given query and number of results, returning JSON-formatted results. **Make sure to be very specific in your search phrase. Ask for specific information, not general information.**

    :param query: The search query.
    :param num_results: The number of results to return.
    :return: A JSON-formatted string containing the search results.
    """
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

    id = get_or_set_task_id(uuid)
    parent_id = (
        get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None
    )

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

    id = get_or_set_task_id(uuid)
    parent_id = (
        get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None
    )

    # Flowchart Update
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{id} -->|Completed| {parent_id}")
    add_to_flowchart(f'{id} --> |Result| ("`{response}`")')
    # Real-time Hierarchy Update
    if uuid in task_nodes:
        task_nodes[uuid].label = Text(f"{task} ✅", style="green")

    if live:
        live.update(task_tree)


def on_tool_call_executed(
    task, uuid, tool_name, tool_args, tool_response, success=True
):

    tool_task_id = f"{uuid}: {tool_name}"
    add_to_flowchart(
        f"{get_or_set_task_id(uuid)} -->|Tool call| {get_or_set_task_id(tool_task_id)}[{tool_name}]"
    )
    add_to_flowchart(
        f"{get_or_set_task_id(tool_task_id)} --> {get_or_set_task_id(uuid)}"
    )

    text_json = json.dumps(tool_args, indent=0).replace("\\n", "")
    # Real-time Hierarchy Update
    tool_text = Text(f"{tool_name} 🔧 {text_json}", style="blue")
    if not success:
        tool_text.stylize("bold red")
        task_nodes[tool_task_id] = task_nodes[uuid].add(tool_text)
        task_nodes[tool_task_id].label = Text(f"{tool_name} ❌", style="red")
    else:
        task_nodes[tool_task_id] = task_nodes[uuid].add(tool_text)

    if live:
        live.update(task_tree)


def main():
    # Ensure 'live' can be assigned in this function
    parser = argparse.ArgumentParser(description="Run the LLM agent.")
    parser.add_argument("task", type=str, help="The task to execute.")
    parser.add_argument("--u_inst", type=str, help="The task to execute.", default="")
    parser.add_argument(
        "--max_layers", type=int, default=3, help="The maximum number of layers."
    )
    parser.add_argument(
        "--output", type=str, default="output.md", help="The output file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getenv("DEFAULT_LLM"),
        help="The name of the LLM to use",
    )
    parser.add_argument("--task_limit", type=str, default="[3,2,2,0]")

    parser.add_argument("--merger", type=str, default="ai")
    args = parser.parse_args()

    tool_llm = llm.bind_tools([web_search])  # , exec_python])
    with tracer.start_as_current_span("A") as span:
        # Create the agent
        agent = RecursiveAgent(  # Adjusted: Removed llm_agent_x prefix
            task=args.task,
            u_inst=args.u_inst,
            tracer_span=span,
            agent_options=RecursiveAgentOptions(  # Adjusted: Removed llm_agent_x prefix
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
                tools_dict={
                    "web_search": web_search
                },  # "exec_python": exec_python, "exec": exec_python},
                task_limits=TaskLimit.from_array(
                    eval(args.task_limit)
                ),  # Adjusted: Removed llm_agent_x prefix
                merger={"ai": LLMMerger, "append": AppendMerger}[args.merger],
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


if __name__ == "__main__":
    main()
