import argparse
import json
from os import getenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
import llm_agent_x
from llm_agent_x import int_to_base26
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=True)
llm = ChatOpenAI(
        base_url=getenv("OPENAI_BASE_URL"),
        api_key=getenv("OPENAI_API_KEY"),
        model="qwen2.5-coder-long:latest",
        temperature=0,
    )
# Initialize Searx search
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))

flowchart = ["flowchart TD"]
task_ids: dict[str, str] = {}


def get_or_set_task_id(id: str) -> str|None:
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
    Perform a web search using the Searx search engine.
    Returns: List of search results. Each result is a dictionary with keys: title, link, snippet.
    """
    # print(f"Performing web search with query: {query} and num_results: {num_results}")
    try:
        results = search.results(query, num_results=num_results)

        return json.dumps(results)
    except Exception as error:
        print(error)
        return json.dumps([])

def pre_tasks_executed(task, uuid, parent_agent_uuid):
    id = get_or_set_task_id(uuid)

    parent_id = get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None

    if parent_agent_uuid is not None:
        add_to_flowchart(f"{parent_id} -->|Subtask| {id}[{task}]")
    else:
        add_to_flowchart(f"{id}[{task}]")

def on_task_executed(task, uuid, response, parent_agent_uuid):
    id = get_or_set_task_id(uuid)

    parent_id = get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None

    if parent_agent_uuid is not None:
        add_to_flowchart(f"{id} -->|Completed| {parent_id}")


def on_tool_call_executed(task, uuid, tool_name, tool_args,tool_response):
    add_to_flowchart(
        f"{get_or_set_task_id(uuid)} -->|Tool call| {get_or_set_task_id(f"{uuid}: {tool_name}")}[{tool_name}]"
    )
    add_to_flowchart(
        f"{get_or_set_task_id(f"{uuid}: {tool_name}")} --> {get_or_set_task_id(uuid)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM agent.")
    parser.add_argument("task", type=str, help="The task to execute.")
    parser.add_argument(
        "--max_layers", type=int, default=3, help="The maximum number of layers."
    )
    parser.add_argument(
        "--output", type=str, default="output.md", help="The output file path"
    )
    parser.add_argument(
        "--model", type=str, default=getenv("DEFAULT_LLM"), help="The name of the LLM to use"
    )
    args = parser.parse_args()

    search_llm = llm.bind_tools([web_search])
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
            search_llm=search_llm,
            tools=[],
            allow_search=True,
            allow_tools=False,
            tools_dict={
                "web_search": web_search,
            },
        ),
    )

    # Ensure the output directory exists

    # Call agent.run()
    response = agent.run()

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "flowchart.mmd").open("w") as flowchart_o:
        flowchart_o.write(render_flowchart())

    with (output_dir / args.output).open("w") as output:
        output.write(response)
