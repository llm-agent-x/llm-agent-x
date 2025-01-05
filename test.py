import argparse
import json
from typing import List
from os import getenv
from icecream import ic
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
import llm_agent_x
from llm_agent_x import int_to_base26

# Initialize Searx search
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = getenv("OUTPUT_DIR", "./output/")
def web_search(query: str, num_results: int) -> List:
    """
    Perform a web search using the Searx search engine.
    """
    try:
        results = search.results(query, num_results=num_results)
        return json.dumps(results)
    except Exception as e:
        ic(f"Error during web search: {e}")
        return json.dumps([])

# Mermaid.js flowchart data
flowchart = ["graph TD"]
tasks_dict: dict = {}

def get_task_label(task):
    if task not in tasks_dict:
        tasks_dict[task] = int_to_base26(len(tasks_dict))
    return tasks_dict[task]

# Event handlers with better formatting for nodes and edges
def on_subtasks_created(parts, task_thread, **kwargs):
    tasks = parts["tasks"] if isinstance(parts, dict) else parts
    for part in tasks:
        task_label = get_task_label(part["task"])
        parent_label = get_task_label(task_thread[-1])
        flowchart.append(f'    {parent_label} -->|Subtask| {task_label}[{part["task"]}]')

def on_task_executed(result, task_thread, **kwargs):
    task_label = get_task_label(result.get("task", "Unknown Task"))
    parent_label = get_task_label(task_thread[-1])
    # flowchart.append(f'    {parent_label} -->|Executed| {task_label}[{result.get("task", "Unknown Task")}]')

def on_tool_call_executed(tool_call, tool_response, task_thread, **kwargs):
    tool_name = tool_call.get("name", "Unknown Tool")
    parent_label = get_task_label(task_thread[-1])
    tool_label = get_task_label(tool_name)
    flowchart.append(f'    {parent_label} -->|Tool Used| {tool_label}[{tool_name}]')

# Main function
def main(task, output_file):
    # Initialize the LLM
    llm = ChatOpenAI(
        model="qwen2.5-coder-long:latest",
        base_url=getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    task_label = get_task_label(task)
    flowchart.append(f"    {task_label}[{task}]")
    
    # Initialize the agent
    agent = llm_agent_x.Agent(
        task=task,
        llm=llm,
        max_layers=3,
        max_agents=5,
        search_tool=web_search,
        on_subtasks_created=on_subtasks_created,
        on_task_executed=on_task_executed,
        on_tool_call_executed=on_tool_call_executed,
    )
    
    # Run the agent
    agent_response = agent.run()
    ic(agent_response)
    
    # Write results to output file
    with open(output_dir + output_file, "w") as f:
        f.write(agent_response.get("result", "No result"))

    # Write the flowchart to a Mermaid.js file
    with open(output_dir + "flowchart.mmd", "w") as f:
        f.write("\n".join(flowchart))

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Research tool using a custom LLM agent.")
    parser.add_argument("task", type=str, help="The research task to be performed.")
    parser.add_argument("output_file", type=str, help="The file where the output will be saved.")
    
    args = parser.parse_args()
    main(task=args.task, output_file=args.output_file)
