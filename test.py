import argparse
import json
from typing import List
from os import getenv
from icecream import ic
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
import llm_agent_x
from llm_agent_x import int_to_base26
import uuid

from dotenv import load_dotenv

load_dotenv()

# Initialize Searx search
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = getenv("OUTPUT_DIR", "./output/")

# Initialize log structure
log = {"tasks": []}

def web_search(query: str, num_results: int) -> List:
    """
    Perform a web search using the Searx search engine.
    Returns: List of search results. Each result is a dictionary with keys: title, link, snippet.
    """
    print(f"Performing web search with query: {query} and num_results: {num_results}")
    try:
        results = search.results(query, num_results=num_results)
        
        return json.dumps(results)
    except Exception as error:
        print(error)
        return json.dumps([])

# Mermaid.js flowchart data
flowchart = ["graph TD"]
tasks_dict: dict = {}

def get_task_label(agent_id):
    if agent_id not in tasks_dict:
        tasks_dict[agent_id] = int_to_base26(len(tasks_dict))
    return tasks_dict[agent_id]

def add_flowchart_line(line):
    flowchart.append(f'{line}')

# Event handlers with better formatting for nodes and edges
def on_subtasks_created(parts, task_thread, agent_id, agent_ids, **kwargs):
    tasks = parts["tasks"] if isinstance(parts, dict) else parts
    for i, part in enumerate(tasks):
        task_label = get_task_label(agent_ids[i])
        parent_label = get_task_label(agent_id)
        ic(parent_label)
        if f'    {parent_label} -->|Subtask| {task_label}[\"{part["task"]}\"]' not in flowchart:
            add_flowchart_line(f'    {parent_label} -->|Subtask| {task_label}[\"{part["task"]}\"]')

def on_task_executed(result, task_thread, agent_id, **kwargs):
    task_label = get_task_label(agent_id)
    parent_label = get_task_label(agent_id)
    log_entry = {
        "task": result.get("task", "Unknown Task"),
        "result": result.get("result", "No result"),
        "subtasks": []
    }
    current_log = log
    for task in task_thread:
        for subtask in current_log["tasks"]:
            if subtask["task"] == task:
                current_log = subtask
                break
    if "tasks" not in current_log:
        current_log["tasks"] = []
    current_log["tasks"].append(log_entry)

def on_tool_call_executed(tool_call, tool_response, task_thread, agent_id, **kwargs):
    tool_name = tool_call.get("name", "Unknown Tool")
    parent_label = get_task_label(agent_id)
    tool_label = get_task_label(f"{tool_name}_{len(flowchart)}")
    response_label = get_task_label(f"response_{len(flowchart)}")
    add_flowchart_line(f'    {parent_label} -->|Tool Used| {tool_label}[{tool_name}]')
    add_flowchart_line(f'    {tool_label} -->|Response| {parent_label}')

# Main function
def main(task, output_file):
    print(f"Starting main function with task: {task} and output_file: {output_file}")
    # Initialize the LLM
    llm = ChatOpenAI(
        model="qwen2.5-coder-long:latest",
        base_url=getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    agent_id = str(uuid.uuid4())
    task_label = get_task_label(agent_id)
    add_flowchart_line(f"    {task_label}[\"{task}\"]")
    
    # Initialize the agent
    agent = llm_agent_x.Agent(
        task=task,
        agent_id=agent_id,
        llm=llm,
        max_layers=2,
        max_agents=[5, 3, 2],
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

    # Write the log to a JSON file
    with open(output_dir + "log.json", "w") as f:
        json.dump(log, f, indent=4)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Research tool using a custom LLM agent.")
    parser.add_argument("task", type=str, help="The research task to be performed.")
    parser.add_argument("output_file", type=str, help="The file where the output will be saved.")
    
    args = parser.parse_args()
    main(task=args.task, output_file=args.output_file)
