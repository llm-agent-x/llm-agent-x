import json
from pathlib import Path
from rich.tree import Tree
from typing import Dict
from llm_agent_x import int_to_base26

from rich.text import Text

from llm_agent_x.console import console, task_tree, live

task_nodes: Dict[str, Tree] = {}  # Explicit type hint for task_nodes


def get_or_set_task_id(id: str) -> str | None:
    if id not in task_ids:
        result = int_to_base26(len(task_ids))
        task_ids[id] = result
        return result
    else:
        return task_ids.get(id)


def add_to_flowchart(line: str):
    flowchart.append(f"    {line}")


def pre_tasks_executed(task, uuid, parent_agent_uuid):
    id = get_or_set_task_id(uuid)
    parent_id = (
        get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None
    )

    # Flowchart Update (always runs)
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{parent_id} -->|Subtask| {id}[{task}]")
    else:
        add_to_flowchart(f"{id}[{task}]")

    # Real-time Hierarchy Update (conditional on live display being active)
    if live:
        task_text = Text(task, style="bold yellow")
        if parent_agent_uuid is None:
            task_nodes[uuid] = task_tree.add(task_text)  # Top-level task
        elif parent_agent_uuid in task_nodes:  # Check if parent node exists
            task_nodes[uuid] = task_nodes[parent_agent_uuid].add(task_text)  # Subtask
        else:
            # This case should ideally not happen if tasks are processed in order.
            # Adding as a direct child of the root tree if parent is missing for some reason.
            console.print(
                f"[yellow]Warning: Parent node {parent_agent_uuid} for task '{task}' not found in tree. Adding as top-level.[/yellow]"
            )
            task_nodes[uuid] = task_tree.add(task_text)

        if live is not None:  # Check that live has been initialized before updating.
            live.update(task_tree)


def on_task_executed(task, uuid, response, parent_agent_uuid):
    id = get_or_set_task_id(uuid)
    parent_id = (
        get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None
    )

    # Flowchart Update (always runs)
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{id} -->|Completed| {parent_id}")
    add_to_flowchart(f'{id} --> |Result| ("`{response}`")')

    # Real-time Hierarchy Update (conditional on live display being active)
    if live:
        if uuid in task_nodes:
            task_nodes[uuid].label = Text(f"{task} ✅", style="green")
        else:
            console.print(
                f"[yellow]Warning: Task node {uuid} for task '{task}' not found in tree for completion update.[/yellow]"
            )
        if live is not None:  # Check that live has been initialized before updating.
            live.update(task_tree)


def on_tool_call_executed(
    task, uuid, tool_name, tool_args, tool_response, success=True, tool_call_id=None
):
    tool_task_id = f"{uuid}: {tool_name}"  # Unique ID for the tool call visualization

    # Flowchart Update (always runs)
    add_to_flowchart(
        f"{get_or_set_task_id(uuid)} -->|Tool call| {get_or_set_task_id(tool_task_id)}[{tool_name}]"
    )
    add_to_flowchart(
        f"{get_or_set_task_id(tool_task_id)} --> {get_or_set_task_id(uuid)}"
    )

    # Real-time Hierarchy Update (conditional on live display being active)
    if live:
        text_json = json.dumps(tool_args, indent=0).replace("\\n", "")
        tool_text = Text(f"{tool_name} 🔧 {text_json}", style="blue")

        if uuid in task_nodes:  # Check if parent task node exists
            if not success:
                tool_text.stylize("bold red")
                # Create a new node for the tool call itself, even if it failed
                tool_node = task_nodes[uuid].add(tool_text)
                tool_node.label = Text(
                    f"{tool_name} ❌", style="red"
                )  # Update its label to show failure
                task_nodes[tool_task_id] = (
                    tool_node  # Store reference if needed elsewhere, though not strictly necessary for this structure
                )
            else:
                task_nodes[tool_task_id] = task_nodes[uuid].add(tool_text)
        else:
            console.print(
                f"[yellow]Warning: Parent task node {uuid} for tool '{tool_name}' not found in tree.[/yellow]"
            )
        if live is not None:  # Check that live has been initialized before updating.
            live.update(task_tree)


# Flowchart Tracking
flowchart = ["flowchart TD"]
task_ids: dict[str, str] = {}


def render_flowchart():
    return "\n".join(flowchart)


def save_flowchart(output_dir: Path):
    # Save Flowchart
    flowchart_file = output_dir / "flowchart.mmd"
    with flowchart_file.open("w") as flowchart_o:
        flowchart_o.write(render_flowchart())
    console.print(f"Flowchart saved to {flowchart_file}")
