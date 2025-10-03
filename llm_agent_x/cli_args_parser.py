# In llm_agent_x/cli_args_parser.py

import argparse
from os import getenv
from dotenv import load_dotenv

load_dotenv(".env", override=True)

parser = argparse.ArgumentParser(
    description="Run an LLM agent. Choose an agent type and provide the relevant options.",
    formatter_class=argparse.RawTextHelpFormatter,  # Allows for better help text formatting
)

# --- Primary Agent Selection ---
parser.add_argument(
    "agent_type",
    type=str,
    choices=["recursive", "dag"],
    help="The type of agent to run:\n"
    "  recursive - A hierarchical agent that recursively breaks down tasks.\n"
    "  dag       - An agent that executes a directed acyclic graph of tasks.",
)

parser.add_argument(
    "task", type=str, help="The main task or objective for the agent to execute."
)

# --- Common Options ---
common_group = parser.add_argument_group("Common Options (apply to all agents)")
common_group.add_argument(
    "--model",
    type=str,
    default=getenv("DEFAULT_LLM", "gpt-4o-mini"),
    help="The name of the LLM to use (e.g., gpt-4o-mini, gpt-4o).",
)
common_group.add_argument(
    "--u_inst",
    type=str,
    help="General user instructions to guide the agent's behavior.",
    default="",
)
common_group.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to save the final response to a file.",
)
common_group.add_argument(
    "--enable-python-execution",
    action="store_true",
    help="Enable the exec_python tool for the agent. (Requires Docker for sandbox mode)",
)
common_group.add_argument(
    "--dev-mode",
    action="store_true",
    help="Enable development mode (e.g., for icecream outputs).",
)


# --- Recursive Agent Options ---
recursive_group = parser.add_argument_group("Recursive Agent Options")
recursive_group.add_argument(
    "--task_type",
    type=str,
    choices=["research", "search", "basic", "text/reasoning"],
    default="research",
    help="[RECURSIVE-ONLY] The type of the initial task.",
)
recursive_group.add_argument(
    "--task_limit",
    type=str,
    default="[3,2,2,0]",
    help="[RECURSIVE-ONLY] Task limits per layer as a Python list string e.g., '[3,2,2,0]'",
)
recursive_group.add_argument(
    "--merger",
    type=str,
    default="ai",
    choices=["ai", "append", "algorithmic"],
    help="[RECURSIVE-ONLY] Merger type for subtask results.",
)
recursive_group.add_argument(
    "--align_summaries",
    type=bool,
    default=True,
    help="[RECURSIVE-ONLY] Whether to align summaries with user instructions.",
)
recursive_group.add_argument(
    "--no-tree",
    action="store_true",
    help="[RECURSIVE-ONLY] Disable the real-time MermaidJS tree view.",
)
recursive_group.add_argument(
    "--default_subtask_type",
    type=str,
    default="basic",
    choices=["research", "search", "basic", "text/reasoning"],
    help="[RECURSIVE-ONLY] The default task type for all subtasks.",
)
recursive_group.add_argument(
    "--mcp-config",
    type=str,
    help="[RECURSIVE-ONLY] Path to the MCP config file.",
)


# --- DAG Agent Options ---
dag_group = parser.add_argument_group("DAG Agent Options")
dag_group.add_argument(
    "--dag-documents",
    type=str,
    help="[DAG-ONLY] Path to a JSON file defining initial documents/tasks for the DAG. "
    "Format: [{'name': 'doc_name', 'content': 'doc_content'}, ...]",
)
dag_group.add_argument(
    "--max-grace-attempts",
    type=int,
    default=1,
    help="[DAG-ONLY] The number of extra retries granted by the retry analyst.",
)

dag_group.add_argument(
    "--global-proposal-limit",
    type=int,
    default=5,
)

parser.add_argument(
    "--disable-web-search",
    type=bool,
    help="[DAG-ONLY] Whether to disable the web search tool.",
)
