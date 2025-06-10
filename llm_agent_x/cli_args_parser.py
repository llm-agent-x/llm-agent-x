import argparse
from os import getenv
from dotenv import load_dotenv

load_dotenv(".env", override=True)

parser = argparse.ArgumentParser(description="Run the LLM agent.")
parser.add_argument("task", type=str, help="The task to execute.")

parser.add_argument(
    "--task_type",
    type=str,
    choices=["research", "search", "basic", "text/reasoning"],
    default="research",
    help="The type of task to execute.",
)
parser.add_argument(
    "--u_inst", type=str, help="User instructions for the task.", default=""
)
parser.add_argument(
    "--max_layers",
    type=int,
    default=3,
    help="The maximum number of layers (deprecated, use task_limit).",
)
parser.add_argument(
    "--output", type=str, default=None, help="The output file path"
)
parser.add_argument(
    "--model",
    type=str,
    default=getenv(
        "DEFAULT_LLM", "gpt-4o-mini"
    ),  # Ensure default matches initialization
    help="The name of the LLM to use",
)
parser.add_argument(
    "--task_limit",
    type=str,
    default="[3,2,2,0]",
    help="Task limits per layer as a Python list string e.g., '[3,2,2,0]'",
)
parser.add_argument(
    "--merger",
    type=str,
    default="ai",
    choices=["ai", "append", "algorithmic"],
    help="Merger type: 'ai' or 'append'.",
)

parser.add_argument(
    "--align_summaries",
    type=bool,
    default=True,
    help="Whether to align summaries with user instructions.",
)
parser.add_argument(
    "--no-tree", action="store_true", help="Disable the real-time tree view."
)
parser.add_argument(
    "--default_subtask_type",
    type=str,
    default="basic",
    choices=["research", "search", "basic", "text/reasoning"],
    help="The default task type to apply to all subtasks. Should be one of 'research', 'search', 'basic', or 'text/reasoning'.",
)

parser.add_argument(
    "--enable-python-execution",
    help="Enable the exec_python tool for the agent. (Requires Docker for sandbox mode)",
)

parser.add_argument(
    "--mcp-config",
    type=str,
    help="Path to the MCP config file",
)