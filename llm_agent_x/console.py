# Initialize Console
from rich.console import Console
from rich.tree import Tree
from typing import Optional
from rich.live import Live

console = Console()
task_tree = Tree("Agent Execution")
live: Optional[Live] = None
