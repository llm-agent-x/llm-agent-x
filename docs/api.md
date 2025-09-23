# Python API Usage

LLM Agent X offers two distinct agent architectures for programmatic use, each suited for different kinds of tasks.

*   **`RecursiveAgent`**: A hierarchical agent that excels at tasks that can be broken down into a clear, tree-like structure. It operates by recursively decomposing a task into sub-tasks until they are simple enough to be executed directly. This is the original agent architecture and is ideal for straightforward, structured problems.

*   **`DAGAgent`**: An evolution of the recursive model, the `DAGAgent` (Directed Acyclic Graph Agent) treats tasks as nodes in a graph. This allows for complex, non-linear dependencies where a task can depend on multiple other tasks, not just a single parent. It features a more sophisticated, multi-phase planning and execution process, making it suitable for complex problems that require adaptation and overcoming uncertainty.

This guide covers the API for both architectures.

## `RecursiveAgent`

The `RecursiveAgent` is the original hierarchical agent in LLM Agent X. It's ideal for tasks that can be neatly broken down into a tree of sub-tasks.

### Core Concepts (`RecursiveAgent`)

-   **`RecursiveAgent`**: The main class that takes a task, decomposes it if necessary, and executes it.
-   **`RecursiveAgentOptions`**: A Pydantic model to configure the behavior of `RecursiveAgent`.
-   **Task Limits**: A configuration (`task_limits`) that defines how many subtasks can be created at each level of recursion, controlling the shape of the execution tree.

### Getting Started (`RecursiveAgent`)

Here's a basic example of how to use `RecursiveAgent`:

```python
import asyncio
from llm_agent_x.agents.recursive_agent import RecursiveAgent, RecursiveAgentOptions, TaskLimit
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from llm_agent_x.tools.brave_web_search import brave_web_search

async def main():
    # 1. Configure the LLM
    client = AsyncOpenAI()
    llm = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))

    # 2. Define Agent Options
    agent_options = RecursiveAgentOptions(
        llm=llm,
        tools=[brave_web_search],
        tools_dict={"web_search": brave_web_search},
        task_limits=TaskLimit.from_array([2, 1, 0]), # Max 2 subtasks, then 1, then 0
    )

    # 3. Create and Run the Agent
    agent = RecursiveAgent(
        task="Research the benefits of renewable energy sources and summarize them.",
        u_inst="Focus on solar and wind power. Keep the summary concise.",
        agent_options=agent_options,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### `RecursiveAgentOptions`

This Pydantic model holds the configuration for a `RecursiveAgent`. Key fields include:
-   `llm`, `tools`, `tools_dict`, `task_limits`.
-   `merger`: The strategy for merging results from subtasks (`LLMMerger`, `AppendMerger`).
-   `pre_task_executed`, `on_task_executed`: Callbacks for monitoring the agent's lifecycle.

---

## `DAGAgent`

The `DAGAgent` provides a more powerful and flexible approach to task execution by modeling the workflow as a Directed Acyclic Graph (DAG). This is ideal for complex scenarios where tasks have intricate dependencies.

### Core Concepts (`DAGAgent`)

-   **`TaskRegistry`**: The central hub for a `DAGAgent`'s operation. It holds all tasks, documents, and their states (e.g., pending, running, complete). You initialize a registry and populate it with initial data and root tasks.
-   **`Task`**: A Pydantic model representing a single node in the graph. Each task has a unique ID, a description, a status, and a set of dependencies on other tasks.
-   **Hybrid Planning**: The `DAGAgent` uses a two-stage planning process:
    1.  **Initial Planning (Top-Down)**: When a task is marked with `needs_planning=True`, a "planner" agent creates an initial, high-level execution plan, breaking the root task into several sub-tasks with defined dependencies.
    2.  **Adaptive Decomposition (Bottom-Up)**: During execution, a task can be flagged with `can_request_new_subtasks=True`. If such a task is still too complex, an "explorer" agent can propose a new set of more granular sub-tasks. These are then reviewed and integrated into the main graph.

### Getting Started with `DAGAgent`

Here is a typical workflow for using the `DAGAgent`:

```python
import asyncio
from llm_agent_x.agents.dag_agent import DAGAgent, TaskRegistry, Task
from llm_agent_x.tools.brave_web_search import brave_web_search # Example tool

async def run_dag_agent():
    # 1. Initialize the Task Registry
    registry = TaskRegistry()

    # 2. Add Initial Data (Optional)
    registry.add_document(
        "Financial_Report_Q1",
        "Q1 revenue was $1.2M with a profit of $200k."
    )
    registry.add_document(
        "Market_Analysis_Q1",
        "Competitor A launched a new product, impacting our market share by 5%."
    )

    # 3. Define the Root Task
    root_task = Task(
        id="ROOT_Q1_BRIEFING",
        desc="Create a comprehensive investor briefing for Q1. First, plan to analyze financial reports and market data. Then, synthesize the findings into a summary.",
        needs_planning=True,
    )
    registry.add_task(root_task)

    # 4. Initialize and Configure the DAGAgent
    agent = DAGAgent(
        registry=registry,
        llm_model="gpt-4o-mini",
        tools=[brave_web_search]
    )

    # 5. Run the Agent
    await agent.run()

    # 6. Retrieve the Final Result
    final_result_task = registry.tasks.get("ROOT_Q1_BRIEFING")
    if final_result_task and final_result_task.status == 'complete':
        print("\n--- Agent's Final Report ---")
        print(final_result_task.result)
    else:
        print(f"Agent failed to complete the root task. Status: {final_result_task.status}")

if __name__ == "__main__":
    asyncio.run(main())
```

### `TaskRegistry`

The `TaskRegistry` is the cornerstone of the `DAGAgent`.

-   `registry.add_task(task: Task)`: Adds a new task to the registry.
-   `registry.add_document(name: str, content: dict) -> str`: Adds a data source as a completed task. Returns the unique ID of the document task.
-   `registry.add_dependency(task_id: str, dep_id: str)`: Manually creates a dependency between two tasks.

### `Task` Model

The `Task` model has several key fields for controlling execution:

-   `id` (str): A unique identifier for the task.
-   `desc` (str): The description of what needs to be done.
-   `deps` (Set[str]): A set of other task IDs that this task depends on.
-   `status` (str): The current lifecycle status (e.g., `pending`, `running`, `complete`, `failed`).
-   `result` (Optional[str]): The output of the task once completed.
-   `needs_planning` (bool): If `True`, the agent will first create a sub-plan to execute this task.
-   `can_request_new_subtasks` (bool): If `True`, the agent can propose new, more granular sub-tasks during execution if it deems it necessary.

---

## Common Concepts

### Tools

Both `RecursiveAgent` and `DAGAgent` can be equipped with tools. Tools are standard Python functions that the agent can choose to execute to gather information or perform actions.

-   **Signature**: Tools should be well-documented with type hints and clear docstrings.
-   **Registration**:
    -   For `RecursiveAgent`, provide tools to `RecursiveAgentOptions` via the `tools` list and `tools_dict`.
    -   For `DAGAgent`, provide a list of tool functions to the constructor via the `tools` argument.

### Error Handling

Both agents can fail. It's good practice to wrap the `agent.run()` call in a `try...except` block to handle any exceptions gracefully.

### Tracing with OpenTelemetry

Both agents are instrumented with OpenTelemetry. If you configure an OpenTelemetry tracer, you can visualize the execution flow in compatible systems like Jaeger or Arize Phoenix.
