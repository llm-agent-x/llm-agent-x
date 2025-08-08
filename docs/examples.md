# Examples

This page provides examples of how to use LLM Agent X, both from the command line and programmatically. For a more comprehensive set of examples, including sample outputs, please refer to the [`samples`](../samples/) directory in the repository.

## Command-Line Interface (CLI) Examples

These examples demonstrate common use cases for the `llm-agent-x` CLI tool. See the [CLI Documentation](./cli.md) for a full list of arguments.

### Recursive Agent

The `recursive` agent is well-suited for tasks that can be broken down into a hierarchy of sub-tasks.

1.  **Basic Research Task:**
    Ask the agent to research a topic. It will use its search tool and decompose the task as needed.
    ```sh
    llm-agent-x recursive "Research the impact of renewable energy on climate change mitigation."
    ```

2.  **Controlling Task Decomposition:**
    Use `--task_limit` to control how many subtasks can be generated at each level.
    ```sh
    llm-agent-x recursive "Develop a brief marketing plan for a new eco-friendly coffee shop." --task_limit "[2,2,0]"
    ```
    This allows 2 subtasks at the first level, 2 sub-subtasks for each of those, and no further decomposition.

3.  **Enabling Python Execution (with Sandbox):**
    Allow the agent to write and execute Python code.
    ```sh
    llm-agent-x recursive "Write a Python script to calculate the factorial of 5 and explain the code." --enable-python-execution
    ```

### DAG Agent

The `dag` (Directed Acyclic Graph) agent excels at tasks with complex dependencies that don't fit a simple hierarchical structure.

1.  **Analysis from Pre-defined Documents:**
    The `dag` agent often starts with a set of initial data sources. First, create a `docs.json` file:
    ```json
    [
      {"name": "Q2_Revenue", "content": "Q2 2024 Revenue: $50M, Net Profit: $10M."},
      {"name": "Q2_Sales_Report", "content": "The new product line accounted for 80% of sales growth."}
    ]
    ```
    Then, run the agent, pointing to your documents:
    ```sh
    llm-agent-x dag "Analyze Q2 financial performance and create a summary." --dag-documents docs.json --output q2_summary.md
    ```

2.  **Complex Research with Tool Use:**
    The `dag` agent can also start without initial documents and rely entirely on tools. Here, we provide an empty list `[]` for the documents and enable web search (which is on by default).
    ```sh
    llm-agent-x dag "Investigate the supply chain of cobalt and its ethical implications." --dag-documents '[]'
    ```

## Python API Examples

### RecursiveAgent API

This example demonstrates a basic programmatic use of `RecursiveAgent`.

```python
import asyncio
from llm_agent_x.agents.recursive_agent import RecursiveAgent, RecursiveAgentOptions, TaskLimit
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from llm_agent_x.tools.brave_web_search import brave_web_search

async def run_recursive_agent():
    client = AsyncOpenAI()
    llm = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))

    agent_options = RecursiveAgentOptions(
        llm=llm,
        tools=[brave_web_search],
        tools_dict={"web_search": brave_web_search},
        task_limits=TaskLimit.from_array([2, 1, 0]),
        allow_search=True,
        allow_tools=True,
        mcp_servers=[],
    )

    agent = RecursiveAgent(
        task="Explore the pros and cons of remote work for software development teams.",
        u_inst="Provide a balanced view with three points for each side.",
        agent_options=agent_options
    )

    print("--- Running RecursiveAgent ---")
    result = await agent.run()
    print("\n--- Agent's Final Report ---")
    print(result)
    print(f"\nEstimated Cost: ${agent.cost:.4f}")

if __name__ == "__main__":
    asyncio.run(run_recursive_agent())
```

### DAGAgent API

This example shows how to set up and run the `DAGAgent` programmatically.

```python
import asyncio
from llm_agent_x.agents.dag_agent import DAGAgent, TaskRegistry, Task
from llm_agent_x.tools.brave_web_search import brave_web_search

async def run_dag_agent():
    # 1. Initialize the Task Registry
    registry = TaskRegistry()

    # 2. Add initial data (documents) to the registry
    registry.add_document(
        "Financial_Report_Q1",
        "Q1 revenue was $1.2M with a profit of $200k."
    )
    registry.add_document(
        "Market_Analysis_Q1",
        "Competitor A launched a new product, impacting our market share by 5%."
    )

    # 3. Define the root task for the agent to tackle
    root_task = Task(
        id="ROOT_Q1_BRIEFING",
        desc="Create a comprehensive investor briefing for Q1. First, plan to analyze financial reports and market data. Then, synthesize the findings into a summary.",
        needs_planning=True, # This tells the agent to start by creating a plan
    )
    registry.add_task(root_task)

    # 4. Initialize and configure the DAGAgent
    agent = DAGAgent(
        registry=registry,
        llm_model="gpt-4o-mini",
        tools=[brave_web_search] # Provide tools for tasks that might need them
    )

    print("--- Running DAGAgent ---")
    registry.print_status_tree() # Shows the initial state

    await agent.run()

    print("\n--- DAG Execution Complete ---")
    registry.print_status_tree() # Shows the final state

    final_result = registry.tasks.get("ROOT_Q1_BRIEFING")
    if final_result:
        print("\n--- Agent's Final Report ---")
        print(final_result.result)
        total_cost = sum(t.cost for t in registry.tasks.values())
        print(f"\nEstimated Cost: ${total_cost:.4f}")

if __name__ == "__main__":
    asyncio.run(run_dag_agent())
```

## More Examples

For more detailed examples, including the structure of input tasks and the corresponding outputs generated by LLM Agent X, please see the files within the [`samples/`](../samples/) directory of the project repository.
