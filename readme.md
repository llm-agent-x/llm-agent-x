# LLM Agent X

## Overview

LLM Agent X is a Python-based framework for creating and running autonomous agents that can perform complex tasks. It provides a flexible architecture that allows language models to decompose high-level objectives into smaller, manageable steps and execute them using a variety of tools.

The framework currently features two primary agent architectures:
-   **Recursive Agent**: A hierarchical agent that recursively breaks down tasks into a tree of sub-tasks. It's ideal for problems that have a clear, structured decomposition.
-   **DAG Agent**: A more advanced agent that models tasks as a Directed Acyclic Graph (DAG). This allows for more complex, non-linear dependencies between tasks and is suitable for problems requiring sophisticated planning and adaptation.

> ⚠️ **Security Warning**: This project is a research demonstration and is not secure. It allows language models to execute arbitrary code, which can be dangerous. Use with trusted inputs only and preferably in a sandboxed environment.

## Features

-   **Multiple Agent Architectures**: Choose between a simple `recursive` agent or a more powerful `dag` agent.
-   **Tool Use**: Equip agents with tools like web search (`brave_web_search`) and code execution (`exec_python`).
-   **Task Decomposition**: Agents can autonomously break down complex goals into smaller sub-tasks.
-   **Extensible**: Designed to be integrated into other applications via a Python API.
-   **Observability**: Integrates with OpenTelemetry for tracing agent execution.
-   **Optional Sandbox**: Includes a Dockerized sandbox for safe execution of Python code.

## Installation

1.  **Clone the repository (optional):**
    ```sh
    git clone https://github.com/cvaz1306/llm_agent_x.git
    cd llm_agent_x
    ```

2.  **Install the package:**
    ```sh
    pip install .
    ```
    For development, use editable mode: `pip install -e .`

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys.
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    BRAVE_API_KEY="your_brave_search_api_key" # If using brave_web_search

    # Optional
    OUTPUT_DIR=./output/
    DEFAULT_LLM=gpt-4o-mini
    ```

## Usage (CLI)

The command-line interface allows you to quickly run agents on a given task.

### Basic Syntax

```sh
llm-agent-x <agent_type> "Your task description" [options]
```

### Examples

**1. Run the `recursive` agent for a research task:**
```sh
llm-agent-x recursive "Research the pros and cons of using nuclear energy for power generation." --output nuclear_report.md
```

**2. Run the `dag` agent to analyze pre-existing data:**

First, create a `documents.json` file:
```json
[
  {"name": "Q3_Financials", "content": "Our Q3 revenue was $75M, beating estimates."},
  {"name": "Q3_Press_Release", "content": "The successful launch of Product Z drove significant growth in the third quarter."}
]
```

Then, run the agent:
```sh
llm-agent-x dag "Create a summary of the Q3 performance for an internal memo." --dag-documents documents.json
```

**3. Run the `dag` agent with web search enabled:**
If the `dag` agent needs to find its own information, you can start it with an empty set of documents.
```sh
llm-agent-x dag "Investigate and report on the current state of quantum computing hardware." --dag-documents '[]'
```

For a full list of CLI arguments and advanced options, see the [CLI Documentation](./docs/cli.md).

## Usage (Python API)

For more control and integration, use the Python API.

### RecursiveAgent API Example
```python
import asyncio
from llm_agent_x.agents.recursive_agent import RecursiveAgent, RecursiveAgentOptions, TaskLimit
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from llm_agent_x.tools.brave_web_search import brave_web_search

async def main():
    client = AsyncOpenAI()
    llm = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))

    agent_options = RecursiveAgentOptions(
        llm=llm,
        tools=[brave_web_search],
        task_limits=TaskLimit.from_array([2, 1, 0])
    )

    agent = RecursiveAgent(
        task="Explore the future of remote work.",
        agent_options=agent_options
    )

    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### DAGAgent API Example
```python
import asyncio
from llm_agent_x.agents.dag_agent import DAGAgent, TaskRegistry, Task

async def main():
    registry = TaskRegistry()
    registry.add_document("Initial_Data", "The company's goal is to expand into the European market in 2025.")

    root_task = Task(
        id="ROOT_TASK",
        desc="Create a market entry strategy for Europe.",
        needs_planning=True,
    )
    registry.add_task(root_task)

    agent = DAGAgent(registry=registry, llm_model="gpt-4o-mini")

    await agent.run()

    final_result = registry.tasks.get("ROOT_TASK")
    print(final_result.result)

if __name__ == "__main__":
    asyncio.run(main())
```

See the [Examples Documentation](./docs/examples.md) for more detailed examples.

## Documentation

-   [**Installation**](./docs/installation.md): How to set up the project.
-   [**CLI Reference**](./docs/cli.md): Detailed command-line usage.
-   [**API Reference**](./docs/api.md): Guide to using the Python API.
-   [**Examples**](./docs/examples.md): Practical examples for both CLI and API.
-   [**Sandbox**](./docs/sandbox.md): Information on the optional code execution sandbox.

## Dependencies

Project dependencies are managed with Poetry and are listed in `pyproject.toml`.

> Note: `torch` is an optional dependency for certain models but may be required at runtime.

## License

This project is licensed under the MIT License.
