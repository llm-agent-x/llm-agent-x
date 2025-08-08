# Command-Line Interface (CLI)

LLM Agent X provides a command-line interface (`llm-agent-x`) for executing tasks using different agent architectures. This document details its usage and arguments.

> **Note:** `cli.py` (which `llm-agent-x` invokes) is primarily a demonstration of the project's capabilities. For advanced use cases or integration into other applications, consider using the [API directly](./api.md).

## Basic Usage

The basic syntax for running the agent is:

```sh
llm-agent-x <agent_type> "Your task description" [options]
```

## Arguments

### Agent Selection

| Argument       | Description                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent_type`   | (Positional) The type of agent to run. Choices: `recursive`, `dag`.                                                                          |
| `task`         | (Positional) The main task or objective for the agent to execute.                                                                            |

### Common Options (Apply to all agents)

| Argument                    | Description                                                                                                | Default Value                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `--model`                   | The name of the Language Model to use (e.g., `gpt-4o-mini`, `gpt-4-turbo`).                                  | Value of `DEFAULT_LLM` env var, or `gpt-4o-mini` |
| `--u_inst`                  | General user instructions to guide the agent's behavior.                                                   | `""` (empty string)                            |
| `--output`                  | Path to save the final response. If not specified, output is printed to console. Saved relative to `OUTPUT_DIR`. | `None`                                         |
| `--enable-python-execution` | Enable the `exec_python` tool, allowing the agent to execute Python code. See [Python Sandbox](./sandbox.md). | Not set (Python execution is disabled)       |
| `--dev-mode`                | Enable development mode (provides more verbose logging and potentially other debug features).                | Not set                                        |

### Recursive Agent Options

These options are only applicable when `agent_type` is `recursive`.

| Argument                 | Description                                                                                                | Default Value       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------- |
| `--task_type`            | The type of the initial task. Choices: `research`, `search`, `basic`, `text/reasoning`.                      | `research`          |
| `--task_limit`           | A string representation of a Python list defining the maximum number of subtasks allowed at each layer.      | `"[3,2,2,0]"`       |
| `--merger`               | Strategy for merging results. Choices: `ai`, `append`, `algorithmic`.                                      | `ai`                |
| `--align_summaries`      | Whether to align subtask summaries with user instructions.                                                 | `True`              |
| `--no-tree`              | If specified, disables the real-time, console-based tree view of task execution.                             | Not set             |
| `--default_subtask_type` | The default task type for all subtasks.                                                                    | `basic`             |
| `--mcp-config`           | Path to a JSON configuration file for Meta-Cognitive Processes (MCP) servers.                                | `None`              |

### DAG Agent Options

These options are only applicable when `agent_type` is `dag`.

| Argument               | Description                                                                                             | Default Value |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | ------------- |
| `--dag-documents`      | Path to a JSON file defining initial documents/tasks for the DAG. Required for the `dag` agent.         | `None`        |
| `--max-grace-attempts` | The number of extra retries that can be granted by the retry analyst for a failing task.                | `1`           |


## Examples

### Recursive Agent Examples

1.  **Basic Research Task:**
    ```sh
    llm-agent-x recursive "Research the history of artificial intelligence, focusing on key breakthroughs."
    ```

2.  **Task with Custom Task Limits and Python Execution:**
    ```sh
    llm-agent-x recursive "Write a Python script to fetch today's weather from an API and suggest appropriate attire." --task_limit "[2,1,0]" --enable-python-execution
    ```
    *(Note: This requires the agent to be prompted to use the `exec_python` tool and may require the sandbox to be running.)*

3.  **Using a specific merger strategy:**
    ```sh
    llm-agent-x recursive "Compile a list of popular sci-fi novels from the last decade with a brief synopsis for each." --merger append --output scifi_novels.md
    ```

### DAG Agent Examples

1.  **Basic Analysis Task with Initial Documents:**

    First, create a `documents.json` file:
    ```json
    [
      {"name": "Q2_Revenue", "content": "Q2 2024 Revenue for our company was $50M, with a net profit of $10M."},
      {"name": "Q2_Sales_Report", "content": "The new product line launched in Q2 accounted for 80% of sales growth."}
    ]
    ```

    Then, run the `dag` agent:
    ```sh
    llm-agent-x dag "Analyze the Q2 financial performance and create a summary for the board." --dag-documents documents.json --output q2_summary.md
    ```

2.  **Complex Task Requiring Web Search:**

    The `dag` agent can also use tools if they are enabled.
    ```sh
    llm-agent-x dag "Research the latest trends in renewable energy and their economic impact." --dag-documents '[]' --enable-python-execution
    ```
    *(Note: Providing an empty JSON array `[]` for documents is a way to start the DAG with just a root task that must rely on tools like web search.)*

## Next Steps

-   Understand how to use LLM Agent X [programmatically via its API](./api.md).
-   See more detailed [CLI and API examples](./examples.md).
-   Learn about the [Python Execution Sandbox](./sandbox.md) for safe code execution.
