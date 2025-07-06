# Command-Line Interface (CLI)

LLM Agent X provides a command-line interface (`llm-agent-x`) for executing tasks. This document details its usage, arguments, and provides examples.

> **Note:** `cli.py` (which `llm-agent-x` invokes) is primarily a demonstration of the project's capabilities. For advanced use cases or integration into other applications, consider using the [API directly](./api.md) or adapting `cli.py` to your needs.

## Basic Usage

The basic syntax for running the agent is:

```sh
llm-agent-x "Your task description" [options]
```

## Arguments

Below are the available command-line arguments:

| Argument                    | Description                                                                                                | Default Value                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `task` (Positional)       | The main task description for the agent to execute. (Required)                                             | N/A                                            |
| `--task_type`               | The type of the main task. Choices: `research`, `search`, `basic`, `text/reasoning`.                       | `research`                                     |
| `--u_inst`                  | Additional user instructions for the task.                                                                 | `""` (empty string)                            |
| `--max_layers`              | **Deprecated.** Use `--task_limit` instead. The maximum number of recursive layers for task decomposition. | `3`                                            |
| `--output`                  | Path to save the final output. If not specified, output is printed to console. Saved relative to `OUTPUT_DIR`. | `None`                                         |
| `--model`                   | The name of the Language Model to use (e.g., `gpt-4o-mini`, `gpt-4-turbo`).                                  | Value of `DEFAULT_LLM` env var, or `gpt-4o-mini` |
| `--task_limit`              | A string representation of a Python list defining the maximum number of subtasks allowed at each layer. Example: `"[3,2,2,0]"`. | `"[3,2,2,0]"`                                  |
| `--merger`                  | Strategy for merging results from subtasks. Choices: `ai` (uses LLM), `append` (concatenates), `algorithmic`. | `ai`                                           |
| `--align_summaries`         | Whether to align summaries with user instructions. (Boolean)                                               | `True`                                         |
| `--no-tree`                 | If specified, disables the real-time, console-based tree view of task execution.                             | Not set (tree view is enabled by default)      |
| `--default_subtask_type`    | The default task type to apply to all subtasks. Choices: `research`, `search`, `basic`, `text/reasoning`.    | `basic`                                        |
| `--enable-python-execution` | Enable the `exec_python` tool, allowing the agent to execute Python code. See [Python Sandbox](./sandbox.md). | Not set (Python execution is disabled)       |
| `--mcp-config`              | Path to a JSON configuration file for Meta-Cognitive Processes (MCP) servers.                               | `None`                                         |
| `--dev-mode`                | Enable development mode (provides more verbose logging and potentially other debug features).                | Not set                                        |

### Details on Specific Arguments:

*   **`task`**: This is the primary input to the agent. It should be a clear and concise description of what you want the agent to accomplish.
*   **`--model`**: Ensure the model name corresponds to one available through your configured OpenAI API (or other LLM provider if adapted). The default can be set via the `DEFAULT_LLM` environment variable (see [Installation](./installation.md)).
*   **`--task_limit`**: This controls the "shape" of the task decomposition. For example, `"[3,2,2,0]"` means:
    *   Layer 0 (initial task): Can be split into a maximum of 3 subtasks.
    *   Layer 1 (subtasks of Layer 0): Each can be split into a maximum of 2 subtasks.
    *   Layer 2 (subtasks of Layer 1): Each can be split into a maximum of 2 subtasks.
    *   Layer 3 (subtasks of Layer 2): Cannot be split further (limit is 0).
*   **`--output`**: The filename provided will be saved in the directory specified by the `OUTPUT_DIR` environment variable (defaults to `./output/`).
*   **`--enable-python-execution`**: When this flag is used, the agent gains the ability to write and execute Python code to solve tasks. This is a powerful feature but should be used with caution. For safety, it's highly recommended to use this in conjunction with the [Python Sandbox](./sandbox.md).
*   **`--mcp-config`**: This allows specifying external servers or processes that can perform meta-cognitive functions, potentially guiding or reflecting on the agent's own processes. The configuration file should be a JSON detailing the MCP server(s) and their transport mechanisms (e.g., stdio, HTTP).

## Examples

1.  **Basic Research Task:**
    ```sh
    llm-agent-x "Research the history of artificial intelligence, focusing on key breakthroughs."
    ```

2.  **Task with Specific Output File and Model:**
    ```sh
    llm-agent-x "Summarize the latest advancements in quantum computing." --output quantum_summary.txt --model gpt-4-turbo
    ```

3.  **Task with Custom Task Limits and Python Execution Enabled:**
    ```sh
    llm-agent-x "Write a Python script to fetch today's weather from an API and then analyze the data to suggest appropriate attire." --task_limit "[2,1,0]" --enable-python-execution
    ```
    *(Note: For the above example, ensure the Python sandbox is running if you want isolated execution, and the agent is prompted appropriately to use the `exec_python` tool.)*

4.  **Task without Real-time Tree View:**
    ```sh
    llm-agent-x "What are the main differences between renewable and non-renewable energy sources?" --no-tree
    ```

5.  **Using a specific merger strategy:**
    ```sh
    llm-agent-x "Compile a list of popular science fiction novels from the last decade and provide a brief synopsis for each." --merger append --output scifi_novels.md
    ```

## Environment Variables

The CLI also respects environment variables set in a `.env` file or your shell environment. See the [Installation](./installation.md) guide for a list of relevant variables like `OPENAI_API_KEY`, `DEFAULT_LLM`, `OUTPUT_DIR`, etc.

## Next Steps

-   Understand how to use LLM Agent X [programmatically via its API](./api.md).
-   Learn about the [Python Execution Sandbox](./sandbox.md) for safe code execution.
