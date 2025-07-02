# Python API Usage

LLM Agent X can be integrated into your Python projects, allowing you to leverage its task decomposition and execution capabilities programmatically. The core component for this is the `RecursiveAgent`.

## Core Concepts

-   **`RecursiveAgent`**: The main class that takes a task, decomposes it if necessary, and executes it. It can use tools and delegate to child agents.
-   **`RecursiveAgentOptions`**: A Pydantic model to configure the behavior of `RecursiveAgent`.
-   **Tools**: Python functions that the agent can decide to call to gather information or perform actions (e.g., `brave_web_search`, `exec_python`).
-   **LLM**: The language model used by the agent for reasoning, task splitting, and summarization. LLM Agent X uses `pydantic-ai` which supports various models, primarily demonstrated with OpenAI's GPT series.
-   **Task Limits**: Configuration that defines how many subtasks can be created at each level of recursion.

## Getting Started

Here's a basic example of how to use `RecursiveAgent`:

```python
import asyncio
from llm_agent_x import RecursiveAgent, RecursiveAgentOptions, TaskLimit
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from llm_agent_x.tools.brave_web_search import brave_web_search # Example tool
# Ensure NLTK data is available (RecursiveAgent might use it indirectly)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)


async def main():
    # 1. Configure the LLM
    # Ensure OPENAI_API_KEY is set in your environment
    client = AsyncOpenAI()
    llm = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))

    # 2. Define Agent Options
    agent_options = RecursiveAgentOptions(
        llm=llm,
        tools=[brave_web_search], # List of available tools
        tools_dict={"web_search": brave_web_search, "brave_web_search": brave_web_search}, # Mapping for the LLM to identify tools
        task_limits=TaskLimit.from_array([2, 1, 0]), # Max 2 subtasks at layer 0, 1 at layer 1, 0 at layer 2
        allow_search=True, # Allow agent to use search tools if available
        allow_tools=True, # Allow agent to use any provided tools
        mcp_servers=[], # Meta-Cognitive Process servers (optional)
        # You can also specify callbacks like on_task_executed, pre_task_executed
    )

    # 3. Create the Agent
    task_description = "Research the benefits of renewable energy sources and summarize them."
    user_instructions = "Focus on solar and wind power. Keep the summary concise."

    agent = RecursiveAgent(
        task=task_description,
        u_inst=user_instructions,
        agent_options=agent_options,
        # task_type_override can be 'research', 'search', 'basic', 'text/reasoning'
    )

    # 4. Run the Agent
    try:
        print(f"Starting agent for task: {task_description}")
        result = await agent.run()
        print("\nFinal Result:")
        print(result)
        print(f"\nEstimated Cost: ${agent.cost:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## `RecursiveAgent`

The `RecursiveAgent` is initialized with the following key parameters:

-   `task` (str or `TaskObject`): The description of the task to be performed.
-   `u_inst` (str): Specific user instructions to guide the agent.
-   `agent_options` (`RecursiveAgentOptions`): Configuration object for the agent.
-   `tracer` (Optional `opentelemetry.trace.Tracer`): For OpenTelemetry tracing.
-   `tracer_span` (Optional `opentelemetry.trace.Span`): Parent span for tracing.
-   `allow_subtasks` (bool, default `True`): Whether this agent instance is allowed to create subtasks.
-   `current_layer` (int, default `0`): The current recursion depth.
-   `parent` (Optional `RecursiveAgent`): The parent agent, if this is a subtask.
-   `context` (Optional `TaskContext`): Contextual information for the task.
-   `task_type_override` (Optional str): Overrides the default task type (e.g., "research", "basic").
-   `max_fix_attempts` (int, default `2`): How many times to attempt self-correction if verification fails.

The primary method is `await agent.run()`, which executes the task and returns the final result as a string.

## `RecursiveAgentOptions`

This Pydantic model holds the configuration for an agent. Key fields include:

-   `llm` (Any): The language model instance (e.g., `OpenAIModel` from `pydantic-ai`).
-   `tools` (List[Callable]): A list of Python functions that the agent can use.
-   `tools_dict` (Dict[str, Callable]): A dictionary mapping tool names (as the LLM might call them) to the actual tool functions.
-   `task_limits` (`TaskLimit`): Defines the maximum number of subtasks per layer (e.g., `TaskLimit.from_array([3, 2, 1, 0])`).
-   `search_tool` (Any, Optional): A specific tool designated for search operations.
-   `merger` (Any, default `LLMMerger`): Class responsible for merging results from subtasks. Options include `LLMMerger`, `AppendMerger`, `AlgorithmicMerger`.
-   `allow_search` (bool, default `True`): If `True` and a search tool is available, the agent can use it.
-   `allow_tools` (bool, default `False`): If `True`, the agent can use any of the tools provided in the `tools` list.
-   `mcp_servers` (List[`MCPServer`], default `[]`): List of Meta-Cognitive Process server instances.
-   `pre_task_executed`, `on_task_executed`, `on_tool_call_executed` (Callable, Optional): Callbacks for monitoring agent lifecycle events.
    -   `pre_task_executed(task: str, uuid: str, parent_agent_uuid: Optional[str])`
    -   `on_task_executed(task: str, uuid: str, result: str, parent_agent_uuid: Optional[str])`
    -   `on_tool_call_executed(tool_name: str, tool_input: Any, tool_output: Any, task_uuid: str)`
-   `similarity_threshold` (float, default `0.8`): If a subtask is too similar to its parent, it might be executed as a single task instead of further splitting.
-   `align_summaries` (bool, default `True`): Whether to run an additional LLM call to align merged summaries with the original task and user instructions.
-   `max_fix_attempts` (int, default `2`): Default number of self-correction attempts if a task fails verification.

## Tools

Tools are standard Python functions that the agent can choose to execute.

-   **Signature**: Tools should be well-documented with type hints and clear docstrings, especially the first line of the docstring, as this is often what the LLM uses to understand the tool's purpose.
-   **Registration**: Provide tools to `RecursiveAgentOptions` via the `tools` list and `tools_dict`.
    -   `tools`: A list of the callable functions.
    -   `tools_dict`: A dictionary where keys are names the LLM might use to refer to the tool (e.g., "web_search", "python_executor") and values are the corresponding functions.
-   **Example Tool (`brave_web_search`)**:
    ```python
    from llm_agent_x.tools.brave_web_search import brave_web_search
    # ...
    agent_options = RecursiveAgentOptions(
        # ...
        tools=[brave_web_search],
        tools_dict={"web_search": brave_web_search, "brave_web_search": brave_web_search},
        allow_search=True, # Important for search tools
        allow_tools=True   # General tool allowance
    )
    ```
-   **Python Execution (`exec_python`)**:
    LLM Agent X includes an `exec_python` tool for executing Python code.
    To enable it:
    ```python
    from llm_agent_x.tools.exec_python import exec_python
    # ...
    agent_options = RecursiveAgentOptions(
        # ...
        tools=[brave_web_search, exec_python], # Add exec_python
        tools_dict={
            "web_search": brave_web_search,
            "exec_python": exec_python,
            "exec": exec_python # Alias
        },
        allow_tools=True
    )
    ```
    When using `exec_python`, it's highly recommended to also set up and use the [Python Sandbox](./sandbox.md) for security. The `exec_python` tool can be configured to use it.

## Callbacks

You can hook into the agent's lifecycle using callbacks defined in `RecursiveAgentOptions`:

-   `pre_task_executed(task: str, uuid: str, parent_agent_uuid: Optional[str])`: Called before a task (or subtask) starts execution.
-   `on_task_executed(task: str, uuid: str, result: str, parent_agent_uuid: Optional[str])`: Called after a task (or subtask) finishes execution.
-   `on_tool_call_executed(tool_name: str, tool_input: Any, tool_output: Any, task_uuid: str)`: Called after a tool is used by the agent.

Example:
```python
def my_task_tracker(task, uuid, result, parent_uuid):
    print(f"Task Completed (UUID: {uuid}): {task} -> Result: {result[:50]}...")

agent_options = RecursiveAgentOptions(
    # ... other options
    on_task_executed=my_task_tracker
)
```

## Error Handling

The `agent.run()` method can raise `TaskFailedException` if a task fails critically after exhausting retries or if an unrecoverable error occurs. It's good practice to wrap the `run()` call in a `try...except` block.

## Advanced: Task Types and Merging

-   **Task Types**: The `task_type_override` parameter in `RecursiveAgent` (or `type` in `TaskObject`) can influence how the agent approaches a task (e.g., "research", "basic", "search"). "basic" tasks are often simpler and may not involve extensive web searching or complex summarization.
-   **Merging Strategies**: When subtasks complete, their results need to be merged. The `merger` option in `RecursiveAgentOptions` controls this:
    -   `LLMMerger` (default): Uses an LLM to synthesize a coherent summary from subtask results.
    -   `AppendMerger`: Simply concatenates the results from subtasks.
    -   `AlgorithmicMerger`: A more structured, non-LLM based approach to merging (details may vary based on implementation).

## Tracing with OpenTelemetry

If you provide a `tracer` and `tracer_span` to `RecursiveAgent`, it will create spans for its operations, allowing you to visualize the execution flow in distributed tracing systems like Jaeger or Arize Phoenix.

This API provides a flexible way to incorporate LLM Agent X's recursive reasoning and tool use into various Python applications. Remember to manage API keys and other sensitive configurations securely, typically through environment variables.Tool output for `create_file_with_block`:
