# LLM Agent X Documentation

## Overview

LLM Agent X is a task execution framework that leverages language models to perform complex tasks by recursively decomposing them into subtasks and using tools like web search.

This documentation provides a comprehensive guide to installing, using, and understanding LLM Agent X.

## Key Features

*   **Recursive Task Decomposition:** Breaks down complex tasks into smaller, manageable subtasks.
*   **Tool Usage:** Can utilize external tools like web search (Brave Search) and Python code execution.
*   **Configurable:** Offers various options to customize agent behavior, including LLM choice, task limits, and result merging strategies.
*   **Extensible:** Designed to be integrated into other Python projects.
*   **Optional Sandbox:** Provides an isolated Docker environment for safe Python code execution.
*   **Real-time Tree View:** Offers a console-based visualization of the task execution flow.

## Getting Started

-   **[Installation](./installation.md):** Learn how to install LLM Agent X and set up your environment.
-   **[Command-Line Interface (CLI)](./cli.md):** Understand how to use the `llm-agent-x` command-line tool.
-   **[API Usage](./api.md):** Discover how to use LLM Agent X programmatically in your own projects.
-   **[Python Sandbox](./sandbox.md):** Find out more about the optional Python execution sandbox.
-   **[Examples](./examples.md):** See practical examples of how LLM Agent X can be used.

## Dependencies

Project dependencies are managed with Poetry and are listed in the `pyproject.toml` file.

> ⚠️ `torch` is optional in Poetry but **required** at runtime. You must install the correct version for your hardware manually using the appropriate `--index-url`.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
