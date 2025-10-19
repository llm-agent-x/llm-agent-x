# Interactive Mode Architecture

LLM Agent X is designed to run as a persistent, interactive service. This mode is ideal for complex, long-running tasks that may require human supervision, intervention, or dynamic goal changes. The system is composed of several containerized services that work together, with a web UI for operator control.

## Architecture Overview

The interactive mode operates on a distributed, microservice-style architecture orchestrated with Docker Compose:

1.  **Mission Control UI:**
    *   A **Next.js** web application that provides the main user interface.
    *   Connects to the Gateway's API and Socket.IO server to display the task graph and allow the operator to send commands.

2.  **Gateway Server:**
    *   A **FastAPI** web server that acts as the primary entry point for all user interactions.
    *   Exposes a **REST API** for creating tasks and a **Socket.IO** server to broadcast real-time state updates.
    *   Communicates with the agent worker by publishing directives to a **RabbitMQ** message queue.

3.  **Agent Worker:**
    *   A long-running process that runs the `InteractiveDAGAgent`.
    *   Listens for directives from the Gateway on a **RabbitMQ** queue.
    *   Executes the agent's lifecycle, including planning, task execution, and state changes.
    *   Publishes comprehensive state updates back to a **RabbitMQ** exchange, which are then picked up by the Gateway and broadcast to all connected clients.

4.  **RabbitMQ:**
    *   The message broker that decouples the Gateway and Worker, ensuring reliable and resilient communication between them.

This separation ensures that the core agent logic is decoupled from the web interface, allowing for robust, scalable, and resilient operation.

## How to Run

The entire application stack is managed via Docker Compose. To run the system in interactive mode, please follow the main installation guide, which covers cloning the repository, setting up your environment variables, and launching all services with a single command.

**➡️ [Running the Application](./installation.md)**

## Interacting with the Agent

The primary way to interact with the agent is through the Mission Control UI, which uses the Gateway API behind the scenes. Once the application is running, you can:

-   **Launch New Tasks:** Use the form to define a high-level objective.
-   **Manage Documents:** Add, edit, or delete documents that the agent can use as context.
-   **Monitor Progress:** Watch the DAG View update in real-time as the agent creates and executes tasks.
-   **Inspect Tasks:** Click on any task node to see its details, dependencies, and results in the inspector pane.
-   **Issue Directives:** Use the command palette in the inspector to pause, cancel, redirect, or manually complete tasks.
-   **Answer Questions:** If an agent needs clarification, the task will pause, and an input box will appear in the inspector for you to provide an answer.

For detailed information on the programmatic interface for custom integrations, see the **[Gateway API Reference](./api.md)**.