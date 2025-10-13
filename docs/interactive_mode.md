# Interactive Mode

LLM Agent X can be run as a persistent, interactive service. This mode is ideal for complex, long-running tasks that may require human supervision, intervention, or dynamic goal changes. It exposes a web server with a REST API and a real-time user interface for monitoring and controlling the agent's execution.

## Architecture Overview

The interactive mode operates on a distributed, two-component architecture:

1.  **Gateway Server (`gateway.py`):**
    *   A lightweight **FastAPI** web server that acts as the primary entry point for all user interactions.
    *   Exposes a **REST API** for creating tasks and sending control directives.
    *   Hosts a **Socket.IO** server to broadcast real-time state updates to connected clients (like the Mission Control UI).
    *   Communicates with the agent worker via a **RabbitMQ** message queue, publishing directives to be consumed by the worker.

2.  **Agent Worker (`worker.py`):**
    *   A long-running process that initializes and runs the `InteractiveDAGAgent`.
    *   Listens for directives from the Gateway on a **RabbitMQ** queue.
    *   Executes the agent's lifecycle, including planning, task execution, and state changes.
    *   Publishes comprehensive state updates back to a **RabbitMQ** exchange, which are then picked up by the Gateway and broadcast to the UI.

This separation ensures that the core agent logic is decoupled from the web interface, allowing for robust, scalable, and resilient operation.

## How to Run

To run the system in interactive mode, you need to start three services: a RabbitMQ instance, the Gateway, and at least one Worker.

### Prerequisites

*   Dependencies met
*   An environment file (`.env`) configured.

### Running with Docker Compose

1.  **Start all services:**
    From the root of the project, run:
    ```sh
    poetry run llm-agent-x-gateway
    ```
    In a separate terminal:
    ```sh
    poetry run llm-agent-x-worker
    ```
    In another separate terminal:
    ```sh
    cd mi* && npm run dev
    ```
3.  **Access the Mission Control UI:**
    Once the services are running, open a web browser and navigate to the Mission Control UI (typically `http://localhost:3000` if you are running it locally).

4.  **Shutting Down:**
    Press `Ctrl+C` in the terminals.

## Gateway API Endpoints

The Gateway exposes the following REST endpoints:

#### `GET /api/tasks`

-   **Description:** Retrieves a snapshot of the current state of all tasks known to the Gateway. The state is cached and updated in real-time via the worker's broadcasts.
-   **Response:**
    ```json
    {
      "tasks": {
        "TASK_ID_1": { "...task state object..." },
        "TASK_ID_2": { "...task state object..." }
      }
    }
    ```

#### `POST /api/tasks`

-   **Description:** Creates a new root task for the agent to begin working on.
-   **Request Body:**
    ```json
    {
      "desc": "The high-level objective for the agent.",
      "mcp_servers": [] // Optional: configuration for MCP servers
    }
    ```
-   **Response:**
    ```json
    { "status": "new task submitted" }
    ```

#### `POST /api/tasks/{task_id}/directive`

-   **Description:** Sends a specific control command (a "directive") to a running task.
-   **URL Parameter:** `task_id` - The ID of the task to control.
-   **Request Body:**
    ```json
    {
      "command": "DIRECTIVE_NAME",
      "payload": "..." // Value depends on the command
    }
    ```
-   **Response:**
    ```json
    { "status": "directive sent" }
    ```

## Available Directives

Directives are the core mechanism for interacting with the agent. They are sent to the `/api/tasks/{task_id}/directive` endpoint.

| Command              | Payload                                                     | Description                                                                                                                              |
| -------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `PAUSE`              | `null`                                                      | Pauses the execution of the specified task. The task's state is preserved.                                                               |
| `RESUME`             | `null`                                                      | Resumes a task that was previously paused by a human operator.                                                                           |
| `ANSWER_QUESTION`    | `string`                                                    | Provides an answer to a question the agent asked when it entered the `waiting_for_user_response` state. The agent will consume the answer and resume. |
| `REDIRECT`           | `string`                                                    | Provides a new instruction or clarification to the task. This forces the agent to re-evaluate its approach.                              |
| `MANUAL_OVERRIDE`    | `string`                                                    | Forces a task to be marked as `complete` and sets its result to the provided payload. This is useful for manually completing a step.    |
| `CANCEL`             | `string` (optional reason)                                  | Marks a task as `cancelled`. The task is not deleted but is removed from the active execution flow.                                     |
| `PRUNE_TASK`         | `string` (optional reason)                                  | Permanently removes a task and all its children from the graph. This is a destructive action used to clean up the task tree.              |
| `TERMINATE`          | `string` (optional reason)                                  | Forcibly marks a task as `failed`.                                                                                                       |

## Socket.IO Events

The Gateway broadcasts state changes via Socket.IO, allowing a UI to update in real-time.

-   **Event:** `task_update`
-   **Payload:**
    ```json
    {
      "task": {
        "id": "TASK_ID",
        "desc": "Task description",
        "status": "running",
        "result": null,
        "children": ["CHILD_ID_1"],
        // ... and all other fields from the Task model
      }
    }
    ```
    Clients should listen for this event to receive the latest state of any task that has been modified by the agent worker.
