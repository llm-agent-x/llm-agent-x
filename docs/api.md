# Gateway API Reference

When running in interactive mode, LLM Agent X exposes a **Gateway** service. This service provides a REST API for controlling the agent and managing documents, and a Socket.IO endpoint for receiving real-time state updates. This is the primary programmatic interface for the application.

## REST API

The Gateway's REST API is the entry point for creating tasks, managing documents, and sending commands (directives) to the agent swarm.

---

### Task Management

#### `GET /api/tasks`

-   **Description:** Retrieves a snapshot of the current state of all tasks and documents known to the Gateway.
-   **Method:** `GET`
-   **Response (200 OK):** A JSON object where keys are task IDs and values are the full task state objects.
    ```json
    {
      "tasks": {
        "TASK_ID_1": {
          "id": "TASK_ID_1",
          "desc": "Description of the task...",
          "status": "running",
          "..." : "..."
        }
      }
    }
    ```

#### `POST /api/tasks`

-   **Description:** Creates a new root task for the agent to begin working on.
-   **Method:** `POST`
-   **Request Body:**
    ```json
    {
      "desc": "The high-level objective for the agent.",
      "mcp_servers": [] // Optional: configuration for MCP servers
    }
    ```
-   **Response (200 OK):**
    ```json
    { "status": "new task submitted" }
    ```

#### `POST /api/tasks/{task_id}/directive`

-   **Description:** Sends a specific control command (a "directive") to a running task.
-   **Method:** `POST`
-   **URL Parameter:** `task_id` - The ID of the task to control.
-   **Request Body:**
    ```json
    {
      "command": "DIRECTIVE_NAME",
      "payload": "..." // Value depends on the command
    }
    ```
-   **Response (200 OK):**
    ```json
    { "status": "directive sent" }
    ```

---

### Document Management

#### `GET /api/documents`

-   **Description:** Retrieves a list of all current documents in the system.
-   **Method:** `GET`
-   **Response (200 OK):** An array of document objects.
    ```json
    [
      {
        "id": "DOC_ID_1",
        "name": "My Report",
        "content": "The content of the document..."
      }
    ]
    ```

#### `POST /api/documents`

-   **Description:** Adds a new document to the agent's context.
-   **Method:** `POST`
-   **Request Body:**
    ```json
    {
      "name": "New Document Name",
      "content": "The text content of the document."
    }
    ```
-   **Response (200 OK):**
    ```json
    { "status": "new document submitted" }
    ```

#### `PUT /api/documents/{document_id}`

-   **Description:** Updates the name and/or content of an existing document.
-   **Method:** `PUT`
-   **URL Parameter:** `document_id` - The ID of the document to update.
-   **Request Body:**
    ```json
    {
      "name": "Updated Document Name",
      "content": "New and updated content."
    }
    ```
-   **Response (200 OK):**
    ```json
    { "status": "update directive sent" }
    ```

#### `DELETE /api/documents/{document_id}`

-   **Description:** Deletes a document from the agent's context. This is a destructive action.
-   **Method:** `DELETE`
-   **URL Parameter:** `document_id` - The ID of the document to delete.
-   **Response (200 OK):**
    ```json
    { "status": "delete directive sent" }
    ```

---

### State Management

#### `GET /api/state/download`

-   **Description:** Downloads the entire current graph state (all tasks and documents) as a JSON file.
-   **Method:** `GET`
-   **Response (200 OK):** A JSON file attachment containing the full `task_state_cache`.

#### `POST /api/state/upload`

-   **Description:** Uploads a JSON state file to completely reset and replace the agent's current graph.
-   **Method:** `POST`
-   **Request Body:** A multipart/form-data request with a single file field named `file`.
-   **Response (200 OK):**
    ```json
    { "status": "State upload directive sent successfully." }
    ```

---

### System Health

#### `GET /health`

-   **Description:** A simple health check endpoint to verify that the Gateway service is running.
-   **Method:** `GET`
-   **Response (200 OK):**
    ```json
    { "status": "healthy" }
    ```

---

## Available Directives

Directives are the core mechanism for interacting with an individual task. They are sent to the `/api/tasks/{task_id}/directive` endpoint.

| Command              | Payload                                                     | Description                                                                                                                              |
| -------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `PAUSE`              | `null`                                                      | Pauses the execution of the specified task. The task's state is preserved.                                                               |
| `RESUME`             | `null`                                                      | Resumes a task that was previously paused by a human operator.                                                                           |
| `ANSWER_QUESTION`    | `string`                                                    | Provides an answer to a question the agent asked when it entered the `waiting_for_user_response` state. The agent will consume the answer and resume. |
| `REDIRECT`           | `string`                                                    | Provides a new instruction or clarification to the task. This forces the agent to re-evaluate its approach.                              |
| `MANUAL_OVERRIDE`    | `string`                                                    | Forces a task to be marked as `complete` and sets its result to the provided payload. This is useful for manually completing a step.    |
| `CANCEL`             | `string` (optional reason)                                  | **Soft-deletes** a task by marking its status as `cancelled`. The task remains in the graph but is removed from the active execution flow. |
| `PRUNE_TASK`         | `string` (optional reason)                                  | **Hard-deletes** a task by permanently removing it and all its children from the graph. This is a destructive action.              |
| `TERMINATE`          | `string` (optional reason)                                  | Forcibly marks a task as `failed`.                                                                                                       |

## Socket.IO Events

The Gateway broadcasts state changes from the agent worker via Socket.IO, allowing a UI or other clients to update in real-time.

-   **Event:** `task_update`
-   **Description:** Emitted whenever a task's state changes. This is the primary event for monitoring the system.
-   **Payload:** A JSON object containing the complete, updated state of a single task.
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