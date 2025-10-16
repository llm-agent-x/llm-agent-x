# LLM Agent X Documentation

## Overview

LLM Agent X is an interactive, multi-agent framework for performing complex tasks with real-time human supervision. It uses a message-driven architecture to coordinate between a user interface, a gateway, and one or more agent workers, all managed within a Dockerized environment.

## Core Concepts

The application is composed of three primary services that work together:

-   **Mission Control UI:** A web interface for launching new objectives, visualizing the agent's task graph, inspecting progress, and providing real-time commands and feedback to the agent.
-   **Gateway:** A central API server that handles all communication. It exposes a REST API for commands and a Socket.IO endpoint for broadcasting real-time state updates from the agent.
-   **Agent Worker:** The "brain" of the system. This service runs the `InteractiveDAGAgent`, which listens for tasks, executes its planning and action cycles, and publishes its state changes back to the Gateway.

## Key Features

-   **Interactive DAG Agent**: A persistent agent that models tasks as a graph, allowing for adaptive planning and execution.
-   **Real-time UI**: A web-based "Mission Control" for launching tasks, visualizing the task graph, and providing real-time guidance.
-   **Dockerized Environment**: The entire application stack is containerized for easy setup and deployment with Docker Compose.
-   **Human-in-the-Loop Control**: Operators can pause, resume, cancel, and redirect tasks, or answer questions posed by the agent.
-   **REST & Socket.IO API**: The gateway provides programmatic access for custom integrations and real-time state monitoring.

## Getting Started

The recommended way to use LLM Agent X is through its interactive mode, which provides a full user interface. Start here to get the application running in minutes.

1.  **[Running the Application](./installation.md):** The primary guide to set up and run the entire LLM Agent X stack using Docker.
2.  **[Usage Examples](./examples.md):** See practical examples of how to use the Mission Control UI to manage agents and documents.

## For Developers & Advanced Users

Once you have the application running, you may want to understand its architecture or integrate with it programmatically.

-   **[Interactive Mode Architecture](./interactive_mode.md):** An in-depth look at how the services work together.
-   **[Gateway API Reference](./api.md):** Detailed documentation for the REST and Socket.IO API for custom integrations.
-   **[Python Sandbox](./sandbox.md):** Learn about the optional sandbox for safe code execution.
-   **[Legacy CLI & Examples](./cli.md):** Documentation for the original, non-interactive command-line tool.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.