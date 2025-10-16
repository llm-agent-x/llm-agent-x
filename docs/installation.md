# Running the Application

This guide explains how to run the complete LLM Agent X application stack—including the Mission Control UI, the API Gateway, and the Agent Worker—using Docker Compose.

## Prerequisites

-   [Docker](https://www.docker.com/get-started) and Docker Compose (usually included with Docker Desktop).
-   [Git](https://git-scm.com/) for cloning the repository.
-   An **OpenAI API Key**.
-   A **Brave Search API Key** (for the web search tool).

## Step 1: Clone the Repository

Open your terminal and clone the project repository:

```sh
git clone https://github.com/cvaz1306/llm_agent_x.git
cd llm_agent_x
```

## Step 2: Configure Environment Variables

The project uses a `.env` file to manage secret keys and configuration. Create one by copying the example file:

```sh
cp .env.example .env
```

Now, open the newly created `.env` file in a text editor and add your API keys. It should look like this:

```env
# Required: Add your API keys here
OPENAI_API_KEY="sk-..."
BRAVE_API_KEY="..."

# --- Pre-configured for Docker ---
# These variables are already set up for the Docker environment.
# You generally do not need to change them.
RABBITMQ_HOST=rabbitmq
GATEWAY_RELOAD=false
NEXT_PUBLIC_API_URL=http://localhost:8000
```

-   `OPENAI_API_KEY`: Essential for the agent worker to use language models.
-   `BRAVE_API_KEY`: Required for the `brave_web_search` tool.

## Step 3: Launch with Docker Compose

From the root directory of the project, run the following command:

```sh
docker-compose up --build
```

This command will:
1.  **Build** the Docker images for all services (`ui`, `gateway`, `worker`, `sandbox`) if they don't already exist.
2.  **Create and start** containers for all services, including the RabbitMQ message broker.
3.  **Network** the containers so they can communicate with each other.
4.  **Stream logs** from all containers to your terminal, so you can see what's happening.

The first time you run this, it may take a few minutes to download the base images and build everything. Subsequent launches will be much faster.

## Step 4: Access Mission Control

Once all the services have started (you'll see log output from the `ui`, `gateway`, and `worker`), open your web browser and navigate to:

**[http://localhost:3000](http://localhost:3000)**

You should see the Mission Control UI, ready to accept new tasks.

## Shutting Down

To stop all the running services, press `Ctrl+C` in the terminal where `docker-compose` is running.