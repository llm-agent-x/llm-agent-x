# Programmatic Usage Examples

This page provides code examples for developers who want to integrate with or build upon the LLM Agent X framework.

For the best experience, it is recommended to run the full application stack via Docker Compose (see **[Running the Application](./installation.md)**) and interact with it through the Gateway API. Using the agent library directly is an advanced use case for custom implementations.

## 1. Interacting with the Gateway API (Recommended)

These examples demonstrate how to create a Python client to communicate with the running Gateway service.

### Example 1.1: Launching a New Task via REST API

This script sends a request to the Gateway to create a new root task.

```python
import requests
import json

GATEWAY_URL = "http://localhost:8000"

def launch_new_task(description: str):
    """Sends a new task description to the agent gateway."""
    try:
        response = requests.post(
            f"{GATEWAY_URL}/api/tasks",
            json={"desc": description}
        )
        response.raise_for_status()  # Raises an exception for bad status codes
        print("Successfully submitted new task:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error submitting task: {e}")

if __name__ == "__main__":
    task_desc = "Investigate the current state of quantum computing hardware and create a summary report."
    launch_new_task(task_desc)
```

### Example 1.2: Monitoring State in Real-Time with Socket.IO

This script connects to the Gateway's Socket.IO server to receive and print real-time `task_update` events.

```python
import asyncio
import socketio

# Create a new Socket.IO client
sio = socketio.AsyncClient()
GATEWAY_URL = "http://localhost:8000"

@sio.event
async def connect():
    print("Successfully connected to the Gateway WebSocket.")

@sio.event
async def disconnect():
    print("Disconnected from the Gateway WebSocket.")

@sio.on("task_update")
async def on_task_update(data):
    """Handles incoming task_update events."""
    task = data.get("task", {})
    print(f"--- Task Update Received ---")
    print(f"  ID:     {task.get('id')}")
    print(f"  Status: {task.get('status')}")
    print(f"  Desc:   {task.get('desc', '')[:70]}...")
    print("-" * 26)


async def main():
    """Connects to the server and waits for events."""
    try:
        await sio.connect(GATEWAY_URL, socketio_path="/ws/socket.io")
        print("Client is running. Press Ctrl+C to disconnect.")
        await sio.wait()  # Wait indefinitely for events
    except socketio.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")
    finally:
        if sio.connected:
            await sio.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient shutting down.")
```

---

## 2. Using the Agent Library Directly (Advanced)

These examples show how to import and run the agent classes directly within a Python script. This approach bypasses the Gateway/Worker architecture and is suitable for custom backends or standalone applications.

### Example 2.1: Running the `DAGAgent`

This shows the standard workflow for initializing and running the `DAGAgent` programmatically.

```python
import asyncio
from llm_agent_x.agents.dag_agent import DAGAgent, TaskRegistry, Task
from llm_agent_x.tools.brave_web_search import brave_web_search

async def run_dag_agent():
    # 1. Initialize the Task Registry
    registry = TaskRegistry()

    # 2. Add initial data (documents) to the registry
    registry.add_document(
        "Financial_Report_Q1",
        "Q1 revenue was $1.2M with a profit of $200k."
    )
    registry.add_document(
        "Market_Analysis_Q1",
        "Competitor A launched a new product, impacting our market share by 5%."
    )

    # 3. Define the root task for the agent to tackle
    root_task = Task(
        id="ROOT_Q1_BRIEFING",
        desc="Create a comprehensive investor briefing for Q1. First, plan to analyze financial reports and market data. Then, synthesize the findings into a summary.",
        needs_planning=True,
    )
    registry.add_task(root_task)

    # 4. Initialize and configure the DAGAgent
    agent = DAGAgent(
        registry=registry,
        llm_model="gpt-4o-mini",
        tools=[brave_web_search]
    )

    # 5. Run the agent and wait for completion
    print("--- Running DAGAgent ---")
    await agent.run()

    # 6. Retrieve and print the final result
    print("\n--- DAG Execution Complete ---")
    final_result_task = registry.tasks.get("ROOT_Q1_BRIEFING")
    if final_result_task and final_result_task.status == 'complete':
        print("\n--- Agent's Final Report ---")
        print(final_result_task.result)
    else:
        print("Agent failed to complete the root task.")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY and BRAVE_API_KEY are set in your environment
    asyncio.run(run_dag_agent())
```

### Example 2.2: Running the `RecursiveAgent`

This example demonstrates a basic programmatic use of the legacy `RecursiveAgent`.

```python
import asyncio
from llm_agent_x.agents.recursive_agent import RecursiveAgent, RecursiveAgentOptions, TaskLimit
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from llm_agent_x.tools.brave_web_search import brave_web_search

async def run_recursive_agent():
    # 1. Configure the LLM and tools
    client = AsyncOpenAI()
    llm = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))
    tools_dict = {"web_search": brave_web_search}

    # 2. Define Agent Options
    agent_options = RecursiveAgentOptions(
        llm=llm,
        tools=[brave_web_search],
        tools_dict=tools_dict,
        task_limits=TaskLimit.from_array([2, 1, 0]),
        allow_search=True,
        allow_tools=True,
        mcp_servers=[],
    )

    # 3. Create and Run the Agent
    agent = RecursiveAgent(
        task="Explore the pros and cons of remote work for software development teams.",
        u_inst="Provide a balanced view with three points for each side.",
        agent_options=agent_options
    )

    print("--- Running RecursiveAgent ---")
    result = await agent.run()

    # 4. Print the final result
    print("\n--- Agent's Final Report ---")
    print(result)

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY and BRAVE_API_KEY are set in your environment
    asyncio.run(run_recursive_agent())
```