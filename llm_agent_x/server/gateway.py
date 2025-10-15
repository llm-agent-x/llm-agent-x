import asyncio
import json
import logging
import os
import threading
import uuid

import pika
import socketio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pika.adapters.blocking_connection import BlockingChannel
from dotenv import load_dotenv
from starlette.responses import JSONResponse

load_dotenv(".env", override=True)

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("APIGateway")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

# --- Global State & Event Loop ---
task_state_cache = {}
main_event_loop = None  # <<< NEW: Global variable to hold the main event loop

# --- FastAPI App Setup ---
app = FastAPI(title="LLM Agent X Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],  # Add any other origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Socket.IO Setup ---
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app_asgi = socketio.ASGIApp(sio, other_asgi_app=app)


# --- RabbitMQ Connection for Publishing Directives ---
def get_rabbitmq_channel() -> BlockingChannel:
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    return connection.channel()

def get_rabbitmq_connection_and_channel():
    """
    Creates and returns a new RabbitMQ connection and channel.
    This should be called for each request that needs to publish.
    """
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
        channel = connection.channel()
        # Ensure the queue exists. This is an idempotent operation.
        channel.queue_declare(queue=DIRECTIVES_QUEUE, durable=True)
        return connection, channel
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        return None, None

DIRECTIVES_QUEUE = "directives_queue"

# --- API Endpoints (Unchanged) ---
@app.get("/api/tasks")
async def get_all_tasks():
    logger.info("Received request for all tasks.")
    # Filter out documents from the general task list
    tasks_only = {
        task_id: task
        for task_id, task in task_state_cache.items()
        if task.get("task_type", "task") == "task"
    }
    return {"tasks": tasks_only}


@app.post("/api/tasks")
async def add_root_task(request: Request):
    logger.info("Received request to add root task.")
    connection, channel = None, None
    try:
        body = await request.json()
        desc = body.get("desc")
        mcp_servers = body.get("mcp_servers", [])

        if not desc:
            raise ValueError("Description cannot be empty")

        message = {
            "task_id": str(uuid.uuid4()),
            "command": "ADD_ROOT_TASK",
            "payload": {
                "desc": desc,
                "needs_planning": True,
                "mcp_servers": mcp_servers,
            },
        }

        connection, channel = get_rabbitmq_connection_and_channel()
        if not channel:
            return JSONResponse(status_code=503, content={"message": "Service unavailable: Cannot connect to message queue."})

        channel.basic_publish(
            exchange="",
            routing_key=DIRECTIVES_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
        )
        logger.info(f"Published new task directive to queue: {desc}")
        return {"status": "new task submitted"}

    except Exception as e:
        logger.exception(f"Error in add_root_task: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})
    finally:
        if connection and connection.is_open:
            connection.close()

@app.get("/api/documents")
async def get_all_documents():
    """Fetches all entities marked as documents from the state cache."""
    logger.info("Received request for all documents.")
    documents = []
    for task_id, task_data in task_state_cache.items():
        if task_data.get("task_type") == "document":
            doc_state = task_data.get("document_state", {})
            # Extract name from description, assuming "Document: {name}" format
            name = task_data.get("desc", "").replace("Document: ", "", 1)
            documents.append({
                "id": task_id,
                "name": name,
                "content": doc_state.get("content", ""),
            })
    return documents

@app.post("/api/documents")
async def add_document(request: Request):
    """Receives a new document from the UI and sends a directive to the agent."""
    logger.info("Received request to add document.")
    connection, channel = None, None
    try:
        body = await request.json()
        name = body.get("name")
        content = body.get("content")

        # Validate that name and content are provided
        if not name or content is None:
            raise ValueError("Document name and content cannot be empty")

        # Construct the message for the agent
        message = {
            "task_id": str(uuid.uuid4()),  # The backend generates the unique ID
            "command": "ADD_DOCUMENT",
            "payload": {
                "name": name,
                "content": content,
            },
        }

        connection, channel = get_rabbitmq_connection_and_channel()
        if not channel:
            return JSONResponse(
                status_code=503,
                content={
                    "message": "Service unavailable: Cannot connect to message queue."
                },
            )

        channel.basic_publish(
            exchange="",
            routing_key=DIRECTIVES_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
            ),
        )
        logger.info(f"Published new document directive to queue: {name}")
        return {"status": "new document submitted"}

    except Exception as e:
        logger.exception(f"Error in add_document: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error"},
        )
    finally:
        if connection and connection.is_open:
            connection.close()
@app.put("/api/documents/{document_id}")
async def update_document(document_id: str, request: Request):
    """Sends a directive to update an existing document."""
    logger.info(f"Received request to update document {document_id}.")
    connection, channel = None, None
    try:
        body = await request.json()
        message = {
            "task_id": document_id,
            "command": "UPDATE_DOCUMENT",
            "payload": {
                "name": body.get("name"),
                "content": body.get("content"),
            },
        }
        connection, channel = get_rabbitmq_connection_and_channel()
        if not channel:
            return JSONResponse(status_code=503, content={"message": "Service unavailable."})

        channel.basic_publish(
            exchange="",
            routing_key=DIRECTIVES_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
        )
        return {"status": "update directive sent"}
    except Exception as e:
        logger.exception(f"Error in update_document: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal server error"})
    finally:
        if connection and connection.is_open:
            connection.close()


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Sends a directive to delete (prune) a document."""
    logger.info(f"Received request to delete document {document_id}.")
    connection, channel = None, None
    try:
        message = {
            "task_id": document_id,
            "command": "DELETE_DOCUMENT",
            "payload": {}
        }
        connection, channel = get_rabbitmq_connection_and_channel()
        if not channel:
            return JSONResponse(status_code=503, content={"message": "Service unavailable."})

        channel.basic_publish(
            exchange="",
            routing_key=DIRECTIVES_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
        )
        return {"status": "delete directive sent"}
    except Exception as e:
        logger.exception(f"Error in delete_document: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal server error"})
    finally:
        if connection and connection.is_open:
            connection.close()

# --- MODIFIED listen_for_state_updates FUNCTION ---
# The existing RabbitMQ listener for state updates needs a small tweak
# to handle documents correctly.

def callback(ch, method, properties, body):
    global task_state_cache
    update = json.loads(body)
    task_id = update.get("id")
    if task_id:
        # Check if the task is being pruned/deleted
        if update.get("status") == "pruned":
            if task_id in task_state_cache:
                del task_state_cache[task_id]
                logger.info(f"Removed pruned task/document {task_id} from cache.")
                # We can optionally emit a deletion event to the UI here if needed
        else:
            task_state_cache[task_id] = update
            logger.debug(f"Received state update for {task_id}, broadcasting.")

        # The existing sio.emit will now broadcast updates for tasks AND documents
        # The frontend will need to handle these updates appropriately.
        if main_event_loop:
            asyncio.run_coroutine_threadsafe(
                sio.emit("task_update", {"task": update}), main_event_loop
            )
    ch.basic_ack(delivery_tag=method.delivery_tag)
@app.post("/api/tasks/{task_id}/directive")
async def post_directive(task_id: str, request: Request):
    logger.info(f"Received directive for task {task_id}")
    connection, channel = None, None
    try:
        directive = await request.json()
        message = {
            "task_id": task_id,
            "command": directive.get("command"),
            "payload": directive.get("payload"),
        }

        connection, channel = get_rabbitmq_connection_and_channel()
        if not channel:
            return JSONResponse(status_code=503, content={"message": "Service unavailable: Cannot connect to message queue."})

        channel.basic_publish(
            exchange="",
            routing_key=DIRECTIVES_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
        )
        logger.info(f"Published directive to queue: {message}")
        return {"status": "directive sent"}

    except Exception as e:
        logger.exception(f"Error in post_directive: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})
    finally:
        if connection and connection.is_open:
            connection.close()


# --- RabbitMQ Listener for State Updates (runs in background) ---
def listen_for_state_updates():
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    exchange_name = "state_updates_exchange"
    channel.exchange_declare(exchange=exchange_name, exchange_type="fanout")
    result = channel.queue_declare(queue="", exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange=exchange_name, queue=queue_name)
    logger.info("Waiting for state updates from the agent...")

    def callback(ch, method, properties, body):
        global task_state_cache
        update = json.loads(body)
        task_id = update.get("id")
        if task_id:
            task_state_cache[task_id] = update
            logger.debug(f"Received state update for {task_id}, broadcasting.")

            # --- MODIFIED: Use the captured main_event_loop ---
            if main_event_loop:
                asyncio.run_coroutine_threadsafe(
                    sio.emit("task_update", {"task": update}), main_event_loop
                )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    try:
        channel.start_consuming()
    except Exception as e:
        logger.error(f"RabbitMQ consumer thread stopped unexpectedly: {e}")
    finally:
        connection.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


# --- MODIFIED: The startup event now captures the loop ---
@app.on_event("startup")
async def startup_event():
    global main_event_loop
    # Get the currently running event loop and store it globally
    main_event_loop = asyncio.get_running_loop()

    # Start the listener thread
    listener_thread = threading.Thread(target=listen_for_state_updates, daemon=True)
    listener_thread.start()
