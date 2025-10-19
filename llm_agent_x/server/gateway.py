import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager  # <-- FIX: Import the async version
from datetime import timezone, datetime

import pika
import socketio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, HTTPException, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from starlette.responses import JSONResponse

load_dotenv(".env", override=True)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("APIGateway")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

# --- Global State & Threading Locks ---
task_state_cache = {}
main_event_loop = None
pika_connection: BlockingConnection = None
pika_connection_lock = threading.Lock()
shutdown_event = threading.Event()

DIRECTIVES_QUEUE = "directives_queue"

def pika_heartbeat_thread():
    """Periodically processes pika events to keep the publisher connection alive."""
    while not shutdown_event.is_set():
        try:
            if pika_connection and pika_connection.is_open:
                with pika_connection_lock:
                    pika_connection.process_data_events()
            time.sleep(10)  # Sleep for an interval shorter than the heartbeat
        except Exception as e:
            logger.error(f"Error in Gateway Pika heartbeat thread: {e}", exc_info=True)
            break
    logger.info("Gateway Pika heartbeat thread shutting down.")

# --- FastAPI Lifespan Manager for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop, pika_connection
    main_event_loop = asyncio.get_running_loop()

    listener_thread = threading.Thread(target=listen_for_state_updates, daemon=True)
    listener_thread.start()

    try:
        # FIX: Add heartbeat=60 parameter to the connection
        pika_connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST, heartbeat=60))
        logger.info("Successfully established persistent RabbitMQ connection for publishing.")

        # FIX: Start the heartbeat thread
        heartbeat = threading.Thread(target=pika_heartbeat_thread, daemon=True)
        heartbeat.start()
    except pika.exceptions.AMQPConnectionError as e:
        logger.critical(f"Failed to connect to RabbitMQ on startup: {e}")
        pika_connection = None

    yield
    # --- Code to run on shutdown ---
    shutdown_event.set()
    if pika_connection and pika_connection.is_open:
        pika_connection.close()
        logger.info("Closed RabbitMQ publisher connection.")


# --- FastAPI App & Socket.IO Setup ---
app = FastAPI(title="LLM Agent X Gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app_asgi = socketio.ASGIApp(sio, other_asgi_app=app)


# --- RabbitMQ Dependency for FastAPI Endpoints ---
def get_pika_channel() -> BlockingChannel:
    if not pika_connection or not pika_connection.is_open:
        raise HTTPException(status_code=503, detail="Message queue service is unavailable.")

    with pika_connection_lock:
        channel = pika_connection.channel()
        channel.queue_declare(queue=DIRECTIVES_QUEUE, durable=True)
        try:
            yield channel
        finally:
            if channel and channel.is_open:
                channel.close()


# --- API Endpoints ---
@app.get("/api/tasks")
async def get_all_tasks():
    logger.info("Received request for all tasks and documents.")
    return {"tasks": task_state_cache}


@app.post("/api/tasks")
async def add_root_task(request: Request, channel: BlockingChannel = Depends(get_pika_channel)):
    try:
        body = await request.json()
        message = {
            "task_id": str(uuid.uuid4()),
            "command": "ADD_ROOT_TASK",
            "payload": {
                "desc": body.get("desc"),
                "needs_planning": True,
                "mcp_servers": body.get("mcp_servers", []),
            },
        }
        channel.basic_publish(
            exchange="", routing_key=DIRECTIVES_QUEUE, body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        return {"status": "new task submitted"}
    except Exception as e:
        logger.exception(f"Error in add_root_task: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/documents")
async def get_all_documents():
    documents = [
        {
            "id": task_id,
            "name": task_data.get("desc", "").replace("Document: ", "", 1),
            "content": task_data.get("document_state", {}).get("content", ""),
        }
        for task_id, task_data in task_state_cache.items()
        if task_data.get("task_type") == "document"
    ]
    return documents


@app.post("/api/documents")
async def add_document(request: Request, channel: BlockingChannel = Depends(get_pika_channel)):
    try:
        body = await request.json()
        message = {
            "task_id": str(uuid.uuid4()),
            "command": "ADD_DOCUMENT",
            "payload": {"name": body.get("name"), "content": body.get("content")},
        }
        channel.basic_publish(
            exchange="", routing_key=DIRECTIVES_QUEUE, body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        return {"status": "new document submitted"}
    except Exception as e:
        logger.exception(f"Error in add_document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.put("/api/documents/{document_id}")
async def update_document(document_id: str, request: Request, channel: BlockingChannel = Depends(get_pika_channel)):
    try:
        body = await request.json()
        message = {
            "task_id": document_id,
            "command": "UPDATE_DOCUMENT",
            "payload": {"name": body.get("name"), "content": body.get("content")},
        }
        channel.basic_publish(
            exchange="", routing_key=DIRECTIVES_QUEUE, body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        return {"status": "update directive sent"}
    except Exception as e:
        logger.exception(f"Error in update_document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, channel: BlockingChannel = Depends(get_pika_channel)):
    try:
        message = {"task_id": document_id, "command": "DELETE_DOCUMENT", "payload": {}}
        channel.basic_publish(
            exchange="", routing_key=DIRECTIVES_QUEUE, body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        return {"status": "delete directive sent"}
    except Exception as e:
        logger.exception(f"Error in delete_document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/tasks/{task_id}/directive")
async def post_directive(task_id: str, request: Request, channel: BlockingChannel = Depends(get_pika_channel)):
    try:
        directive = await request.json()
        message = {
            "task_id": task_id,
            "command": directive.get("command"),
            "payload": directive.get("payload"),
        }
        channel.basic_publish(
            exchange="", routing_key=DIRECTIVES_QUEUE, body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        return {"status": "directive sent"}
    except Exception as e:
        logger.exception(f"Error in post_directive: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/state/download")
async def download_state():
    if not task_state_cache:
        return JSONResponse(status_code=404, content={"message": "No state to download."})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"graph_state_{timestamp}.json"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return JSONResponse(content=task_state_cache, headers=headers)


@app.post("/api/state/upload")
async def upload_state(file: UploadFile = File(...), channel: BlockingChannel = Depends(get_pika_channel)):
    try:
        contents = await file.read()
        state_data = json.loads(contents)
        if not isinstance(state_data, dict):
            raise HTTPException(status_code=400, detail="Invalid file format.")

        message = {"task_id": "system", "command": "RESET_STATE", "payload": state_data}
        channel.basic_publish(
            exchange="", routing_key=DIRECTIVES_QUEUE, body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        return {"status": "State upload directive sent successfully."}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in uploaded file.")
    except Exception as e:
        logger.exception(f"Error in upload_state: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health():
    return {"status": "healthy"}


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
        try:
            update = json.loads(body)
            task_id = update.get("id")
            if not task_id: return

            if update.get("status") == "pruned" and task_id in task_state_cache:
                del task_state_cache[task_id]
                logger.info(f"Removed pruned task/document {task_id} from cache.")
            else:
                task_state_cache[task_id] = update
                logger.debug(f"Received state update for {task_id}, broadcasting.")

            if main_event_loop:
                asyncio.run_coroutine_threadsafe(
                    sio.emit("task_update", {"task": update}), main_event_loop
                )
        except Exception as e:
            logger.error(f"Error in state update callback: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    try:
        channel.start_consuming()
    except Exception as e:
        logger.error(f"RabbitMQ consumer thread stopped unexpectedly: {e}")
    finally:
        if connection.is_open:
            connection.close()