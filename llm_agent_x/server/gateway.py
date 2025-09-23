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

load_dotenv(".env", override=True)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("APIGateway")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

# --- Global State (In-memory cache of task states) ---
task_state_cache = {}

# --- FastAPI App Setup ---
app = FastAPI(title="LLM Agent X Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Socket.IO Setup ---
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app_asgi = socketio.ASGIApp(sio, app)


# --- RabbitMQ Connection for Publishing Directives ---
def get_rabbitmq_channel() -> BlockingChannel:
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    return connection.channel()


DIRECTIVES_QUEUE = 'directives_queue'
rabbit_channel = get_rabbitmq_channel()
rabbit_channel.queue_declare(queue=DIRECTIVES_QUEUE, durable=True)


# --- API Endpoints ---
@app.get("/api/tasks")
async def get_all_tasks():
    return {"tasks": task_state_cache}


@app.post("/api/tasks")
async def add_root_task(request: Request):
    """Accepts a new root task from the UI and publishes it as a directive."""
    body = await request.json()
    desc = body.get("desc")

    if not desc:
        return {"status": "error", "message": "Description cannot be empty"}, 400

    message = {
        "task_id": str(uuid.uuid4()),  # A unique ID for the message itself
        "command": "ADD_ROOT_TASK",
        "payload": {
            "desc": desc,
            "needs_planning": True  # Assume all new tasks from UI need planning
        }
    }
    rabbit_channel.basic_publish(
        exchange='',
        routing_key=DIRECTIVES_QUEUE,
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
    )
    logger.info(f"Published new task directive to queue: {desc}")
    return {"status": "new task submitted"}



@app.post("/api/tasks/{task_id}/directive")
async def post_directive(task_id: str, request: Request):
    # ... (this existing endpoint is unchanged)
    directive = await request.json()
    message = {"task_id": task_id, "command": directive.get("command"), "payload": directive.get("payload")}
    rabbit_channel.basic_publish(
        exchange='',
        routing_key=DIRECTIVES_QUEUE,
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
    )
    logger.info(f"Published directive to queue: {message}")
    return {"status": "directive sent"}


# --- RabbitMQ Listener for State Updates (runs in background) ---
def listen_for_state_updates():
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    exchange_name = 'state_updates_exchange'
    channel.exchange_declare(exchange=exchange_name, exchange_type='fanout')

    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange=exchange_name, queue=queue_name)
    logger.info('Waiting for state updates from the agent...')

    def callback(ch, method, properties, body):
        global task_state_cache
        update = json.loads(body)
        task_id = update.get("id")
        if task_id:
            task_state_cache[task_id] = update
            logger.debug(f"Received state update for {task_id}, broadcasting.")
            asyncio.run_coroutine_threadsafe(sio.emit('task_update', {'task': update}), sio.loop)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    channel.start_consuming()


@app.on_event("startup")
async def startup_event():
    listener_thread = threading.Thread(target=listen_for_state_updates, daemon=True)
    listener_thread.start()