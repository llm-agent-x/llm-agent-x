# In llm_agent_x/cli.py

import argparse
import asyncio
import json
import sys
import time
from os import getenv
from pathlib import Path
import nltk
from typing import Literal
import atexit

# --- Agent Imports ---
from llm_agent_x.agents.recursive_agent import (
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
    TaskFailedException,
)
from llm_agent_x.agents.dag_agent import (
    DAGAgent,
    TaskRegistry,
    Task,
)

# --- Backend and Tool Imports ---
from llm_agent_x.backend import (
    AppendMerger,
    LLMMerger,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from icecream import ic

from llm_agent_x.backend.callbacks.mermaidjs_callbacks import (
    pre_tasks_executed,
    on_task_executed,
    on_tool_call_executed,
)
from llm_agent_x.console import console, live
from llm_agent_x.cli_args_parser import parser  # Your updated parser
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from llm_agent_x.tools.brave_web_search import brave_web_search
from llm_agent_x.tools.exec_python import exec_python
from llm_agent_x.backend.utils import ic_dev, TaskType

# --- Global Setup (remains the same) ---
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(
    endpoint=getenv("ARIZE_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
)
provider.add_span_processor(BatchSpanProcessor(exporter))


def shutdown_telemetry():
    provider.shutdown()


atexit.register(shutdown_telemetry)

nltk.download("punkt_tab", force=False, quiet=True)
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))


# --- Main Application Logic ---
def main():
    args = parser.parse_args()
    ic.disable()
    if args.dev_mode:
        ic.enable()

    # --- Shared Setup ---
    client = AsyncOpenAI(max_retries=3)
    available_tools = []
    if not args.disable_web_search:
        available_tools.append(brave_web_search)
    if args.enable_python_execution:
        available_tools.append(exec_python)

    output_dir.mkdir(parents=True, exist_ok=True)
    agent_instance = None
    total_cost = 0.0

    try:
        with tracer.start_as_current_span(f"llm_agent_x.{args.agent_type}.run") as span:
            span.set_attribute("agent.type", args.agent_type)
            span.set_attribute("agent.task", args.task)

            # ===============================================================
            # --- Agent Dispatch Logic ---
            # ===============================================================
            if args.agent_type == "recursive":
                console.print(
                    f"Initializing [bold yellow]RecursiveAgent[/bold yellow] for task: '{args.task}'"
                )
                # This setup is specific to the RecursiveAgent
                model_for_recursive = OpenAIModel(
                    args.model, provider=OpenAIProvider(openai_client=client)
                )
                tools_dict_for_agent = {"web_search": brave_web_search}
                mcp_servers = []

                if args.mcp_config:
                    # MCP config loading logic...
                    try:
                        with open(args.mcp_config, "r") as f:
                            config = json.load(f)
                        assert isinstance(config, dict)
                        for key, value in config.items():
                            mcp_client = None
                            if value.get("transport") == "streamable_http":
                                mcp_client = MCPServerStreamableHTTP(
                                    url=value.get("url")
                                )
                            if value.get("transport") == "stdio":
                                mcp_client = MCPServerStdio(
                                    command=value.get("command"), args=value.get("args")
                                )
                            if mcp_client:
                                mcp_servers.append(mcp_client)
                    except (
                        FileNotFoundError,
                        json.JSONDecodeError,
                        AssertionError,
                    ) as e:
                        console.print(
                            f"[bold red]Error loading MCP config:[/bold red] {e}"
                        )
                        sys.exit(1)

                agent_instance = RecursiveAgent(
                    task=args.task,
                    task_type_override=args.task_type,
                    u_inst=args.u_inst,
                    tracer=tracer,
                    tracer_span=span,
                    agent_options=RecursiveAgentOptions(
                        search_tool=brave_web_search,
                        pre_task_executed=pre_tasks_executed,
                        on_task_executed=on_task_executed,
                        on_tool_call_executed=on_tool_call_executed,
                        llm=model_for_recursive,
                        tools=available_tools,
                        mcp_servers=mcp_servers,
                        allow_tools=True,
                        tools_dict=tools_dict_for_agent,
                        task_limits=TaskLimit.from_array(eval(args.task_limit)),
                        merger={
                            "ai": LLMMerger,
                            "append": AppendMerger,
                        }[args.merger],
                    ),
                )
                # Run logic for Recursive agent
                response = ""
                try:
                    active_live = live if not args.no_tree else None
                    if active_live:
                        with active_live:
                            response = asyncio.run(agent_instance.run())
                    else:
                        response = asyncio.run(agent_instance.run())
                    total_cost = agent_instance.cost
                except TaskFailedException as e:
                    console.print_exception()
                    response = (
                        f"ERROR: Task '{args.task}' failed. See logs for details."
                    )

            elif args.agent_type == "dag":
                console.print(
                    f"Initializing [bold cyan]DAGAgent[/bold cyan] for task: '{args.task}'"
                )
                if not args.dag_documents:
                    raise ValueError(
                        "DAG agent requires initial documents. Use the --dag-documents flag."
                    )

                # This setup is specific to the DAGAgent
                registry = TaskRegistry()
                try:
                    with open(args.dag_documents, "r") as f:
                        documents = json.load(f)
                    for doc in documents:
                        registry.add_document(doc["name"], doc["content"])
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    console.print(
                        f"[bold red]Error loading DAG documents:[/bold red] {e}"
                    )
                    sys.exit(1)

                root_task = Task(
                    id="ROOT_TASK",
                    desc=args.task,
                    needs_planning=True,
                )
                registry.add_task(root_task)

                agent_instance = DAGAgent(
                    registry=registry,
                    llm_model=args.model,  # DAGAgent takes the model name string
                    tools=available_tools,
                    tracer=tracer,
                    max_grace_attempts=args.max_grace_attempts,
                    global_proposal_limit=args.global_proposal_limit,
                )
                # Run logic for DAG agent
                console.print("\n--- Initial Task Status ---")
                registry.print_status_tree()
                console.print("\n--- Executing DAG... ---")
                asyncio.run(agent_instance.run())
                console.print("\n--- DAG Execution Complete ---")
                registry.print_status_tree()
                response = registry.tasks.get("ROOT_TASK").result
                total_cost = sum(t.cost for t in registry.tasks.values())

            # ===============================================================
            # --- Common Post-Execution Logic ---
            # ===============================================================
            if args.output:
                output_file = output_dir / args.output
                with output_file.open("w", encoding="utf-8") as output_f:
                    output_f.write(str(response))
                console.print(
                    f"\nAgent response saved to {output_file}", style="bold magenta"
                )

            console.print("\nFinal Response:\n", style="bold green")
            console.print(str(response))
            console.print(
                f"\nTotal estimated cost: ${total_cost:.4f}", style="bold yellow"
            )
            span.set_attribute("agent.total_cost", total_cost)

    except Exception as e:
        console.print_exception()
        console.print(f"A critical error occurred: {e}", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()
