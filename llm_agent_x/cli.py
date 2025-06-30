import argparse
import asyncio
import json
import sys
import time
from os import getenv
from pathlib import Path
import nltk
from typing import Literal
import atexit  # <-- Import atexit

from llm_agent_x import (
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
    TaskFailedException,
)
from llm_agent_x.backend import (
    AppendMerger,
    LLMMerger,
    AlgorithmicMerger,
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
from llm_agent_x.cli_args_parser import parser
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from llm_agent_x.tools.brave_web_search import brave_web_search
from llm_agent_x.tools.exec_python import exec_python
from llm_agent_x.backend.utils import ic_dev, TaskType
# Keep all initialization code at the global level
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(
    endpoint=getenv("ARIZE_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
)
provider.add_span_processor(BatchSpanProcessor(exporter))


# --- FIX: Register the shutdown function with atexit ---
# This function will be called reliably upon script exit.
def shutdown_telemetry():
    # print("atexit: Shutting down OpenTelemetry provider...")
    provider.shutdown()
    # print("atexit: Shutdown complete.")


atexit.register(shutdown_telemetry)
# ---------------------------------------------------------

client = AsyncOpenAI(max_retries=3)
model = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))

nltk.download("punkt_tab", force=False)

output_dir = Path(getenv("OUTPUT_DIR", "./output/"))


def main():
    global live

    args = parser.parse_args()

    # The main try/finally block is now only for application logic, not global resources
    try:
        default_subtask_type: TaskType = args.default_subtask_type  # type: ignore

        # Prepare tools based on the CLI flag
        available_tools = [brave_web_search]
        tools_dict_for_agent = {
            "web_search": brave_web_search,
            "brave_web_search": brave_web_search,
        }

        mcp_config = args.mcp_config
        mcp_servers = []
        if mcp_config:
            try:
                with open(mcp_config, "r") as f:
                    config = json.load(f)
                ic_dev(config)

                assert type(config) == dict
                for key, value in config.items():
                    mcp_client = None
                    if value.get("transport") == "streamable_http":
                        mcp_client = MCPServerStreamableHTTP(url=value.get("url"))
                    if value.get("transport") == "stdio":
                        mcp_client = MCPServerStdio(
                            command=value.get("command"), args=value.get("args")
                        )
                    assert mcp_client is not None
                    mcp_servers.append(mcp_client)
            except FileNotFoundError:
                raise FileNotFoundError(f"Config file '{mcp_config}' not found.")

        ic_dev(tools_dict_for_agent.values())

        if args.enable_python_execution:
            available_tools.append(exec_python)
            tools_dict_for_agent["exec_python"] = exec_python
            tools_dict_for_agent["exec"] = exec_python  # Alias

        output_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_tree:
            console.print("Starting agent with real-time tree view...")
            live = live
        else:
            console.print("Starting agent without real-time tree view...")
            live = None

        with tracer.start_as_current_span("agent run") as span:
            ic("Running agent...")
            agent = RecursiveAgent(
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
                    llm=model,
                    tools=available_tools,
                    mcp_servers=mcp_servers,
                    allow_search=True,
                    allow_tools=True,
                    tools_dict=tools_dict_for_agent,
                    task_limits=TaskLimit.from_array(eval(args.task_limit)),
                    merger={
                        "ai": LLMMerger,
                        "append": AppendMerger,
                        "algorithmic": AlgorithmicMerger,
                    }[args.merger],
                ),
            )

            response = ""  # Default response
            try:
                if live is not None:
                    with live:
                        response = asyncio.run(agent.run())
                else:
                    response = asyncio.run(agent.run())

            except TaskFailedException as e:
                console.print_exception()
                console.print(f"Task '{args.task}' failed: {e}", style="bold red")
                response = f"ERROR: Task '{args.task}' failed. See logs for details."
            except Exception as e:
                console.print_exception()
                console.print(f"An unexpected error occurred: {e}", style="bold red")
                response = f"ERROR: An unexpected error occurred. See logs for details."
            finally:
                if live is not None:
                    live.stop()
                live = None

            if args.output is not None:
                output_file = output_dir / args.output
                with output_file.open("w", encoding="utf-8") as output_f:
                    output_f.write(response)
                console.print(f"Agent response saved to {output_file}")
            console.print("\nFinal Response:\n", style="bold green")
            console.print(response)

            ic(agent.cost)

    except Exception as e:
        # Catch any exceptions from the main logic so we still exit cleanly
        console.print_exception()
        console.print(
            f"A critical error occurred in main execution: {e}", style="bold red"
        )


if __name__ == "__main__":
    print("Starting agent...")
    main()
    print("Main function finished. Exiting.")
