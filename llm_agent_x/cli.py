import argparse
import asyncio
import json
import sys
from os import getenv, environ
from pathlib import Path
import nltk
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Optional  # Import Dict, Optional
from enum import Enum  # Import Enum for TaskType
from typing import Literal
from sumy.parsers.html import HtmlParser

from llm_agent_x import (  # Changed from . to llm_agent_x
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
    TaskObject,  # Import TaskObject
    TaskFailedException,  # Import TaskFailedException
)
from llm_agent_x.backend import (
    AppendMerger,
    LLMMerger,
    AlgorithmicMerger,
)  # Changed from .backend to llm_agent_x.backend
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from icecream import ic

from llm_agent_x.backend.callbacks.mermaidjs_callbacks import (
    pre_tasks_executed,
    on_task_executed,
    on_tool_call_executed,
    save_flowchart,
)
from llm_agent_x.console import console, task_tree, live
from llm_agent_x.constants import openai_api_key, openai_base_url
from llm_agent_x.llm_manager import llm, model_tree
from llm_agent_x.tools.brave_web_search import brave_web_search

from llm_agent_x.cli_args_parser import parser
from llm_agent_x.tools.exec_python import exec_python


nltk.download("punkt_tab", force=False)

# Setup (only needed once)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(
    endpoint=getenv("ARIZE_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))

# Load environment variables


# Initialize LLM and Search
# llm = ChatOpenAI(
#     base_url=openai_base_url,
#     api_key=openai_api_key,
#     model=getenv("DEFAULT_LLM", "gpt-4o-mini"),
#     temperature=0.5,
# )
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))

TaskType = Literal["research", "search", "basic", "text/reasoning"]


def main():
    global live

    args = parser.parse_args()

    default_subtask_type: TaskType = args.default_subtask_type  # type: ignore

    # Prepare tools based on the CLI flag
    available_tools = [brave_web_search]
    tools_dict_for_agent = {
        "web_search": brave_web_search,
        "brave_web_search": brave_web_search,
    }

    mcp_config = args.mcp_config
    if mcp_config:
        try:
            with open(mcp_config, "r") as f:
                config = json.load(f)
            mcp_client = MultiServerMCPClient(config)
            mcp_tools = asyncio.run(mcp_client.get_tools())
            available_tools.extend(mcp_tools)

            # tools_dict_for_agent.update(mcp_client.get_tools_dict())
            # mcp_client.get_tools_dict() doesn't exist, so we must construct a dictionary based on each tool's __name__ and the tool itself
            for tool in mcp_tools:
                ic(tool.name)
                ic(type(tool.name))
                if tool.name not in tools_dict_for_agent:
                    tools_dict_for_agent[tool.name] = tool
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{mcp_config}' not found.")
    ic(tools_dict_for_agent.values())

    if args.enable_python_execution:
        available_tools.append(exec_python)
        tools_dict_for_agent["exec_python"] = exec_python
        tools_dict_for_agent["exec"] = exec_python  # Alias

    tool_llm = llm.bind_tools(available_tools)
    model_tree.update("llm.tools", tool_llm)
    model_tree.update("llm.small.tools", tool_llm)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize live display (Rich Tree)
    if not args.no_tree:
        console.print("Starting agent with real-time tree view...")
        live = live
    else:
        console.print("Starting agent without real-time tree view...")
        live = None  # Clear live display manager after use

    with tracer.start_as_current_span("agent run") as span:
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
                llm=(model_tree),
                tools=[],
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

        try:
            if live is not None:
                with live:  # Execute the agent
                    response = agent.run()
            else:
                response = agent.run()

        except TaskFailedException as e:
            console.print_exception()  # Output exception to console
            console.print(f"Task '{args.task}' failed: {e}", style="bold red")
            response = f"ERROR: Task '{args.task}' failed. See logs for details."
        except Exception as e:
            console.print_exception()
            console.print(f"An unexpected error occurred: {e}", style="bold red")
            response = f"ERROR: An unexpected error occurred. See logs for details."

        finally:  # Ensure cleanup regardless of result
            if live is not None:
                live.stop()  # Ensure live display is stopped
            live = None

        save_flowchart(output_dir)

        # Save Response
        if args.output is not None:
            output_file = output_dir / args.output
            with output_file.open("w") as output_f:
                output_f.write(response)
            console.print(f"Agent response saved to {output_file}")
        console.print("\nFinal Response:\n", style="bold green")
        console.print(response)


if __name__ == "__main__":
    main()
