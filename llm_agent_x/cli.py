import argparse
import sys
from os import getenv, environ
from pathlib import Path
from dotenv import load_dotenv
import nltk
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from typing import Optional  # Import Dict, Optional
from enum import Enum # Import Enum for TaskType
from typing import Literal
from sumy.parsers.html import HtmlParser

from llm_agent_x import ( # Changed from . to llm_agent_x
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
    TaskObject, # Import TaskObject
    TaskFailedException # Import TaskFailedException
)
from llm_agent_x.backend import AppendMerger, LLMMerger, AlgorithmicMerger # Changed from .backend to llm_agent_x.backend
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from llm_agent_x.backend.callbacks.mermaidjs_callbacks import pre_tasks_executed, on_task_executed, on_tool_call_executed, save_flowchart
from llm_agent_x.console import console, task_tree, live
from llm_agent_x.constants import openai_api_key, openai_base_url
from llm_agent_x.tools.brave_web_search import brave_web_search
from llm_agent_x.tools.exec_python import exec_python

nltk.download('punkt_tab', force=False)

# Setup (only needed once)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(endpoint=getenv("ARIZE_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"))
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))

# Load environment variables
load_dotenv(".env", override=True)


# Initialize LLM and Search
llm = ChatOpenAI(
    base_url=openai_base_url,
    api_key=openai_api_key,
    model=getenv("DEFAULT_LLM", "gpt-4o-mini"),
    temperature=0.5,
)
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))

TaskType = Literal["research", "search", "basic", "text/reasoning"]

def main():
    global live  # Allow assignment to the global 'live' variable

    parser = argparse.ArgumentParser(description="Run the LLM agent.")
    parser.add_argument("task", type=str, help="The task to execute.")
    parser.add_argument(
        "--u_inst", type=str, help="User instructions for the task.", default=""
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=3,
        help="The maximum number of layers (deprecated, use task_limit).",
    )
    parser.add_argument(
        "--output", type=str, default="output.md", help="The output file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getenv(
            "DEFAULT_LLM", "gpt-4o-mini"
        ),  # Ensure default matches initialization
        help="The name of the LLM to use",
    )
    parser.add_argument(
        "--task_limit",
        type=str,
        default="[3,2,2,0]",
        help="Task limits per layer as a Python list string e.g., '[3,2,2,0]'",
    )
    parser.add_argument(
        "--merger",
        type=str,
        default="ai",
        choices=["ai", "append", "algorithmic"],
        help="Merger type: 'ai' or 'append'.",
    )
    parser.add_argument(
        "--no-tree", action="store_true", help="Disable the real-time tree view."
    )
    parser.add_argument(
        "--default_subtask_type",
        type=str,
        default="basic",
        choices=["research", "search", "basic", "text/reasoning"],
        help="The default task type to apply to all subtasks. Should be one of 'research', 'search', 'basic', or 'text/reasoning'.",
    )
    parser.add_argument(
        "--enable-python-execution",
        action="store_true",
        help="Enable the exec_python tool for the agent. (Requires Docker for sandbox mode)",
    )

    args = parser.parse_args()
    
    default_subtask_type: TaskType = args.default_subtask_type # type: ignore



    # Update LLM if model argument is different from default used for global llm
    global llm  # Allow modification of global llm
    if args.model != getenv("DEFAULT_LLM", "gpt-4o-mini"):
        llm = ChatOpenAI(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=args.model,
            temperature=0.5,
        )

    # Prepare tools based on the CLI flag
    available_tools = [brave_web_search]
    tools_dict_for_agent = {
        "web_search": brave_web_search,
        "brave_web_search": brave_web_search,
    }

    if args.enable_python_execution:
        available_tools.append(exec_python)
        tools_dict_for_agent["exec_python"] = exec_python
        tools_dict_for_agent["exec"] = exec_python # Alias

    tool_llm = llm.bind_tools(available_tools)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize live display (Rich Tree)
    if not args.no_tree:
        console.print("Starting agent with real-time tree view...")
        live = Live(
            task_tree,
            console=console,
            auto_refresh=True,
            vertical_overflow="visible",
        )
    else:
        console.print("Starting agent without real-time tree view...")
        live = None  # Clear live display manager after use

    with tracer.start_as_current_span("agent run") as span:
        agent = RecursiveAgent(
            task=args.task,
            u_inst=args.u_inst,
            tracer=tracer,
            tracer_span=span,
            agent_options=RecursiveAgentOptions(
                search_tool=brave_web_search,
                pre_task_executed=pre_tasks_executed,
                on_task_executed=on_task_executed,
                on_tool_call_executed=on_tool_call_executed,
                llm=llm,
                tool_llm=tool_llm, # This tool_llm is now correctly bound
                tools=[], # This 'tools' list in RecursiveAgentOptions seems to be different from the bound tools. Keep as is unless its purpose is clarified.
                allow_search=True, # Assuming brave_web_search is always allowed if available
                allow_tools=True if args.enable_python_execution else False,
                tools_dict=tools_dict_for_agent,
                task_limits=TaskLimit.from_array(eval(args.task_limit)),
                merger={"ai": LLMMerger, "append": AppendMerger, "algorithmic": AlgorithmicMerger}[args.merger],
            ),
        )

        try:
            if live is not None:
                with live: # Execute the agent
                    response = agent.run()
            else:
                response = agent.run()
                
        except TaskFailedException as e:
            console.print_exception() # Output exception to console
            console.print(f"Task '{args.task}' failed: {e}", style="bold red")
            response = f"ERROR: Task '{args.task}' failed. See logs for details."
        except Exception as e:
            console.print_exception()
            console.print(f"An unexpected error occurred: {e}", style="bold red")
            response = f"ERROR: An unexpected error occurred. See logs for details."

        finally: # Ensure cleanup regardless of result
            if live is not None:
                 live.stop()  # Ensure live display is stopped
            live = None


        save_flowchart(output_dir)

        # Save Response
        output_file = output_dir / args.output
        with output_file.open("w") as output_f:
            output_f.write(response)
        console.print(f"Agent response saved to {output_file}")
        console.print("\nFinal Response:\n", style="bold green")
        console.print(response)


if __name__ == "__main__":
    main()