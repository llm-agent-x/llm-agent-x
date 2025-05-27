import argparse
import json
import time
from os import getenv
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper

from . import (
    int_to_base26,
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
)  # Adjusted import
from .backend import AppendMerger, LLMMerger  # Adjusted import

# Load environment variables
load_dotenv(
    ".env", override=True
)  # This might need adjustment if .env is not in the right place relative to cli.py

# Initialize LLM and Search

llm = ChatOpenAI(
    base_url=getenv("OPENAI_BASE_URL"),
    api_key=getenv("OPENAI_API_KEY"),
    model=getenv("DEFAULT_LLM", "gpt-4-0613"),
    temperature=0,
)
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))


def web_search(query: str, num_results: int) -> str:
    """
    Perform a web search with the given query and number of results, returning JSON-formatted results. **Make sure to be very specific in your search phrase. Ask for specific information, not general information.**

    :param query: The search query.
    :param num_results: The number of results to return.
    :return: A JSON-formatted string containing the search results.
    """
    try:
        results = search.results(query, num_results=num_results)
        return json.dumps(results)
    except Exception as error:
        print(error)
        return json.dumps([])


def exec_python(code, globals=None, locals=None):
    """
    Execute the given code with empty globals and locals.

    This function executes the provided code string in an isolated
    environment where both the global and local namespaces are empty,
    preventing the code from accessing or modifying external variables.

    Parameters:
    code (str): The code to be executed.
    globals (dict, optional): A dictionary of global variables. Defaults to None.
    locals (dict, optional): A dictionary of local variables. Defaults to None.

    Returns:
    None
    """
    exec(code, {}, {})


def main():
    # Ensure 'live' can be assigned in this function
    parser = argparse.ArgumentParser(description="Run the LLM agent.")
    parser.add_argument("task", type=str, help="The task to execute.")
    parser.add_argument("--u_inst", type=str, help="The task to execute.", default="")
    parser.add_argument(
        "--max_layers", type=int, default=3, help="The maximum number of layers."
    )
    parser.add_argument(
        "--output", type=str, default="output.md", help="The output file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getenv("DEFAULT_LLM"),
        help="The name of the LLM to use",
    )
    parser.add_argument("--task_limit", type=str, default="[3,2,2,0]")

    parser.add_argument("--merger", type=str, default="ai")
    args = parser.parse_args()

    tool_llm = llm.bind_tools([web_search])  # , exec_python])
    # Create the agent
    agent = RecursiveAgent(  # Adjusted: Removed llm_agent_x prefix
        task=args.task,
        u_inst=args.u_inst,
        agent_options=RecursiveAgentOptions(  # Adjusted: Removed llm_agent_x prefix
            max_layers=args.max_layers,
            search_tool=web_search,
            llm=llm,
            tool_llm=tool_llm,
            tools=[],
            allow_search=True,
            allow_tools=False,
            tools_dict={
                "web_search": web_search
            },  # "exec_python": exec_python, "exec": exec_python},
            task_limits=TaskLimit.from_array(
                eval(args.task_limit)
            ),  # Adjusted: Removed llm_agent_x prefix
            merger={"ai": LLMMerger, "append": AppendMerger}[args.merger],
        ),
    )
    response = agent.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save Response
    with (output_dir / args.output).open("w") as output:
        output.write(response)


if __name__ == "__main__":
    main()
