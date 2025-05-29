import argparse
import json
import time
from os import getenv, environ
from pathlib import Path
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.tree import Tree
from rich.live import Live
from rich.text import Text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from bs4 import BeautifulSoup  # New import

from . import (
    int_to_base26,
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
)  # Adjusted import
from .backend import AppendMerger, LLMMerger

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Setup (only needed once)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))

# Load environment variables
load_dotenv(
    ".env", override=True
)  # This might need adjustment if .env is not in the right place relative to cli.py
# Initialize Console
console = Console()
live = None  #  display manager, will be initialized conditionally
task_tree = Tree(
    "Agent Execution"
)  # Root of the real-time task tree, used if live display is active
task_nodes = {}  # Store references to tree nodes, used if live display is active

# Initialize LLM and Search
llm = ChatOpenAI(
    base_url=getenv("OPENAI_BASE_URL"),
    api_key=getenv("OPENAI_API_KEY"),
    model=getenv("DEFAULT_LLM", "gpt-4o-mini"),
    temperature=0.5,
)
search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))

# Flowchart Tracking
flowchart = ["flowchart TD"]
task_ids: dict[str, str] = {}


def get_or_set_task_id(id: str) -> str | None:
    if id not in task_ids:
        result = int_to_base26(len(task_ids))
        task_ids[id] = result
        return result
    else:
        return task_ids.get(id)


def add_to_flowchart(line: str):
    flowchart.append(f"    {line}")


def render_flowchart():
    return "\n".join(flowchart)


def get_page_text_content(soup: BeautifulSoup) -> str:
    """
    Extracts text content from a BeautifulSoup object.
    Tries to be a bit smarter than just soup.get_text() by removing script/style.
    Still, this is a heuristic and might not be perfect for all pages.
    """
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text, try to preserve some structure with spaces
    text = soup.get_text(separator=" ", strip=True)

    # Optional: Further clean-up (e.g., multiple newlines, excessive whitespace)
    # text = "\n".join([line for line in text.splitlines() if line.strip()])
    # text = re.sub(r'\s+', ' ', text).strip()
    return text


def brave_web_search(query: str, num_results: int) -> str:
    """
    Perform a web search with the given query and number of results using the Brave Search API, returning JSON-formatted results. **Make sure to be very specific in your search phrase. Ask for specific information, not general information.**

    :param query: The search query.
    :param num_results: The desired number of results. This will be passed as the 'count' parameter to the Brave API.
    :return: A JSON-formatted string containing the search results, or a JSON-formatted error message if the API call fails or returns an error.
    """
    import time
    import requests
    import json
    from os import getenv
    from bs4 import BeautifulSoup

    # Some constants for request handling
    SCRAPE_TIMEOUT_SECONDS = 10
    max_scrape_chars = 5000  # Example max character count for scrape
    REQUEST_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def get_page_text_content(element):
        """
        Extract and normalize text content from an HTML element.

        :param element: The HTML element to extract content from.
        :return: A string containing the cleaned-up text content of the element.
        """
        return element.get_text(" ", strip=True)

    api_key = getenv("BRAVE_API_KEY")
    if not api_key or api_key == "YOUR_ACTUAL_BRAVE_API_KEY_HERE":
        return json.dumps(
            {
                "error": "BRAVE_API_KEY environment variable not set or is a placeholder.",
                "results": [],
            }
        )

    base_url = "https://api.search.brave.com/res/v1/web/search"
    brave_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": num_results}

    response_obj = None

    max_retries = 3  # Maximum number of retries for 429 errors
    retry_wait_time = 10  # Wait time in seconds before retrying on a 429 error
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # --- 1. Brave Search API Call ---
            search_response = requests.get(
                base_url, headers=brave_headers, params=params, timeout=10
            )
            response_obj = search_response

            # Handle 429 Too Many Requests
            if search_response.status_code == 429:
                retry_count += 1
                if retry_count <= max_retries:
                    print(
                        f"Rate limit reached. Retrying in {retry_wait_time} seconds... (Retry {retry_count}/{max_retries})"
                    )
                    time.sleep(retry_wait_time)
                    continue  # Retry the request
                else:
                    return json.dumps(
                        {
                            "error": "Rate limit exceeded. Exhausted all retries.",
                            "status_code": 429,
                            "results": [],
                        }
                    )

            search_response.raise_for_status()
            json_response_data = search_response.json()

            extracted_results_for_llm = []

            if json_response_data.get("web") and json_response_data["web"].get(
                "results"
            ):
                search_results_to_process = json_response_data["web"]["results"]

                for i, result in enumerate(search_results_to_process):
                    title = result.get("title")
                    url = result.get("url")
                    snippet = result.get(
                        "description", ""
                    )  # Default to empty string if no description

                    content_for_llm = snippet  # Default to snippet

                    if title and url:  # We need a URL to attempt scraping
                        print(
                            f"  Processing result {i+1}/{len(search_results_to_process)}: '{title}' ({url})"
                        )
                        try:
                            # --- 2. Scrape Individual Webpage ---
                            print(f"    Attempting to scrape content from {url}...")
                            page_response = requests.get(
                                url,
                                headers=REQUEST_HEADERS,
                                timeout=SCRAPE_TIMEOUT_SECONDS,
                                allow_redirects=True,
                            )
                            page_response.raise_for_status()  # Check for HTTP errors for the page itself

                            # Ensure content type is HTML before parsing
                            content_type = page_response.headers.get(
                                "Content-Type", ""
                            ).lower()
                            if "text/html" in content_type:
                                # Using lxml is generally faster and more robust
                                soup = BeautifulSoup(
                                    page_response.content, "lxml"
                                )  # or 'html.parser'

                                # More targeted text extraction (example, adjust as needed)
                                # Try to find common main content containers
                                main_content_text = ""
                                main_content_elements = soup.find_all(
                                    ["article", "main"]
                                )
                                if main_content_elements:
                                    for el in main_content_elements:
                                        main_content_text += (
                                            get_page_text_content(el) + " "
                                        )
                                    main_content_text = main_content_text.strip()

                                if (
                                    not main_content_text
                                ):  # Fallback to body if no specific main content found
                                    if soup.body:
                                        main_content_text = get_page_text_content(
                                            soup.body
                                        )
                                    else:  # Fallback if no body tag (highly unlikely for valid HTML)
                                        main_content_text = get_page_text_content(soup)

                                if main_content_text:
                                    print(
                                        f"    Scraped {len(main_content_text)} chars. Max allowed: {max_scrape_chars}"
                                    )
                                    if 0 < len(main_content_text) <= max_scrape_chars:
                                        content_for_llm = main_content_text
                                        print(
                                            f"    Using scraped content (length: {len(main_content_text)})."
                                        )
                                    elif len(main_content_text) > max_scrape_chars:
                                        print(
                                            f"    Scraped content too long ({len(main_content_text)} chars), falling back to snippet."
                                        )
                                        content_for_llm = (
                                            snippet
                                            + " [Note: Full content exceeded character limit]"
                                        )
                                    else:  # Scraped text was empty
                                        print(
                                            f"    Scraped content was empty, using snippet."
                                        )
                                else:
                                    print(
                                        f"    Could not extract meaningful text, using snippet."
                                    )
                            else:
                                print(
                                    f"    Skipping scrape: Content-Type is '{content_type}', not HTML. Using snippet."
                                )
                                content_for_llm = (
                                    snippet + " [Note: Content was not HTML]"
                                )

                        except requests.exceptions.Timeout:
                            print(f"    Scraping timed out for {url}. Using snippet.")
                            content_for_llm = (
                                snippet + " [Note: Page timed out during scraping]"
                            )
                        except requests.exceptions.HTTPError as e:
                            print(
                                f"    HTTP error {e.response.status_code} while scraping {url}. Using snippet."
                            )
                            content_for_llm = (
                                snippet
                                + f" [Note: HTTP {e.response.status_code} during scraping]"
                            )
                        except requests.exceptions.RequestException as e:
                            print(f"    Error scraping {url}: {e}. Using snippet.")
                            content_for_llm = snippet + " [Note: Error during scraping]"
                        except (
                            Exception
                        ) as e:  # Catch-all for any other scraping/parsing errors
                            print(
                                f"    Unexpected error scraping/parsing {url}: {e}. Using snippet."
                            )
                            content_for_llm = (
                                snippet + " [Note: Unexpected error during scraping]"
                            )

                        extracted_results_for_llm.append(
                            {
                                "title": title,
                                "url": url,
                                "content": content_for_llm.strip(),  # Ensure no leading/trailing ws
                            }
                        )
                    else:  # If no title or URL from Brave search
                        if title and snippet:  # Still add if we have title and snippet
                            extracted_results_for_llm.append(
                                {
                                    "title": title,
                                    "url": url or "N/A",
                                    "content": snippet.strip(),
                                }
                            )

                return extracted_results_for_llm

            if "errors" in json_response_data:
                return json.dumps(
                    {
                        "error": "API returned errors in response",
                        "details": json_response_data["errors"],
                        "results": [],
                    }
                )
            if not extracted_results_for_llm:
                return []  # No web results found or processed

        except Exception as e:
            return json.dumps(
                {"error": f"An unexpected error occurred: {str(e)}", "results": []}
            )


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


def pre_tasks_executed(task, uuid, parent_agent_uuid):
    id = get_or_set_task_id(uuid)
    parent_id = (
        get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None
    )

    # Flowchart Update (always runs)
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{parent_id} -->|Subtask| {id}[{task}]")
    else:
        add_to_flowchart(f"{id}[{task}]")

    # Real-time Hierarchy Update (conditional on live display being active)
    if live:
        task_text = Text(task, style="bold yellow")
        if parent_agent_uuid is None:
            task_nodes[uuid] = task_tree.add(task_text)  # Top-level task
        elif parent_agent_uuid in task_nodes:  # Check if parent node exists
            task_nodes[uuid] = task_nodes[parent_agent_uuid].add(task_text)  # Subtask
        else:
            # This case should ideally not happen if tasks are processed in order.
            # Adding as a direct child of the root tree if parent is missing for some reason.
            console.print(
                f"[yellow]Warning: Parent node {parent_agent_uuid} for task '{task}' not found in tree. Adding as top-level.[/yellow]"
            )
            task_nodes[uuid] = task_tree.add(task_text)

        live.update(task_tree)


def on_task_executed(task, uuid, response, parent_agent_uuid):
    id = get_or_set_task_id(uuid)
    parent_id = (
        get_or_set_task_id(parent_agent_uuid) if parent_agent_uuid is not None else None
    )

    # Flowchart Update (always runs)
    if parent_agent_uuid is not None:
        add_to_flowchart(f"{id} -->|Completed| {parent_id}")
    add_to_flowchart(f'{id} --> |Result| ("`{response}`")')

    # Real-time Hierarchy Update (conditional on live display being active)
    if live:
        if uuid in task_nodes:
            task_nodes[uuid].label = Text(f"{task} ✅", style="green")
        else:
            console.print(
                f"[yellow]Warning: Task node {uuid} for task '{task}' not found in tree for completion update.[/yellow]"
            )
        live.update(task_tree)


def on_tool_call_executed(
    task, uuid, tool_name, tool_args, tool_response, success=True, tool_call_id=None
):
    tool_task_id = f"{uuid}: {tool_name}"  # Unique ID for the tool call visualization

    # Flowchart Update (always runs)
    add_to_flowchart(
        f"{get_or_set_task_id(uuid)} -->|Tool call| {get_or_set_task_id(tool_task_id)}[{tool_name}]"
    )
    add_to_flowchart(
        f"{get_or_set_task_id(tool_task_id)} --> {get_or_set_task_id(uuid)}"
    )

    # Real-time Hierarchy Update (conditional on live display being active)
    if live:
        text_json = json.dumps(tool_args, indent=0).replace("\\n", "")
        tool_text = Text(f"{tool_name} 🔧 {text_json}", style="blue")

        if uuid in task_nodes:  # Check if parent task node exists
            if not success:
                tool_text.stylize("bold red")
                # Create a new node for the tool call itself, even if it failed
                tool_node = task_nodes[uuid].add(tool_text)
                tool_node.label = Text(
                    f"{tool_name} ❌", style="red"
                )  # Update its label to show failure
                task_nodes[tool_task_id] = (
                    tool_node  # Store reference if needed elsewhere, though not strictly necessary for this structure
                )
            else:
                task_nodes[tool_task_id] = task_nodes[uuid].add(tool_text)
        else:
            console.print(
                f"[yellow]Warning: Parent task node {uuid} for tool '{tool_name}' not found in tree.[/yellow]"
            )

        live.update(task_tree)


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
        choices=["ai", "append"],
        help="Merger type: 'ai' or 'append'.",
    )
    parser.add_argument(
        "--no-tree", action="store_true", help="Disable the real-time tree view."
    )

    args = parser.parse_args()

    # Update LLM if model argument is different from default used for global llm
    global llm  # Allow modification of global llm
    if args.model != getenv("DEFAULT_LLM", "gpt-4o-mini"):
        llm = ChatOpenAI(
            base_url=getenv("OPENAI_BASE_URL"),
            api_key=getenv("OPENAI_API_KEY"),
            model=args.model,
            temperature=0.5,
        )

    tool_llm = llm.bind_tools([brave_web_search])  # , exec_python])

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
                tool_llm=tool_llm,
                tools=[],
                allow_search=True,
                allow_tools=False,  # Set to True if you want tools like exec_python to be considered by the agent's planning
                tools_dict={
                    "web_search": brave_web_search,
                    "brave_web_search": brave_web_search,
                    # "exec_python": exec_python, "exec": exec_python # Uncomment if exec_python is to be used
                },
                task_limits=TaskLimit.from_array(eval(args.task_limit)),
                merger={"ai": LLMMerger, "append": AppendMerger}[args.merger],
            ),
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        response = ""
        if not args.no_tree:
            console.print("Starting agent with real-time tree view...")
            with Live(
                task_tree,
                console=console,
                auto_refresh=True,
                vertical_overflow="visible",
            ) as live_display:
                live = live_display  # Assign to global variable for callbacks
                response = agent.run()  # Execute the agent
            live = None  # Clear live display manager after use
        else:
            console.print("Starting agent without real-time tree view...")
            # 'live' remains None, callbacks will skip tree updates
            response = agent.run()

        # Save Flowchart
        flowchart_file = output_dir / "flowchart.mmd"
        with flowchart_file.open("w") as flowchart_o:
            flowchart_o.write(render_flowchart())
        console.print(f"Flowchart saved to {flowchart_file}")

        # Save Response
        output_file = output_dir / args.output
        with output_file.open("w") as output_f:
            output_f.write(response)
        console.print(f"Agent response saved to {output_file}")
        console.print("\nFinal Response:\n", style="bold green")
        console.print(response)


if __name__ == "__main__":
    main()
