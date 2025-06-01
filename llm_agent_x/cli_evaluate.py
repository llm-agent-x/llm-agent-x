import argparse
import base64
import json
import time
from os import (
    getenv,
    environ,
)  # environ was in cli.py, adding for consistency if needed later
from pathlib import Path
from dotenv import load_dotenv
import requests  # Added for brave_web_search
from rich.console import Console
from bs4 import BeautifulSoup  # Added for brave_web_search
from langchain_openai import ChatOpenAI

# from langchain_community.utilities import SearxSearchWrapper # Removed, brave search is used
from icecream import ic
from typing import Dict, List # Import Dict, List

from llm_agent_x import (  # Changed from . to llm_agent_x
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
)  # Adjusted import
from llm_agent_x.backend import AppendMerger, LLMMerger  # Adjusted import

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Setup (only needed once)
trace.set_tracer_provider(TracerProvider())
tracer_global_scope = trace.get_tracer(
    __name__
)  # Renamed to avoid conflict with tracer in main

exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))


# Load environment variables
load_dotenv(
    ".env", override=True
)  # This might need adjustment if .env is not in the right place relative to cli.py

# Initialize Console
console = (
    Console()
)  # Added from cli.py for consistency, though not heavily used in eval

# Initialize LLM
llm = ChatOpenAI(
    base_url=getenv("OPENAI_BASE_URL"),
    api_key=getenv("OPENAI_API_KEY"),
    model=getenv("DEFAULT_LLM", "gpt-4o-mini"),  # Matched cli.py
    temperature=0.5,  # Matched cli.py
)
# search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080")) # Removed
output_dir = Path(getenv("OUTPUT_DIR", "./output/"))


# Copied brave_web_search from cli.py
def brave_web_search(query: str, num_results: int) -> str:
    """
    Perform a web search with the given query and number of results using the Brave Search API, returning JSON-formatted results. **Make sure to be very specific in your search phrase. Ask for specific information, not general information.**

    :param query: The search query.
    :param num_results: The desired number of results. This will be passed as the 'count' parameter to the Brave API.
    :return: A JSON-formatted string containing the search results, or a JSON-formatted error message if the API call fails or returns an error.
    """
    import time  # Moved import inside as per original cli.py style for this function
    import requests
    import json
    from os import getenv
    from bs4 import (
        BeautifulSoup,
    )  # Already imported globally, but brave_web_search also had it locally. Kept local for self-containment.

    # Some constants for request handling
    SCRAPE_TIMEOUT_SECONDS = 10
    max_scrape_chars = 5000  # Example max character count for scrape
    REQUEST_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def get_page_text_content_internal(
        element,
    ):  # Renamed to avoid conflict if a global one existed
        """
        Extract and normalize text content from an HTML element.

        :param element: The HTML element to extract content from.

        :return: A string containing the cleaned-up text content of the element.
        """
        return element.get_text(" ", strip=True)

    api_key = getenv("BRAVE_API_KEY")
    if (
        not api_key or api_key == "YOUR_ACTUAL_BRAVE_API_KEY_HERE"
    ):  # Placeholder check from cli.py
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
            response_obj = search_response  # Store for potential debugging

            # Handle 429 Too Many Requests
            if search_response.status_code == 429:
                retry_count += 1
                if retry_count <= max_retries:
                    # Using rich console for printing if available, else standard print
                    (console if "console" in globals() else print)(
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
                        (console if "console" in globals() else print)(
                            f"  Processing result {i+1}/{len(search_results_to_process)}: '{title}' ({url})"
                        )
                        try:
                            # --- 2. Scrape Individual Webpage ---
                            (console if "console" in globals() else print)(
                                f"    Attempting to scrape content from {url}..."
                            )
                            page_response = requests.get(
                                url,
                                headers=REQUEST_HEADERS,
                                timeout=SCRAPE_TIMEOUT_SECONDS,
                                allow_redirects=True,
                            )
                            page_response.raise_for_status()  # Check for HTTP errors for the page itself

                            content_type = page_response.headers.get(
                                "Content-Type", ""
                            ).lower()
                            if "text/html" in content_type:
                                soup = BeautifulSoup(page_response.content, "lxml")

                                main_content_text = ""
                                main_content_elements = soup.find_all(
                                    ["article", "main"]
                                )
                                if main_content_elements:
                                    for el in main_content_elements:
                                        main_content_text += (
                                            get_page_text_content_internal(el) + " "
                                        )
                                    main_content_text = main_content_text.strip()

                                if not main_content_text:
                                    if soup.body:
                                        main_content_text = (
                                            get_page_text_content_internal(soup.body)
                                        )
                                    else:
                                        main_content_text = (
                                            get_page_text_content_internal(soup)
                                        )

                                if main_content_text:
                                    (console if "console" in globals() else print)(
                                        f"    Scraped {len(main_content_text)} chars. Max allowed: {max_scrape_chars}"
                                    )
                                    if 0 < len(main_content_text) <= max_scrape_chars:
                                        content_for_llm = main_content_text
                                        (console if "console" in globals() else print)(
                                            f"    Using scraped content (length: {len(main_content_text)})."
                                        )
                                    elif len(main_content_text) > max_scrape_chars:
                                        (console if "console" in globals() else print)(
                                            f"    Scraped content too long ({len(main_content_text)} chars), falling back to snippet."
                                        )
                                        content_for_llm = (
                                            snippet
                                            + " [Note: Full content exceeded character limit]"
                                        )
                                    else:
                                        (console if "console" in globals() else print)(
                                            f"    Scraped content was empty, using snippet."
                                        )
                                else:
                                    (console if "console" in globals() else print)(
                                        f"    Could not extract meaningful text, using snippet."
                                    )
                            else:
                                (console if "console" in globals() else print)(
                                    f"    Skipping scrape: Content-Type is '{content_type}', not HTML. Using snippet."
                                )
                                content_for_llm = (
                                    snippet + " [Note: Content was not HTML]"
                                )

                        except requests.exceptions.Timeout:
                            (console if "console" in globals() else print)(
                                f"    Scraping timed out for {url}. Using snippet."
                            )
                            content_for_llm = (
                                snippet + " [Note: Page timed out during scraping]"
                            )
                        except requests.exceptions.HTTPError as e:
                            (console if "console" in globals() else print)(
                                f"    HTTP error {e.response.status_code} while scraping {url}. Using snippet."
                            )
                            content_for_llm = (
                                snippet
                                + f" [Note: HTTP {e.response.status_code} during scraping]"
                            )
                        except requests.exceptions.RequestException as e:
                            (console if "console" in globals() else print)(
                                f"    Error scraping {url}: {e}. Using snippet."
                            )
                            content_for_llm = snippet + " [Note: Error during scraping]"
                        except Exception as e:
                            (console if "console" in globals() else print)(
                                f"    Unexpected error scraping/parsing {url}: {e}. Using snippet."
                            )
                            content_for_llm = (
                                snippet + " [Note: Unexpected error during scraping]"
                            )

                        extracted_results_for_llm.append(
                            {
                                "title": title,
                                "url": url,
                                "content": content_for_llm.strip(),
                            }
                        )
                    else:
                        if title and snippet:
                            extracted_results_for_llm.append(
                                {
                                    "title": title,
                                    "url": url or "N/A",
                                    "content": snippet.strip(),
                                }
                            )

                #  The agent expects a JSON string, so we should return json.dumps(extracted_results_for_llm)
                return json.dumps(extracted_results_for_llm)

            if "errors" in json_response_data:
                return json.dumps(
                    {
                        "error": "API returned errors in response",
                        "details": json_response_data["errors"],
                        "results": [],
                    }
                )
            if (
                not extracted_results_for_llm
            ):  # Should be empty list if no web results, not [] as string
                return json.dumps(
                    []
                )  # No web results found or processed, return empty list as JSON string

        except Exception as e:  # Catch any other exception during the API call itself
            return json.dumps(
                {"error": f"An unexpected error occurred: {str(e)}", "results": []}
            )
    # If loop finishes due to retries exhausted, this part is not reached because of earlier returns.
    # This case should be handled within the loop. If it were to be reached, means an issue.
    return json.dumps(
        {
            "error": "Failed to get search results after multiple retries or due to an unknown issue.",
            "results": [],
        }
    )


def exec_python(code, globals=None, locals=None):  # Unchanged, kept for completeness
    """
    Execute the given code with empty globals and locals.
    """
    exec(code, {}, {})


def main():
    global llm  # Allow modification of global llm

    tracer = trace.get_tracer(__name__)  # Get tracer for main function

    parser = argparse.ArgumentParser(description="Run the LLM agent for evaluation.")
    parser.add_argument(
        "prompts", type=str, help="The path of the file containing prompts to evaluate."
    )
    parser.add_argument(
        "--u_inst", type=str, help="User instructions for the task.", default=""
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=3,
        help="The maximum number of layers (deprecated, use task_limit).",
    )  # Kept for CLI compatibility, but not used in RecursiveAgentOptions
    parser.add_argument(
        "--output", type=str, default="output.json", help="The output file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getenv("DEFAULT_LLM", "gpt-4o-mini"),  # Matched cli.py
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
    args = parser.parse_args()

    # Update LLM if model argument is different from default used for global llm
    if args.model != getenv("DEFAULT_LLM", "gpt-4o-mini"):
        llm = ChatOpenAI(
            base_url=getenv("OPENAI_BASE_URL"),
            api_key=getenv("OPENAI_API_KEY"),
            model=args.model,
            temperature=0.5,  # Matched cli.py
        )

    tool_llm = llm.bind_tools([brave_web_search])  # Updated to brave_web_search

    with open(args.prompts, "r") as f:
        try:
            prompts: List[str] = json.load(f) # Add prompt type
            if not isinstance(prompts, list): # if prompts is not a List
                raise TypeError("Expected prompts to be a List of strings")
            for prompt in prompts:
                if not isinstance(prompt, str): # Check each entry in prompt if its a str
                    raise TypeError("Each entry of Prompts should be a String")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in prompts file: {e}")
        except TypeError as e:
            raise ValueError(f"Issue with List: {e}")

    responses = []
    for i, prompt_task in enumerate(
        prompts
    ):  # Changed 'prompt' to 'prompt_task' to avoid conflict
        console.print(
            f'\n[bold cyan]Processing prompt {i+1}/{len(prompts)}: "{prompt_task}"[/bold cyan]'
        )
        with tracer.start_as_current_span(
            f"evaluation_agent_run_prompt_{i}_{args.model}"
        ) as span:
            span.set_attribute("task.prompt", prompt_task)
            span.set_attribute("agent.model", args.model)

            try:
                agent = RecursiveAgent(
                    task=prompt_task,
                    u_inst=args.u_inst,
                    tracer=tracer,  # Pass tracer
                    tracer_span=span,  # Pass current span
                    agent_options=RecursiveAgentOptions(
                        # max_layers=args.max_layers, # Removed, task_limit is used
                        search_tool=brave_web_search,  # Updated
                        llm=llm,
                        tool_llm=tool_llm,
                        tools=[],  # No additional non-search tools by default
                        allow_search=True,
                        allow_tools=False,  # Matched cli.py
                        tools_dict={
                            "web_search": brave_web_search,  # For compatibility if agent calls "web_search"
                            "brave_web_search": brave_web_search,
                            # "exec_python": exec_python, "exec": exec_python # Uncomment if needed
                        },
                        task_limits=TaskLimit.from_array(eval(args.task_limit)),
                        merger={"ai": LLMMerger, "append": AppendMerger}[args.merger],
                        # Callbacks like pre_task_executed are not added to keep eval script simpler
                    ),
                )
                response = agent.run()
                responses.append(
                    {
                        "prompt": prompt_task,
                        "response": base64.b64encode(response.encode()).decode(),
                    }
                )
                span.set_attribute("task.response.length", len(response))
                span.set_status(trace.StatusCode.OK)
            except Exception as e:
                console.print(
                    f'[bold red]Error processing prompt "{prompt_task}": {e}[/bold red]'
                )
                responses.append(
                    {"prompt": prompt_task, "error": str(e), "response": ""}
                )
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, description=str(e))
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / args.output
    with save_path.open("w") as output_file:  # Renamed 'output' to 'output_file'
        console.print(f"Saving responses to {save_path}...")
        ic(str(save_path))  # ic expects basic types for printing, Path might be complex
        output_file.write(
            json.dumps(responses, indent=2)
        )  # Added indent for readability
    console.print(f"Evaluation finished. Results saved to {save_path}")


if __name__ == "__main__":
    main()