import argparse
from langchain_openai import ChatOpenAI
import llm_agent_x
from icecream import ic
import json
from typing import List
from langchain_community.utilities import SearxSearchWrapper
from os import getenv

search = SearxSearchWrapper(searx_host=getenv("SEARX_HOST", "http://localhost:8080"))

def web_search(query: str, num_results: int) -> List:
    """
    Perform a web search using the Searx search engine.

    Args:
        query (str): The search query string.
        num_results (int): The number of search results to return.

    Returns:
        str: The search results as a JSON string.
    """
    return json.dumps(search.results(query, num_results=num_results))

def main(task, output_file):
    # Initialize the LLM
    llm = ChatOpenAI(model="qwen2.5-coder-long:latest", base_url=getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), api_key=getenv("OPENAI_API_KEY"), temperature=0, )
    
    # Initialize the agent with the task
    agent = llm_agent_x.Agent(task=task, llm=llm, max_layers=2, max_agents=5, search_tool = web_search)
    
    # Run the agent and capture the response
    agent_response = agent.run()
    ic(agent_response)
    
    # Write the response to the output file
    with open(output_file, "w") as f:
        f.write(f"{agent_response}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Research tool using a custom LLM agent.")
    parser.add_argument("task", type=str, help="The research task to be performed.")
    parser.add_argument("output_file", type=str, help="The file where the output will be saved.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(task=args.task, output_file=args.output_file)
