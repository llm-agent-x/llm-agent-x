import asyncio
import json
import time
from typing import Dict, List
# from langchain_openai import ChatOpenAI
from llm_agent_x.constants import openai_base_url, openai_api_key
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
# from langchain_core.output_parsers import JsonOutputParser
from inspect import getdoc, signature
from icecream import ic
from llm_agent_x.tools.training_stub_tools import get_random_subset_from_distribution
from os import getenv
from rich import console
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import UnexpectedModelBehavior


distribution = {
    "Communication": 0.1,
    "FileManagement": 0.1,
    "DataProcessing": 0.1,
    "InternetInteraction": 0.1,
    "Utility": 0.1,
    "DatabaseInteraction": 0.1,
    "SystemControl": 0.1,
    "UserInteraction": 0.1,
    # "Authentication": 0.1,
    # "Orchestration": 0.1,
}

c = console.Console()

load_dotenv(".env", override=True)

class Task(BaseModel):
    task: str
    tools_required: List[str]
    details: Dict = Field(description="Additional details for the task. Can be empty. **Make up** any details in here that may come in handy for the task, like auth tokens, user metadata, etc.")

agent = Agent(model="gpt-4.1-nano", system_prompt="You are a helpful assistant.", output_type=Task, result_retries=3)

async def generate_task():
    # Make a line across the console to divide iterations
    c.rule("")

    start = time.time()
    # use the numbers to select tools
    tools = get_random_subset_from_distribution(distribution, 3)

    # pick between 0 and 3 more tools
    more_tools = get_random_subset_from_distribution(distribution, 3)

    end = time.time()
    elapsed = end - start
    ic(elapsed)
    # generate string to describe the tools chosen, given their names, arguments, and their docstrings using the inspect module
    tools_description = "\n\n\n".join(
        [
            f"{tool.__name__}:\n    {getdoc(tool)}\n   Arguments: {signature(tool)}"
            for tool in tools
        ]
    )
    # ic(tools_description)

    response = await agent.run(
        user_prompt=f"I need you to generate a command for an advanced agent that uses these tools: \n\n"
        + tools_description
        + "\n\n If you can, use all the tools. Make the prompt super detailed. Include any details "
        "needed for an agent (human or otherwise) to actually complete the task. \n\n"
        "For instance, if the tools included a `search` tool, a current time tool, and a location tool, \n"
        "the prompt could be something like: \n\n"
        f"{Task(task='Get my location. Find restaurants open at the current time, near me.', tools_required=['search', 'get_user_local_time', 'get_user_location'], details={'user':{'allergies': 'peanuts', 'diet': None}}).model_dump_json()}\n\n"
        "Or, if the tool was `send_email`, the prompt could be something like: \n\n"
        f"{Task(task='Send an email to the project manager about the status of the project.', tools_required=['send_email'], details={'status': 'in progress', 'estimate': '2 weeks', 'project_manager': {'name': 'John Doe', 'email': '1d3l4@example.com', 'phone': '123-456-7890'}}).model_dump_json()}\n\n"

    )

    ic(response.output)
    return response.output

    



# generate string to describe the tools chosen, given their names, arguments, and their docstrings using the inspect module
tasks = []
for i in range(10):
    try:
        tasks.append(json.loads(asyncio.run(generate_task()).model_dump_json()))
    except UnexpectedModelBehavior as e:
        ic("Skipping due to unexpected model behavior.")

with open("tasks.json", "w") as f:
    json.dump(tasks, f, indent=4)