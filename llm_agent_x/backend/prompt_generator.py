import time
from langchain_openai import ChatOpenAI
from llm_agent_x.constants import openai_base_url, openai_api_key
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from inspect import getdoc, signature
from icecream import ic
from llm_agent_x.tools.training_stub_tools import get_random_subset_from_distribution
from os import getenv
from rich import console

# # select n numbers that add up to 1
# distribution = {
#     "Communication": 1 / 3,
#     "FileManagement": 1 / 3,
#     "DatabaseInteraction": 1 / 3,

# }

distribution = {
    "Communication": 0.1,
    "FileManagement": 0.1,
    "DataProcessing": 0.1,
    "InternetInteraction": 0.1,
    "Utility": 0.1,
    "DatabaseInteraction": 0.1,
    "SystemControl": 0.1,
    "UserInteraction": 0.1,
    "Authentication": 0.1,
    "Orchestration": 0.1,
}

c = console.Console()

load_dotenv(".env", override=True)

llm = ChatOpenAI(
    base_url=openai_base_url,
    api_key=openai_api_key,
    model="gpt-4.1-nano",  # getenv("DEFAULT_LLM", "gpt-4o-mini"),
    temperature=0.5,
)

# generate string to describe the tools chosen, given their names, arguments, and their docstrings using the inspect module
for i in range(3):
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
    ic(tools_description)

    chat_history = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage(
            "I need you to generate a task that uses these tools: \n\n"
            + tools_description
            + "\n\n If you can, use all the tools. Make it brief (1-3 sentences)."
        ),
    ]

    # create a task based on the tools with llm
    start = time.time()
    task = llm.invoke(chat_history)
    end = time.time()
    elapsed_llm_1 = end - start
    ic(elapsed_llm_1)
    chat_history.append(task)

    chat_history.append(
        HumanMessage(
            "Put it in JSON, with a `task` field, holding a command describing the task, as a string. I want 1-3 sentences. Also, any chance you could do it all in one line?"
        )
    )
    start = time.time()
    task = llm.invoke(chat_history)
    end = time.time()
    elapsed_llm_2 = end - start

    parser = JsonOutputParser()

    task = parser.parse(task.content)

    ic(task)
    ic(elapsed_llm_2)
