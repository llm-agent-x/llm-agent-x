from langchain_openai import ChatOpenAI
from llm_agent_x import SequentialCodeAgent, SequentialCodeAgentOptions
from test import llm
from icecream import ic

agent = SequentialCodeAgent(options=SequentialCodeAgentOptions(llm=llm), execute=exec)
response = agent.run("Get the date and time. then multiply the day of the month by the month number.")
ic(response)