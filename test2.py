from langchain_openai import ChatOpenAI
from llm_agent_x import SequentialCodeAgent, SequentialCodeAgentOptions
from test import llm
from icecream import ic


agent = SequentialCodeAgent(options=SequentialCodeAgentOptions(llm=llm))

ic(agent.run())