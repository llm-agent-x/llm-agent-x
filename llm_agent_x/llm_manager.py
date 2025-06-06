from llm_agent_x.backend.dot_tree import DotTree
from llm_agent_x.constants import openai_base_url, openai_api_key


from langchain_openai import ChatOpenAI


from os import getenv

from llm_agent_x.tools.brave_web_search import brave_web_search


llm_manager = "hello"
# Initialize LLM and Search
llm = ChatOpenAI(
    base_url=openai_base_url,
    api_key=openai_api_key,
    model=getenv("DEFAULT_LLM", "gpt-4o-mini"),
    temperature=0.5,
)
llm_tiny = ChatOpenAI(
    base_url=openai_base_url,
    api_key=openai_api_key,
    model="gpt-4.1-nano",  # getenv("DEFAULT_LLM", "gpt-4o-mini"),
    temperature=0.5,
)
model_tree = DotTree()
model_tree.update("llm", llm)
model_tree.update("llm.small", llm)
model_tree.update("llm.tiny", llm_tiny)
model_tree.update("llm.small.tiny", llm_tiny)
