from pydantic import BaseModel
from langchain_core.language_models.llms import LLM

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.output_parsers import JsonOutputParser


class SequentialCodeAgentOptions(BaseModel):
    llm: LLM = None


class SequentialCodeAgent:
    def __init__(self, options: SequentialCodeAgentOptions):
        self.options = options
        self.llm = self.options.llm.bind(stop="```\n")

    def run(self):
        self.llm.invoke("Generate a simple python script.")