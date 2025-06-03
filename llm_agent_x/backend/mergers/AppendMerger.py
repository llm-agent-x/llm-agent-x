import re
from typing import List, Any
from dataclasses import dataclass
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from .LLMMerger import MergeOptions


class AppendMerger:
    def __init__(self, options: MergeOptions):
        self.options = options
        self.llm = options.llm

    def merge_documents(self, documents: List[str]) -> str:
        if not documents:
            return ""

        return "\n\n".join(documents)
