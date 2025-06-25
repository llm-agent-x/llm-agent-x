import re
from typing import List, Any
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from ..config_classes.MergerConfig import MergeChunk, MergeOptions
from pydantic_ai import Agent


class LLMMerger:
    def __init__(self, options: MergeOptions):
        self.options = options
        self.llm = options.llm
        self.merge_agent = Agent(
            model=self.llm, system_prompt="Merge these documents:", output_type=str
        )

    def merge_documents(self, documents: List[str]) -> str:
        assert documents
        if len(documents) == 1:
            return documents[0]

        merged_text = self.merge_agent.run_sync(
            "Here are the documents to merge:\n\n" + "\n\n---\n\n".join(documents)
        ).output
        return merged_text
