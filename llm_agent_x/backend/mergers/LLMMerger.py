import re
from typing import List, Any
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from ..config_classes.MergerConfig import MergeChunk, MergeOptions


class LLMMerger:
    def __init__(self, options: MergeOptions):
        self.options = options
        self.llm = options.llm

    def summarize_single_chunk(self, chunk, n=5):
        sys_msg = (
            "Summarize this, but make the ending able to flow into more content:\n"
        )
        words = chunk.split()

        # Get the first five words
        start = words[:n]

        # Join the first five words back into a string
        ffw = " ".join(start)
        result = self.llm.invoke(
            [
                SystemMessage(sys_msg),
                HumanMessage(chunk),
                AIMessage(ffw),
            ]
        )
        return ffw + result.content

    def merge_documents(self, documents: List[str]) -> str:
        if not documents:
            return ""

        merged_text = self.summarize_single_chunk(documents[0]) + "\n"
        sys_msg = (
            "Reproduce the previous context, and then, in <merged></merged> tags, "
            "add the next document, making sure that the two documents flow into each other. "
            "Ensure the end of the merged document can transition into a new document unless this is the final document."
        )

        for i in range(1, len(documents)):
            context = " ".join(merged_text.split()[-self.options.context_window :])
            user_prompt = f"{context}\n<merge>\n{documents[i]}\n</merge>"

            response = self.llm.bind(stop="</merged>").invoke(
                [
                    SystemMessage(sys_msg),
                    HumanMessage(user_prompt),
                    AIMessage(f"{context}\n<merged>\n"),
                ]
            )

            merged_content = re.search(r"<merged>(.*?)$", response.content, re.DOTALL)
            merged_text += (
                "\n\n"
                + (merged_content.group(1).strip() if merged_content else documents[i])
                + "\n"
            )
        merged_text = merged_text.rstrip("</merged>")
        return merged_text
