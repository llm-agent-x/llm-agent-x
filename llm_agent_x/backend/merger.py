from typing import List, Any
import re
from dataclasses import dataclass
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import tiktoken
from tqdm import tqdm
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

@dataclass
class MergeChunk:
    text: str
    source_doc: int


class MergeOptions(BaseModel):
    llm: Any
    token_count_model_name: str = "gpt-4-0613"
    context_window: int = 50
    chunk_size: int = 800


class LLMMerger:
    def __init__(self, options: MergeOptions):
        self.options = options
        self.llm = options.llm
        self.encoding = tiktoken.encoding_for_model(options.token_count_model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_documents(self, documents: List[str]) -> List[MergeChunk]:
        chunks = []
        for doc_idx, doc in enumerate(documents):
            while doc:
                if self.count_tokens(doc) <= self.options.chunk_size:
                    chunks.append(MergeChunk(text=doc, source_doc=doc_idx))
                    break
                split_idx = doc[: self.options.chunk_size].rfind("\n")
                if split_idx == -1:
                    split_idx = self.options.chunk_size
                chunks.append(MergeChunk(text=doc[:split_idx], source_doc=doc_idx))
                doc = doc[split_idx:]
        return chunks

    def merge_documents(self, documents: List[str]) -> str:
        chunks = self.chunk_documents(documents)
        merged_text = f"{chunks[0].text}\n<merged>"

        for i in tqdm(range(1, len(chunks))):
            context = " ".join(merged_text.split()[-self.options.context_window:])
            prompt = f"""Previous context:
{context}

<merge>
{chunks[i].text}
</merge>

Merge the content within the merge tags into a coherent continuation of the previous context. Ensure:
1. The merged text flows naturally from the context
2. All important information is preserved
3. Redundancy is eliminated
4. Markdown formatting is preserved
5. Section headers and structure are maintained appropriately
6. All technical details and specifics are retained

Begin your response with the previous context followed by your merged text in <merged> tags."""

            response = self.llm.bind(stop="</merged>").invoke([
                SystemMessage(prompt),
                HumanMessage(context + "\n<merge>\n" + chunks[i].text + "\n</merge>"),
                AIMessage(f"{context}\n<merged>"),
            ],)
            

            merged_content = re.search(r"<merged>(.*?)$", response.content, re.DOTALL)
            merged_text += (
                "\n\n"
                + (
                    merged_content.group(1).strip()
                    if merged_content
                    else chunks[i].text
                )
                + "\n</merged>"
            )

        return merged_text.rstrip("</merged>")