from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any


@dataclass
class MergeChunk:
    text: str
    source_doc: int


class MergeOptions(BaseModel):
    llm: Any
    context_window: int = 50
