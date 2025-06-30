from typing import Literal

from icecream import ic

from llm_agent_x.backend.wrappers.is_dev import is_dev

ic_dev = is_dev(ic)
TaskType = Literal["research", "search", "basic", "text/reasoning"]
