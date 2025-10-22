import hashlib
from datetime import datetime, timezone
from typing import Set, Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field

from llm_agent_x.core.interrupts import Interrupt

# --- All Pydantic Models related to the Task graph structure go here ---

class HumanInjectedDependency(BaseModel):
    """Represents a dependency added by a human operator post-planning."""
    source_task_id: str
    depth: Literal["shallow", "deep"] = "shallow"

class UserQuestion(BaseModel):
    """
A specific output type for agents to ask clarifying questions to the human operator.
The task will pause until the human provides an answer.
    """
    question: str = Field(description="The clarifying question to ask the human operator.")
    priority: int = Field(
        description="A numerical score from 1 (lowest) to 10 (highest) representing the urgency or criticality of getting an answer.",
        ge=1, le=10)
    details: Dict[str, Any] = Field(default_factory=dict,
                                    description="Any additional context or structured data relevant to the question.")

def generate_hash(content: str) -> str:
    """Generates a SHA256 hash for a string."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

class DocumentState(BaseModel):
    """Holds the versioned content of a document."""
    content: str
    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.content_hash:
            self.content_hash = generate_hash(self.content)

class TokenBucket(BaseModel):
    """Manages the communication token economy for a task."""
    tokens: float = 100.0
    max_tokens: int = 100
    refill_rate: float = 10.0  # Tokens per minute
    last_refilled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def refill(self):
        """Adds tokens to the bucket based on elapsed time."""
        now = datetime.now(timezone.utc)
        time_delta_seconds = (now - self.last_refilled_at).total_seconds()
        if time_delta_seconds > 0:
            tokens_to_add: int = int((time_delta_seconds / 60.0) * self.refill_rate)
            self.tokens = min(self.max_tokens, int(self.tokens) + tokens_to_add)
            self.last_refilled_at = now

    def spend(self, amount: int) -> bool:
        """Spends tokens from the bucket. Returns True on success, False on failure."""
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"
    is_critical: bool = Field(False, description="If True, this task cannot be automatically pruned.")
    counts_toward_limit: bool = Field(True, description="If False, this task does not count towards the task limit.")
    task_type: Literal["task", "document"] = "task"
    document_state: Optional[DocumentState] = None
    result: Optional[str] = None
    cost: float = 0.0
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    dep_results: Dict[str, Any] = Field(default_factory=dict)
    shared_notebook: Dict[str, Any] = Field(default_factory=dict)
    needs_planning: bool = False
    already_planned: bool = False
    can_request_new_subtasks: bool = False
    fix_attempts: int = 0
    max_fix_attempts: int = 2
    verification_scores: List[float] = Field(default_factory=list)
    grace_attempts: int = 0
    otel_context: Optional[Dict] = Field(None, exclude=True)
    span: Optional[Any] = Field(None, exclude=True)

    # --- DEPRECATED: Replaced by interrupt_queue ---
    human_directive: Optional[str] = Field(None, description="A direct, corrective instruction from the operator.")
    user_response: Optional[str] = Field(None, description="The human operator's response to an agent's question.")

    human_injected_deps: List[HumanInjectedDependency] = Field(default_factory=list)
    current_question: Optional[UserQuestion] = Field(None)
    last_llm_history: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)
    executor_llm_history: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    agent_role_paused: Optional[str] = Field(None)
    mcp_servers: List[Dict[str, Any]] = Field(default_factory=list)

    # +++ START: ADD NEW FIELDS FOR COMMUNICATION SYSTEM +++
    interrupt_queue: List[Interrupt] = Field(default_factory=list)
    tags: Set[str] = Field(default_factory=set)
    comm_token_bucket: TokenBucket = Field(default_factory=TokenBucket)
    # +++ END: ADD NEW FIELDS +++

    class Config:
        arbitrary_types_allowed = True
