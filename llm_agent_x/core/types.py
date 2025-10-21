# llm_agent_x/core/types.py

import hashlib
from datetime import datetime, timezone
from typing import Set, Dict, Any, Optional, List, Literal

from pydantic import BaseModel, Field


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

class Task(BaseModel):
    id: str
    desc: str
    deps: Set[str] = Field(default_factory=set)
    status: str = "pending"  # can be: pending, planning, proposing, waiting_for_children, running, complete, failed, cancelled, pruned, paused_by_human, waiting_for_user_response
    is_critical: bool = Field(False,
                              description="If True, this task cannot be automatically pruned by the graph manager.")

    counts_toward_limit: bool = Field(True,
                                      description="If False, this task does not count towards the limit of tasks that can exist")
    task_type: Literal["task", "document"] = "task"
    document_state: Optional[DocumentState] = None

    result: Optional[str] = None
    cost: float = 0.0
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    dep_results: Dict[str, Any] = Field(default_factory=dict)
    shared_notebook: Dict[str, Any] = Field(default_factory=dict,
                                            description="A key-value store visible to the agent for persisting important state across interactions.")

    # --- HYBRID PLANNING & ADAPTATION FIELDS ---
    needs_planning: bool = False
    already_planned: bool = False
    can_request_new_subtasks: bool = False

    # --- RETRY & TRACING FIELDS ---
    fix_attempts: int = 0
    max_fix_attempts: int = 2
    verification_scores: List[float] = Field(default_factory=list)
    grace_attempts: int = 0

    # We remove Span and Any to avoid complex imports in this core types file
    # otel_context: Optional[Any] = None
    # span: Optional[Span] = None
    otel_context: Optional[Dict] = Field(None, exclude=True) # Stored as dict if needed, excluded from serialization
    span: Optional[Any] = Field(None, exclude=True) # Stored as Any, excluded from serialization


    human_directive: Optional[str] = Field(None, description="A direct, corrective instruction from the operator.")

    human_injected_deps: List[HumanInjectedDependency] = Field(default_factory=list, description="Dependencies manually injected by an operator during a pause.")
    
    current_question: Optional[UserQuestion] = Field(None,
                                                     description="The question an agent is currently asking the human operator.")
    user_response: Optional[str] = Field(None, description="The human operator's response to an agent's question.")

    # --- NEW CONTEXT FIELDS ---
    last_llm_history: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)
    executor_llm_history: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)

    execution_log: List[Dict[str, Any]] = Field(default_factory=list,
                                                description="A log of real-time execution steps (tool calls, thoughts) for the UI.")

    agent_role_paused: Optional[str] = Field(None,
                                             description="Stores the name of the agent role that paused for human input.")

    # --- MCP FIEDS ---
    mcp_servers: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True