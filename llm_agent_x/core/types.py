# llm_agent_x/core/types.py

import hashlib
from datetime import datetime, timezone
from typing import Set, Dict, Any, Optional, List, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from llm_agent_x.core.interrupts import Interrupt
if TYPE_CHECKING:
    from llm_agent_x.state_manager import AbstractStateManager


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

class verification(BaseModel):
    reason: str
    message_for_user: str
    score: float = Field(description="A numerical score from 1 (worst) to 10 (best).")

    def get_successful(self):
        return self.score > 5


class TokenBucket(BaseModel):
    """Manages communication tokens for a task."""
    tokens: float = 100.0
    max_tokens: int = 100
    refill_rate: float = 2.0  # Tokens per second
    last_refilled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def refill(self):
        """Refills the bucket based on elapsed time."""
        now = datetime.now(timezone.utc)
        time_delta = (now - self.last_refilled_at).total_seconds()
        if time_delta > 0:
            new_tokens = self.tokens + (time_delta * self.refill_rate)
            self.tokens = min(new_tokens, self.max_tokens)
            self.last_refilled_at = now

    def spend(self, amount: int) -> bool:
        """Spends tokens if available. Returns True on success, False on failure."""
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class Task(BaseModel):
    id: str
    desc: str

    metadata: Optional[Dict[str,Any]] = {}

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

    verification_history: List[verification] = Field(default_factory=list)
    grace_attempts: int = 0

    # We remove Span and Any to avoid complex imports in this core types file
    # otel_context: Optional[Any] = None
    # span: Optional[Span] = None
    otel_context: Optional[Dict] = Field(None, exclude=True) # Stored as dict if needed, excluded from serialization
    span: Optional[Any] = Field(None, exclude=True) # Stored as Any, excluded from serialization

    interrupt_queue: List[Interrupt] = Field(default_factory=list, exclude=True)
    tags: Set[str] = Field(default_factory=set)
    comm_token_bucket: TokenBucket = Field(default_factory=TokenBucket)
    last_cycle_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    human_injected_deps: List[HumanInjectedDependency] = Field(default_factory=list, description="Dependencies manually injected by an operator during a pause.")

    current_question: Optional[UserQuestion] = Field(None,
                                                     description="The question an agent is currently asking the human operator.")

    human_directive: Optional[str] = Field(None,
                                           description="The human's response to the agent's question.")

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


class RetryDecision(BaseModel):
    """The decision on whether to attempt another fix for a failing task."""

    should_retry: bool = Field(
        description="Set to true if the trend of scores suggests success is likely."
    )
    reason: str = Field(
        description="A brief explanation for the decision, citing the score trend."
    )
    next_step_suggestion: str = Field(
        description="A specific, actionable suggestion for the next attempt to improve the result."
    )


class InformationNeedDecision(BaseModel):
    """The decision on whether a task's failure is due to a critical lack of information."""

    is_needed: bool = Field(
        description="Set to true ONLY if the task is fundamentally impossible to complete without specific external information from a human."
    )
    reason: str = Field(
        description="A brief explanation for the decision, justifying why the information gap is or is not the root cause."
    )


class TaskToPrune(BaseModel):
    """A task that is a candidate for pruning."""

    task_id: str = Field(description="The unique ID of the task to be pruned.")
    reason: str = Field(
        description="A brief justification for why this task is the least critical."
    )


class PruningDecision(BaseModel):
    """The final decision on which tasks to prune from the graph."""

    tasks_to_prune: List[TaskToPrune] = Field(
        description="A list of tasks that have been selected for removal."
    )


class Dependency(BaseModel):
    reason: str
    local_id: str


class DependencySelection(BaseModel):
    """The decision on which dependencies are most critical for a task."""

    reasoning: str = Field(description="A brief explanation of the selection strategy.")
    approved_dependency_ids: List[str] = Field(
        description="A list of the most critical dependency task IDs to use."
    )


class NewSubtask(BaseModel):
    local_id: str
    desc: str
    deps: List[Dependency] = Field(default_factory=list)
    can_request_new_subtasks: bool = Field(
        False,
        description="Set to true ONLY for complex, integrative, or uncertain tasks that might need further dynamic decomposition later.",
    )


class ExecutionPlan(BaseModel):
    needs_subtasks: bool
    subtasks: List[NewSubtask] = Field(default_factory=list)


class ProposedSubtask(BaseModel):
    local_id: str
    desc: str
    importance: int = Field(
        description="An integer score from 1 (least critical) to 100 (most critical) representing this task's necessity.",
        ge=1,
        le=100,
    )
    deps: List[str] = Field(
        default_factory=list,
        description="A list of local_ids of other PROPOSED tasks that this one depends on.",
    )


class AdaptiveDecomposerResponse(BaseModel):
    """
    The structured response from the adaptive_decomposer agent.
    It can propose new tasks to be created and/or request dependencies on existing tasks.
    """

    tasks: List[ProposedSubtask] = Field(
        default_factory=list,
        description="A list of new, granular sub-tasks to be created to achieve the parent objective.",
    )
    dep_requests: List[str] = Field(
        default_factory=list,
        description="A list of IDs of EXISTING tasks whose results are needed as new dependencies for the current task.",
    )


class TaskForMerging(BaseModel):
    """Represents a single task in a plan being evaluated for merging."""

    local_id: str
    desc: str
    deps: List[str] = Field(description="List of local_ids this task depends on.")


class MergedTask(BaseModel):
    """Represents a new, consolidated task that replaces one or more original tasks."""

    new_local_id: str = Field(
        description="A new, descriptive local ID for the merged task (e.g., 'plan_and_book_venue')."
    )
    new_desc: str = Field(
        description="A new, comprehensive description for the merged task."
    )
    subsumed_task_ids: List[str] = Field(
        description="A list of the original local_ids that this new task replaces."
    )


class MergingDecision(BaseModel):
    """The plan for merging overly granular or redundant tasks."""

    merged_tasks: List[MergedTask] = Field(
        description="A list of new tasks that consolidate others."
    )
    kept_task_ids: List[str] = Field(
        description="A list of local_ids for tasks that were NOT merged and should be kept as-is."
    )


class ProposalResolutionPlan(BaseModel):
    """The final, pruned list of approved sub-tasks."""

    approved_tasks: List[ProposedSubtask]


class ContextualAnswer(BaseModel):
    """The result of searching internal task history for an answer."""

    is_found: bool = Field(
        description="True if a definitive answer was found in the provided context."
    )
    answer: Optional[str] = Field(
        None, description="The answer found in the context, if any."
    )
    source_task_id: Optional[str] = Field(
        None, description="The ID of the task that contained the answer."
    )
    reasoning: str = Field(
        description="A brief explanation of why the context is or is not sufficient to answer the question."
    )


class RedundancyDecision(BaseModel):
    """The decision on which proposed tasks are not redundant and should proceed."""

    non_redundant_tasks: List[ProposedSubtask] = Field(
        description="A list of the tasks from the proposal that are unique and not covered by existing work."
    )


class TaskDescription(BaseModel):
    """A single, discrete step in a plan."""

    local_id: str = Field(
        description="A temporary, unique identifier for this task within the current plan."
    )
    desc: str = Field(description="A clear and concise description of the task.")
    can_request_new_subtasks: bool = Field(
        False,
        description="Set to true ONLY for complex, integrative, or uncertain tasks that might need further dynamic decomposition later.",
    )
    deps: List[str] = Field(
        default_factory=list,
        description="A list of GLOBAL IDs of existing tasks or documents this new task depends on.",
    )

    tags: List[str] = Field(
        default_factory=list,
        description="A list of descriptive tags for grouping and targeting this task (e.g., 'planning', 'finance', 'outreach')."
    )


class TaskChain(BaseModel):
    """Represents a sequence of tasks that MUST be executed in a specific order."""

    chain: List[TaskDescription] = Field(
        description="An ordered list of tasks forming a sequential chain."
    )


class ChainedExecutionPlan(BaseModel):
    """
    The full execution plan, composed of one or more parallel chains of tasks.
    Each chain represents a sequence of dependent tasks. Different chains can be executed in parallel.
    """

    task_chains: List[TaskChain] = Field(
        description="A list of task chains. All chains can run in parallel."
    )




class TaskContext(BaseModel):
    """
    A context object that provides a convenient, state-aware view for a single task.
    The 'task' attribute is automatically loaded upon initialization.
    """
    # --- CHANGE 1: Use ConfigDict to allow arbitrary types ---
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    state_manager: "AbstractStateManager"
    task: Optional["Task"] = None


    def model_post_init(self, context: Any) -> None:
        """
        After the model is initialized, use the state_manager to load the
        full task object into the 'task' field.
        """
        # By the time this runs, self.state_manager is a real object, not a string.
        if self.state_manager:
            self.task = self.state_manager.get_task(self.task_id)
