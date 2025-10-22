import uuid
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

class Interrupt(ABC):
    """Abstract base class for all events that can interrupt a task's normal flow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    priority: int  # Higher number means higher priority

    @abstractmethod
    def get_interrupt_prompt(self) -> str:
        """Generates the specific text the agent will see for this interrupt."""
        pass

    def __lt__(self, other: 'Interrupt') -> bool:
        """
Comparison for the priority queue (heapq).
heapq is a min-heap, so we invert the logic to make it a max-priority queue.
        """
        return self.priority > other.priority

class HumanDirectiveInterrupt(BaseModel, Interrupt, ABC):
    """An interrupt initiated by a human operator."""
    priority: int = 100  # Highest priority
    command: str
    payload: Any

    def get_interrupt_prompt(self) -> str:
        return (
            f"--- CRITICAL HUMAN DIRECTIVE RECEIVED ---\n"
            f"COMMAND: {self.command}\n"
            f"PAYLOAD: {self.payload}\n"
            f"You MUST address this directive immediately. Acknowledge the directive and adjust your plan or action accordingly."
        )

class AgentMessageInterrupt(BaseModel, Interrupt, ABC):
    """An interrupt representing a message from another agent task."""
    priority: int = 50  # Standard priority
    source_task_id: str
    message: str

    def get_interrupt_prompt(self) -> str:
        return (
            f"--- INCOMING MESSAGE from Task {self.source_task_id} ---\n"
            f"MESSAGE: {self.message}\n"
            f"Process this message. Does it change your current goal? Does it provide information you need? "
            f"Decide on your next action, which may include responding or continuing your work."
        )