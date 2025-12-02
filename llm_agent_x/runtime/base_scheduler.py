from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """Abstract base class for all schedulers."""

    @abstractmethod
    async def run(self):
        """Starts the scheduler's main loop or process."""
        pass

    @abstractmethod
    def shutdown(self):
        """Signals the scheduler to gracefully shut down."""
        pass