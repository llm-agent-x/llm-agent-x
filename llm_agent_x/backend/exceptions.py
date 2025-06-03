class TaskFailedException(Exception):
    """Raised when a task is failed"""

    def __init__(self, message):
        self.message = message
        super().__init__(message)
