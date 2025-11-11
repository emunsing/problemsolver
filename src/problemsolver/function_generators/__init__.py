from typing import Callable, Any


class ProblemFunction:
    """A callable class that represents a transformed optimization function."""

    def __init__(self):
        self.optimizer: Callable | None = None
        self.fitted: Any | None = None

    def fit_and_report_loss(self, **kwargs):
        # Run the optimizer on the function and report the loss
        pass


