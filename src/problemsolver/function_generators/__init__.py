from typing import Callable
import numpy as np

class ProblemFunction:
    """A callable class that represents a transformed optimization function."""

    def __init__(self, func_z: Callable[[np.ndarray], float], optimum_z: np.ndarray):
        self.func_z: Callable
        self.optimum_z: np.ndarray
        self.optimum_x : np.ndarray
        self.optimizer: Callable | None = None

    def __call__(self, x: np.ndarray) -> float:
        pass