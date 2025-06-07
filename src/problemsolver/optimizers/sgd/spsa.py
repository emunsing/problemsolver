from typing import Callable, Optional
import numpy as np

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    epsilon: float = 1e-3,
    n_directions: int = 10,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Minimizer using Simultaneous Perturbation Stochastic Approximation (SPSA) gradient descent
    SPSA optimizer with absolute (x) and relative (f) stopping criteria.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Scalar objective.
    initial_guess : List[float]
        Starting point.
    learning_rate : float
        Step size.
    n_iterations : int
        Maximum iterations.
    epsilon : float
        Perturbation scale.
    n_directions : int
        Number of random directions per iteration.
    atol : float
        Absolute tolerance on step norm.
    rtol : float
        Relative tolerance on objective change.
    seed : Optional[int]
        RNG seed.

    Returns
    -------
    List[float]
        Estimated minimizer.
    """
    rng = np.random.default_rng(seed)
    x = np.array(initial_guess, dtype=float)
    f_old = fun(x)

    for _ in range(n_iterations):
        # Estimate gradient via SPSA
        grad_est = np.zeros_like(x)
        for _ in range(n_directions):
            u = rng.choice([-1.0, 1.0], size=x.size)
            f_plus  = fun(x + epsilon * u)
            f_minus = fun(x - epsilon * u)
            grad_est += ((f_plus - f_minus) / (2 * epsilon)) * u
        grad_est /= n_directions

        # Take step
        x_new = x - learning_rate * grad_est
        f_new = fun(x_new)

        # Compute stopping metrics
        delta_x = np.linalg.norm(x_new - x)
        delta_f = abs(f_new - f_old) / max(1.0, abs(f_old))

        # Check both criteria
        if (delta_x < atol) and (delta_f < rtol):
            break

        # Update for next iter
        x, f_old = x_new, f_new

    return x 