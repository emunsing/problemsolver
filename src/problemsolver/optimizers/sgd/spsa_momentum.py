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
    beta1: float = 0.9,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    SPSA with Momentum

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
    beta1 : float
        Momentum
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

    # momentum/Adam buffers
    m = np.zeros_like(x)   # first moment (or velocity for momentum)

    for _ in range(n_iterations):
        # 1) SPSA gradient estimate
        grad_est = np.zeros_like(x)
        for _ in range(n_directions):
            u = rng.choice([-1.0, 1.0], size=x.size)
            f_plus  = fun(x + epsilon * u)
            f_minus = fun(x - epsilon * u)
            grad_est += ((f_plus - f_minus) / (2 * epsilon)) * u
        grad_est /= n_directions

        # classical momentum: v = beta1 * v + (1 - beta1) * grad
        m = beta1 * m + (1 - beta1) * grad_est
        x_new = x - learning_rate * m

        f_new = fun(x_new)
        delta_x = np.linalg.norm(x_new - x)
        delta_f = abs(f_new - f_old) / max(1.0, abs(f_old))
        if (delta_x < atol) and (delta_f < rtol):
            x = x_new
            break

        # 4) Roll forward
        x, f_old = x_new, f_new

    return x 