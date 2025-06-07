import numpy as np
from typing import Callable

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    initial_temp: float = 1.0,
    final_temp: float = 1e-3,
    alpha: float = 0.9,
    n_iter_per_temp: int = 100,
    max_iters_without_improvement: int = 10,
    step_size: float = 0.1,
    seed: int = None
) -> np.ndarray:
    """
    Simulated Annealing optimizer.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Starting point.
    initial_temp : float
        Starting temperature.
    final_temp : float
        Temperature at which to stop.
    alpha : float
        Multiplicative cooling factor (0 < alpha < 1).
    n_iter_per_temp : int
        Number of candidate moves evaluated at each temperature.
    step_size : float
        Std-dev of Gaussian perturbation for candidate moves.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Best-found position.
    """
    rng = np.random.default_rng(seed)
    x_best = x_curr = np.array(initial_guess, dtype=float)
    f_best = f_curr = fun(x_curr)
    T = initial_temp
    iters_without_improvement = 0

    while T > final_temp:
        for _ in range(n_iter_per_temp):
            # Propose a new candidate by Gaussian perturbation
            x_new = x_curr + rng.normal(scale=step_size, size=x_curr.shape)
            f_new = fun(x_new)

            # Accept if better, or with Boltzmann probability otherwise
            Δ = f_new - f_curr
            if Δ <= 0 or rng.random() < np.exp(-Δ / T):
                x_curr, f_curr = x_new, f_new
                # Track global best
                if f_new < f_best:
                    x_best, f_best = x_new, f_new
                    iters_without_improvement = 0
                else:
                    iters_without_improvement += 1
                    if iters_without_improvement >= max_iters_without_improvement:
                        break

        # Cool down
        T *= alpha

    return x_best 