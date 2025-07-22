import numpy as np
from typing import Callable, Annotated
from problemsolver.utils import Interval

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    pop_size: Annotated[int, Interval(low=20, high=200, step=10, log=False)] = 30,
    alpha: Annotated[float, Interval(low=0.1, high=1.0, step=0.05, log=False)] = 0.5,
    beta0: Annotated[float, Interval(low=0.1, high=1.0, step=0.05, log=False)] = 0.5,
    gamma: Annotated[float, Interval(low=0.1, high=10.0, step=None, log=True)] = 1.0,
    n_iterations: int = 100,
    max_iters_without_improvement: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Firefly Algorithm optimizer.  Mimics fireflies’ bioluminescent communication by moving each firefly toward brighter (better) peers with an attractiveness that decays with distance, plus a random perturbation to explore.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Center point for initializing population.
    pop_size : int
        Number of fireflies.
    alpha : float
        Randomization parameter.
    beta0 : float
        Attractiveness at distance = 0.
    gamma : float
        Light absorption coefficient.
    n_iterations : int
        Number of iterations.
    bounds : np.ndarray, optional
        Array of shape (2, n_dim): [lower_bounds, upper_bounds].
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Best‐found solution.
    """

    rng = np.random.default_rng(seed)
    dim = initial_guess.size

    # Unbounded initialization around initial guess
    spread = 0.1 * np.maximum(1.0, np.abs(initial_guess))
    pop = initial_guess + rng.standard_normal((pop_size, dim)) * spread

    # Initial evaluation
    intensity = np.array([fun(x) for x in pop])
    best_idx = np.argmin(intensity)
    best, best_val = pop[best_idx].copy(), intensity[best_idx]
    iters_without_improvement = 0

    for _ in range(n_iterations):
        # All pairwise differences and distances
        diff    = pop[:, None, :] - pop[None, :, :]               # (n, n, dim)
        d2      = np.sum(diff**2, axis=2)                         # (n, n)
        beta    = beta0 * np.exp(-gamma * d2)                     # (n, n)
        brighter= (intensity[None, :] < intensity[:, None])       # (n, n)

        # Compute movement toward all brighter fireflies
        move = -np.einsum('ij,ijk->ik', beta * brighter, diff)    # (n, dim)
        rand = alpha * (rng.random((pop_size, dim)) - 0.5)

        pop += move + rand                                        # update population

        # Single evaluation per firefly
        intensity = np.array([fun(x) for x in pop])

        # Track best
        idx = np.argmin(intensity)
        if intensity[idx] < best_val:
            best, best_val = pop[idx].copy(), intensity[idx]
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1
            if iters_without_improvement >= max_iters_without_improvement:
                break

    return best