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
    bounds: np.ndarray = None,
    max_iters_without_improvement: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Firefly Algorithm optimizer.

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

    # Initialize population
    if bounds is not None:
        lb, ub = bounds
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
    else:
        # Initialize around initial guess
        spread = 0.1 * np.maximum(1.0, np.abs(initial_guess))
        pop = initial_guess + rng.standard_normal((pop_size, dim)) * spread

    # Evaluate initial population
    intensity = np.array([fun(x) for x in pop])
    best_idx = np.argmin(intensity)
    best = pop[best_idx].copy()
    best_val = intensity[best_idx]
    iters_without_improvement = 0

    for _ in range(n_iterations):
        # For each firefly
        for i in range(pop_size):
            # Compare with all other fireflies
            for j in range(pop_size):
                if intensity[j] < intensity[i]:  # brighter firefly found
                    # Calculate distance
                    r = np.linalg.norm(pop[i] - pop[j])
                    # Calculate attractiveness
                    beta = beta0 * np.exp(-gamma * r**2)
                    # Move firefly i toward j
                    pop[i] = pop[i] + beta * (pop[j] - pop[i]) + alpha * (rng.random(dim) - 0.5)

                    # Apply bounds if provided
                    if bounds is not None:
                        pop[i] = np.minimum(np.maximum(pop[i], lb), ub)

                    # Update intensity
                    intensity[i] = fun(pop[i])

        # Update best solution
        idx = np.argmin(intensity)
        if intensity[idx] < best_val:
            best = pop[idx].copy()
            best_val = intensity[idx]
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1
            if iters_without_improvement >= max_iters_without_improvement:
                break

    return best 