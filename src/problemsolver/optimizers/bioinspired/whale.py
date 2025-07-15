import numpy as np
from typing import Callable, Annotated
from problemsolver.utils import Interval

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    pop_size: Annotated[int, Interval(low=20, high=200, step=10, log=False)] = 30,
    a_start: Annotated[float, Interval(low=1.0, high=5.0, step=0.5, log=False)] = 2.0,
    a_end: Annotated[float, Interval(low=0.0, high=1.0, step=0.1, log=False)] = 0.0,
    max_iters: int = 100,
    bounds: np.ndarray = None,
    max_iters_without_improvement: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Whale Optimization Algorithm.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Center point for initializing population.
    pop_size : int
        Population size.
    a_start : float
        Initial search coefficient.
    a_end : float
        Final search coefficient.
    max_iters : int
        Maximum number of iterations.
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
    fitness = np.array([fun(x) for x in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_val = fitness[best_idx]
    iters_without_improvement = 0

    for t in range(max_iters):
        # Update a linearly from a_start to a_end
        a = a_start - t * (a_start - a_end) / max_iters

        for i in range(pop_size):
            # Random parameters
            r = rng.random()
            A = 2 * a * rng.random() - a  # Eq. (2.3)
            C = 2 * rng.random()          # Eq. (2.4)
            l = rng.uniform(-1, 1)        # parameter for spiral
            p = rng.random()              # probability for hunting choice

            if p < 0.5:
                # Encircling prey
                if abs(A) < 1:
                    D = abs(C * best - pop[i])
                    pop[i] = best - A * D
                # Search for prey
                else:
                    rand_idx = rng.integers(pop_size)
                    rand_whale = pop[rand_idx]
                    D = abs(C * rand_whale - pop[i])
                    pop[i] = rand_whale - A * D
            # Spiral bubble-net attack
            else:
                D = abs(best - pop[i])
                pop[i] = D * np.exp(2 * l) * np.cos(2 * np.pi * l) + best

            # Apply bounds if provided
            if bounds is not None:
                pop[i] = np.minimum(np.maximum(pop[i], lb), ub)

            # Evaluate new position
            val = fun(pop[i])
            if val < best_val:
                best = pop[i].copy()
                best_val = val
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

        if iters_without_improvement >= max_iters_without_improvement:
            break

    return best 