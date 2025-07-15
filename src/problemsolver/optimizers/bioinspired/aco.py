import numpy as np
from typing import Callable, Annotated
from problemsolver.utils import Interval

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    n_ants: Annotated[int, Interval(low=20, high=200, step=10, log=False)] = 50,
    archive_size: int = 10,
    q: Annotated[float, Interval(low=0.1, high=1.0, step=0.05, log=False)] = 0.5,
    xi: Annotated[float, Interval(low=0.5, high=0.95, step=0.05, log=False)] = 0.85,
    n_iterations: int = 100,
    max_iters_without_improvement: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Continuous Ant Colony Optimization (ACO) for function minimization.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Starting point.
    n_ants : int
        Number of ants per iteration.
    archive_size : int
        Size of solution archive.
    q : float
        Locality of search (selection pressure) [0,1].
    xi : float
        Pheromone evaporation rate [0,1].
    n_iterations : int
        Number of iterations.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Best‐found solution.
    """
    rng = np.random.default_rng(seed)
    dim = initial_guess.size

    # Initialize solution archive with random solutions around initial guess
    spread = 0.1 * np.maximum(1.0, np.abs(initial_guess))
    archive = initial_guess + rng.standard_normal((archive_size, dim)) * spread
    archive_fitness = np.array([fun(x) for x in archive])

    # Sort archive by fitness
    sort_idx = np.argsort(archive_fitness)
    archive = archive[sort_idx]
    archive_fitness = archive_fitness[sort_idx]

    # Calculate weights for solution selection
    weights = 1 / (q * archive_size * np.sqrt(2 * np.pi)) * \
             np.exp(-0.5 * ((np.arange(1, archive_size + 1)) / (q * archive_size)) ** 2)
    weights /= np.sum(weights)

    best = archive[0].copy()
    best_val = archive_fitness[0]
    iters_without_improvement = 0

    for _ in range(n_iterations):
        # Generate new solutions
        new_solutions = np.zeros((n_ants, dim))
        for i in range(n_ants):
            # For each dimension
            for j in range(dim):
                # Select solution from archive based on weights
                selected = rng.choice(archive_size, p=weights)
                # Calculate standard deviation
                sigma = xi * np.sum(np.abs(archive[:, j] - archive[selected, j])) / (archive_size - 1)
                # Sample new value from Gaussian
                new_solutions[i, j] = archive[selected, j] + rng.normal(0, sigma)

        # Evaluate new solutions
        new_fitness = np.array([fun(x) for x in new_solutions])

        # Update archive with best solutions
        all_solutions = np.vstack([archive, new_solutions])
        all_fitness = np.concatenate([archive_fitness, new_fitness])
        sort_idx = np.argsort(all_fitness)[:archive_size]
        archive = all_solutions[sort_idx]
        archive_fitness = all_fitness[sort_idx]

        # Update best solution
        if archive_fitness[0] < best_val:
            best = archive[0].copy()
            best_val = archive_fitness[0]
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1
            if iters_without_improvement >= max_iters_without_improvement:
                break

    return best 