import numpy as np
from typing import Callable, Annotated
from problemsolver.utils import Interval

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    pop_size: Annotated[int, Interval(low=20, high=200, step=10, log=False)] = 50,
    F: Annotated[float, Interval(low=0.1, high=2.0, step=0.1, log=False)] = 0.8,
    CR: Annotated[float, Interval(low=0.1, high=1.0, step=0.05, log=False)] = 0.9,
    n_generations: int = 200,
    bounds: np.ndarray = None,
    max_iters_without_improvement: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Differential Evolution optimizer.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Center point for initializing population.
    pop_size : int
        Population size.
    F : float
        Differential weight [0,2].
    CR : float
        Crossover probability [0,1].
    n_generations : int
        Number of generations.
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
    fitness = np.array([fun(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_val = fitness[best_idx]
    iters_without_improvement = 0

    for _ in range(n_generations):
        for i in range(pop_size):
            # Select three random individuals, different from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)

            # Create trial vector through mutation and crossover
            mutant = pop[a] + F * (pop[b] - pop[c])
            trial = np.zeros_like(mutant)
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]

            # Apply bounds if provided
            if bounds is not None:
                trial = np.minimum(np.maximum(trial, lb), ub)

            # Selection
            f_trial = fun(trial)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

                # Update best solution
                if f_trial < best_val:
                    best = trial.copy()
                    best_val = f_trial
                    iters_without_improvement = 0
                else:
                    iters_without_improvement += 1

        if iters_without_improvement >= max_iters_without_improvement:
            break

    return best 