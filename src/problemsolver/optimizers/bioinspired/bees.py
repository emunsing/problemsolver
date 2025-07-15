import numpy as np
from typing import Callable, Annotated
from problemsolver.utils import Interval

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    n_sites: Annotated[int, Interval(low=3, high=20, step=1, log=False)] = 5,
    n_elite: Annotated[int, Interval(low=1, high=5, step=1, log=False)] = 2,
    n_elite_bees: Annotated[int, Interval(low=5, high=30, step=5, log=False)] = 10,
    n_other_bees: Annotated[int, Interval(low=2, high=15, step=1, log=False)] = 5,
    n_scouts: Annotated[int, Interval(low=10, high=50, step=5, log=False)] = 20,
    patch_size: Annotated[float, Interval(low=0.01, high=1.0, step=None, log=True)] = 0.1,
    shrink_factor: Annotated[float, Interval(low=0.8, high=0.99, step=0.01, log=False)] = 0.9,
    n_iterations: int = 100,
    max_iters_without_improvement: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Artificial Bee Colony optimizer.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Center point for initializing sites.
    n_sites : int
        Number of sites to explore.
    n_elite : int
        Number of elite sites.
    n_elite_bees : int
        Number of bees recruited for elite sites.
    n_other_bees : int
        Number of bees recruited for non‐elite sites.
    n_scouts : int
        Number of scout bees for random search.
    patch_size : float
        Initial size of search patches.
    shrink_factor : float
        Factor to shrink patches by each iteration.
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

    # Initialize sites around initial guess
    spread = 0.1 * np.maximum(1.0, np.abs(initial_guess))
    sites = initial_guess + rng.standard_normal((n_sites, dim)) * spread
    fitness = np.array([fun(site) for site in sites])

    # Track best solution
    best_idx = np.argmin(fitness)
    best = sites[best_idx].copy()
    best_val = fitness[best_idx]
    iters_without_improvement = 0
    current_patch_size = patch_size

    for _ in range(n_iterations):
        # Sort sites by fitness
        sort_idx = np.argsort(fitness)
        sites = sites[sort_idx]
        fitness = fitness[sort_idx]

        new_sites = []
        new_fitness = []

        # Elite sites
        for i in range(n_elite):
            site = sites[i]
            for _ in range(n_elite_bees):
                # Search around site
                new_site = site + rng.uniform(-current_patch_size, current_patch_size, dim)
                new_sites.append(new_site)
                new_fitness.append(fun(new_site))

        # Other selected sites
        for i in range(n_elite, n_sites):
            site = sites[i]
            for _ in range(n_other_bees):
                # Search around site
                new_site = site + rng.uniform(-current_patch_size, current_patch_size, dim)
                new_sites.append(new_site)
                new_fitness.append(fun(new_site))

        # Scout bees
        for _ in range(n_scouts):
            # Random search
            new_site = initial_guess + rng.standard_normal(dim) * spread
            new_sites.append(new_site)
            new_fitness.append(fun(new_site))

        # Convert to arrays
        new_sites = np.array(new_sites)
        new_fitness = np.array(new_fitness)

        # Select best sites for next iteration
        all_sites = np.vstack([sites, new_sites])
        all_fitness = np.concatenate([fitness, new_fitness])
        sort_idx = np.argsort(all_fitness)[:n_sites]
        sites = all_sites[sort_idx]
        fitness = all_fitness[sort_idx]

        # Update best solution
        if fitness[0] < best_val:
            best = sites[0].copy()
            best_val = fitness[0]
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        # Shrink patch size
        current_patch_size *= shrink_factor

        if iters_without_improvement >= max_iters_without_improvement:
            break

    return best 