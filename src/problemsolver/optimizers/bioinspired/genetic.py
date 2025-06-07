import numpy as np
from typing import Callable

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    pop_size: int = 50,
    n_generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    tournament_size: int = 3,
    max_iters_without_improvement: int = 10,
    elitism: int = 2,
    seed: int = None
) -> np.ndarray:
    """
    Genetic Algorithm optimizer.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Center point for initializing population.
    pop_size : int
        Number of individuals in population.
    n_generations : int
        Number of generations to run.
    crossover_rate : float
        Probability of crossover between pairs.
    mutation_rate : float
        Probability of mutating each gene.
    mutation_scale : float
        Std‐dev of Gaussian mutation perturbations.
    tournament_size : int
        Number of individuals competing in tournament selection.
    elitism : int
        Number of best individuals to carry over each generation.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Best‐found solution.
    """
    rng = np.random.default_rng(seed)
    dim = initial_guess.size

    # Initialize population around the initial guess
    pop = initial_guess + rng.standard_normal((pop_size, dim)) * mutation_scale

    # Evaluate initial fitness
    fitness = np.array([fun(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    iters_without_improvement = 0

    def tournament_select(pop, fitness):
        # randomly pick tournament_size individuals, return the best one
        idxs = rng.choice(pop_size, size=tournament_size, replace=False)
        winner = idxs[np.argmin(fitness[idxs])]
        return pop[winner]

    for gen in range(n_generations):
        new_pop = []

        # Elitism: carry over top `elitism` individuals
        elite_idxs = np.argsort(fitness)[:elitism]
        for idx in elite_idxs:
            new_pop.append(pop[idx].copy())

        # Crossover
        for _ in range(pop_size - elitism):
            if rng.random() < crossover_rate:
                parent1 = tournament_select(pop, fitness)
                parent2 = tournament_select(pop, fitness)
                child = (parent1 + parent2) / 2
                new_pop.append(child)

        # Mutation
        for i in range(pop_size):
            if rng.random() < mutation_rate:
                new_pop[i] += rng.standard_normal(dim) * mutation_scale

        # Evaluate new fitness
        new_fitness = np.array([fun(ind) for ind in new_pop])
        worst_idx = np.argmax(new_fitness)
        worst = new_pop[worst_idx].copy()
        worst_fit = new_fitness[worst_idx]

        # Replace worst individual with best individual from previous generation
        new_pop[worst_idx] = best
        new_fitness[worst_idx] = best_fit

        # Update fitness
        fitness = new_fitness

        # Update best individual
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        # Check for improvement
        if best_fit < best_fit:
            best = best
            best_fit = best_fit
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        # Update population
        pop = new_pop

        # Check termination condition
        if iters_without_improvement >= max_iters_without_improvement:
            break

    return best 