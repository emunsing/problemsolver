import numpy as np
from typing import Callable, Annotated
from problemsolver.utils import Interval

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    swarm_size: Annotated[int, Interval(low=10, high=100)] = 30,
    inertia: Annotated[float, Interval(low=0.4, high=0.9)] = 0.7,
    alpha: Annotated[float, Interval(low=0.2, high=0.8)] = 0.5,
    max_iters_without_improvement: int = 20,
    n_iterations: int = 1000,
    tol: float = 1e-6,
    seed: int = None
) -> np.ndarray:
    """
    Particle Swarm Optimization (PSO).

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        The center point around which to initialize particles.
    swarm_size : int
        Number of particles.
    inertia : float
        Inertia weight (w).
    alpha: float in [0, 1]
        Proportion of guidance given to cognitive weight c1; social weight given as [1 - alpha].
    n_iterations : int
        Maximum number of iterations.
    tol : float
        Tolerance on improvement of global best.
    seed : int, optional
        RNG seed.
    max_iters_without_improvement: int, optional
        Early stop if no change found

    Returns
    -------
    np.ndarray
        Best-found position.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(initial_guess, dtype=float)
    n_dim = x0.size

    # init in a small ball around x0
    spread = 0.1 * np.maximum(1.0, np.abs(x0))
    positions = x0 + rng.standard_normal((swarm_size, n_dim)) * spread
    velocities = rng.standard_normal((swarm_size, n_dim))

    # Personal and global bests
    pbest_pos = positions.copy()
    pbest_val = np.array([fun(p) for p in positions])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    iters_without_improvement = 0

    for _ in range(n_iterations):
        # Update velocities and positions
        r1 = rng.random((swarm_size, n_dim))
        r2 = rng.random((swarm_size, n_dim))
        velocities = (
            inertia * velocities
            + alpha * r1 * (pbest_pos - positions)
            + (1-alpha) * r2 * (gbest_pos - positions)
        )
        positions += velocities

        # Evaluate and update personal bests
        vals = np.array([fun(p) for p in positions])
        better = vals < pbest_val
        pbest_pos[better] = positions[better]
        pbest_val[better] = vals[better]

        # Update global best
        idx = np.argmin(pbest_val)
        if pbest_val[idx] + tol < gbest_val:
            gbest_val = pbest_val[idx]
            gbest_pos = pbest_pos[idx].copy()
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1
            if iters_without_improvement >= max_iters_without_improvement:
                break

    return gbest_pos 