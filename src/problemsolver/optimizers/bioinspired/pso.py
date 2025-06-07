import numpy as np
from typing import Callable

def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    swarm_size: int = 30,
    inertia: float = 0.7,
    cognitive: float = 1.4,
    social: float = 1.4,
    n_iterations: int = 500,
    max_iters_without_improvement: int = 10,
    bounds: np.ndarray = None,
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
    cognitive : float
        Cognitive coefficient (c1).
    social : float
        Social coefficient (c2).
    n_iterations : int
        Maximum number of iterations.
    tol : float
        Tolerance on improvement of global best.
    bounds : np.ndarray, optional
        Array of shape (2, n_dim): [lower_bounds, upper_bounds].
        If provided, particles are clamped to these bounds.
    seed : int, optional
        RNG seed.

    Returns
    -------
    np.ndarray
        Best-found position.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(initial_guess, dtype=float)
    n_dim = x0.size

    # Initialize particle positions + velocities
    if bounds is not None:
        lb, ub = bounds
        positions = rng.uniform(lb, ub, size=(swarm_size, n_dim))
    else:
        # init in a small ball around x0
        spread = 0.1 * np.maximum(1.0, np.abs(x0))
        positions = x0 + rng.standard_normal((swarm_size, n_dim)) * spread

    velocities = np.zeros((swarm_size, n_dim), dtype=float)

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
            + cognitive * r1 * (pbest_pos - positions)
            + social    * r2 * (gbest_pos - positions)
        )
        positions += velocities

        # Clamp to bounds if provided
        if bounds is not None:
            positions = np.minimum(np.maximum(positions, lb), ub)

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