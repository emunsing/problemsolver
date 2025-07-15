import numpy as np
from typing import Callable, List, Optional, Tuple, Annotated
from problemsolver.utils import Interval


# --- (Reuse the finite-difference L-BFGS from before) ---
def _finite_diff_grad(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: Optional[float] = None
) -> np.ndarray:
    n = x.size
    g = np.zeros(n, float)
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)*(1+np.linalg.norm(x))
    for i in range(n):
        dx = np.zeros_like(x); dx[i] = eps
        g[i] = (fun(x+dx) - fun(x-dx)) / (2*eps)
    return g

def minimize_lbfgs(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    m: int = 10,
    tol: float = 1e-6,
    maxiter: int = 200
) -> np.ndarray:
    x = x0.astype(float)
    n = x.size
    s_list, y_list, rho_list = [], [], []
    f = fun(x); g = _finite_diff_grad(fun, x)
    for _ in range(maxiter):
        if np.linalg.norm(g, np.inf) < tol:
            break
        # two-loop recursion
        q = -g.copy(); alphas = []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * s.dot(q)
            alphas.append(a)
            q -= a * y
        if y_list:
            gamma = s_list[-1].dot(y_list[-1]) / y_list[-1].dot(y_list[-1])
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y, rho), a in zip(zip(s_list, y_list, rho_list), reversed(alphas)):
            b = rho * y.dot(r)
            r += s * (a - b)
        p = r
        # backtracking line search
        alpha, f_new = 1.0, f
        gTp = g.dot(p)
        while True:
            x_new = x + alpha * p
            f_new = fun(x_new)
            if f_new <= f + 1e-4 * alpha * gTp or alpha < 1e-16:
                break
            alpha *= 0.5
        g_new = _finite_diff_grad(fun, x_new)
        s = x_new - x; y = g_new - g; sy = s.dot(y)
        if sy > 1e-12:
            if len(s_list) == m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s); y_list.append(y); rho_list.append(1.0 / sy)
        x, f, g = x_new, f_new, g_new
    return x

# --- Dual-Annealing Global Optimizer ---
def dual_annealing(
    fun: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    n_iterations: int = 200,
    temp_max: Annotated[float, Interval(low=100.0, high=10000.0, step=100.0, log=True)] = 5230.0,
    temp_min: float = 1e-3,
    visit: Annotated[float, Interval(low=1.5, high=5.0, step=0.1, log=False)] = 2.62,
    accept: Annotated[float, Interval(low=0.1, high=5.0, step=0.1, log=False)] = 1.0,
    local_search_interval: int = 20,
    local_m: int = 6,
    local_tol: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Dual-Annealing–style optimizer.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    bounds : np.ndarray, shape (2, n_dim)
        Lower and upper bounds for each dimension.
    initial_guess : np.ndarray, optional
        Starting point; if None, begins at the center of bounds.
    n_iterations : int
        Total global annealing steps (hops).
    temp_max, temp_min : float
        Starting and ending “temperatures.”
    visit : float
        Visiting distribution parameter (>1 for Cauchy-like steps).
    accept : float
        Acceptance distribution parameter.
    local_search_interval : int
        Run a local L-BFGS every this many global steps.
    local_m, local_tol : int, float
        History size and convergence tol for the local L-BFGS.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    x_best : np.ndarray
        Best-found solution.
    """
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    n_dim = lb.size

    # Initialize current point
    if initial_guess is None:
        x_curr = lb + 0.5 * (ub - lb)
    else:
        x_curr = np.clip(initial_guess, lb, ub).astype(float)

    f_curr = fun(x_curr)
    x_best, f_best = x_curr.copy(), f_curr

    for k in range(1, n_iterations+1):
        # Temperature schedule (geometric)
        T = temp_max * (temp_min / temp_max)**(k / n_iterations)

        # 1) Generate a trial point via generalized Cauchy step
        u = rng.standard_cauchy(size=n_dim)  # heavy-tailed
        step = (ub - lb) * (u / np.abs(u)**(1/visit))
        x_trial = x_curr + step
        # project back into bounds
        x_trial = np.minimum(np.maximum(x_trial, lb), ub)

        # 2) Accept or reject (Metropolis criterion)
        f_trial = fun(x_trial)
        delta = f_trial - f_curr
        if (delta < 0) or (rng.random() < np.exp(-delta / (accept * T))):
            x_curr, f_curr = x_trial, f_trial
            # update best
            if f_curr < f_best:
                x_best, f_best = x_curr.copy(), f_curr

        # 3) Local search occasionally
        if (k % local_search_interval) == 0:
            x_loc = minimize_lbfgs(fun, x_curr, m=local_m, tol=local_tol)
            f_loc = fun(x_loc)
            if f_loc < f_best:
                x_best, f_best = x_loc.copy(), f_loc
            # restart current to local min
            x_curr, f_curr = x_loc, f_loc

    return x_best