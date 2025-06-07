import numpy as np
from typing import Callable, Optional, Tuple

def _finite_diff_grad(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: Optional[float] = None
) -> np.ndarray:
    """Central‐difference gradient approximation."""
    n = x.size
    grad = np.zeros(n, dtype=float)
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(x))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        f_plus  = fun(x + dx)
        f_minus = fun(x - dx)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def _backtracking_line_search(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    f0: float,
    g0: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5
) -> Tuple[float, float]:
    """
    Backtracking‐Armijo line search:
      find alpha in {alpha0 * rho^k} such that
      f(x + alpha p) <= f0 + c1*alpha*g0^T p
    Returns (alpha, f_new).
    """
    alpha = alpha0
    gTp = g0.dot(p)
    while True:
        x_new = x + alpha * p
        f_new = fun(x_new)
        if f_new <= f0 + c1 * alpha * gTp:
            return alpha, f_new
        alpha *= rho
        if alpha < 1e-16:
            # stepsize too small
            return alpha, f_new

def minimize_lbfgs(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    m: int = 10,
    tol: float = 1e-6,
    maxiter: int = 500
) -> np.ndarray:
    """
    Limited‐memory BFGS (L-BFGS) optimizer.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Starting point.
    m : int
        History size (number of (s,y) pairs to keep).
    tol : float
        Convergence tolerance on gradient norm.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        The approximate minimizer.
    """
    x = initial_guess.astype(float)
    n = x.size

    # Storage for s_k = x_{k+1} - x_k and y_k = g_{k+1} - g_k
    s_list = []  # up to m past s vectors
    y_list = []  # up to m past y vectors
    rho_list = []  # 1 / (y_k^T s_k) for each pair

    f = fun(x)
    g = _finite_diff_grad(fun, x)

    for k in range(1, maxiter+1):
        # check convergence
        if np.linalg.norm(g, ord=np.inf) < tol:
            break

        # --- two-loop recursion to compute H_k * (-g) ---
        q = -g.copy()
        alpha_vals = []
        for si, yi, rhoi in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            alpha_i = rhoi * si.dot(q)
            alpha_vals.append(alpha_i)
            q -= alpha_i * yi

        # scaling of initial Hessian approx
        if y_list:
            last_sy = y_list[-1].dot(s_list[-1])
            gamma_k = last_sy / (y_list[-1].dot(y_list[-1]))
        else:
            gamma_k = 1.0
        # H0 * q
        r = gamma_k * q

        # second loop
        for (si, yi, rhoi), alpha_i in zip(zip(s_list, y_list, rho_list), reversed(alpha_vals)):
            beta_i = rhoi * yi.dot(r)
            r += si * (alpha_i - beta_i)

        p = r  # search direction

        # line search
        alpha, f_new = _backtracking_line_search(fun, x, f, g, p)
        x_new = x + alpha * p
        g_new = _finite_diff_grad(fun, x_new)

        # update history
        s = x_new - x
        y = g_new - g
        sy = s.dot(y)
        if sy > 1e-12:
            if len(s_list) == m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)

        # prepare next iteration
        x, f, g = x_new, f_new, g_new

    return x