import numpy as np
from typing import Callable
from scipy.optimize import minimize as _scipy_minimize

def minimize_lbfgsb_jac(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 15000,
    m: int = 10
) -> np.ndarray:
    """
    L-BFGS-B optimizer wrapper.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Objective to minimize.
    initial_guess : np.ndarray
        Starting point (shape (n_dim,)).
    tol : float
        Convergence tolerance on the projected gradient.
    maxiter : int
        Maximum number of iterations.
    m : int
        Number of corrections to store in the limited-memory matrix (history size).

    Returns
    -------
    np.ndarray
        Estimated minimizer.
    """
    # Finite-difference Jacobian via central differences
    def _grad(x: np.ndarray) -> np.ndarray:
        n = x.size
        grad = np.zeros(n, dtype=float)
        eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(x))
        for i in range(n):
            dx = np.zeros_like(x)
            dx[i] = eps
            f_plus  = fun(x + dx)
            f_minus = fun(x - dx)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    res = _scipy_minimize(
        fun,
        initial_guess,
        method="L-BFGS-B",
        jac=_grad,
        tol=tol,
        options={
            "maxiter": maxiter,
            "maxcor": m,
            "ftol": tol,
            "gtol": tol
        }
    )

    return res.x