import numpy as np
from typing import Callable, Tuple, Optional, Annotated
from problemsolver.utils import Interval

def _finite_diff_grad(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: Optional[float] = None
) -> np.ndarray:
    """Central‐difference gradient."""
    n = x.size
    g = np.zeros(n, float)
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)*(1+np.linalg.norm(x))
    f0 = fun(x)
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
    """
    Pure-NumPy L-BFGS local minimizer.
    """
    x = x0.astype(float)
    n = x.size
    s_list, y_list, rho_list = [], [], []
    f = fun(x); g = _finite_diff_grad(fun, x)
    for _ in range(maxiter):
        if np.linalg.norm(g, np.inf) < tol:
            break
        # Two-loop recursion
        q = -g.copy(); alphas = []
        for s,y,rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            alpha = rho * s.dot(q); alphas.append(alpha)
            q -= alpha*y
        if y_list:
            gamma = s_list[-1].dot(y_list[-1]) / y_list[-1].dot(y_list[-1])
        else:
            gamma = 1.0
        r = gamma*q
        for s,y,rho,alpha in zip(s_list, y_list, rho_list, reversed(alphas)):
            beta = rho * y.dot(r)
            r += s*(alpha - beta)
        p = r
        # Simple backtracking line search
        alpha, f_new = 1.0, f
        gTp = g.dot(p)
        while True:
            x_new = x + alpha*p
            f_new = fun(x_new)
            if f_new <= f + 1e-4*alpha*gTp or alpha<1e-16:
                break
            alpha *= 0.5
        g_new = _finite_diff_grad(fun, x_new)
        s = x_new - x; y = g_new - g; sy = s.dot(y)
        if sy>1e-12:
            if len(s_list)==m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s); y_list.append(y); rho_list.append(1.0/sy)
        x, f, g = x_new, f_new, g_new
    return x


def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    n_hops: Annotated[int, Interval(low=20, high=500, step=20, log=False)] = 100,
    step_size: Annotated[float, Interval(low=0.1, high=2.0, step=0.1, log=False)] = 0.5,
    T: Annotated[float, Interval(low=0.1, high=10.0, step=0.1, log=True)] = 1.0,
    m: int = 10,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Pure-Python Basin-Hopping optimizer.
    """
    # Initial local minimum
    x_best = minimize_lbfgs(fun, initial_guess, m=m, tol=tol)
    f_best = fun(x_best)
    x_curr, f_curr = x_best.copy(), f_best

    for _ in range(n_hops):
        # 1) Random perturbation
        x_trial = x_curr + np.random.randn(*x_curr.shape)*step_size  # hop  [oai_citation:6‡en.wikipedia.org](https://en.wikipedia.org/wiki/Basin-hopping?utm_source=chatgpt.com)
        # 2) Local minimization
        x_loc = minimize_lbfgs(fun, x_trial, m=m, tol=tol)
        f_loc = fun(x_loc)
        # 3) Metropolis acceptance
        Δ = f_loc - f_curr
        if (Δ <= 0) or (np.random.rand() < np.exp(-Δ / T)):  # acceptance  [oai_citation:7‡docs.scipy.org](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html?utm_source=chatgpt.com) [oai_citation:8‡stats.stackexchange.com](https://stats.stackexchange.com/questions/436052/simulated-annealing-vs-basin-hopping-algorithm?utm_source=chatgpt.com)
            x_curr, f_curr = x_loc, f_loc
            if f_loc < f_best:
                x_best, f_best = x_loc.copy(), f_loc
    return x_best