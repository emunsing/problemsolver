import torch
from typing import Annotated, Optional
from problemsolver.utils import Interval


def ns_orthogonalize(
    G: torch.Tensor,
    a: float,
    b: float,
    c: float,
    eps: float,
    ns_steps: int,
) -> torch.Tensor:
    """Quintic Newton–Schulz iteration for Muon (minimal version)."""
    if G.ndim != 2:
        raise ValueError("Muon only supports 2D parameter tensors")

    # Normalize spectral norm
    G = G / G.norm().clamp(min=eps)

    for _ in range(ns_steps):
        gram = G @ G.T                      # [m,m]
        gram_update = b * gram + c * (gram @ gram)
        G = a * G + gram_update @ G

    return G


def adjust_lr(lr: float, fn: Optional[str], shape: torch.Size) -> float:
    A, B = shape[:2]

    if fn is None or fn == "original":
        return lr * (max(1, A / B) ** 0.5)
    elif fn == "match_rms_adamw":
        return lr * (0.2 * max(A, B) ** 0.5)
    else:
        return lr


class TestOptimizer(torch.optim.Optimizer):
    """Minimal Muon optimizer (for 2D weight matrices only)."""

    def __init__(
        self,
        params,
        lr: Annotated[float, Interval(low=1e-5, high=1e-2, log=True)] = 1e-3,
        weight_decay: Annotated[float, Interval(low=0.0, high=0.1, log=True)] = 0.1,
        momentum: Annotated[float, Interval(low=0.7, high=0.999, log=False)] = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: Optional[str] = "original",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    def _init_group(self, group, params_with_grad, grads, bufs):
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim != 2:
                raise ValueError(f"Muon only supports 2D params, got {tuple(p.shape)}")
            if torch.is_complex(p) or p.grad.is_sparse:
                raise ValueError("Muon does not support complex or sparse grads")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            bufs.append(state["momentum_buffer"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            a, b, c = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_fn = group["adjust_lr_fn"]

            params_with_grad, grads, bufs = [], [], []
            self._init_group(group, params_with_grad, grads, bufs)

            for p, g, buf in zip(params_with_grad, grads, bufs):
                # Momentum
                buf.mul_(momentum).add_(g, alpha=1 - momentum)
                update = g + momentum * buf if nesterov else buf

                # Orthogonalize with Newton–Schulz
                update = ns_orthogonalize(update, a, b, c, eps, ns_steps)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Adjust learning rate by matrix shape
                lr_adj = adjust_lr(lr, adjust_fn, p.shape)

                # Update
                p.add_(update, alpha=-lr_adj)

        return loss