import torch
from typing import Annotated
from problemsolver.utils import Interval

class TestOptimizer(torch.optim.Optimizer):
    """A minimal, fully functional Adam optimizer."""

    def __init__(
        self,
        params,
        lr: Annotated[float, Interval(low=1e-5, high=1e-2, log=True)] = 1e-3,
        beta1: Annotated[float, Interval(low=0.8, high=0.99, step=0.05, log=False)] = 0.9,
        weight_decay: Annotated[float, Interval(low=1e-9, high=1e-2, step=1e-3, log=True)] = 1e-9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        # Construct betas tuple
        betas = (beta1, beta2)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
    ):
        """Collect and lazily initialize per-parameter state."""
        has_complex = False
        for p in group['params']:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                has_complex = True
                raise RuntimeError("MyAdam does not support complex parameters")

            grad = p.grad
            params_with_grad.append(p)
            grads.append(grad)

            # Lazy state initialization: first time we see this parameter
            state = self.state[p]
            if len(state) == 0:
                state['step'] = torch.tensor(0., device=p.device)
                state['exp_avg'] = torch.zeros_like(p, device=p.device)     # first moment
                state['exp_avg_sq'] = torch.zeros_like(p, device=p.device)  # second moment

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            state_steps.append(state['step'])

        return has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            params_with_grad, grads = [], []
            exp_avgs, exp_avg_sqs, state_steps = [], [], []

            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
            )

            for i, p in enumerate(params_with_grad):
                grad = grads[i]
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                # Increment step
                step_t += 1

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rates
                bias_correction1 = 1 - beta1 ** step_t.item()
                bias_correction2 = 1 - beta2 ** step_t.item()
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                # Parameter update
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
