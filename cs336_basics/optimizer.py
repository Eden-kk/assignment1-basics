from __future__ import annotations

from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr, 
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = m * beta1 + grad * (1 - beta1)
                v = v * beta2 + grad**2 * (1 - beta2)
                lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p_old = p.data.clone()
                p.data -= lr_t * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p_old

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate at iteration ``it`` under a cosine schedule with warmup."""
    raise NotImplementedError
