from __future__ import annotations

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
