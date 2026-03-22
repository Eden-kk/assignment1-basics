from __future__ import annotations

from collections.abc import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """Compute the average cross-entropy loss across examples."""
    shifted_inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    row_indices = torch.arange(inputs.shape[0], device=inputs.device)
    target_logits = shifted_inputs[row_indices, targets]
    log_denom = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1))
    losses = -target_logits + log_denom
    return torch.mean(losses)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip parameter gradients in-place to have l2 norm at most ``max_l2_norm``."""
    parameters = tuple(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_sq_norm = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for grad in grads:
        total_sq_norm += torch.sum(grad * grad)

    total_norm = torch.sqrt(total_sq_norm)
    eps = 1e-6
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scale
