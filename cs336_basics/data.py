from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of language-modeling inputs and next-token targets."""
    start_indices = torch.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,),
    )
    x = torch.stack(
        [
            torch.tensor(dataset[start.item() : start.item() + context_length], dtype=torch.long)
            for start in start_indices
        ]
    )
    y = torch.stack(
        [
            torch.tensor(dataset[start.item() + 1 : start.item() + 1 + context_length], dtype=torch.long)
            for start in start_indices
        ]
    )
    x = x.to(device)
    y = y.to(device)
    return (x, y)
