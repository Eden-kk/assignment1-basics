from __future__ import annotations

import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )
        mean, std = 0, (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            self.weight,
            x,
            "out_features in_features, ... in_features -> ... out_features",
        )


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
