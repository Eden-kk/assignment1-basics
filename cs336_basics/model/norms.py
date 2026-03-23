from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig, NormType


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        mean_square = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        y = (x / rms) * self.weight
        return y.to(dtype=in_type)


class NoNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_norm(
    config: ModelConfig,
    d_model: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.norm_type == NormType.RMS:
        return RMSNorm(d_model, device=device, dtype=dtype)
    if config.norm_type == NormType.NONE:
        return NoNorm()
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")
