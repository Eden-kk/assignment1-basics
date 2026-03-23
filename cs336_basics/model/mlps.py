from __future__ import annotations

import torch
from torch import nn

from .activations import silu
from .config import FFNType, ModelConfig
from .layers import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_activation = self.w1(x)
        activation = silu(pre_activation)
        gating = self.w3(x)
        return self.w2(activation * gating)


class SiluMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


def build_mlp(
    config: ModelConfig,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.ffn_type == FFNType.SWIGLU:
        return SwiGLU(config.d_model, config.d_ff, device=device, dtype=dtype)
    if config.ffn_type == FFNType.SILU:
        return SiluMLP(config.d_model, config.d_ff, device=device, dtype=dtype)
    raise ValueError(f"Unsupported ffn_type: {config.ffn_type}")
