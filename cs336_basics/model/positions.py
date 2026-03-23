from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .config import ModelConfig, PositionType


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires an even d_k"

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freq_exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = theta ** (-freq_exponents)
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)

        x_pair = rearrange(x, "... seq (pair two) -> ... seq pair two", two=2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        out = torch.stack((out0, out1), dim=-1)
        return rearrange(out, "... seq pair two -> ... seq (pair two)", two=2)


class NoPositionEncoding(nn.Module):
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        del token_positions
        return x


def build_position_encoding(
    config: ModelConfig,
    device: torch.device | None = None,
) -> nn.Module | None:
    if config.position_type == PositionType.ROPE:
        return RotaryPositionalEmbedding(
            theta=config.rope_theta,
            d_k=config.d_model // config.num_heads,
            max_seq_len=config.context_length,
            device=device,
        )
    if config.position_type == PositionType.NONE:
        return None
    raise ValueError(f"Unsupported position_type: {config.position_type}")
