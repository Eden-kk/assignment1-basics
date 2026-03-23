from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .config import BlockStyle, ModelConfig
from .mlps import build_mlp
from .norms import build_norm


class PreNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        position_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        config: ModelConfig | None = None,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        block_config = config or ModelConfig(
            vocab_size=0,
            context_length=0,
            d_model=d_model,
            num_layers=1,
            num_heads=num_heads,
            d_ff=d_ff,
        )
        encoding = position_encoding if position_encoding is not None else rope
        self.ln1 = build_norm(block_config, d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            position_encoding=encoding,
            device=device,
            dtype=dtype,
        )
        self.ln2 = build_norm(block_config, d_model, device=device, dtype=dtype)
        self.ffn = build_mlp(block_config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class PostNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        position_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        config: ModelConfig | None = None,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        block_config = config or ModelConfig(
            vocab_size=0,
            context_length=0,
            d_model=d_model,
            num_layers=1,
            num_heads=num_heads,
            d_ff=d_ff,
        )
        encoding = position_encoding if position_encoding is not None else rope
        self.ln1 = build_norm(block_config, d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            position_encoding=encoding,
            device=device,
            dtype=dtype,
        )
        self.ln2 = build_norm(block_config, d_model, device=device, dtype=dtype)
        self.ffn = build_mlp(block_config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ln1(x + self.attn(x, token_positions))
        x = self.ln2(x + self.ffn(x))
        return x


TransformerBlock = PreNormTransformerBlock


def build_transformer_block(
    config: ModelConfig,
    position_encoding: nn.Module | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.block_style == BlockStyle.PRE_NORM:
        return PreNormTransformerBlock(
            config.d_model,
            config.num_heads,
            config.d_ff,
            position_encoding=position_encoding,
            device=device,
            dtype=dtype,
            config=config,
        )
    if config.block_style == BlockStyle.POST_NORM:
        return PostNormTransformerBlock(
            config.d_model,
            config.num_heads,
            config.d_ff,
            position_encoding=position_encoding,
            device=device,
            dtype=dtype,
            config=config,
        )
    raise ValueError(f"Unsupported block_style: {config.block_style}")
