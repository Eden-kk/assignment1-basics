from __future__ import annotations

import torch
from einops import einsum, rearrange
from torch import nn

from .activations import softmax
from .layers import Linear


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = einsum(
        Q,
        K,
        "... queries d_k, ... keys d_k -> ... queries keys",
    )
    d_k = K.shape[-1]
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_probs = softmax(scores, dim=-1)
    return einsum(
        attn_probs,
        V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        position_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.position_encoding = position_encoding if position_encoding is not None else rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = rearrange(
            self.q_proj(x),
            "... seq_len (num_heads d_q) -> ... num_heads seq_len d_q",
            num_heads=self.num_heads,
        )
        k = rearrange(
            self.k_proj(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        v = rearrange(
            self.v_proj(x),
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))

        if self.position_encoding is not None and token_positions is not None:
            q = self.position_encoding(q, token_positions)
            k = self.position_encoding(k, token_positions)

        multihead_attn = scaled_dot_product_attention(q, k, v, mask)
        multihead_attn = rearrange(
            multihead_attn,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads=self.num_heads,
        )
        return self.output_proj(multihead_attn)
