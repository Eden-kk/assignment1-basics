from __future__ import annotations

import torch
from torch import nn

from .blocks import build_transformer_block
from .config import BlockStyle, FFNType, ModelConfig, NormType, PositionType
from .layers import Embedding, Linear
from .norms import build_norm
from .positions import build_position_encoding


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType | str = NormType.RMS,
        position_type: PositionType | str = PositionType.ROPE,
        ffn_type: FFNType | str = FFNType.SWIGLU,
        block_style: BlockStyle | str = BlockStyle.PRE_NORM,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.config = ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            norm_type=NormType(norm_type),
            position_type=PositionType(position_type),
            ffn_type=FFNType(ffn_type),
            block_style=BlockStyle(block_style),
        )

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.position_encoding = build_position_encoding(self.config, device=device)
        self.layers = nn.ModuleList(
            [
                build_transformer_block(
                    self.config,
                    position_encoding=self.position_encoding,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = build_norm(self.config, d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
