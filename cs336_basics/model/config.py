from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NormType(StrEnum):
    RMS = "rms"
    NONE = "none"


class PositionType(StrEnum):
    ROPE = "rope"
    NONE = "none"


class FFNType(StrEnum):
    SWIGLU = "swiglu"
    SILU = "silu"


class BlockStyle(StrEnum):
    PRE_NORM = "pre_norm"
    POST_NORM = "post_norm"


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10_000.0
    norm_type: NormType = NormType.RMS
    position_type: PositionType = PositionType.ROPE
    ffn_type: FFNType = FFNType.SWIGLU
    block_style: BlockStyle = BlockStyle.PRE_NORM
