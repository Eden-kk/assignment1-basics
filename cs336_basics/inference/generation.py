from __future__ import annotations

from collections.abc import Sequence

import torch

from ..model import TransformerLM, softmax


def sample_next_token(
    model: TransformerLM,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample the next token conditioned on the provided prefix token IDs."""
    logits = model(input_ids)[..., -1, :]
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        # top p sampling
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative_probs <= top_p
        keep_mask[..., 0] = True # return at least 1 sample

        shifted_keep_mask = keep_mask.clone()
        shifted_keep_mask[..., 1:] = keep_mask[..., :-1]
        shifted_keep_mask[..., 0] = True

        filtered_sorted_probs = sorted_probs * shifted_keep_mask
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, filtered_sorted_probs)

        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token


def decode(
    model: TransformerLM,
    prompt_token_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device | str | None = None,
) -> list[int]:
    """Generate a continuation from the provided prompt token IDs."""
    if device is None:
        device = next(model.parameters()).device
    if isinstance(prompt_token_ids, torch.Tensor):
        tokens = prompt_token_ids.tolist()
    else:
        tokens = list(prompt_token_ids)

    for _ in range(max_new_tokens):
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        next_token = sample_next_token(
            model=model, 
            input_ids=input_ids, 
            temperature=temperature,
            top_p=top_p,
        )
        next_token_id = int(next_token.item())
        tokens.append(next_token_id)
        if next_token_id == eos_token_id:
            break
    return tokens
