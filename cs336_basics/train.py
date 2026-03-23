from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time

import numpy as np
import torch

from .train_config import parse_config, TrainConfig
from .data import get_batch
from .transformerlm import TransformerLM
from .tokenizer import Tokenizer
from .generation import decode
from .optimizer import AdamW, get_lr_cosine_schedule, set_learning_rate
from .nn_utils import cross_entropy, gradient_clipping
from .checkpointing import save_checkpoint
from .logging_utils import (
    build_checkpoint_log,
    build_sample_log,
    build_train_metrics,
    build_training_logger,
    build_validation_metrics,
    apply_wandb_sweep_overrides,
    compute_grad_norm,
    compute_throughput,
)


def evaluate(model: torch.nn.Module, data: np.ndarray, config: TrainConfig) -> float:
    """Estimate validation loss on a single held-out batch."""
    model.eval()
    with torch.no_grad():
        x, y = get_batch(
            dataset=data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
        )
    return float(loss.item())


def load_runtime_tokenizer(config: TrainConfig) -> Tokenizer:
    """Load the tokenizer needed for prompt encoding and sample decoding."""
    return Tokenizer.from_files(
        vocab_path=config.tokenizer_vocab_path,
        merges_path=config.tokenizer_merges_path,
        special_tokens=["<|endoftext|>"],
    )


def generate_sample_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    config: TrainConfig,
) -> str:
    """Generate one qualitative sample for experiment tracking."""
    prompt_token_ids = tokenizer.encode(config.sample_prompt)
    eos_token_id = tokenizer.token_to_id(b"<|endoftext|>")
    model.eval()
    with torch.no_grad():
        generated_token_ids = decode(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=config.sample_max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=config.sample_temperature,
            top_p=config.sample_top_p,
            device=config.device,
        )
    return tokenizer.decode(generated_token_ids)


def main() -> None:
    """Entry point for the training script described in section 5.3."""
    config = parse_config()
    config = apply_wandb_sweep_overrides(config)
    print("parsed training config", flush=True)
    train_data = np.load(config.tokenized_train_data_path, mmap_mode="r")
    print(f"loaded train data from {config.tokenized_train_data_path}", flush=True)
    valid_data = np.load(config.tokenized_valid_data_path, mmap_mode="r")
    print(f"loaded valid data from {config.tokenized_valid_data_path}", flush=True)
    tokenizer = load_runtime_tokenizer(config)
    print("loaded tokenizer", flush=True)
    torch.manual_seed(config.seed)
    print(f"set torch seed to {config.seed}", flush=True)
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
    )
    print(f"built model on device={config.device}", flush=True)
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    print("built optimizer", flush=True)
    Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"ensured checkpoint directory at {Path(config.checkpoint_path).parent}", flush=True)

    logger = build_training_logger(config)
    print("initialized training logger", flush=True)
    start_time = time.time()
    latest_eval_loss: float | None = None

    for iteration in range(config.max_iters):
        step_start_time = time.time()
        model.train()
        x, y = get_batch(
            dataset=train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = compute_grad_norm(model.parameters())
        gradient_clipping(model.parameters(), config.grad_clip)

        lr = get_lr_cosine_schedule(
            it=iteration,
            min_learning_rate=config.min_learning_rate,
            max_learning_rate=config.learning_rate,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters,
        )
        set_learning_rate(optimizer, lr)
        optimizer.step()
        step_end_time = time.time()
        elapsed_sec = step_end_time - start_time
        step_sec = step_end_time - step_start_time
        tokens_per_sec, examples_per_sec = compute_throughput(
            batch_size=config.batch_size,
            context_length=config.context_length,
            step_sec=step_sec,
        )

        if iteration % config.log_every == 0:
            print(f"current iter: {iteration}, train_loss: {loss.item():.6f}")
            logger.log_train(
                build_train_metrics(
                    iteration=iteration,
                    elapsed_sec=elapsed_sec,
                    learning_rate=lr,
                    train_loss=float(loss.item()),
                    grad_norm=grad_norm,
                    tokens_per_sec=tokens_per_sec,
                    examples_per_sec=examples_per_sec,
                )
            )

        if iteration % config.eval_every == 0:
            eval_loss = evaluate(model, valid_data, config)
            latest_eval_loss = eval_loss
            print(f"current iter: {iteration}, eval_loss: {eval_loss:.6f}")
            logger.log_validation(
                build_validation_metrics(
                    iteration=iteration,
                    elapsed_sec=elapsed_sec,
                    valid_loss=float(eval_loss),
                )
            )

        if iteration % config.sample_every == 0:
            sample_text = generate_sample_text(model, tokenizer, config)
            logger.log_sample(
                build_sample_log(
                    iteration=iteration,
                    elapsed_sec=elapsed_sec,
                    prompt=config.sample_prompt,
                    generated_text=sample_text,
                    temperature=config.sample_temperature,
                    top_p=config.sample_top_p,
                    max_new_tokens=config.sample_max_new_tokens,
                )
            )

        if iteration > 0 and iteration % config.save_every == 0:
            save_checkpoint(model, optimizer, iteration, config.checkpoint_path)
            logger.log_checkpoint(
                build_checkpoint_log(
                    iteration=iteration,
                    elapsed_sec=elapsed_sec,
                    checkpoint_path=config.checkpoint_path,
                    valid_loss=latest_eval_loss,
                )
            )
        
    logger.finish()

if __name__ == "__main__":
    main()
