from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    run_name: str = "baseline"
    experiment_name: str = "tinystories_baseline"
    log_dir: Path = Path("logs")
    use_wandb: bool = False
    wandb_project: str = "cs336-basics"
    wandb_group: str = "baseline"
    wandb_tags: tuple[str, ...] = ()

    train_data_path: Path = Path("data/TinyStoriesV2-GPT4-train.txt")
    valid_data_path: Path = Path("data/TinyStoriesV2-GPT4-valid.txt")
    tokenized_train_data_path: Path = Path("data/prepare_data_test.npy")
    tokenized_valid_data_path: Path = Path("data/TinyStoriesV2-GPT4-valid.npy")
    tokenizer_vocab_path: Path = Path("assets/tokenizer/gpt2_vocab.json")
    tokenizer_merges_path: Path = Path("assets/tokenizer/gpt2_merges.txt")
    checkpoint_path: Path = Path("checkpoints/latest.pt")

    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 768
    rope_theta: float = 10_000.0

    batch_size: int = 32
    max_iters: int = 10_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 1_000
    cosine_cycle_iters: int = 10_000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    log_every: int = 10
    eval_every: int = 250
    save_every: int = 1_000
    sample_every: int = 500
    sample_prompt: str = "Once upon a time"
    sample_max_new_tokens: int = 128
    sample_temperature: float = 1.0
    sample_top_p: float = 1.0
    device: str = "mps"
    seed: int = 42


DEFAULT_CONFIG = TrainConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build an argument parser for overriding training configuration defaults."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default=DEFAULT_CONFIG.run_name)
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_CONFIG.experiment_name)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_CONFIG.log_dir)
    parser.add_argument("--use-wandb", action="store_true", default=DEFAULT_CONFIG.use_wandb)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_CONFIG.wandb_project)
    parser.add_argument("--wandb-group", type=str, default=DEFAULT_CONFIG.wandb_group)
    parser.add_argument("--wandb-tag", dest="wandb_tags", action="append")

    parser.add_argument("--train-data-path", type=Path, default=DEFAULT_CONFIG.train_data_path)
    parser.add_argument("--valid-data-path", type=Path, default=DEFAULT_CONFIG.valid_data_path)
    parser.add_argument("--tokenized-train-data-path", type=Path, default=DEFAULT_CONFIG.tokenized_train_data_path)
    parser.add_argument("--tokenized-valid-data-path", type=Path, default=DEFAULT_CONFIG.tokenized_valid_data_path)
    parser.add_argument("--tokenizer-vocab-path", type=Path, default=DEFAULT_CONFIG.tokenizer_vocab_path)
    parser.add_argument("--tokenizer-merges-path", type=Path, default=DEFAULT_CONFIG.tokenizer_merges_path)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CONFIG.checkpoint_path)

    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocab_size)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONFIG.context_length)
    parser.add_argument("--d-model", type=int, default=DEFAULT_CONFIG.d_model)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG.num_layers)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_CONFIG.num_heads)
    parser.add_argument("--d-ff", type=int, default=DEFAULT_CONFIG.d_ff)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_CONFIG.rope_theta)

    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_CONFIG.max_iters)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_CONFIG.min_learning_rate)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_CONFIG.warmup_iters)
    parser.add_argument("--cosine-cycle-iters", type=int, default=DEFAULT_CONFIG.cosine_cycle_iters)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--beta1", type=float, default=DEFAULT_CONFIG.beta1)
    parser.add_argument("--beta2", type=float, default=DEFAULT_CONFIG.beta2)
    parser.add_argument("--eps", type=float, default=DEFAULT_CONFIG.eps)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG.grad_clip)

    parser.add_argument("--log-every", type=int, default=DEFAULT_CONFIG.log_every)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_CONFIG.eval_every)
    parser.add_argument("--save-every", type=int, default=DEFAULT_CONFIG.save_every)
    parser.add_argument("--sample-every", type=int, default=DEFAULT_CONFIG.sample_every)
    parser.add_argument("--sample-prompt", type=str, default=DEFAULT_CONFIG.sample_prompt)
    parser.add_argument("--sample-max-new-tokens", type=int, default=DEFAULT_CONFIG.sample_max_new_tokens)
    parser.add_argument("--sample-temperature", type=float, default=DEFAULT_CONFIG.sample_temperature)
    parser.add_argument("--sample-top-p", type=float, default=DEFAULT_CONFIG.sample_top_p)
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG.device)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)

    return parser


def parse_config(argv: Sequence[str] | None = None) -> TrainConfig:
    """Parse command-line overrides and return the resulting training config."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    return TrainConfig(
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_tags=tuple(args.wandb_tags) if args.wandb_tags is not None else DEFAULT_CONFIG.wandb_tags,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        tokenized_train_data_path=args.tokenized_train_data_path,
        tokenized_valid_data_path=args.tokenized_valid_data_path,
        tokenizer_vocab_path=args.tokenizer_vocab_path,
        tokenizer_merges_path=args.tokenizer_merges_path,
        checkpoint_path=args.checkpoint_path,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        sample_max_new_tokens=args.sample_max_new_tokens,
        sample_temperature=args.sample_temperature,
        sample_top_p=args.sample_top_p,
        device=args.device,
        seed=args.seed,
    )
