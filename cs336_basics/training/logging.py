from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb


@dataclass(slots=True)
class TrainMetrics:
    iteration: int
    elapsed_sec: float
    learning_rate: float
    train_loss: float
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    examples_per_sec: float | None = None


@dataclass(slots=True)
class ValidationMetrics:
    iteration: int
    elapsed_sec: float
    valid_loss: float


@dataclass(slots=True)
class SampleLog:
    iteration: int
    elapsed_sec: float
    prompt: str
    generated_text: str
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(slots=True)
class CheckpointLog:
    iteration: int
    elapsed_sec: float
    checkpoint_path: Path
    valid_loss: float | None = None


def build_train_metrics(
    iteration: int,
    elapsed_sec: float,
    learning_rate: float,
    train_loss: float,
    grad_norm: float | None = None,
    tokens_per_sec: float | None = None,
    examples_per_sec: float | None = None,
) -> TrainMetrics:
    """Construct a train-metrics payload for logger backends."""
    return TrainMetrics(
        iteration=iteration,
        elapsed_sec=elapsed_sec,
        learning_rate=learning_rate,
        train_loss=train_loss,
        grad_norm=grad_norm,
        tokens_per_sec=tokens_per_sec,
        examples_per_sec=examples_per_sec,
    )


def build_validation_metrics(
    iteration: int,
    elapsed_sec: float,
    valid_loss: float,
) -> ValidationMetrics:
    """Construct a validation-metrics payload for logger backends."""
    return ValidationMetrics(
        iteration=iteration,
        elapsed_sec=elapsed_sec,
        valid_loss=valid_loss,
    )


def build_checkpoint_log(
    iteration: int,
    elapsed_sec: float,
    checkpoint_path: Path,
    valid_loss: float | None = None,
) -> CheckpointLog:
    """Construct a checkpoint-log payload for logger backends."""
    return CheckpointLog(
        iteration=iteration,
        elapsed_sec=elapsed_sec,
        checkpoint_path=checkpoint_path,
        valid_loss=valid_loss,
    )


def build_sample_log(
    iteration: int,
    elapsed_sec: float,
    prompt: str,
    generated_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> SampleLog:
    """Construct a sample-log payload for logger backends."""
    return SampleLog(
        iteration=iteration,
        elapsed_sec=elapsed_sec,
        prompt=prompt,
        generated_text=generated_text,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


def compute_grad_norm(parameters: Any) -> float:
    """Compute the global L2 norm of all available parameter gradients."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total_sq_norm = torch.zeros((), device=grads[0].device)
    for grad in grads:
        total_sq_norm += torch.sum(grad * grad)
    return float(torch.sqrt(total_sq_norm).item())


def compute_throughput(
    batch_size: int,
    context_length: int,
    step_sec: float,
) -> tuple[float | None, float | None]:
    """Estimate token and example throughput for one optimization step."""
    if step_sec <= 0:
        return None, None

    tokens_this_step = batch_size * context_length
    return tokens_this_step / step_sec, batch_size / step_sec


def _normalize_wandb_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_wandb_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_wandb_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_wandb_value(item) for key, item in value.items()}
    return value


def build_wandb_config(config: Any) -> dict[str, Any]:
    """Convert a training config dataclass into a W&B-friendly config payload."""
    return {
        key: _normalize_wandb_value(value)
        for key, value in asdict(config).items()
    }


def _coerce_wandb_override(current_value: Any, raw_value: Any) -> Any:
    if isinstance(current_value, Path):
        return Path(raw_value)
    if isinstance(current_value, tuple):
        if raw_value is None:
            return ()
        if isinstance(raw_value, list | tuple):
            return tuple(raw_value)
        return (raw_value,)
    if isinstance(current_value, bool):
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            return raw_value.lower() in {"1", "true", "yes", "on"}
        return bool(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return str(raw_value)
    return raw_value


def ensure_wandb_run(config: Any) -> None:
    """Initialize a W&B run once and keep its config in sync across helpers."""
    if wandb.run is None:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            group=config.wandb_group,
            tags=list(config.wandb_tags),
            config=build_wandb_config(config),
        )
    else:
        wandb.config.update(build_wandb_config(config), allow_val_change=True)

    wandb.define_metric("iteration")
    wandb.define_metric("train/*", step_metric="iteration")
    wandb.define_metric("valid/*", step_metric="iteration")
    wandb.define_metric("time/*", step_metric="iteration")
    wandb.define_metric("throughput/*", step_metric="iteration")
    wandb.define_metric("sample/*", step_metric="iteration")
    wandb.define_metric("checkpoint/*", step_metric="iteration")


def apply_wandb_sweep_overrides(config: Any) -> Any:
    """Apply any active W&B sweep parameters to the parsed training config."""
    if not getattr(config, "use_wandb", False):
        return config

    ensure_wandb_run(config)

    overrides: dict[str, Any] = {}
    for key in config.__dataclass_fields__:
        if key not in wandb.config:
            continue
        current_value = getattr(config, key)
        raw_value = wandb.config[key]
        coerced_value = _coerce_wandb_override(current_value, raw_value)
        if coerced_value != current_value:
            overrides[key] = coerced_value

    if not overrides:
        return config

    updated_config = replace(config, **overrides)
    wandb.config.update(build_wandb_config(updated_config), allow_val_change=True)
    return updated_config


class TrainingLogger:
    """Interface for training-time logging backends."""

    def __init__(self, config: Any) -> None:
        raise NotImplementedError

    def log_train(self, metrics: TrainMetrics) -> None:
        raise NotImplementedError

    def log_validation(self, metrics: ValidationMetrics) -> None:
        raise NotImplementedError

    def log_sample(self, sample: SampleLog) -> None:
        raise NotImplementedError

    def log_checkpoint(self, checkpoint: CheckpointLog) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


class WandbTrainingLogger(TrainingLogger):
    """Training logger backed by Weights & Biases."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.run_dir = Path(config.log_dir) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_log_path = Path(config.log_dir) / "experiment_log.md"
        self.latest_train_metrics: TrainMetrics | None = None
        self.latest_validation_metrics: ValidationMetrics | None = None
        self.latest_sample: SampleLog | None = None
        self.latest_checkpoint: CheckpointLog | None = None
        self.best_valid_loss: float | None = None
        self.run_started_at = datetime.now().astimezone()

        ensure_wandb_run(config)

    def log_train(self, metrics: TrainMetrics) -> None:
        self.latest_train_metrics = metrics
        payload = {
            "iteration": metrics.iteration,
            "time/elapsed_sec": metrics.elapsed_sec,
            "time/elapsed_hours": metrics.elapsed_sec / 3600,
            "train/loss": metrics.train_loss,
            "train/lr": metrics.learning_rate,
        }
        if metrics.grad_norm is not None:
            payload["train/grad_norm"] = metrics.grad_norm
        if metrics.tokens_per_sec is not None:
            payload["throughput/tokens_per_sec"] = metrics.tokens_per_sec
        if metrics.examples_per_sec is not None:
            payload["throughput/examples_per_sec"] = metrics.examples_per_sec
        wandb.log(payload)

    def log_validation(self, metrics: ValidationMetrics) -> None:
        self.latest_validation_metrics = metrics
        if self.best_valid_loss is None or metrics.valid_loss < self.best_valid_loss:
            self.best_valid_loss = metrics.valid_loss
        wandb.log({
            "iteration": metrics.iteration,
            "time/elapsed_sec": metrics.elapsed_sec,
            "time/elapsed_hours": metrics.elapsed_sec / 3600,
            "valid/loss": metrics.valid_loss,
        })

    def log_sample(self, sample: SampleLog) -> None:
        self.latest_sample = sample
        wandb.log({
            "iteration": sample.iteration,
            "time/elapsed_sec": sample.elapsed_sec,
            "time/elapsed_hours": sample.elapsed_sec / 3600,
            "sample/prompt": sample.prompt,
            "sample/text": sample.generated_text,
            "sample/temperature": sample.temperature,
            "sample/top_p": sample.top_p,
            "sample/max_new_tokens": sample.max_new_tokens,
        })

    def log_checkpoint(self, checkpoint: CheckpointLog) -> None:
        self.latest_checkpoint = checkpoint
        wandb.log({
            "iteration": checkpoint.iteration,
            "time/elapsed_sec": checkpoint.elapsed_sec,
            "time/elapsed_hours": checkpoint.elapsed_sec / 3600,
            "checkpoint/path": str(checkpoint.checkpoint_path),
            "checkpoint/valid_loss": checkpoint.valid_loss,
        })

    def finish(self) -> None:
        self._append_experiment_log()
        wandb.finish(exit_code=0)

    def _append_experiment_log(self) -> None:
        """Append a concise markdown summary of the run for experiment tracking."""
        self.experiment_log_path.parent.mkdir(parents=True, exist_ok=True)
        latest_checkpoint_path = None
        if self.latest_checkpoint is not None:
            latest_checkpoint_path = self.latest_checkpoint.checkpoint_path

        with open(self.experiment_log_path, mode="a", encoding="utf-8") as f:
            f.write(f"## Run: {self.config.run_name}\n")
            f.write(f"- Timestamp: {self.run_started_at.isoformat()}\n")
            f.write(f"- Experiment: {self.config.experiment_name}\n")
            f.write(f"- W&B Project: {self.config.wandb_project}\n")
            f.write(f"- W&B Group: {self.config.wandb_group}\n")
            f.write(f"- Tags: {', '.join(self.config.wandb_tags) if self.config.wandb_tags else '(none)'}\n")
            f.write(f"- Dataset: {self.config.tokenized_train_data_path}\n")
            f.write(f"- Validation Dataset: {self.config.tokenized_valid_data_path}\n")
            f.write(f"- Model: d_model={self.config.d_model}, layers={self.config.num_layers}, heads={self.config.num_heads}, d_ff={self.config.d_ff}\n")
            f.write(f"- Optimization: lr={self.config.learning_rate}, min_lr={self.config.min_learning_rate}, batch_size={self.config.batch_size}, grad_clip={self.config.grad_clip}\n")
            if self.latest_train_metrics is not None:
                f.write(f"- Latest train loss: {self.latest_train_metrics.train_loss:.6f}\n")
            if self.best_valid_loss is not None:
                f.write(f"- Best valid loss: {self.best_valid_loss:.6f}\n")
            if latest_checkpoint_path is not None:
                f.write(f"- Latest checkpoint: {latest_checkpoint_path}\n")
            if self.latest_sample is not None:
                f.write(f"- Sample prompt: {self.latest_sample.prompt}\n")
                f.write(f"- Sample output: {self.latest_sample.generated_text}\n")
            f.write("\n")


class JsonlTrainingLogger(TrainingLogger):
    """Training logger backed by local JSONL files."""

    def __init__(self, config: Any) -> None:
        raise NotImplementedError

    def log_train(self, metrics: TrainMetrics) -> None:
        raise NotImplementedError

    def log_validation(self, metrics: ValidationMetrics) -> None:
        raise NotImplementedError

    def log_sample(self, sample: SampleLog) -> None:
        raise NotImplementedError

    def log_checkpoint(self, checkpoint: CheckpointLog) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


def build_training_logger(config: Any) -> TrainingLogger:
    """Construct the configured training logger backend."""
    return WandbTrainingLogger(config)
