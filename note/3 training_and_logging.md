# Training and Logging Notes

## Why logging matters for Section 7

Section 7 of the handout is not just about getting training to run. It asks for:

- learning curves
- validation loss comparisons
- wallclock-time plots
- generated text samples
- ablation comparisons
- leaderboard-style experiment tracking

This means the training script needs more than `print(loss)`. It needs a small logging infrastructure.

## Minimum logging infrastructure

A minimal but useful logging setup for this project should record:

1. Run metadata
2. Scalar metrics
3. Time information
4. Text generation samples
5. Checkpoint metadata
6. Experiment grouping information

## 1. Run metadata

At the start of every run, log:

- `run_name`
- `experiment_name`
- `dataset`
- `tokenizer`
- `seed`
- `device`
- model hyperparameters
- optimizer hyperparameters
- decoding hyperparameters for sampling
- optional git commit hash

This is the information you need later to know what produced a particular curve.

## 2. Scalar metrics

The training loop should periodically log scalar values such as:

- `train/loss`
- `valid/loss`
- `train/lr`
- `train/grad_norm`
- `throughput/tokens_per_sec`
- `throughput/examples_per_sec`

At minimum, Section 7 experiments need:

- training loss
- validation loss
- time

## 3. Time information

Section 7 explicitly asks for wallclock-time x-axes and leaderboard runs have a time budget.

So every logging event should include:

- `iteration`
- `time/elapsed_sec`
- `time/elapsed_hours`

Optional but helpful:

- `time/step_sec`
- `time/eta_sec`

## 4. Text generation samples

Loss curves do not fully capture output quality. The handout also asks for generated text.

It is useful to periodically log:

- `sample/prompt`
- `sample/generated_text`
- `sample/temperature`
- `sample/top_p`
- `sample/max_new_tokens`

These samples are especially useful for:

- TinyStories qualitative checks
- OpenWebText qualitative checks
- comparing ablations that have similar losses but different text quality

## 5. Checkpoint metadata

Whenever a checkpoint is saved, also log:

- `checkpoint/path`
- `checkpoint/iteration`
- `checkpoint/valid_loss`
- `checkpoint/elapsed_sec`

This makes it easy to connect saved model states with specific training milestones.

## 6. Experiment grouping

Section 7 includes multiple experiment families:

- TinyStories baseline
- OpenWebText baseline
- remove RMSNorm
- post-norm
- NoPE
- SwiGLU vs SiLU
- custom modification / leaderboard run

So the logging system should support grouping or tagging runs with:

- `group`
- `tags`
- `notes`

Example tags:

- `tinystories`
- `owt`
- `rmsnorm_ablation`
- `post_norm`
- `nope`
- `swiglu_vs_silu`
- `leaderboard`

## Where W&B fits

Weights & Biases (W&B) is a good fit for Section 7 because it can store:

- scalar metrics
- config metadata
- grouped runs
- generated samples
- artifacts such as checkpoints

In this project, the most natural places to call W&B are:

### At run start

Call `wandb.init(...)` after parsing config and constructing the model/optimizer.

Log:

- project name
- run name
- config dict
- experiment group
- tags

### In `log_every`

Call `wandb.log(...)` for:

- `iteration`
- `train/loss`
- `train/lr`
- `time/elapsed_sec`
- optional throughput metrics

### In `eval_every`

Call `wandb.log(...)` for:

- `iteration`
- `valid/loss`
- `time/elapsed_sec`

### In `sample_every`

Call `wandb.log(...)` for text samples:

- prompt
- sampled output
- decoding settings

### In `save_every`

Log checkpoint metadata and optionally save checkpoint artifacts.

## Suggested W&B naming scheme

Keep metric names structured and stable.

Recommended prefixes:

- `train/*`
- `valid/*`
- `time/*`
- `throughput/*`
- `sample/*`
- `checkpoint/*`
- `system/*`

Examples:

- `train/loss`
- `train/lr`
- `valid/loss`
- `time/elapsed_hours`
- `throughput/tokens_per_sec`
- `sample/text`
- `checkpoint/path`

This makes W&B dashboards and plots easier to read.

## Suggested config fields

To support logging cleanly, add configuration fields such as:

- `run_name`
- `experiment_name`
- `log_dir`
- `use_wandb`
- `wandb_project`
- `wandb_group`
- `wandb_tags`
- `sample_every`
- `sample_prompt`
- `sample_max_new_tokens`
- `sample_temperature`
- `sample_top_p`

## Minimal fallback without W&B

If you do not want to depend on W&B immediately, the same logging structure can be implemented with:

- human-readable stdout logs
- `metrics.jsonl` for scalar events
- `samples.jsonl` for generated text
- checkpoint files on disk

This is often enough for:

- plotting learning curves later
- comparing experiments manually
- reproducing a good run

## Practical recommendation for this repo

For this project, a simple path forward is:

1. Keep the main training loop in `train.py`.
2. Add a small logging helper module later if the loop gets cluttered.
3. Log scalar metrics in `log_every`.
4. Log validation metrics in `eval_every`.
5. Add a `sample_every` hook for generated text.
6. Save checkpoint metadata together with checkpoints.
7. Optionally mirror all of this to W&B.

This gives enough structure to satisfy the needs of Section 7 without overengineering the training code.
