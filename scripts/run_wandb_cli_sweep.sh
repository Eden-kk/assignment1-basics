#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${1:-}"
AGENT_COUNT="${AGENT_COUNT:-${2:-}}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data/TinyStoriesV2-GPT4-train.npy}"
VALID_DATA_PATH="${VALID_DATA_PATH:-data/TinyStoriesV2-GPT4-valid.npy}"
TOKENIZER_VOCAB_PATH="${TOKENIZER_VOCAB_PATH:-assets/tokenizer/gpt2_vocab.json}"
TOKENIZER_MERGES_PATH="${TOKENIZER_MERGES_PATH:-assets/tokenizer/gpt2_merges.txt}"
WANDB_PROJECT="${WANDB_PROJECT:-cs336-basics}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <sweep-config-path> [agent-count]"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN"
  echo "Create the project environment first, for example with: uv sync"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing sweep config: $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
  echo "Missing tokenized train data: $TRAIN_DATA_PATH"
  exit 1
fi

if [[ ! -f "$VALID_DATA_PATH" ]]; then
  echo "Missing tokenized valid data: $VALID_DATA_PATH"
  exit 1
fi

if [[ ! -f "$TOKENIZER_VOCAB_PATH" ]]; then
  echo "Missing tokenizer vocab: $TOKENIZER_VOCAB_PATH"
  exit 1
fi

if [[ ! -f "$TOKENIZER_MERGES_PATH" ]]; then
  echo "Missing tokenizer merges: $TOKENIZER_MERGES_PATH"
  exit 1
fi

if [[ -n "${SWEEP_ID:-}" ]]; then
  sweep_id="$SWEEP_ID"
else
  sweep_init_cmd=("$PYTHON_BIN" -m wandb sweep --project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    sweep_init_cmd+=(--entity "$WANDB_ENTITY")
  fi
  sweep_init_cmd+=("$CONFIG_PATH")

  init_output="$("${sweep_init_cmd[@]}" 2>&1)"
  printf '%s\n' "$init_output"

  sweep_id="$(printf '%s\n' "$init_output" | rg -o '[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9]+' -N | tail -n 1)"
  if [[ -z "$sweep_id" ]]; then
    echo "Failed to extract sweep ID from wandb sweep output."
    exit 1
  fi
fi

echo "Using sweep: $sweep_id"

agent_cmd=("$PYTHON_BIN" -m wandb agent)
if [[ -n "$AGENT_COUNT" ]]; then
  agent_cmd+=(--count "$AGENT_COUNT")
fi
agent_cmd+=("$sweep_id")

"${agent_cmd[@]}"
