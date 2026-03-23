#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/scripts/sweeps/layer_norm_ablation.yaml}"

"$ROOT_DIR/scripts/run_wandb_cli_sweep.sh" "$CONFIG_PATH"
