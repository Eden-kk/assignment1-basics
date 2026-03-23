#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting layer norm ablation sweep..."
bash "$ROOT_DIR/scripts/run_layer_norm_ablation.sh"

echo "Starting post-norm ablation sweep..."
bash "$ROOT_DIR/scripts/run_post_norm_ablation.sh"

echo "Starting NoPE ablation sweep..."
bash "$ROOT_DIR/scripts/run_no_pos_emb_ablation.sh"

echo "Starting SwiGLU vs SiLU ablation sweep..."
bash "$ROOT_DIR/scripts/run_swiglu_ablation.sh"
