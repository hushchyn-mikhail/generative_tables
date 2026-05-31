#!/usr/bin/env bash
set -e

DATASET="${1:-adult}"
GPU="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$ROOT_DIR/.pylibs:$ROOT_DIR/submodules/tabsyn:${PYTHONPATH:-}"

python "$ROOT_DIR/experiments/tabsyn_impute/eval_test2_tabsyn.py" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --dataname "$DATASET" \
  --gpu "$GPU" \
  --num_steps 50 \
  --resample_steps 10 \
  --bootstrap 300 \
  --seed 1 \
  --cat_decoder_type "mlp" \
  --outdir "$ROOT_DIR/experiments/tabsyn_impute/results/test2"
