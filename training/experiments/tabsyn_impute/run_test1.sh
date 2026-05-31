#!/usr/bin/env bash
set -e

DATASET="${1:-adult}"
GPU="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$ROOT_DIR/.pylibs:$ROOT_DIR/submodules/tabsyn:${PYTHONPATH:-}"

python "$ROOT_DIR/experiments/tabsyn_impute/eval_test1_tabsyn.py" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --dataname "$DATASET" \
  --gpu "$GPU" \
  --missingratio 0.2 \
  --num_steps 50 \
  --resample_steps 10 \
  --bootstrap 500 \
  --seed 1 \
  --outdir "$ROOT_DIR/experiments/tabsyn_impute/results"
