#!/usr/bin/env bash
set -e

DATASET="${1:-adult}"
GPU="${2:-0}"



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$ROOT_DIR/.pylibs:$ROOT_DIR/submodules/tabdiff:$ROOT_DIR/submodules/tabsyn:${PYTHONPATH:-}"

python "$ROOT_DIR/experiments/tabdiff_impute/eval_test2_tabdiff.py" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --tabdiff_root "$ROOT_DIR/submodules/tabdiff" \
  --dataname "$DATASET" \
  --gpu "$GPU" \
  --resample_rounds 1 \
  --impute_condition x_t \
  --bootstrap 300 \
  --seed 1 \
  --outdir "$ROOT_DIR/experiments/tabdiff_impute/results/test2"
