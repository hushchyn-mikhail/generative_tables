#!/usr/bin/env bash
set -e

DATASET="${1:-adult}"
MODEL_PATH="${2:-}"
GPU="${3:-0}"

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: bash run_test2.sh <dataset> <model_path> [gpu]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/submodules/TabCSDI"

python "$ROOT_DIR/experiments/tabcsdi_tabsyn_onehot/eval_test2_tabcsdi_tabsyn_onehot.py" \
  --dataname "$DATASET" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --model_path "$MODEL_PATH" \
  --seed 1 \
  --nsample 1 \
  --bootstrap 300 \
  --eval_batch_size 32 \
  --device "cuda:${GPU}" \
  --outdir "$ROOT_DIR/experiments/tabcsdi_tabsyn_onehot/results/test2"