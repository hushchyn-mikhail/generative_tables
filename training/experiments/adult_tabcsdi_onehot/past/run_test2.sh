#!/usr/bin/env bash
set -e

MODEL_PATH="$1"
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: bash run_test2.sh save/<run>/model.pth"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/submodules/TabCSDI"

python "$ROOT_DIR/experiments/adult_tabcsdi_onehot/eval_test2_adult_tabsyn_onehot.py" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --model_path "$MODEL_PATH" \
  --nsample 10 \
  --bootstrap 500 \
  --eval_batch_size 512 \
  --outdir "$ROOT_DIR/experiments/adult_tabcsdi_onehot/results"