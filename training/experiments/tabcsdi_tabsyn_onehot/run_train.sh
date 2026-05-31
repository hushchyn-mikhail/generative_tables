#!/usr/bin/env bash
set -e

DATASET="${1:-adult}"
GPU="${2:-0}"
EPOCHS="${3:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/submodules/TabCSDI"

python "$ROOT_DIR/experiments/tabcsdi_tabsyn_onehot/train_tabcsdi_tabsyn_onehot.py" \
  --dataname "$DATASET" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --missingratio 0.2 \
  --trainmissingratio 0.2 \
  --batch_size 1024 \
  --nsample 1 \
  --seed 1 \
  --epochs "$EPOCHS" \
  --device "cuda:${GPU}"