#!/usr/bin/env bash
set -e

# перейти в папку, где лежит этот скрипт
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# корень проекта = на 2 уровня выше (experiments/adult_tabcsdi_onehot/.. -> experiments/.. -> project root)
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/submodules/TabCSDI"

python "$ROOT_DIR/experiments/adult_tabcsdi_onehot/exe_adult_tabsyn_onehot.py" \
  --tabsyn_root "$ROOT_DIR/submodules/tabsyn" \
  --missingratio 0.2 \
  --trainmissingratio 0.2 \
  --seed 1 \
  --epochs 200