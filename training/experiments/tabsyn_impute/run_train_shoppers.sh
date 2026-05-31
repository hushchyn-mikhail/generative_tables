#!/usr/bin/env bash
set -e

GPU="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/submodules/tabsyn:$PYTHONPATH"

cd "$ROOT_DIR/submodules/tabsyn"


python main.py --dataname shoppers --method vae --mode train --gpu "$GPU" --cat_decoder_type "mlp"
python main.py --method tabsyn --mode train  --dataname shoppers --block_noise_mode learned --gpu "$GPU"



echo "Trained shoppers."
