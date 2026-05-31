#!/usr/bin/env bash
set -e

GPU="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/submodules/tabsyn:$PYTHONPATH"


cd "$ROOT_DIR/submodules/tabsyn"

# 1) train VAE
# python main.py --dataname adult --method vae --mode train --gpu "$GPU"

# # 2) train diffusion (TabSyn)
# python main.py --dataname adult --method tabsyn --mode train --gpu "$GPU"


# python main.py --dataname adult --method vae --mode train --cat_decoder_type linear --gpu "$GPU"
# #python main.py --dataname magic --method tabsyn --mode train --gpu "$GPU"
# python main.py --method adult --mode train  --dataname magic --block_noise_mode data --gpu "$GPU"

python main.py --dataname adult --method vae --mode train --gpu "$GPU" --cat_decoder_type "mlp"
python main.py --method tabsyn --mode train  --dataname adult --block_noise_mode learned --gpu "$GPU"


echo " Trained Adult. Checkpoints:"
echo "   VAE:     $ROOT_DIR/submodules/tabsyn/tabsyn/vae/ckpt/adult/"
echo "   Diff:    $ROOT_DIR/submodules/tabsyn/tabsyn/ckpt/adult/"

