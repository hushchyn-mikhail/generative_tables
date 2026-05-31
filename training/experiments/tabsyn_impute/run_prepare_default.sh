#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/submodules/tabsyn"

python process_dataset.py --dataname default
echo "Default processed: $ROOT_DIR/submodules/tabsyn/data/default"
