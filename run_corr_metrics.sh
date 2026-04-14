#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="./metrics/correlation/exp1(mlp vae)"

DATASETS=("adult" "beijing" "default" "magic" "shoppers")
MODELS=("tabsyn" "tabdiff" "catboost" "tabcsdi")

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    python compute_pair_trends_manual.py \
      --dataset-dir "./data/$DATASET" \
      --dataset-name "$DATASET" \
      --pred-path "./$MODEL/prediction_${DATASET}.csv" \
      --output-dir "$OUTPUT_DIR" \
      --model-name "$MODEL"
  done
done