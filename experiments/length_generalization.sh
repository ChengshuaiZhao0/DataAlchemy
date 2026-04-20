#!/usr/bin/env bash
# Length generalization: train at l=4, evaluate at l=2 and l=3.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=length_generalization
mkdir -p "logs/${NAME}"
export DATAALCHEMY_RUN_LOG="logs/${NAME}/run_$(date +%Y%m%d_%H%M%S).log"

python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 2 --output data/F1F2_len2.jsonl
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 3 --output data/F1F2_len3.jsonl
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 4 --output data/F1F2_len4.jsonl

python scripts/pretrain_model.py --config configs/pretrain/${NAME}.yaml
python scripts/model_inference.py --config configs/pretrain/${NAME}.yaml
python scripts/evaluate_predictions.py --config configs/pretrain/${NAME}.yaml

echo "Results: $(pwd)/results/${NAME}"
