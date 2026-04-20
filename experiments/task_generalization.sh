#!/usr/bin/env bash
# Task generalization (CMP): train on F1F1, F1F2, F2F1; evaluate on held-out F2F2.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=task_generalization
mkdir -p "logs/${NAME}"
export DATAALCHEMY_RUN_LOG="logs/${NAME}/run_$(date +%Y%m%d_%H%M%S).log"

python scripts/generate_data.py --transformations "[F1]" "[F1]" --element-length 4 --output data/F1F1.jsonl
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 4 --output data/F1F2.jsonl
python scripts/generate_data.py --transformations "[F2]" "[F1]" --element-length 4 --output data/F2F1.jsonl
python scripts/generate_data.py --transformations "[F2]" "[F2]" --element-length 4 --output data/F2F2.jsonl

python scripts/pretrain_model.py --config configs/pretrain/${NAME}.yaml
python scripts/model_inference.py --config configs/pretrain/${NAME}.yaml
python scripts/evaluate_predictions.py --config configs/pretrain/${NAME}.yaml

echo "Results: $(pwd)/results/${NAME}"
