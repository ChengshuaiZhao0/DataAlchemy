#!/usr/bin/env bash
# Task SFT: small-data fine-tune on F2F2 starting from the task-gen checkpoint.
# Requires experiments/task_generalization.sh to have produced
# ./saves/task_generalization/model/final beforehand.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=task_sft
mkdir -p "logs/${NAME}"
export DATAALCHEMY_RUN_LOG="logs/${NAME}/run_$(date +%Y%m%d_%H%M%S).log"

python scripts/generate_data.py --transformations "[F2]" "[F2]" --element-length 4 --output data/F2F2.jsonl

python scripts/sft_model.py --config configs/sft/${NAME}.yaml
python scripts/model_inference.py --config configs/sft/${NAME}.yaml
python scripts/evaluate_predictions.py --config configs/sft/${NAME}.yaml

echo "Results: $(pwd)/results/${NAME}"
