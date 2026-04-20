#!/usr/bin/env bash
# Format generalization: train on clean single-step F1 and F2, evaluate on
# hybrid-noised variants of each at p=0.10 and p=0.20.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=format_generalization
mkdir -p "logs/${NAME}"
export DATAALCHEMY_RUN_LOG="logs/${NAME}/run_$(date +%Y%m%d_%H%M%S).log"

python scripts/generate_data.py --transformations "[F1]" --element-length 4 --output data/F1.jsonl
python scripts/generate_data.py --transformations "[F2]" --element-length 4 --output data/F2.jsonl

for rule in F1 F2; do
    for p in 0p10 0p20; do
        python scripts/apply_noise.py \
            --inputs data/${rule}.jsonl \
            --output data/${rule}_hybrid_${p}.jsonl \
            --mode hybrid --p "0.${p#0p}" --seed 42
    done
done

python scripts/pretrain_model.py --config configs/pretrain/${NAME}.yaml
python scripts/model_inference.py --config configs/pretrain/${NAME}.yaml
python scripts/evaluate_predictions.py --config configs/pretrain/${NAME}.yaml

echo "Results: $(pwd)/results/${NAME}"
