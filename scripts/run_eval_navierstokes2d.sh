#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_eval_navierstokes2d.sh

LNO_PATH="2026-04-21_23-39-12_light_neural_operator_version_1"
CKPT_PATH="../outputs/${LNO_PATH}/ckpts/best_model.pt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src" || exit 1

INDEX_LIST=($(seq 0 4))
for INDEX in "${INDEX_LIST[@]}"; do
  IMAGE_PATH="../outputs/${LNO_PATH}/evaluation_results_${INDEX}.png"
  python eval.py --ckpt-path "$CKPT_PATH" --index "$INDEX" --output-path "$IMAGE_PATH"
done

echo "Evaluation complete!"
