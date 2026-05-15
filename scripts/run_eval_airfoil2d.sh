#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_eval_airfoil2d.sh

LNO_PATH="2026-05-14_08-50-30_light_neural_operator_airfoil2d_lino"
# LNO_PATH="airfoil2d"
CKPT_PATH="../outputs/${LNO_PATH}/ckpts/best_model.pt"

# Get the script's directory and change to src
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src" || exit 1

INDEX_LIST=($(seq 50 54)) # Generate a list of indices from 0 to 4
for INDEX in "${INDEX_LIST[@]}"; do
  IMAGE_PATH="../outputs/${LNO_PATH}/evaluation_results_${INDEX}.png"
  python eval.py --ckpt-path "$CKPT_PATH" --index "$INDEX" --output-path "$IMAGE_PATH"
done

echo "Evaluation complete!"
