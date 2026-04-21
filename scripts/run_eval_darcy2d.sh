#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_eval.sh

# LNO_PATH="2026-04-20_06-27-54_light_neural_operator_version_1" # standard mode: O(N^2) scattering
LNO_PATH="2026-04-20_23-18-22_light_neural_operator_version_1" # efficient mode: O(NdM) scattering
CKPT_PATH="../outputs/${LNO_PATH}/ckpts/best_model.pt"

# Get the script's directory and change to src
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src" || exit 1

INDEX_LIST=($(seq 0 4)) # Generate a list of indices from 0 to 4
for INDEX in "${INDEX_LIST[@]}"; do
  IMAGE_PATH="../outputs/${LNO_PATH}/evaluation_results_${INDEX}.png"
  python eval.py --ckpt-path "$CKPT_PATH" --index "$INDEX" --output-path "$IMAGE_PATH"
done

echo "Evaluation complete!"
