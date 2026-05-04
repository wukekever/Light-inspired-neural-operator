#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_eval_navierstokes2d.sh

set -e

LNO_PATH="navierstokes2d_64"  # efficient mode: O(NdM) scattering
CKPT_PATH="../outputs/${LNO_PATH}/ckpts/best_model.pt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src" || exit 1

INDEX_LIST=($(seq 20 5 40))  # Evaluate samples at these time indices (e.g., 10, 15, 20, 25, 30)

for INDEX in "${INDEX_LIST[@]}"; do
  IMAGE_PATH="../outputs/${LNO_PATH}/evaluation_results_${INDEX}.png"

  echo "Evaluating Navier-Stokes sample index ${INDEX}..."
  python eval.py \
    --ckpt-path "$CKPT_PATH" \
    --index "$INDEX" \
    --output-path "$IMAGE_PATH"
done

echo "Evaluation complete!"