#!/bin/bash
## Usage: bash ./scripts/run_eval_temporal_error_navierstokes2d.sh

set -e

LNO_PATH="2026-04-21_23-39-12_light_neural_operator_version_1"
CKPT_PATH="../outputs/${LNO_PATH}/ckpts/best_model.pt"
OUTPUT_DIR="../outputs/${LNO_PATH}/temporal_error"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src" || exit 1

mkdir -p "$OUTPUT_DIR"

python eval_temporal_error.py \
  --ckpt-path "$CKPT_PATH" \
  --output-path "${OUTPUT_DIR}/ns_temporal_error.pdf"

echo "Temporal error evaluation complete!"