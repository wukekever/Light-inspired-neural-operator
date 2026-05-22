#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage:
##   bash ./scripts/run_eval_darcy2d_ablation.sh [index_start] [index_end] [device]
##
## By default, this script searches ../outputs for run folders whose names end with
## the tags used in scripts/run_train_darcy2d_ablation.sh:
##   darcy2d_full, darcy2d_no_reflection, darcy2d_no_refraction, darcy2d_no_scattering
## It evaluates best_model.pt for each matched run folder.

set -euo pipefail

INDEX_START="${1:-0}"
INDEX_END="${2:-2}"
DEVICE="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
OUTPUT_ROOT="${REPO_ROOT}/outputs"

TAGS=(
  "darcy2d_full"
  "darcy2d_no_reflection"
  "darcy2d_no_refraction"
  "darcy2d_no_scattering"

)

find_latest_run() {
  local tag="$1"
  local latest=""
  if [[ -d "${OUTPUT_ROOT}" ]]; then
    latest=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "*${tag}" | sort | tail -n 1)
  fi
  echo "${latest}"
}

cd "${REPO_ROOT}/src" || exit 1

for TAG in "${TAGS[@]}"; do
  RUN_DIR="$(find_latest_run "${TAG}")"

  if [[ -z "${RUN_DIR}" ]]; then
    echo "[Skip] no output directory found for tag: ${TAG}"
    continue
  fi

  CKPT_PATH="${RUN_DIR}/ckpts/best_model.pt"
  RUN_NAME="$(basename "${RUN_DIR}")"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[Skip] checkpoint not found: ${CKPT_PATH}"
    continue
  fi

  echo "[Darcy2D ablation eval] ${RUN_NAME}"
  for INDEX in $(seq "${INDEX_START}" "${INDEX_END}"); do
    IMAGE_PATH="../outputs/${RUN_NAME}/evaluation_results_${INDEX}.png"
    CMD=(python eval_darcy2d_ablation.py --ckpt-path "${CKPT_PATH}" --index "${INDEX}" --output-path "${IMAGE_PATH}")
    if [[ -n "${DEVICE}" ]]; then
      CMD+=(--device "${DEVICE}")
    fi
    echo "  index=${INDEX}"
    "${CMD[@]}"
  done
done

echo "Darcy2D ablation evaluation complete!"
