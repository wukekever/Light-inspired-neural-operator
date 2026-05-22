#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_train_darcy2d_ablation.sh [extra run_darcy2d_ablation.py args]

set -euo pipefail

cd "$(dirname "$0")/.." && cd src

COMMON_ARGS=("$@")

run_one() {
  local tag="$1"
  shift
  echo "[LiNO Darcy ablation] ${tag}: components=$*"
  python run_darcy2d_ablation.py \
    --tag "${tag}" \
    --light-components "$@" \
    "${COMMON_ARGS[@]}"
}

run_one "darcy2d_full" reflection refraction scattering
run_one "darcy2d_no_reflection" refraction scattering
run_one "darcy2d_no_refraction" reflection scattering
run_one "darcy2d_no_scattering" reflection refraction

