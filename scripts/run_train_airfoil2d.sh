#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_train_airfoil2d.sh

set -euo pipefail

cd "$(dirname "$0")/.." && cd src

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "Python is not installed or not found in PATH." >&2
  exit 1
fi

"$PYTHON" run_airfoil2d.py \
  --dataset_name airfoil2d \
  --data-path ./datasets/NACA
