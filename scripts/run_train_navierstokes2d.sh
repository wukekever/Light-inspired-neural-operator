#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_train_navierstokes2d.sh

set -euo pipefail

cd "$(dirname "$0")/.." && cd src
python run_navierstokes2d.py
