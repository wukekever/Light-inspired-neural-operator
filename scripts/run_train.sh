#!/bin/bash
## Working directory: Light-inspired-neural-operator
## Usage: bash ./scripts/run_train.sh

cd "$(dirname "$0")/.." && cd src
python main.py
