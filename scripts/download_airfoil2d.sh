#!/bin/bash
# Working directory: Light-inspired-neural-operator
# Usage: bash ./scripts/download_airfoil2d.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/src/datasets/NACA"

mkdir -p "${OUT_DIR}"

declare -A FILE_IDS=(
  ["NACA_Cylinder_X.npy"]="1rJUPtIhTAsG8TQnqV5mljjgQlyz0NJvJ"
  ["NACA_Cylinder_Y.npy"]="16EY0obqsccypaDFVY0wlsXX73SlD5TMy"
  ["NACA_Cylinder_Q.npy"]="1AjW0t0YolY680J6xTQJ_g5bqTSJAZDZc"
)

all_present=1
for filename in "${!FILE_IDS[@]}"; do
  if [ ! -f "${OUT_DIR}/${filename}" ]; then
    all_present=0
    break
  fi
done

if [ "${all_present}" -eq 1 ]; then
  echo "Airfoil dataset already present under: ${OUT_DIR}"
  exit 0
fi

cat <<'MSG'
This script downloads only the three required Geo-FNO NACA airfoil files into:

  src/datasets/NACA

Expected files:
  - NACA_Cylinder_X.npy
  - NACA_Cylinder_Y.npy
  - NACA_Cylinder_Q.npy

If automatic download fails because of Google Drive quota/permission restrictions,
please manually download the files and place them in src/datasets/NACA.
MSG

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown locally with pip..."
  python3 -m pip install gdown
fi

download_file() {
  local filename="$1"
  local file_id="$2"
  local output_path="${OUT_DIR}/${filename}"
  local url="https://drive.google.com/uc?id=${file_id}"

  if [ -f "${output_path}" ]; then
    echo "Found existing file: ${output_path}"
    return 0
  fi

  echo "Downloading ${filename}..."
  python3 -m gdown "${url}" -O "${output_path}"

  if [ ! -s "${output_path}" ]; then
    echo "Failed to download ${filename}, or downloaded file is empty."
    rm -f "${output_path}"
    exit 1
  fi

  echo "Installed ${output_path}"
}

download_file "NACA_Cylinder_X.npy" "${FILE_IDS[NACA_Cylinder_X.npy]}"
download_file "NACA_Cylinder_Y.npy" "${FILE_IDS[NACA_Cylinder_Y.npy]}"
download_file "NACA_Cylinder_Q.npy" "${FILE_IDS[NACA_Cylinder_Q.npy]}"

echo "Airfoil dataset is ready under: ${OUT_DIR}"