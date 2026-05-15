#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/src/datasets/NACA"

mkdir -p "${OUT_DIR}"

NEEDED=(
  "NACA_Cylinder_X.npy"
  "NACA_Cylinder_Y.npy"
  "NACA_Cylinder_Q.npy"
)

all_present=1
for f in "${NEEDED[@]}"; do
  if [ ! -f "${OUT_DIR}/${f}" ]; then
    all_present=0
  fi
done

if [ "${all_present}" -eq 1 ]; then
  echo "Airfoil dataset already present under: ${OUT_DIR}"
  exit 0
fi

cat <<'MSG'
This script downloads the Geo-FNO NACA airfoil files into src/datasets/NACA.
Expected files after download:
  - NACA_Cylinder_X.npy
  - NACA_Cylinder_Y.npy
  - NACA_Cylinder_Q.npy

The upstream dataset is hosted by the Geo-FNO authors on Google Drive.
If automatic download fails because of Google Drive quota/permission restrictions,
manually download the Geo-PDE datasets from the Geo-FNO repository README and place
these three .npy files in src/datasets/NACA.
MSG

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown locally with pip..."
  python3 -m pip install gdown
fi

# Geo-FNO README dataset folder. gdown will download all files in the folder;
# the script then searches for the NACA files and copies them to OUT_DIR.
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing"
TMP_DIR="${OUT_DIR}/.download_tmp"
mkdir -p "${TMP_DIR}"

echo "Downloading Geo-PDE data folder to temporary directory..."
python -m gdown --folder "${GDRIVE_FOLDER}" -O "${TMP_DIR}" || {
  echo "Automatic download failed. Please download manually from the Geo-FNO README dataset link."
  exit 1
}

for f in "${NEEDED[@]}"; do
  match="$(find "${TMP_DIR}" -name "${f}" -type f | head -n 1 || true)"
  if [ -z "${match}" ]; then
    echo "Could not find ${f} in downloaded folder. Please place it manually in ${OUT_DIR}."
    exit 1
  fi
  cp "${match}" "${OUT_DIR}/${f}"
  echo "Installed ${OUT_DIR}/${f}"
done

rm -rf "${TMP_DIR}"
echo "Airfoil dataset is ready under: ${OUT_DIR}"
