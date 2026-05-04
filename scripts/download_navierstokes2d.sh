#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set Google Drive file ID and output filename
FILE_ID="1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d"  # NavierStokes_V1e-3_N5000_T50
OUTPUT="${REPO_ROOT}/src/datasets/NavierStokes2D/navierstokes.zip"
OUT_DIR="$(dirname "$OUTPUT")"
MAT="${OUT_DIR}/ns_V1e-3_N5000_T50.mat"
COOKIES="${OUT_DIR}/.navierstokes_cookies.txt"
RESPONSE_HTML="${OUT_DIR}/.navierstokes_response.html"

mkdir -p "$OUT_DIR"

if [ -f "$MAT" ]; then
    echo "Dataset already present, skipping: $MAT"
    exit 0
fi

if [ -f "$OUTPUT" ]; then
    echo "Archive already exists, skipping download: $OUTPUT"
else
    # Fetch the download page and capture the response
    RESPONSE=$(wget --quiet --save-cookies "$COOKIES" \
        --keep-session-cookies --no-check-certificate \
        "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
        -O -)

    # Save response for debugging
    echo "$RESPONSE" > "$RESPONSE_HTML"

    # Parse hidden form fields for the actual download
    DOWNLOAD_URL="https://drive.usercontent.google.com/download"
    ID=$(echo "$RESPONSE" | grep -o 'name="id" value="[^"]*"' | cut -d'"' -f4)
    EXPORT=$(echo "$RESPONSE" | grep -o 'name="export" value="[^"]*"' | cut -d'"' -f4)
    CONFIRM=$(echo "$RESPONSE" | grep -o 'name="confirm" value="[^"]*"' | cut -d'"' -f4)
    UUID=$(echo "$RESPONSE" | grep -o 'name="uuid" value="[^"]*"' | cut -d'"' -f4)

    # Ensure all required parameters were extracted
    if [ -z "$ID" ] || [ -z "$EXPORT" ] || [ -z "$CONFIRM" ] || [ -z "$UUID" ]; then
        echo "Failed to extract all required download parameters"
        echo "ID: $ID"
        echo "EXPORT: $EXPORT"
        echo "CONFIRM: $CONFIRM"
        echo "UUID: $UUID"
        exit 1
    fi

    # Build the full download URL
    FULL_URL="${DOWNLOAD_URL}?id=${ID}&export=${EXPORT}&confirm=${CONFIRM}&uuid=${UUID}"

    echo "Starting download..."
    echo "URL: $FULL_URL"

    wget --load-cookies "$COOKIES" \
        --no-check-certificate \
        "$FULL_URL" \
        -O "$OUTPUT"

    echo "Download succeeded: $OUTPUT"
    rm -f "$COOKIES" "$RESPONSE_HTML"
fi

echo "Extracting: $OUTPUT -> $OUT_DIR"
unzip -o "$OUTPUT" -d "$OUT_DIR"

rm -f "$OUTPUT"
echo "Removed archive: $OUTPUT"
echo "Done."
