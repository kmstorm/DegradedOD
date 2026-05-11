#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# Download ExDark (Exclusively Dark Image Dataset)
# Usage: bash scripts/download_exdark.sh [./data/exdark]
# Estimated size: ~1.5GB
#
# Google Drive sources:
#   Images:      https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx
#   Annotations: https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i
# ────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_DIR="${1:-./data/exdark}"
mkdir -p "$DATA_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║   ExDark Dataset Downloader                  ║"
echo "║   Target: $DATA_DIR"
echo "╚══════════════════════════════════════════════╝"

# Install gdown if needed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install -q gdown
fi

IMAGES_ID="1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"
ANNOS_ID="1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"

# ── Step 1: Download & extract images ──
IMG_COUNT=$(find "$DATA_DIR/images" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.JPG" -o -iname "*.JPEG" \) 2>/dev/null | wc -l)
if [ "$IMG_COUNT" -gt 7000 ]; then
    echo "[SKIP] Images already present ($IMG_COUNT files)"
else
    echo "[1/2] Downloading ExDark images (~1.5GB)..."
    gdown "$IMAGES_ID" -O "$DATA_DIR/exdark_images.zip"
    echo "       Extracting..."
    unzip -q -o "$DATA_DIR/exdark_images.zip" -d "$DATA_DIR"
    rm -f "$DATA_DIR/exdark_images.zip"

    # Organize: move class dirs into images/
    mkdir -p "$DATA_DIR/images"
    for cls in Bicycle Boat Bottle Bus Car Cat Chair Cup Dog Motorbike People Table; do
        if [ -d "$DATA_DIR/$cls" ]; then
            mv "$DATA_DIR/$cls" "$DATA_DIR/images/"
        fi
    done
fi

# ── Step 2: Download & extract annotations ──
ANN_COUNT=$(find "$DATA_DIR/annotations" -type f -iname "*.txt" 2>/dev/null | wc -l)
if [ "$ANN_COUNT" -gt 7000 ]; then
    echo "[SKIP] Annotations already present ($ANN_COUNT files)"
else
    echo "[2/2] Downloading ExDark annotations..."
    gdown "$ANNOS_ID" -O "$DATA_DIR/exdark_annos.zip"
    echo "       Extracting..."
    unzip -q -o "$DATA_DIR/exdark_annos.zip" -d "$DATA_DIR"
    rm -f "$DATA_DIR/exdark_annos.zip"

    # Organize: move into annotations/
    mkdir -p "$DATA_DIR/annotations"
    # Handle ExDark_Annno/ subfolder if present
    if [ -d "$DATA_DIR/ExDark_Annno" ]; then
        for cls in Bicycle Boat Bottle Bus Car Cat Chair Cup Dog Motorbike People Table; do
            if [ -d "$DATA_DIR/ExDark_Annno/$cls" ]; then
                mv "$DATA_DIR/ExDark_Annno/$cls" "$DATA_DIR/annotations/"
            fi
        done
        rm -rf "$DATA_DIR/ExDark_Annno"
    fi
fi

# ── Cleanup ──
rm -rf "$DATA_DIR/__MACOSX"

# ── Verify ──
echo ""
IMG_COUNT=$(find "$DATA_DIR/images" -type f 2>/dev/null | wc -l)
ANN_COUNT=$(find "$DATA_DIR/annotations" -type f -iname "*.txt" 2>/dev/null | wc -l)
echo "Images:      $IMG_COUNT (expected 7363)"
echo "Annotations: $ANN_COUNT (expected 7363)"

if [ "$IMG_COUNT" -eq 7363 ] && [ "$ANN_COUNT" -eq 7363 ]; then
    echo ""
    echo "✅ ExDark download complete: $DATA_DIR"
else
    echo ""
    echo "⚠️  Counts don't match expected. Check: ls $DATA_DIR"
fi
