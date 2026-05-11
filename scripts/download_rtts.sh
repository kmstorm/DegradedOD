#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# Download RTTS (Real-world Task-driven Testing Set)
# Part of the RESIDE-β benchmark for real-world haze evaluation
# Usage: bash scripts/download_rtts.sh [--data-dir ./data/rtts]
# Estimated size: ~500MB
# Source: https://github.com/Boyiliee/RESIDE-dataset-link
#
# NOTE: RTTS is hosted on Dropbox / Baidu Yun. The Dropbox short
# link (https://bit.ly/3c4gl3z) may expire. If it fails, manually
# download from Baidu Yun:
#   URL: https://pan.baidu.com/s/1nuJOdjr
#   Password: n3v8
# ────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_DIR="${1:-./data/rtts}"
mkdir -p "$DATA_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║   RTTS Dataset Downloader                    ║"
echo "║   Target: $DATA_DIR"
echo "╚══════════════════════════════════════════════╝"

# RTTS Dropbox link (shortened)
RTTS_URL="https://bit.ly/3c4gl3z"
RTTS_ZIP="$DATA_DIR/RTTS.zip"

# Check if already downloaded
if [ -d "$DATA_DIR/JPEGImages" ] && [ "$(ls -1 "$DATA_DIR/JPEGImages" 2>/dev/null | wc -l)" -gt 4000 ]; then
    echo "[SKIP] RTTS already exists ($(ls -1 "$DATA_DIR/JPEGImages" | wc -l) images)"
else
    echo "[1/2] Downloading RTTS from Dropbox (~500MB)..."
    echo "       If this fails, manually download from:"
    echo "       Baidu Yun: https://pan.baidu.com/s/1nuJOdjr (pwd: n3v8)"
    echo ""

    # Try wget with redirect following
    if wget -c -q --show-progress -L "$RTTS_URL" -O "$RTTS_ZIP" 2>/dev/null; then
        echo "[1/2] Download complete. Extracting..."
        unzip -q -o "$RTTS_ZIP" -d "$DATA_DIR"
        rm -f "$RTTS_ZIP"

        # RTTS may extract into a subdirectory — handle both cases
        if [ -d "$DATA_DIR/RTTS" ]; then
            # Move contents up if extracted into RTTS/ subfolder
            if [ -d "$DATA_DIR/RTTS/JPEGImages" ]; then
                mv "$DATA_DIR/RTTS/"* "$DATA_DIR/" 2>/dev/null || true
                rmdir "$DATA_DIR/RTTS" 2>/dev/null || true
            fi
        fi
    else
        echo ""
        echo "❌ Dropbox download failed. Please download manually:"
        echo "   1. Go to: https://bit.ly/3c4gl3z"
        echo "      OR Baidu Yun: https://pan.baidu.com/s/1nuJOdjr (pwd: n3v8)"
        echo "   2. Extract to: $DATA_DIR/"
        echo "   3. Ensure this structure:"
        echo "      $DATA_DIR/"
        echo "      ├── JPEGImages/    # 4,322 hazy images (.png)"
        echo "      ├── Annotations/   # VOC XML annotations"
        echo "      └── ImageSets/Main/test.txt"
        exit 1
    fi
fi

# ── Verify ──
echo ""
echo "[2/2] Verification:"

IMG_COUNT=0
ANN_COUNT=0

if [ -d "$DATA_DIR/JPEGImages" ]; then
    IMG_COUNT=$(ls -1 "$DATA_DIR/JPEGImages" 2>/dev/null | wc -l)
fi
if [ -d "$DATA_DIR/Annotations" ]; then
    ANN_COUNT=$(ls -1 "$DATA_DIR/Annotations" 2>/dev/null | wc -l)
fi

echo "  JPEGImages/:  $IMG_COUNT images"
echo "  Annotations/: $ANN_COUNT XML files"

if [ "$IMG_COUNT" -gt 4000 ] && [ "$ANN_COUNT" -gt 4000 ]; then
    echo ""
    echo "✅ RTTS download complete: $DATA_DIR"
else
    echo ""
    echo "⚠️  File counts seem low. Expected ~4,322 images and annotations."
    echo "    Dataset may still be usable — check the files manually."
fi

echo ""
echo "Expected structure:"
echo "  $DATA_DIR/"
echo "  ├── JPEGImages/     # 4,322 hazy .png images"
echo "  ├── Annotations/    # VOC XML per image"
echo "  └── ImageSets/"
echo "      └── Main/"
echo "          └── test.txt"
echo ""
echo "VOC XML classes: person, car, bus, bicycle, motorbike"
