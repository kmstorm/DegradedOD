#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# Download COCO 2017 Train Images + Annotations
# Usage: bash scripts/download_coco.sh [./data/coco]
# Estimated size: ~19GB (18GB images + ~250MB annotations)
#
# Images source:  HuggingFace mirror (g292725651/coco2017)
# Annotations:    Official COCO server
# ────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_DIR="${1:-./data/coco}"
mkdir -p "$DATA_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║   COCO 2017 Dataset Downloader               ║"
echo "║   Target: $DATA_DIR"
echo "╚══════════════════════════════════════════════╝"

# Install huggingface_hub if needed
if ! command -v hf &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub[cli]
fi

# ── Train images (~19GB) via HuggingFace ──
TRAIN_ZIP="$DATA_DIR/train2017.zip"

if [ -d "$DATA_DIR/train2017" ] && [ "$(ls -1 "$DATA_DIR/train2017" | wc -l)" -gt 100000 ]; then
    echo "[SKIP] train2017/ already exists ($(ls -1 "$DATA_DIR/train2017" | wc -l) files)"
else
    echo "[1/3] Downloading COCO 2017 train images (~19GB) from HuggingFace..."
    echo "       Source: huggingface.co/datasets/g292725651/coco2017"
    echo ""

    # Remove stale partial file if exists
    rm -f "$TRAIN_ZIP"

    hf download g292725651/coco2017 train2017.zip \
        --repo-type dataset \
        --local-dir "$DATA_DIR"

    echo ""
    echo "[1/3] Extracting train2017..."
    unzip -q -o "$TRAIN_ZIP" -d "$DATA_DIR"
    rm -f "$TRAIN_ZIP"
    echo "[1/3] ✓ train2017: $(ls -1 "$DATA_DIR/train2017" | wc -l) images"
fi

# ── Annotations from HuggingFace mirror ──
if [ -f "$DATA_DIR/annotations/instances_train2017.json" ]; then
    echo "[2/3] [SKIP] annotations/ already exists"
else
    echo "[2/3] Downloading annotations directly from HuggingFace mirror..."
    mkdir -p "$DATA_DIR/annotations"
    
    hf download merve/coco annotations/instances_train2017.json \
        --repo-type dataset \
        --local-dir "$DATA_DIR"
        
    hf download merve/coco annotations/instances_val2017.json \
        --repo-type dataset \
        --local-dir "$DATA_DIR"
        
    echo "[2/3] ✓ annotations downloaded"
fi

# ── Verify ──
echo ""
echo "[3/3] Verification:"
TRAIN_COUNT=$(ls -1 "$DATA_DIR/train2017" 2>/dev/null | wc -l)
ANN_COUNT=$(ls -1 "$DATA_DIR/annotations" 2>/dev/null | wc -l)
echo "  train2017/:   $TRAIN_COUNT images (expected 118,287)"
echo "  annotations/: $ANN_COUNT files"
echo ""

if [ -f "$DATA_DIR/annotations/instances_train2017.json" ] && [ "$TRAIN_COUNT" -gt 100000 ]; then
    echo "✅ COCO 2017 download complete: $DATA_DIR"
else
    echo "❌ Something is missing. Check the output above for errors."
    exit 1
fi
