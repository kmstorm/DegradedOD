"""
Phase A: Memory Bank Construction

Runs frozen Grounding DINO on clean COCO train images, extracts per-proposal
features from the encoder hook, and builds a FAISS index for retrieval.

Usage:
    conda activate MODD
    CUDA_VISIBLE_DEVICES=5 python scripts/build_memory_bank.py

    # With overrides:
    CUDA_VISIBLE_DEVICES=5 python scripts/build_memory_bank.py \
        memory.max_images=1000 \
        memory.min_detection_score=0.3
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gdino_wrapper import GroundingDINOWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_memory_bank")


# ──────────────────────────────────────────────────────────────────────
# COCO class vocabulary (all 80 classes)
# ──────────────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def build_text_prompt(classes: list[str]) -> str:
    """Build G-DINO text prompt: 'person . car . dog .'"""
    return " . ".join(classes) + " ."


# ──────────────────────────────────────────────────────────────────────
# Feature extraction loop
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_from_dataset(
    model: GroundingDINOWrapper,
    img_dir: Path,
    text_prompt: str,
    min_score: float = 0.4,
    max_images: int | None = None,
    save_dir: Path = Path("outputs/memory_bank"),
    save_interval: int = 5000,
    resume_from: int = 0,
) -> dict:
    """
    Extract per-proposal features from all images.

    For each detection above min_score:
      - Capture encoder features (vision-language fused)
      - Capture decoder query features (per-proposal)
      - Store with metadata (class, score, image_id, bbox)

    Args:
        model: Frozen GroundingDINOWrapper.
        img_dir: Path to COCO train2017/ directory.
        text_prompt: G-DINO text prompt.
        min_score: Minimum detection score to keep.
        max_images: Process only first N images (None = all).
        save_dir: Output directory for features and index.
        save_interval: Save checkpoint every N images.
        resume_from: Resume from this image index.

    Returns:
        Dict with statistics.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    # Get sorted image list
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    total_images = len(img_files)

    if max_images is not None:
        img_files = img_files[:max_images]
        total_images = len(img_files)

    logger.info(f"Processing {total_images} images from {img_dir}")
    logger.info(f"Min detection score: {min_score}")
    logger.info(f"Text prompt: {text_prompt[:80]}...")

    # Storage buffers
    all_query_features = []    # per-proposal decoder embeddings
    all_encoder_features = []  # per-proposal encoder features (pooled)
    all_metadata = []          # (class_label, score, image_id, bbox)

    # Stats
    stats = {
        "total_images": total_images,
        "processed": 0,
        "skipped": 0,
        "total_detections": 0,
        "total_stored": 0,
        "class_counts": {},
    }

    start_time = time.time()

    for idx, img_file in enumerate(img_files):
        if idx < resume_from:
            continue

        img_path = img_dir / img_file
        image_id = img_file.rsplit(".", 1)[0]  # e.g. "000000000009"

        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_file}: {e}")
            stats["skipped"] += 1
            continue

        # Run G-DINO with feature extraction
        try:
            proposal_features = model.extract_features(img, text_prompt=text_prompt)
        except Exception as e:
            logger.warning(f"Inference failed on {img_file}: {e}")
            stats["skipped"] += 1
            continue

        detection = proposal_features.detection
        stats["total_detections"] += len(detection)

        # Filter by score
        if len(detection) == 0:
            stats["processed"] += 1
            continue

        mask = detection.scores >= min_score
        n_kept = mask.sum().item()

        if n_kept == 0:
            stats["processed"] += 1
            continue

        # Extract features for high-confidence detections
        kept_indices = torch.where(mask)[0]

        for det_idx in kept_indices:
            det_idx_int = det_idx.item()

            # Query feature (decoder output for this proposal)
            if det_idx_int < proposal_features.query_features.shape[0]:
                query_feat = proposal_features.query_features[det_idx_int].cpu().float().numpy()
            else:
                continue

            # Metadata
            score = detection.scores[det_idx_int].item()
            label = detection.labels[det_idx_int]
            bbox = detection.boxes[det_idx_int].cpu().numpy().tolist()

            all_query_features.append(query_feat)
            all_metadata.append({
                "class": label,
                "score": float(score),
                "image_id": image_id,
                "bbox": bbox,
            })

            # Track class distribution
            stats["class_counts"][label] = stats["class_counts"].get(label, 0) + 1

        stats["total_stored"] += n_kept
        stats["processed"] += 1

        # Progress logging
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1 - resume_from) / elapsed
            eta = (total_images - idx - 1) / speed if speed > 0 else 0
            logger.info(
                f"[{idx + 1}/{total_images}] "
                f"stored={stats['total_stored']:,} features, "
                f"speed={speed:.1f} img/s, "
                f"ETA={eta / 3600:.1f}h"
            )

        # Periodic checkpoint
        if (idx + 1) % save_interval == 0 and len(all_query_features) > 0:
            _save_checkpoint(
                save_dir, all_query_features, all_metadata,
                stats, checkpoint_idx=idx + 1,
            )

    # Final save
    elapsed_total = time.time() - start_time
    stats["elapsed_seconds"] = elapsed_total
    stats["features_per_image"] = (
        stats["total_stored"] / stats["processed"]
        if stats["processed"] > 0 else 0
    )

    logger.info(
        f"Extraction complete: {stats['total_stored']:,} features "
        f"from {stats['processed']:,} images in {elapsed_total / 3600:.1f}h"
    )

    if len(all_query_features) > 0:
        _save_final(save_dir, all_query_features, all_metadata, stats)

    return stats


def _save_checkpoint(
    save_dir: Path,
    features: list[np.ndarray],
    metadata: list[dict],
    stats: dict,
    checkpoint_idx: int,
) -> None:
    """Save intermediate checkpoint."""
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    feat_array = np.stack(features)
    np.save(ckpt_dir / f"features_{checkpoint_idx}.npy", feat_array)

    with open(ckpt_dir / f"metadata_{checkpoint_idx}.json", "w") as f:
        json.dump({"metadata": metadata, "stats": stats}, f)

    logger.info(f"Checkpoint saved: {len(features):,} features at idx {checkpoint_idx}")


def _save_final(
    save_dir: Path,
    features: list[np.ndarray],
    metadata: list[dict],
    stats: dict,
) -> None:
    """Save final features + build FAISS index."""
    import faiss

    feat_array = np.stack(features).astype(np.float32)
    dim = feat_array.shape[1]

    logger.info(f"Building FAISS index: {feat_array.shape[0]:,} vectors, dim={dim}")

    # L2-normalize for cosine similarity
    faiss.normalize_L2(feat_array)

    # Build flat index (exact search)
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after L2-norm
    index.add(feat_array)

    # Save index
    faiss.write_index(index, str(save_dir / "index.faiss"))
    logger.info(f"FAISS index saved: {save_dir / 'index.faiss'}")

    # Save features (for potential re-indexing)
    np.save(save_dir / "features.npy", feat_array)

    # Save metadata
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "metadata": metadata,
            "stats": stats,
            "dim": dim,
            "num_entries": len(metadata),
        }, f, indent=2)

    logger.info(
        f"Memory bank saved: {len(metadata):,} entries, "
        f"dim={dim}, index_type=FlatIP"
    )

    # Print class distribution
    logger.info("Class distribution (top 20):")
    sorted_classes = sorted(
        stats["class_counts"].items(), key=lambda x: -x[1]
    )
    for cls, count in sorted_classes[:20]:
        logger.info(f"  {cls:20s}: {count:,}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    """Build memory bank from COCO train images."""
    import argparse

    parser = argparse.ArgumentParser(description="Build memory bank (Phase A)")
    parser.add_argument("--data-dir", type=str, default="./data/coco",
                        help="COCO dataset root")
    parser.add_argument("--img-dir", type=str, default="train2017",
                        help="Image subdirectory")
    parser.add_argument("--output-dir", type=str, default="./outputs/memory_bank",
                        help="Output directory for features and FAISS index")
    parser.add_argument("--model-id", type=str,
                        default="IDEA-Research/grounding-dino-base",
                        help="HuggingFace model ID")
    parser.add_argument("--min-score", type=float, default=0.4,
                        help="Minimum detection score to store")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to process (None=all)")
    parser.add_argument("--save-interval", type=int, default=5000,
                        help="Save checkpoint every N images")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from image index")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    args = parser.parse_args()

    # Setup device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("Phase A: Memory Bank Construction")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Data: {args.data_dir}/{args.img_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Min score: {args.min_score}")
    logger.info(f"Max images: {args.max_images or 'all'}")

    # Load model
    model = GroundingDINOWrapper(
        model_id=args.model_id,
        device=device,
        dtype=torch.float16,
        box_threshold=0.15,     # Lower threshold to catch more proposals
        text_threshold=0.15,
    )

    # Build text prompt from all 80 COCO classes
    text_prompt = build_text_prompt(COCO_CLASSES)
    logger.info(f"Text prompt ({len(COCO_CLASSES)} classes): {text_prompt[:100]}...")

    # Run extraction
    img_dir = Path(args.data_dir) / args.img_dir
    stats = extract_features_from_dataset(
        model=model,
        img_dir=img_dir,
        text_prompt=text_prompt,
        min_score=args.min_score,
        max_images=args.max_images,
        save_dir=Path(args.output_dir),
        save_interval=args.save_interval,
        resume_from=args.resume_from,
    )

    logger.info("=" * 60)
    logger.info("Phase A Complete!")
    logger.info(f"Total features stored: {stats['total_stored']:,}")
    logger.info(f"Images processed: {stats['processed']:,}")
    logger.info(f"Features per image: {stats['features_per_image']:.1f}")
    logger.info(f"Time: {stats.get('elapsed_seconds', 0) / 3600:.1f}h")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
