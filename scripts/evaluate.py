"""
Evaluation script for MODD.

Runs the trained model on target-domain datasets (ExDark, RTTS) and
computes detection metrics: mAP@0.5, mAP@[0.5:0.95], per-class AP.

Usage:
    # Evaluate G-DINO baseline (no adaptation) on ExDark
    CUDA_VISIBLE_DEVICES=5 python scripts/evaluate.py \
        --dataset exdark --data-dir ./data/exdark

    # Evaluate with trained adaptation modules
    CUDA_VISIBLE_DEVICES=5 python scripts/evaluate.py \
        --dataset exdark --data-dir ./data/exdark \
        --checkpoint ./outputs/checkpoints/best_phase_b.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gdino_wrapper import GroundingDINOWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ──────────────────────────────────────────────────────────────────────
# ExDark evaluation helper1s
# ──────────────────────────────────────────────────────────────────────

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]

# ExDark → COCO class mapping
EXDARK_TO_COCO = {
    "Bicycle": "bicycle",
    "Boat": "boat",
    "Bottle": "bottle",
    "Bus": "bus",
    "Car": "car",
    "Cat": "cat",
    "Chair": "chair",
    "Cup": "cup",
    "Dog": "dog",
    "Motorbike": "motorcycle",
    "People": "person",
    "Table": "dining table",
}


def load_exdark_annotations(ann_dir: Path, img_file: str, class_name: str) -> list[dict]:
    """
    Load ExDark annotations for an image.

    ExDark annotation format per line:
        % bbGt version=3    (header, skip)
        ClassName left top width height 0 0 0 0 0 0 0

    Returns list of {"class": str, "bbox": [x1, y1, x2, y2]}
    """
    ann_file = ann_dir / class_name / (img_file + ".txt")
    annotations = []

    if not ann_file.exists():
        return annotations

    with open(ann_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = parts[0]
            left, top, width, height = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            x1, y1 = left, top
            x2, y2 = left + width, top + height

            coco_cls = EXDARK_TO_COCO.get(cls, cls.lower())
            annotations.append({
                "class": coco_cls,
                "bbox": [x1, y1, x2, y2],
            })

    return annotations


def compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision using 11-point interpolation."""
    if not precisions or not recalls:
        return 0.0

    # Add sentinel values
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = 0.0
        for r, pr in zip(recalls, precisions):
            if r >= t:
                p = max(p, pr)
        ap += p / 11.0

    return ap


def evaluate_detections(
    all_detections: list[dict],
    all_gts: list[dict],
    iou_threshold: float = 0.5,
    classes: list[str] | None = None,
) -> dict:
    """
    Compute per-class AP and mAP.

    Args:
        all_detections: List of {"class": str, "score": float, "bbox": [x1,y1,x2,y2], "image_id": str}
        all_gts: List of {"class": str, "bbox": [x1,y1,x2,y2], "image_id": str}
        iou_threshold: IoU threshold for matching.
        classes: List of class names to evaluate. If None, auto-discover.

    Returns:
        Dict with per-class AP and overall mAP.
    """
    if classes is None:
        classes = sorted(set(d["class"] for d in all_gts))

    results = {}
    aps = []

    for cls in classes:
        # Filter detections and GTs for this class
        cls_dets = [d for d in all_detections if d["class"] == cls]
        cls_gts = [g for g in all_gts if g["class"] == cls]

        if len(cls_gts) == 0:
            results[cls] = {"ap": 0.0, "num_gt": 0, "num_det": len(cls_dets)}
            continue

        # Sort detections by score (descending)
        cls_dets.sort(key=lambda x: -x["score"])

        # Group GTs by image
        gt_by_image = {}
        for g in cls_gts:
            img_id = g["image_id"]
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append({"bbox": g["bbox"], "matched": False})

        # Match detections to GTs
        tp = []
        fp = []

        for det in cls_dets:
            img_id = det["image_id"]
            img_gts = gt_by_image.get(img_id, [])

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(img_gts):
                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0 and not img_gts[best_gt_idx]["matched"]:
                tp.append(1)
                fp.append(0)
                img_gts[best_gt_idx]["matched"] = True
            else:
                tp.append(0)
                fp.append(1)

        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = (tp_cumsum / (tp_cumsum + fp_cumsum)).tolist()
        recalls = (tp_cumsum / len(cls_gts)).tolist()

        ap = compute_ap(precisions, recalls)
        aps.append(ap)

        results[cls] = {
            "ap": ap,
            "num_gt": len(cls_gts),
            "num_det": len(cls_dets),
            "precision": precisions[-1] if precisions else 0.0,
            "recall": recalls[-1] if recalls else 0.0,
        }

    mAP = np.mean(aps) if aps else 0.0
    results["mAP"] = mAP

    return results


# ──────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_exdark(
    model: GroundingDINOWrapper,
    data_dir: str,
    max_images: int | None = None,
) -> dict:
    """
    Evaluate G-DINO on the ExDark dataset.

    Args:
        model: GroundingDINOWrapper (with or without adaptation modules).
        data_dir: Path to ExDark root (containing images/ and annotations/).
        max_images: Limit evaluation to N images.

    Returns:
        Dict with mAP and per-class metrics.
    """
    data_dir = Path(data_dir)
    img_root = data_dir / "images"
    ann_root = data_dir / "annotations"

    # Build text prompt from ExDark classes (mapped to COCO names)
    coco_names = list(EXDARK_TO_COCO.values())
    text_prompt = " . ".join(coco_names) + " ."

    all_detections = []
    all_gts = []
    total_images = 0
    skipped = 0

    logger.info(f"Evaluating on ExDark: {data_dir}")
    logger.info(f"Text prompt: {text_prompt}")

    start_time = time.time()

    for class_name in EXDARK_CLASSES:
        class_img_dir = img_root / class_name
        if not class_img_dir.exists():
            logger.warning(f"Missing class directory: {class_img_dir}")
            continue

        img_files = sorted([
            f for f in os.listdir(class_img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        for img_file in img_files:
            if max_images is not None and total_images >= max_images:
                break

            img_path = class_img_dir / img_file
            image_id = f"{class_name}/{img_file}"

            try:
                img = Image.open(str(img_path)).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                skipped += 1
                continue

            # Load GT annotations
            gt_anns = load_exdark_annotations(ann_root, img_file, class_name)
            for ann in gt_anns:
                ann["image_id"] = image_id
            all_gts.extend(gt_anns)

            # Run detection
            try:
                result = model.detect(img, text_prompt=text_prompt)
            except Exception as e:
                logger.warning(f"Inference failed on {img_path}: {e}")
                skipped += 1
                continue

            for i in range(len(result)):
                all_detections.append({
                    "class": result.labels[i],
                    "score": result.scores[i].item(),
                    "bbox": result.boxes[i].tolist(),
                    "image_id": image_id,
                })

            total_images += 1

            if total_images % 500 == 0:
                elapsed = time.time() - start_time
                speed = total_images / elapsed
                logger.info(
                    f"  [{total_images} images] "
                    f"{len(all_detections)} detections, "
                    f"{speed:.1f} img/s"
                )

        if max_images is not None and total_images >= max_images:
            break

    elapsed = time.time() - start_time
    logger.info(
        f"Inference complete: {total_images} images, "
        f"{len(all_detections)} detections, "
        f"{skipped} skipped, {elapsed:.1f}s"
    )

    # Compute metrics
    coco_classes = sorted(set(EXDARK_TO_COCO.values()))

    # mAP@0.5
    results_50 = evaluate_detections(
        all_detections, all_gts,
        iou_threshold=0.5, classes=coco_classes,
    )

    # mAP@[0.5:0.95]
    aps_multi = []
    for iou_thr in np.arange(0.5, 1.0, 0.05):
        r = evaluate_detections(
            all_detections, all_gts,
            iou_threshold=iou_thr, classes=coco_classes,
        )
        aps_multi.append(r["mAP"])
    mAP_50_95 = np.mean(aps_multi)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"ExDark Evaluation Results ({total_images} images)")
    logger.info("=" * 60)
    logger.info(f"  mAP@0.5:        {results_50['mAP']:.4f}")
    logger.info(f"  mAP@[0.5:0.95]: {mAP_50_95:.4f}")
    logger.info("")
    logger.info("  Per-class AP@0.5:")
    for cls in coco_classes:
        if cls in results_50:
            r = results_50[cls]
            logger.info(
                f"    {cls:20s}  AP={r['ap']:.4f}  "
                f"GT={r['num_gt']}  Det={r['num_det']}  "
                f"P={r.get('precision', 0):.3f}  R={r.get('recall', 0):.3f}"
            )
    logger.info("=" * 60)

    return {
        "mAP_50": results_50["mAP"],
        "mAP_50_95": mAP_50_95,
        "per_class": results_50,
        "total_images": total_images,
        "total_detections": len(all_detections),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MODD on target domains")
    parser.add_argument("--dataset", type=str, default="exdark",
                        choices=["exdark", "rtts"],
                        help="Target dataset to evaluate on")
    parser.add_argument("--data-dir", type=str, default="./data/exdark",
                        help="Dataset root directory")
    parser.add_argument("--model-id", type=str,
                        default="IDEA-Research/grounding-dino-base",
                        help="G-DINO model ID")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained MODD adaptation checkpoint")
    parser.add_argument("--memory-bank", type=str, default="./outputs/memory_bank_final",
                        help="Path to memory bank directory")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to N images")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--baseline-threshold", type=float, default=0.50,
                        help="Final output threshold for baseline G-DINO")
    parser.add_argument("--modd-output-threshold", type=float, default=0.50,
                        help="Final output threshold for MODD direct/refined outputs")
    parser.add_argument("--modd-low-threshold", type=float, default=0.10,
                        help="Discard detections below this score before retrieval")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    use_modd = args.checkpoint is not None

    if use_modd:
        # ── MODD evaluation (with trained adaptation modules) ──
        from src.models.memory_bank import MemoryBank
        from src.models.modd_detector import MODDConfig, MODDDetector

        logger.info(f"Loading MODDDetector with checkpoint: {args.checkpoint}")

        # Load memory bank
        memory_bank = MemoryBank.load(args.memory_bank)
        logger.info(f"Memory bank: {memory_bank.num_entries} entries")

        # Create detector and load trained weights
        modd_config = MODDConfig(
            model_id=args.model_id,
            dtype="float16",
            box_threshold=args.modd_output_threshold,
            low_conf_threshold=args.modd_low_threshold,
            high_conf_threshold=args.modd_output_threshold,
        )
        detector = MODDDetector(modd_config)

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        detector.load_state_dict(checkpoint["model_state_dict"], strict=False)
        detector = detector.to(device)
        detector.eval()
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
            f"loss={checkpoint.get('best_metric', '?')}"
        )

        # Evaluate with MODD pipeline
        results = evaluate_exdark_modd(
            detector, memory_bank, args.data_dir, max_images=args.max_images,
        )
    else:
        # ── Baseline G-DINO evaluation (no adaptation) ──
        logger.info(f"Loading G-DINO baseline: {args.model_id}")
        model = GroundingDINOWrapper(
            model_id=args.model_id,
            device=device,
            dtype=torch.float16,
            box_threshold=args.baseline_threshold,
            text_threshold=0.20,
        )

        if args.dataset == "exdark":
            results = evaluate_exdark(model, args.data_dir, max_images=args.max_images)
        else:
            logger.error(f"Dataset '{args.dataset}' not yet supported")
            return

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


@torch.no_grad()
def evaluate_exdark_modd(
    detector,
    memory_bank,
    data_dir: str,
    max_images: int | None = None,
) -> dict:
    """
    Evaluate MODD (with trained adaptation modules) on ExDark.

    Uses full pipeline: G-DINO → confidence split → retrieve → refine → merge.
    """
    data_dir = Path(data_dir)
    img_root = data_dir / "images"
    ann_root = data_dir / "annotations"

    coco_names = list(EXDARK_TO_COCO.values())
    text_prompt = " . ".join(coco_names) + " ."

    all_detections = []
    all_gts = []
    total_images = 0
    skipped = 0

    logger.info(f"Evaluating MODD on ExDark: {data_dir}")
    logger.info(f"Text prompt: {text_prompt}")

    start_time = time.time()

    for class_name in EXDARK_CLASSES:
        class_img_dir = img_root / class_name
        if not class_img_dir.exists():
            logger.warning(f"Missing class directory: {class_img_dir}")
            continue

        img_files = sorted([
            f for f in os.listdir(class_img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        for img_file in img_files:
            if max_images is not None and total_images >= max_images:
                break

            img_path = class_img_dir / img_file
            image_id = f"{class_name}/{img_file}"

            try:
                img = Image.open(str(img_path)).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                skipped += 1
                continue

            # Load GT annotations
            gt_anns = load_exdark_annotations(ann_root, img_file, class_name)
            for ann in gt_anns:
                ann["image_id"] = image_id
            all_gts.extend(gt_anns)

            # Run MODD detection (full pipeline with memory bank)
            try:
                result = detector.detect(img, text_prompt, memory_bank)
            except Exception as e:
                logger.warning(f"MODD inference failed on {img_path}: {e}")
                skipped += 1
                continue

            for i in range(len(result)):
                all_detections.append({
                    "class": result.labels[i],
                    "score": result.scores[i].item(),
                    "bbox": result.boxes[i].tolist(),
                    "image_id": image_id,
                })

            total_images += 1

            if total_images % 500 == 0:
                elapsed = time.time() - start_time
                speed = total_images / elapsed
                logger.info(
                    f"  [{total_images} images] "
                    f"{len(all_detections)} detections, "
                    f"{speed:.1f} img/s"
                )

        if max_images is not None and total_images >= max_images:
            break

    elapsed = time.time() - start_time
    logger.info(
        f"MODD inference complete: {total_images} images, "
        f"{len(all_detections)} detections, "
        f"{skipped} skipped, {elapsed:.1f}s"
    )

    # Compute metrics
    coco_classes = sorted(set(EXDARK_TO_COCO.values()))

    results_50 = evaluate_detections(
        all_detections, all_gts,
        iou_threshold=0.5, classes=coco_classes,
    )

    aps_multi = []
    for iou_thr in np.arange(0.5, 1.0, 0.05):
        r = evaluate_detections(
            all_detections, all_gts,
            iou_threshold=iou_thr, classes=coco_classes,
        )
        aps_multi.append(r["mAP"])
    mAP_50_95 = np.mean(aps_multi)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"MODD ExDark Results ({total_images} images)")
    logger.info("=" * 60)
    logger.info(f"  mAP@0.5:        {results_50['mAP']:.4f}")
    logger.info(f"  mAP@[0.5:0.95]: {mAP_50_95:.4f}")
    logger.info("")
    logger.info("  Per-class AP@0.5:")
    for cls in coco_classes:
        if cls in results_50:
            r = results_50[cls]
            logger.info(
                f"    {cls:20s}  AP={r['ap']:.4f}  "
                f"GT={r['num_gt']}  Det={r['num_det']}  "
                f"P={r.get('precision', 0):.3f}  R={r.get('recall', 0):.3f}"
            )
    logger.info("=" * 60)

    return {
        "mAP_50": results_50["mAP"],
        "mAP_50_95": mAP_50_95,
        "per_class": results_50,
        "total_images": total_images,
        "total_detections": len(all_detections),
    }


if __name__ == "__main__":
    main()
