"""
Evaluation metrics for MODD.

Provides:
  - mAP computation (wrapping pycocotools)
  - Per-class AP breakdown
  - Delta-mAP (clean vs. degraded performance gap)
  - Retrieval precision@K
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of xyxy boxes.

    Args:
        boxes1: (N, 4) boxes.
        boxes2: (M, 4) boxes.

    Returns:
        (N, M) IoU matrix.
    """
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-8)

    return iou


def compute_ap(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision for a single class in a single image.

    Args:
        pred_boxes: (N, 4) predicted boxes in xyxy.
        pred_scores: (N,) confidence scores.
        gt_boxes: (M, 4) ground truth boxes in xyxy.
        iou_threshold: IoU threshold for matching.

    Returns:
        AP value.
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by score (descending)
    sorted_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    # Compute IoU
    iou_matrix = compute_iou(pred_boxes, gt_boxes)

    # Match predictions to GT (greedy)
    n_gt = len(gt_boxes)
    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):
        best_iou = 0
        best_gt = -1
        for j in range(n_gt):
            if gt_matched[j]:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 11-point interpolation AP
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_recall = precision[recall >= t]
        if len(precisions_at_recall) > 0:
            ap += precisions_at_recall.max()
    ap /= 11.0

    return ap


class DetectionMetrics:
    """
    Accumulates detection results and computes aggregate metrics.

    Usage:
        metrics = DetectionMetrics()
        for image_results in dataset:
            metrics.add(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels,
            )
        results = metrics.compute()
        # results = {"mAP_50": 0.65, "mAP_50_95": 0.42, "per_class_ap": {...}}
    """

    def __init__(self, iou_thresholds: list[float] | None = None) -> None:
        self.iou_thresholds = iou_thresholds or [0.5]
        self._predictions: list[dict] = []
        self._ground_truths: list[dict] = []

    def add(
        self,
        pred_boxes: np.ndarray | torch.Tensor,
        pred_scores: np.ndarray | torch.Tensor,
        pred_labels: list[str],
        gt_boxes: np.ndarray | torch.Tensor,
        gt_labels: list[str],
    ) -> None:
        """Add predictions and ground truths for one image."""
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.detach().cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.detach().cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.detach().cpu().numpy()

        self._predictions.append({
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        })
        self._ground_truths.append({
            "boxes": gt_boxes,
            "labels": gt_labels,
        })

    def compute(self) -> dict[str, Any]:
        """Compute aggregate metrics across all accumulated images."""
        all_classes = set()
        for gt in self._ground_truths:
            all_classes.update(gt["labels"])
        for pred in self._predictions:
            all_classes.update(pred["labels"])

        results = {}

        for iou_thr in self.iou_thresholds:
            per_class_ap = {}

            for cls in sorted(all_classes):
                aps = []
                for pred, gt in zip(self._predictions, self._ground_truths):
                    # Filter to current class
                    pred_mask = [l == cls for l in pred["labels"]]
                    gt_mask = [l == cls for l in gt["labels"]]

                    p_boxes = pred["boxes"][pred_mask] if any(pred_mask) else np.empty((0, 4))
                    p_scores = pred["scores"][pred_mask] if any(pred_mask) else np.empty(0)
                    g_boxes = gt["boxes"][gt_mask] if any(gt_mask) else np.empty((0, 4))

                    if len(g_boxes) == 0 and len(p_boxes) == 0:
                        continue

                    ap = compute_ap(p_boxes, p_scores, g_boxes, iou_thr)
                    aps.append(ap)

                if aps:
                    per_class_ap[cls] = np.mean(aps)

            thr_key = f"{int(iou_thr * 100)}"
            results[f"mAP_{thr_key}"] = (
                np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
            )
            results[f"per_class_ap_{thr_key}"] = per_class_ap

        return results

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._predictions.clear()
        self._ground_truths.clear()


def retrieval_precision_at_k(
    retrieved_classes: list[str],
    query_class: str,
    k: int | None = None,
) -> float:
    """
    Compute precision@K for retrieval quality evaluation.

    Args:
        retrieved_classes: List of class names of retrieved supports.
        query_class: The class of the query proposal.
        k: Evaluate at top-K. If None, use all retrieved.

    Returns:
        Precision@K value (0 to 1).
    """
    if k is not None:
        retrieved_classes = retrieved_classes[:k]

    if len(retrieved_classes) == 0:
        return 0.0

    correct = sum(1 for c in retrieved_classes if c == query_class)
    return correct / len(retrieved_classes)


def delta_map(clean_map: float, degraded_map: float) -> dict[str, float]:
    """
    Compute the performance gap between clean and degraded conditions.

    Args:
        clean_map: mAP on clean images.
        degraded_map: mAP on degraded images.

    Returns:
        Dict with absolute drop and relative drop.
    """
    abs_drop = clean_map - degraded_map
    rel_drop = abs_drop / max(clean_map, 1e-8) * 100
    return {
        "absolute_drop": abs_drop,
        "relative_drop_pct": rel_drop,
        "clean_map": clean_map,
        "degraded_map": degraded_map,
    }
