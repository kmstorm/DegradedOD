"""
Visualization utilities for MODD detection results.

Provides functions to:
  - Draw detection boxes and labels on images
  - Visualize retrieved memory supports alongside proposals
  - Create side-by-side clean vs. degraded comparisons
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


# Color palette (BGR) for drawing boxes — visually distinct
COLORS = [
    (255, 76, 76),    # coral
    (76, 255, 76),    # green
    (76, 76, 255),    # blue
    (255, 255, 76),   # yellow
    (255, 76, 255),   # magenta
    (76, 255, 255),   # cyan
    (255, 165, 76),   # orange
    (165, 76, 255),   # purple
    (76, 255, 165),   # teal
    (255, 76, 165),   # pink
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray | torch.Tensor,
    labels: list[str],
    scores: np.ndarray | torch.Tensor | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.

    Args:
        image: (H, W, 3) image in RGB uint8 or float32 [0,1].
        boxes: (N, 4) boxes in xyxy format.
        labels: (N,) text labels.
        scores: Optional (N,) confidence scores.
        thickness: Box line thickness.
        font_scale: Text font scale.

    Returns:
        Annotated image as uint8.
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # Convert to uint8 if needed
    vis = image.copy()
    if vis.dtype == np.float32 or vis.dtype == np.float64:
        vis = (vis * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB → BGR for OpenCV
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # Build class → color mapping
    unique_labels = sorted(set(labels))
    color_map = {l: COLORS[i % len(COLORS)] for i, l in enumerate(unique_labels)}

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        color = color_map[label]

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        if scores is not None:
            text = f"{label} {scores[i]:.2f}"
        else:
            text = label

        # Draw label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            vis, text, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )

    # Convert BGR → RGB
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def create_comparison(
    clean: np.ndarray,
    degraded: np.ndarray,
    detections_clean: dict | None = None,
    detections_degraded: dict | None = None,
    title_clean: str = "Clean",
    title_degraded: str = "Degraded",
) -> np.ndarray:
    """
    Create a side-by-side comparison of clean and degraded images.

    Args:
        clean: (H, W, 3) clean image.
        degraded: (H, W, 3) degraded image.
        detections_clean: Optional dict with "boxes", "labels", "scores".
        detections_degraded: Optional dict with "boxes", "labels", "scores".
        title_clean: Title for clean panel.
        title_degraded: Title for degraded panel.

    Returns:
        (H, W*2 + gap, 3) concatenated comparison image.
    """
    # Draw detections if provided
    if detections_clean:
        clean = draw_detections(
            clean,
            detections_clean["boxes"],
            detections_clean["labels"],
            detections_clean.get("scores"),
        )
    if detections_degraded:
        degraded = draw_detections(
            degraded,
            detections_degraded["boxes"],
            detections_degraded["labels"],
            detections_degraded.get("scores"),
        )

    # Ensure uint8
    for img_ref_name in ["clean", "degraded"]:
        img = locals()[img_ref_name]
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
            if img_ref_name == "clean":
                clean = img
            else:
                degraded = img

    # Resize to same height
    h = max(clean.shape[0], degraded.shape[0])
    if clean.shape[0] != h:
        scale = h / clean.shape[0]
        clean = cv2.resize(clean, (int(clean.shape[1] * scale), h))
    if degraded.shape[0] != h:
        scale = h / degraded.shape[0]
        degraded = cv2.resize(degraded, (int(degraded.shape[1] * scale), h))

    # Create gap
    gap = np.ones((h, 10, 3), dtype=np.uint8) * 200

    # Concatenate
    comparison = np.concatenate([clean, gap, degraded], axis=1)

    return comparison


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save image to disk (handles RGB→BGR conversion)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
