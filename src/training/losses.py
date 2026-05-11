"""
Loss functions for MODD Phase B training.

Three loss components:
  1. DetectionLoss — DETR-style box regression (L1 + GIoU) + focal classification
  2. ContrastiveLoss — InfoNCE to align degraded features to clean prototypes
  3. RetrievalLoss — Triplet margin to ensure retrieved memories are relevant

All losses accept batched inputs and return scalar tensors.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Utility: Generalized IoU
# ──────────────────────────────────────────────────────────────────────

def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (N, 4) xyxy
        boxes2: (N, 4) xyxy

    Returns:
        (N,) GIoU values in [-1, 1]
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter_area

    iou = inter_area / (union + 1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


# ──────────────────────────────────────────────────────────────────────
# 1. Detection Loss (DETR-style)
# ──────────────────────────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """
    DETR-style detection loss combining:
      - L1 box regression loss
      - Generalized IoU loss
      - Focal classification loss (optional, for score refinement)

    Args:
        l1_weight: Weight for L1 box loss.
        giou_weight: Weight for GIoU loss.
        focal_alpha: Focal loss alpha parameter.
        focal_gamma: Focal loss gamma parameter.
    """

    def __init__(
        self,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        pred_scores: torch.Tensor | None = None,
        gt_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_boxes: (N, 4) predicted boxes xyxy
            gt_boxes: (N, 4) ground-truth boxes xyxy (matched)
            pred_scores: (N,) predicted confidence scores
            gt_scores: (N,) target scores (e.g., IoU with GT)

        Returns:
            Dict with "l1", "giou", "score", and "total" losses.
        """
        losses = {}

        if pred_boxes.numel() == 0:
            zero = pred_boxes.sum() * 0.0
            return {"l1": zero, "giou": zero, "score": zero, "total": zero}

        # L1 loss
        l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")
        losses["l1"] = l1_loss

        # GIoU loss
        giou = generalized_iou(pred_boxes, gt_boxes)
        giou_loss = (1 - giou).mean()
        losses["giou"] = giou_loss

        # Focal loss on scores (if provided)
        if pred_scores is not None and gt_scores is not None:
            score_loss = self._focal_loss(pred_scores, gt_scores)
            losses["score"] = score_loss
        else:
            losses["score"] = pred_boxes.sum() * 0.0

        # Total
        losses["total"] = (
            self.l1_weight * losses["l1"]
            + self.giou_weight * losses["giou"]
            + losses["score"]
        )

        return losses

    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Binary focal loss for score refinement (AMP-safe)."""
        pred = pred.float().clamp(1e-6, 1 - 1e-6)
        target = target.float()
        # Disable autocast — BCE is unsafe under AMP
        with torch.amp.autocast("cuda", enabled=False):
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
        return (focal_weight * bce).mean()


# ──────────────────────────────────────────────────────────────────────
# 2. Contrastive Loss (InfoNCE)
# ──────────────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for aligning degraded proposal features
    to their clean-domain counterparts.

    Given a batch of (degraded_feature, clean_feature) pairs, the loss
    encourages each degraded feature to be similar to its matched clean
    feature and dissimilar to all other clean features in the batch.

    Args:
        temperature: Softmax temperature (lower = sharper distribution).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        degraded_features: torch.Tensor,
        clean_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            degraded_features: (N, D) L2-normalized degraded proposal features
            clean_features: (N, D) L2-normalized clean proposal features

        Returns:
            Scalar InfoNCE loss.
        """
        if degraded_features.shape[0] == 0:
            return degraded_features.sum() * 0.0

        # Normalize
        degraded_features = F.normalize(degraded_features, dim=-1)
        clean_features = F.normalize(clean_features, dim=-1)

        # Similarity matrix: (N, N)
        logits = torch.mm(degraded_features, clean_features.t()) / self.temperature

        # Labels: diagonal is positive pair
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Cross-entropy loss (symmetric)
        loss_d2c = F.cross_entropy(logits, labels)
        loss_c2d = F.cross_entropy(logits.t(), labels)

        return (loss_d2c + loss_c2d) / 2


# ──────────────────────────────────────────────────────────────────────
# 3. Retrieval Loss (Triplet Margin)
# ──────────────────────────────────────────────────────────────────────

class RetrievalLoss(nn.Module):
    """
    Triplet margin loss to improve retrieval quality.

    For each degraded proposal (anchor), the positive is the
    closest same-class memory entry, and the negative is a
    different-class entry.

    Args:
        margin: Triplet margin.
        distance: "cosine" or "euclidean".
    """

    def __init__(
        self,
        margin: float = 0.3,
        distance: str = "cosine",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: (N, D) degraded proposal features (queries)
            positive: (N, D) retrieved same-class memory features
            negative: (N, D) retrieved different-class memory features

        Returns:
            Scalar triplet loss.
        """
        if anchor.shape[0] == 0:
            return anchor.sum() * 0.0

        if self.distance == "cosine":
            # Cosine distance: 1 - cosine_similarity
            pos_dist = 1.0 - F.cosine_similarity(anchor, positive)
            neg_dist = 1.0 - F.cosine_similarity(anchor, negative)
        else:
            # Euclidean distance
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────
# Composite Loss
# ──────────────────────────────────────────────────────────────────────

class MODDLoss(nn.Module):
    """
    Combined loss for Phase B training.

    Aggregates DetectionLoss + ContrastiveLoss + RetrievalLoss with
    configurable weights matching phase_b_modules.yaml.

    Args:
        detection_weight: Weight for detection loss.
        contrastive_weight: Weight for contrastive loss.
        retrieval_weight: Weight for retrieval loss.
    """

    def __init__(
        self,
        detection_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        retrieval_weight: float = 0.3,
        temperature: float = 0.07,
        triplet_margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.detection_weight = detection_weight
        self.contrastive_weight = contrastive_weight
        self.retrieval_weight = retrieval_weight

        self.detection_loss = DetectionLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.retrieval_loss = RetrievalLoss(margin=triplet_margin)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        pred_scores: torch.Tensor | None = None,
        gt_scores: torch.Tensor | None = None,
        degraded_features: torch.Tensor | None = None,
        clean_features: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
        positive_features: torch.Tensor | None = None,
        negative_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all losses and return weighted total.

        Returns:
            Dict with individual losses and "total".
        """
        result = {}

        # Detection loss
        det_losses = self.detection_loss(pred_boxes, gt_boxes, pred_scores, gt_scores)
        result["det_l1"] = det_losses["l1"]
        result["det_giou"] = det_losses["giou"]
        result["det_score"] = det_losses["score"]
        result["det_total"] = det_losses["total"]

        # Contrastive loss
        if degraded_features is not None and clean_features is not None:
            result["contrastive"] = self.contrastive_loss(
                degraded_features, clean_features
            )
        else:
            result["contrastive"] = pred_boxes.sum() * 0.0

        # Retrieval loss
        if (
            anchor_features is not None
            and positive_features is not None
            and negative_features is not None
        ):
            result["retrieval"] = self.retrieval_loss(
                anchor_features, positive_features, negative_features
            )
        else:
            result["retrieval"] = pred_boxes.sum() * 0.0

        # Weighted total
        result["total"] = (
            self.detection_weight * result["det_total"]
            + self.contrastive_weight * result["contrastive"]
            + self.retrieval_weight * result["retrieval"]
        )

        return result
