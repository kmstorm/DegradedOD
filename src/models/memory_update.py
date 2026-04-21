"""
MODULE 5: Memory Update — Online EMA update of the memory bank.

Gradually incorporates high-confidence target-domain features back
into the memory bank for test-time adaptation without catastrophic forgetting.

Strategy:
  - Gate: Only insert features with confidence > τ_update
  - Update rule: EMA update of class prototypes
  - Capacity control: FIFO queue per class with max size
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from src.models.memory_bank import MemoryBank

logger = logging.getLogger(__name__)


class MemoryUpdater:
    """
    Online memory bank updater for test-time adaptation.

    Monitors prediction confidence and inserts high-quality target-domain
    features back into the memory bank, enabling gradual domain adaptation.

    Args:
        confidence_threshold: Minimum score to trigger memory insertion.
        ema_momentum: EMA coefficient for prototype updates (higher = slower change).
        max_queue_size: Maximum instance entries per class in the queue.
        enabled: Whether updates are active.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        ema_momentum: float = 0.99,
        max_queue_size: int = 1000,
        enabled: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.ema_momentum = ema_momentum
        self.max_queue_size = max_queue_size
        self.enabled = enabled

        # Statistics
        self._update_count = 0
        self._total_candidates = 0

    def update(
        self,
        memory_bank: MemoryBank,
        features: torch.Tensor,
        class_names: list[str],
        scores: torch.Tensor | list[float],
        source_info: str = "target_domain",
    ) -> int:
        """
        Conditionally update the memory bank with high-confidence target features.

        Args:
            memory_bank: The memory bank to update.
            features: (N, D) refined feature vectors.
            class_names: (N,) predicted class labels.
            scores: (N,) prediction confidence scores.
            source_info: Identifier for the source of these features.

        Returns:
            Number of entries actually inserted.
        """
        if not self.enabled:
            return 0

        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy()
        elif isinstance(scores, list):
            scores_np = np.array(scores)
        else:
            scores_np = scores

        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features

        self._total_candidates += len(scores_np)

        # Filter by confidence threshold
        high_conf_mask = scores_np >= self.confidence_threshold
        n_high_conf = high_conf_mask.sum()

        if n_high_conf == 0:
            return 0

        # Select high-confidence entries
        selected_features = features_np[high_conf_mask]
        selected_classes = [c for c, m in zip(class_names, high_conf_mask) if m]
        selected_scores = scores_np[high_conf_mask]

        # Update class prototypes via EMA
        for feat, cls in zip(selected_features, selected_classes):
            self._ema_update_prototype(memory_bank, cls, feat)

        # Add as instance entries
        added = memory_bank.add(
            features=selected_features,
            class_names=selected_classes,
            scores=selected_scores.tolist(),
            source_image=source_info,
            entry_type="instance",
        )

        self._update_count += added

        if added > 0:
            logger.debug(
                f"Memory update: {added}/{n_high_conf} entries inserted "
                f"(threshold={self.confidence_threshold:.2f})"
            )

        return added

    def _ema_update_prototype(
        self,
        memory_bank: MemoryBank,
        class_name: str,
        feature: np.ndarray,
    ) -> None:
        """
        Update the class prototype using Exponential Moving Average.

        p_new = α * p_old + (1 - α) * f_target

        This ensures smooth adaptation without abrupt changes.
        """
        current_proto = memory_bank.get_prototype(class_name)
        if current_proto is None:
            return  # no prototype to update (class not in memory bank)

        α = self.ema_momentum
        updated = α * current_proto + (1 - α) * feature

        # Normalize if the bank uses cosine similarity
        if memory_bank.similarity == "cosine":
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated = updated / norm

        memory_bank._prototypes[class_name] = updated

    @property
    def stats(self) -> dict[str, int | float]:
        """Return update statistics."""
        return {
            "total_candidates": self._total_candidates,
            "total_inserted": self._update_count,
            "insertion_rate": (
                self._update_count / max(self._total_candidates, 1)
            ),
        }

    def reset_stats(self) -> None:
        """Reset update statistics."""
        self._update_count = 0
        self._total_candidates = 0
