"""
Refined Detection Head — lightweight MLP for final predictions.

Takes refined feature vectors from the Refinement Module and produces
updated bounding box offsets and classification scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RefinedDetectionHead(nn.Module):
    """
    Lightweight detection head applied after feature refinement.

    Produces two outputs:
      1. Box refinement: (N, 4) offsets to adjust initial proposals
      2. Score refinement: (N, 1) updated confidence logit

    The box output is a DELTA applied to the initial proposal boxes, not
    absolute coordinates. This ensures the refinement is a correction
    rather than a full re-prediction.

    Args:
        input_dim: Dimension of refined features.
        hidden_dim: Hidden layer size.
        num_box_params: Box parameterization (4 for xyxy deltas).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_box_params: int = 4,
    ) -> None:
        super().__init__()

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Box refinement branch (predicts deltas)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_box_params),
        )

        # Score refinement branch
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small weights for stable initial behavior."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize box head final layer to near-zero (initially: no box change)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

        # Initialize score head final layer to near-zero (initially: trust original score)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(
        self,
        refined_features: torch.Tensor,
        initial_boxes: torch.Tensor | None = None,
        initial_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute refined box and score predictions.

        Args:
            refined_features: (N, D) or (B, N, D) refined proposal features.
            initial_boxes: Optional (N, 4) or (B, N, 4) initial proposal boxes.
                          If provided, deltas are added to produce final boxes.
            initial_scores: Optional (N,) or (B, N) initial scores.
                           If provided, logit offsets are added.

        Returns:
            Dictionary with:
              - "box_deltas": (N, 4) predicted box adjustments
              - "score_logits": (N, 1) score refinement logits
              - "refined_boxes": (N, 4) if initial_boxes provided
              - "refined_scores": (N,) if initial_scores provided
        """
        shared = self.shared(refined_features)

        box_deltas = self.box_head(shared)
        score_logits = self.score_head(shared)

        output = {
            "box_deltas": box_deltas,
            "score_logits": score_logits.squeeze(-1),
        }

        # Apply deltas to initial predictions if provided
        if initial_boxes is not None:
            output["refined_boxes"] = initial_boxes + box_deltas

        if initial_scores is not None:
            # Convert initial scores to logits, add offset, convert back
            initial_logits = torch.logit(initial_scores.clamp(1e-6, 1 - 1e-6))
            refined_logits = initial_logits + score_logits.squeeze(-1)
            output["refined_scores"] = torch.sigmoid(refined_logits)

        return output
