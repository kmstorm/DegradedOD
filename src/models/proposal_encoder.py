"""
MODULE 2: Proposal Feature Encoder.

Projects Grounding DINO's raw query features into a shared embedding space
suitable for memory bank retrieval. This is a lightweight trainable module
that bridges the gap between degraded-domain and clean-domain feature spaces.

Architecture:
    F_query (D_in) → MLP(D_in → D_hidden → D_out) → LayerNorm → L2-Normalize
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalEncoder(nn.Module):
    """
    Encode G-DINO query features into the memory retrieval embedding space.

    This is the trainable bridge between degraded proposal features and
    clean memory bank entries. During training, it learns to project features
    such that degraded and clean representations of the same object are close.

    Args:
        input_dim: Dimension of G-DINO query features (typically 256).
        hidden_dim: Hidden layer dimension.
        output_dim: Output embedding dimension (must match memory bank feature_dim).
        num_layers: Number of MLP layers (2 or 3).
        dropout: Dropout probability.
        normalize: Whether to L2-normalize output embeddings.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.normalize = normalize
        self.output_dim = output_dim

        # Build MLP layers
        layers: list[nn.Module] = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Xavier uniform for stable training."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Encode query features into memory-compatible embeddings.

        Args:
            query_features: (N, D_in) or (B, N, D_in) proposal features
                           from G-DINO decoder.

        Returns:
            Encoded embeddings: same shape with last dim = output_dim.
            L2-normalized if self.normalize is True.
        """
        x = self.mlp(query_features)
        x = self.layer_norm(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x
