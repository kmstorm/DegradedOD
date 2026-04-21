"""
MODULE 4: Refinement Module — Cross-Attention Transformer Decoder.

Fuses degraded proposal features with retrieved clean-domain supports
via cross-attention to produce refined feature representations.

Architecture:
    Layer 1: Self-attention over degraded proposals
             (models inter-proposal relationships in the degraded scene)
    Layer 2: Cross-attention (Q=degraded, K/V=retrieved supports)
             (injects clean-domain knowledge into each proposal)
    Each layer followed by FFN → LayerNorm → Residual
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RefinementLayer(nn.Module):
    """
    Single refinement layer: self-attention + cross-attention + FFN.

    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention over proposals
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention: Q=proposals, K/V=retrieved supports
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        proposals: torch.Tensor,
        supports: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            proposals: (B, N, D) degraded proposal features.
            supports: (B, N*K, D) or (B, M, D) retrieved support features.
                      Can be flattened (all supports concatenated) or
                      per-proposal (N*K where K supports per proposal).
            support_mask: Optional (B, N*K) boolean mask for padding.

        Returns:
            (B, N, D) refined proposal features.
        """
        # 1. Self-attention over proposals
        residual = proposals
        x, _ = self.self_attn(proposals, proposals, proposals)
        x = self.norm1(residual + self.dropout1(x))

        # 2. Cross-attention: proposals attend to retrieved supports
        residual = x
        x_cross, _ = self.cross_attn(
            query=x,
            key=supports,
            value=supports,
            key_padding_mask=support_mask,
        )
        x = self.norm2(residual + self.dropout2(x_cross))

        # 3. Feed-forward
        residual = x
        x_ff = self.ffn(x)
        x = self.norm3(residual + self.dropout3(x_ff))

        return x


class RefinementModule(nn.Module):
    """
    Multi-layer refinement decoder that fuses degraded features with
    clean-domain supports via cross-attention.

    The key insight: cross-attention LEARNS "what clean features are useful"
    for each degraded proposal, rather than naive nearest-neighbor replacement.

    Args:
        num_layers: Number of refinement layers.
        d_model: Feature dimension throughout.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.layers = nn.ModuleList([
            RefinementLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Optional: learned scale parameter for residual gating
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        proposal_features: torch.Tensor,
        retrieved_features: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Refine degraded proposal features using retrieved clean supports.

        Args:
            proposal_features: (B, N, D) or (N, D) degraded proposals.
            retrieved_features: (B, N, K, D) or (N, K, D) retrieved supports.
                                K supports per proposal.
            support_mask: Optional mask for padded supports.

        Returns:
            (B, N, D) or (N, D) refined features (same shape as input proposals).
        """
        # Handle unbatched input
        squeezed = False
        if proposal_features.dim() == 2:
            proposal_features = proposal_features.unsqueeze(0)
            retrieved_features = retrieved_features.unsqueeze(0)
            squeezed = True

        B, N, D = proposal_features.shape

        # Reshape retrieved: (B, N, K, D) → (B, N*K, D) for cross-attention
        if retrieved_features.dim() == 4:
            K = retrieved_features.shape[2]
            supports_flat = retrieved_features.reshape(B, N * K, D)
        else:
            supports_flat = retrieved_features

        # Apply refinement layers
        x = proposal_features
        for layer in self.layers:
            x = layer(x, supports_flat, support_mask)

        # Gated residual: output = proposals + sigmoid(gate) * refinement
        gate_weight = torch.sigmoid(self.gate)
        refined = proposal_features + gate_weight * (x - proposal_features)

        if squeezed:
            refined = refined.squeeze(0)

        return refined
