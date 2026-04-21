"""
MODULE 3: Memory Retrieval — k-NN search with learned similarity weighting.

Queries the memory bank with encoded proposal features and returns
top-K clean-domain supports with attention-ready similarity weights.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from src.models.memory_bank import MemoryBank

logger = logging.getLogger(__name__)


class MemoryRetrieval(nn.Module):
    """
    Retrieve clean-domain support features from the memory bank.

    For each degraded proposal, finds the top-K most similar entries in
    the memory bank and returns them with softmax-weighted similarity scores.

    Args:
        top_k: Number of nearest neighbors to retrieve.
        temperature: Initial temperature for softmax weighting (learnable).
        class_restricted: If True, restrict search to the predicted class.
    """

    def __init__(
        self,
        top_k: int = 8,
        temperature: float = 0.07,
        class_restricted: bool = True,
    ) -> None:
        super().__init__()

        self.top_k = top_k
        self.class_restricted = class_restricted

        # Learnable temperature for softmax weighting
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return self.log_temperature.exp().item()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        memory_bank: MemoryBank,
        class_hints: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Retrieve top-K supports from the memory bank.

        Args:
            query_embeddings: (N, D) encoded proposal features (from ProposalEncoder).
            memory_bank: The populated MemoryBank instance.
            class_hints: Optional (N,) predicted class labels for class-restricted search.

        Returns:
            Dictionary containing:
              - "retrieved_features": (N, K, D) top-K support feature vectors
              - "similarity_scores": (N, K) raw similarity scores
              - "attention_weights": (N, K) softmax-normalized weights
              - "retrieved_indices": (N, K) indices into the memory bank
              - "retrieved_classes": list of (N, K) class names
        """
        device = query_embeddings.device
        n_queries = query_embeddings.shape[0]
        feat_dim = query_embeddings.shape[1]

        # Determine class filter per query
        if self.class_restricted and class_hints is not None:
            # Search per-query with class filter
            all_distances = []
            all_indices = []

            for i in range(n_queries):
                q = query_embeddings[i:i+1]  # (1, D)
                cls = class_hints[i]
                distances, indices = memory_bank.search(
                    q, top_k=self.top_k, class_filter=cls
                )
                all_distances.append(distances)
                all_indices.append(indices)

            distances = np.concatenate(all_distances, axis=0)  # (N, K)
            indices = np.concatenate(all_indices, axis=0)       # (N, K)
        else:
            # Global search (no class restriction)
            distances, indices = memory_bank.search(
                query_embeddings, top_k=self.top_k
            )

        # Retrieve features by indices
        retrieved_np = memory_bank.get_features_by_indices(indices)  # (N, K, D)
        retrieved_features = torch.from_numpy(retrieved_np).to(device).float()

        # Convert distances to tensor
        similarity_scores = torch.from_numpy(distances).to(device).float()

        # Compute attention weights via temperature-scaled softmax
        temp = self.log_temperature.exp()
        attention_weights = torch.softmax(similarity_scores / temp, dim=-1)

        # Get class names for retrieved entries
        retrieved_meta = memory_bank.get_metadata_by_indices(indices)
        retrieved_classes = [m.class_name for m in retrieved_meta]

        return {
            "retrieved_features": retrieved_features,
            "similarity_scores": similarity_scores,
            "attention_weights": attention_weights,
            "retrieved_indices": torch.from_numpy(indices).to(device).long(),
            "retrieved_classes": retrieved_classes,
        }

    def compute_weighted_support(
        self,
        retrieved_features: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted average of retrieved supports.

        Args:
            retrieved_features: (N, K, D) retrieved support features.
            attention_weights: (N, K) attention weights.

        Returns:
            (N, D) weighted support vectors.
        """
        # (N, K, 1) * (N, K, D) → sum over K → (N, D)
        weighted = attention_weights.unsqueeze(-1) * retrieved_features
        return weighted.sum(dim=1)
