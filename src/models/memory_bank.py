"""
MODULE 1: Memory Bank — FAISS-backed storage for clean-domain features.

Stores three types of entries:
  - Class prototypes: mean feature vector per object class
  - Context embeddings: scene-level features (global)
  - Instance supports: per-detection feature vectors

Backed by FAISS for efficient k-NN retrieval at inference time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning(
        "faiss not installed. Install via: pip install faiss-gpu (or faiss-cpu). "
        "MemoryBank will fall back to brute-force PyTorch search."
    )


@dataclass
class MemoryEntry:
    """Metadata for a single memory bank entry."""
    class_name: str
    score: float
    source_image: str = ""
    entry_type: str = "instance"  # "instance", "prototype", or "context"


class MemoryBank:
    """
    FAISS-backed memory bank for clean-domain feature storage and retrieval.

    Supports three storage modes:
      - Instance supports: individual detection features (many per class)
      - Class prototypes: aggregated mean features (one per class)
      - Context embeddings: scene-level features

    Usage:
        bank = MemoryBank(feature_dim=256)

        # Add entries during memory construction (Phase A)
        bank.add(features=feat_tensor, class_names=["car", "person"], scores=[0.9, 0.85])

        # Build FAISS index for fast retrieval
        bank.build_index()

        # Query during inference
        distances, indices = bank.search(query_features, top_k=8)
        retrieved = bank.get_entries(indices)

        # Save / load
        bank.save("./memory_bank/")
        bank = MemoryBank.load("./memory_bank/")
    """

    def __init__(
        self,
        feature_dim: int = 256,
        max_entries_per_class: int = 1000,
        max_total_entries: int = 100_000,
        similarity: str = "cosine",
    ) -> None:
        """
        Args:
            feature_dim: Dimensionality of feature vectors.
            max_entries_per_class: Maximum instance supports stored per class.
            max_total_entries: Hard cap on total entries.
            similarity: "cosine" (L2-normalized inner product) or "l2".
        """
        self.feature_dim = feature_dim
        self.max_entries_per_class = max_entries_per_class
        self.max_total_entries = max_total_entries
        self.similarity = similarity

        # Storage
        self._features: list[np.ndarray] = []       # list of (D,) arrays
        self._metadata: list[MemoryEntry] = []       # parallel metadata
        self._class_counts: dict[str, int] = {}      # tracks per-class count

        # Class prototypes (maintained separately)
        self._prototypes: dict[str, np.ndarray] = {}      # class_name → (D,)
        self._prototype_counts: dict[str, int] = {}        # running count for online mean

        # FAISS index (built after all entries are added)
        self._index: Any = None
        self._index_built = False

    @property
    def num_entries(self) -> int:
        """Total number of instance entries."""
        return len(self._features)

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the bank."""
        return len(self._class_counts)

    @property
    def class_names(self) -> list[str]:
        """List of class names in the bank."""
        return list(self._class_counts.keys())

    def add(
        self,
        features: torch.Tensor | np.ndarray,
        class_names: list[str],
        scores: list[float] | np.ndarray | torch.Tensor,
        source_image: str = "",
        entry_type: str = "instance",
    ) -> int:
        """
        Add feature entries to the memory bank.

        Args:
            features: (N, D) feature vectors.
            class_names: (N,) class label for each feature.
            scores: (N,) confidence scores.
            source_image: Source image identifier.
            entry_type: "instance", "prototype", or "context".

        Returns:
            Number of entries actually added (may be less if capacity reached).
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        elif isinstance(scores, list):
            scores = np.array(scores)

        assert features.ndim == 2, f"Expected (N, D), got {features.shape}"
        assert features.shape[1] == self.feature_dim, (
            f"Feature dim mismatch: expected {self.feature_dim}, got {features.shape[1]}"
        )
        assert len(class_names) == len(features) == len(scores)

        added = 0
        for feat, cls, score in zip(features, class_names, scores):
            # Check capacity
            if self.num_entries >= self.max_total_entries:
                logger.warning("Memory bank full, skipping remaining entries")
                break

            class_count = self._class_counts.get(cls, 0)
            if class_count >= self.max_entries_per_class:
                continue  # skip, this class is full

            # Normalize for cosine similarity
            if self.similarity == "cosine":
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm

            self._features.append(feat.astype(np.float32))
            self._metadata.append(MemoryEntry(
                class_name=cls,
                score=float(score),
                source_image=source_image,
                entry_type=entry_type,
            ))
            self._class_counts[cls] = class_count + 1

            # Update running class prototype (online mean)
            self._update_prototype(cls, feat)

            added += 1

        if self._index_built:
            logger.warning("Index was already built. Call build_index() again after adding.")
            self._index_built = False

        return added

    def _update_prototype(self, class_name: str, feature: np.ndarray) -> None:
        """Update the running mean prototype for a class."""
        if class_name not in self._prototypes:
            self._prototypes[class_name] = feature.copy()
            self._prototype_counts[class_name] = 1
        else:
            count = self._prototype_counts[class_name]
            # Online mean update: μ_new = μ_old + (x - μ_old) / (n + 1)
            self._prototypes[class_name] += (feature - self._prototypes[class_name]) / (count + 1)
            self._prototype_counts[class_name] = count + 1

    def build_index(self) -> None:
        """Build the FAISS index from all stored features."""
        if self.num_entries == 0:
            logger.warning("No entries to build index from")
            return

        all_features = np.stack(self._features, axis=0).astype(np.float32)

        if faiss is not None:
            if self.similarity == "cosine":
                # Inner product on L2-normalized vectors = cosine similarity
                self._index = faiss.IndexFlatIP(self.feature_dim)
            else:
                self._index = faiss.IndexFlatL2(self.feature_dim)

            self._index.add(all_features)
            logger.info(
                f"FAISS index built: {self._index.ntotal} entries, "
                f"dim={self.feature_dim}, similarity={self.similarity}"
            )
        else:
            # Fallback: store as tensor for brute-force search
            self._index = torch.from_numpy(all_features)
            logger.info(
                f"PyTorch fallback index: {len(all_features)} entries, "
                f"dim={self.feature_dim}"
            )

        self._index_built = True

    def search(
        self,
        query: torch.Tensor | np.ndarray,
        top_k: int = 8,
        class_filter: str | list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the memory bank for the most similar entries.

        Args:
            query: (N, D) or (D,) query feature vectors.
            top_k: Number of nearest neighbors to retrieve.
            class_filter: Optional class name(s) to restrict search.

        Returns:
            Tuple of (distances, indices) each shaped (N, top_k).
            Distances are similarity scores (higher = more similar for cosine).
        """
        if not self._index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        if query.ndim == 1:
            query = query[np.newaxis, :]

        # Normalize queries for cosine
        if self.similarity == "cosine":
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            query = query / norms

        query = query.astype(np.float32)

        if class_filter is not None:
            return self._search_with_filter(query, top_k, class_filter)

        if faiss is not None and isinstance(self._index, faiss.Index):
            distances, indices = self._index.search(query, top_k)
        else:
            distances, indices = self._brute_force_search(query, top_k)

        return distances, indices

    def _search_with_filter(
        self,
        query: np.ndarray,
        top_k: int,
        class_filter: str | list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search restricted to specific class(es)."""
        if isinstance(class_filter, str):
            class_filter = [class_filter]

        # Get indices of entries matching the class filter
        valid_indices = [
            i for i, m in enumerate(self._metadata)
            if m.class_name in class_filter
        ]

        if len(valid_indices) == 0:
            return np.zeros((len(query), top_k)), np.full((len(query), top_k), -1)

        # Extract subset features
        subset_features = np.stack([self._features[i] for i in valid_indices]).astype(np.float32)

        # Brute-force on subset (FAISS subset search is more complex)
        if self.similarity == "cosine":
            sims = query @ subset_features.T  # (N, subset_size)
        else:
            # L2 distance
            diff = query[:, np.newaxis, :] - subset_features[np.newaxis, :, :]
            sims = -np.sum(diff ** 2, axis=-1)

        actual_k = min(top_k, len(valid_indices))
        top_subset_indices = np.argsort(-sims, axis=1)[:, :actual_k]

        distances = np.take_along_axis(sims, top_subset_indices, axis=1)
        # Map back to global indices
        global_indices = np.array([[valid_indices[j] for j in row] for row in top_subset_indices])

        # Pad if needed
        if actual_k < top_k:
            pad_d = np.zeros((len(query), top_k - actual_k))
            pad_i = np.full((len(query), top_k - actual_k), -1)
            distances = np.concatenate([distances, pad_d], axis=1)
            global_indices = np.concatenate([global_indices, pad_i], axis=1)

        return distances, global_indices

    def _brute_force_search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback brute-force search when FAISS is not available."""
        all_feats = self._index  # torch.Tensor from build_index fallback
        if isinstance(all_feats, torch.Tensor):
            all_feats = all_feats.numpy()

        if self.similarity == "cosine":
            sims = query @ all_feats.T
        else:
            diff = query[:, np.newaxis, :] - all_feats[np.newaxis, :, :]
            sims = -np.sum(diff ** 2, axis=-1)

        actual_k = min(top_k, sims.shape[1])
        top_indices = np.argsort(-sims, axis=1)[:, :actual_k]
        distances = np.take_along_axis(sims, top_indices, axis=1)

        return distances, top_indices

    def get_features_by_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Retrieve feature vectors by index.

        Args:
            indices: (N, K) or (K,) indices from search().

        Returns:
            Feature array with matching shape + (D,).
        """
        flat = indices.flatten()
        features = []
        for idx in flat:
            if 0 <= idx < self.num_entries:
                features.append(self._features[idx])
            else:
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
        result = np.stack(features).reshape(*indices.shape, self.feature_dim)
        return result

    def get_metadata_by_indices(self, indices: np.ndarray) -> list[MemoryEntry]:
        """Retrieve metadata by flat indices."""
        return [
            self._metadata[i] if 0 <= i < len(self._metadata)
            else MemoryEntry(class_name="<pad>", score=0.0)
            for i in indices.flatten()
        ]

    def get_prototype(self, class_name: str) -> np.ndarray | None:
        """Get the mean prototype for a class."""
        return self._prototypes.get(class_name)

    def get_all_prototypes(self) -> dict[str, np.ndarray]:
        """Get all class prototypes."""
        return dict(self._prototypes)

    def save(self, output_dir: str | Path) -> None:
        """
        Save the memory bank to disk.

        Saves:
          - features.npy: all feature vectors
          - metadata.json: entry metadata
          - prototypes.npz: class prototypes
          - index.faiss: FAISS index (if using FAISS)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Features
        if self.num_entries > 0:
            all_features = np.stack(self._features)
            np.save(output_dir / "features.npy", all_features)

        # Metadata
        meta_list = [
            {
                "class_name": m.class_name,
                "score": m.score,
                "source_image": m.source_image,
                "entry_type": m.entry_type,
            }
            for m in self._metadata
        ]
        with open(output_dir / "metadata.json", "w") as f:
            json.dump({
                "feature_dim": self.feature_dim,
                "similarity": self.similarity,
                "max_entries_per_class": self.max_entries_per_class,
                "max_total_entries": self.max_total_entries,
                "num_entries": self.num_entries,
                "class_counts": self._class_counts,
                "entries": meta_list,
            }, f, indent=2)

        # Prototypes
        if self._prototypes:
            np.savez(
                output_dir / "prototypes.npz",
                **{k: v for k, v in self._prototypes.items()},
            )

        # FAISS index
        if self._index_built and faiss is not None and isinstance(self._index, faiss.Index):
            faiss.write_index(self._index, str(output_dir / "index.faiss"))

        logger.info(
            f"Memory bank saved to {output_dir}: "
            f"{self.num_entries} entries, {self.num_classes} classes"
        )

    @classmethod
    def load(cls, input_dir: str | Path) -> "MemoryBank":
        """Load a memory bank from disk."""
        input_dir = Path(input_dir)

        with open(input_dir / "metadata.json") as f:
            meta = json.load(f)

        bank = cls(
            feature_dim=meta["feature_dim"],
            max_entries_per_class=meta["max_entries_per_class"],
            max_total_entries=meta["max_total_entries"],
            similarity=meta["similarity"],
        )

        # Load features
        features_path = input_dir / "features.npy"
        if features_path.exists():
            all_features = np.load(features_path)
            bank._features = [all_features[i] for i in range(len(all_features))]

        # Load metadata entries
        bank._metadata = [
            MemoryEntry(**entry) for entry in meta["entries"]
        ]
        bank._class_counts = meta["class_counts"]

        # Load prototypes
        proto_path = input_dir / "prototypes.npz"
        if proto_path.exists():
            proto_data = np.load(proto_path)
            bank._prototypes = {k: proto_data[k] for k in proto_data.files}

        # Load or rebuild FAISS index
        faiss_path = input_dir / "index.faiss"
        if faiss_path.exists() and faiss is not None:
            bank._index = faiss.read_index(str(faiss_path))
            bank._index_built = True
            logger.info(f"Loaded FAISS index: {bank._index.ntotal} entries")
        elif bank.num_entries > 0:
            bank.build_index()

        logger.info(
            f"Memory bank loaded from {input_dir}: "
            f"{bank.num_entries} entries, {bank.num_classes} classes"
        )
        return bank

    def summary(self) -> str:
        """Return a human-readable summary of the memory bank."""
        lines = [
            f"MemoryBank(dim={self.feature_dim}, similarity={self.similarity})",
            f"  Total entries: {self.num_entries} / {self.max_total_entries}",
            f"  Classes: {self.num_classes}",
            f"  Index built: {self._index_built}",
            f"  Per-class counts:",
        ]
        for cls, count in sorted(self._class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {cls}: {count}")
        return "\n".join(lines)
