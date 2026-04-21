"""Tests for MemoryBank — FAISS-backed feature storage and retrieval."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.models.memory_bank import MemoryBank, MemoryEntry


class TestMemoryBank:
    """Unit tests for MemoryBank."""

    def _make_bank(self, dim=64, **kwargs) -> MemoryBank:
        return MemoryBank(feature_dim=dim, **kwargs)

    def _random_features(self, n: int, dim: int = 64) -> np.ndarray:
        feats = np.random.randn(n, dim).astype(np.float32)
        return feats

    def test_init(self):
        bank = self._make_bank()
        assert bank.num_entries == 0
        assert bank.num_classes == 0
        assert bank.feature_dim == 64

    def test_add_entries(self):
        bank = self._make_bank()
        feats = self._random_features(5)
        classes = ["car", "car", "person", "car", "person"]
        scores = [0.9, 0.85, 0.8, 0.7, 0.95]

        added = bank.add(feats, classes, scores)
        assert added == 5
        assert bank.num_entries == 5
        assert bank.num_classes == 2
        assert set(bank.class_names) == {"car", "person"}

    def test_max_entries_per_class(self):
        bank = self._make_bank(max_entries_per_class=3)
        feats = self._random_features(5)
        classes = ["car"] * 5
        scores = [0.9] * 5

        added = bank.add(feats, classes, scores)
        assert added == 3  # capped at max_entries_per_class

    def test_max_total_entries(self):
        bank = self._make_bank(max_total_entries=4)
        feats = self._random_features(6)
        classes = ["a", "b", "c", "d", "e", "f"]
        scores = [0.9] * 6

        added = bank.add(feats, classes, scores)
        assert added == 4

    def test_build_index_and_search(self):
        bank = self._make_bank()
        # Add 20 entries across 4 classes
        for cls in ["car", "person", "bike", "bus"]:
            feats = self._random_features(5)
            bank.add(feats, [cls] * 5, [0.9] * 5)

        bank.build_index()

        # Query with a random feature
        query = self._random_features(1)
        distances, indices = bank.search(query, top_k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)
        assert all(0 <= idx < bank.num_entries for idx in indices[0])

    def test_search_with_class_filter(self):
        bank = self._make_bank()
        # Add entries with known classes
        for cls in ["car", "person"]:
            feats = self._random_features(10)
            bank.add(feats, [cls] * 10, [0.9] * 10)

        bank.build_index()

        query = self._random_features(1)
        distances, indices = bank.search(query, top_k=5, class_filter="car")

        # All retrieved entries should be "car"
        for idx in indices[0]:
            if idx >= 0:
                meta = bank.get_metadata_by_indices(np.array([idx]))
                assert meta[0].class_name == "car"

    def test_get_features_by_indices(self):
        bank = self._make_bank()
        feats = self._random_features(10)
        bank.add(feats, ["cls"] * 10, [0.9] * 10)
        bank.build_index()

        indices = np.array([[0, 1, 2]])
        retrieved = bank.get_features_by_indices(indices)
        assert retrieved.shape == (1, 3, 64)

    def test_prototypes(self):
        bank = self._make_bank(similarity="l2")  # avoid normalization for easy math
        # Add 3 features for "car"
        feats = np.array([
            [1.0, 0.0, 0.0] + [0.0] * 61,
            [0.0, 1.0, 0.0] + [0.0] * 61,
            [0.0, 0.0, 1.0] + [0.0] * 61,
        ], dtype=np.float32)
        bank.add(feats, ["car"] * 3, [0.9] * 3)

        proto = bank.get_prototype("car")
        assert proto is not None
        # Online mean of [1,0,0], [0,1,0], [0,0,1] ≈ [0.333, 0.333, 0.333]
        np.testing.assert_allclose(proto[:3], [1/3, 1/3, 1/3], atol=1e-5)

    def test_save_and_load(self):
        bank = self._make_bank()
        feats = self._random_features(15)
        classes = ["car"] * 5 + ["person"] * 5 + ["bike"] * 5
        bank.add(feats, classes, [0.9] * 15)
        bank.build_index()

        with tempfile.TemporaryDirectory() as tmpdir:
            bank.save(tmpdir)

            loaded = MemoryBank.load(tmpdir)
            assert loaded.num_entries == 15
            assert loaded.num_classes == 3
            assert loaded.feature_dim == 64

            # Verify search works on loaded bank
            query = self._random_features(1)
            d1, i1 = bank.search(query, top_k=3)
            d2, i2 = loaded.search(query, top_k=3)
            np.testing.assert_array_equal(i1, i2)

    def test_empty_search(self):
        bank = self._make_bank()
        bank.build_index()
        # Should handle gracefully (or raise)
        assert bank.num_entries == 0

    def test_batch_search(self):
        bank = self._make_bank()
        feats = self._random_features(50)
        bank.add(feats, ["cls"] * 50, [0.9] * 50)
        bank.build_index()

        queries = self._random_features(5)
        distances, indices = bank.search(queries, top_k=4)
        assert distances.shape == (5, 4)
        assert indices.shape == (5, 4)
