"""Tests for MemoryRetrieval and ProposalEncoder."""

import numpy as np
import torch
import pytest

from src.models.memory_bank import MemoryBank
from src.models.memory_retrieval import MemoryRetrieval
from src.models.proposal_encoder import ProposalEncoder


class TestProposalEncoder:
    """Unit tests for ProposalEncoder."""

    def test_output_shape(self):
        encoder = ProposalEncoder(input_dim=256, hidden_dim=512, output_dim=128)
        x = torch.randn(10, 256)
        out = encoder(x)
        assert out.shape == (10, 128)

    def test_batched_input(self):
        encoder = ProposalEncoder(input_dim=256, output_dim=128)
        x = torch.randn(4, 10, 256)  # (B, N, D)
        out = encoder(x)
        assert out.shape == (4, 10, 128)

    def test_normalization(self):
        encoder = ProposalEncoder(input_dim=64, output_dim=64, normalize=True)
        x = torch.randn(5, 64)
        out = encoder(x)
        norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(norms, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_no_normalization(self):
        encoder = ProposalEncoder(input_dim=64, output_dim=64, normalize=False)
        x = torch.randn(5, 64)
        out = encoder(x)
        norms = torch.norm(out, dim=-1)
        # Should NOT be unit norm
        assert not torch.allclose(norms, torch.ones(5), atol=1e-2)

    def test_gradient_flow(self):
        encoder = ProposalEncoder(input_dim=64, output_dim=64)
        x = torch.randn(5, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (5, 64)


class TestMemoryRetrieval:
    """Unit tests for MemoryRetrieval."""

    def _make_bank(self, dim=64, n_per_class=10):
        bank = MemoryBank(feature_dim=dim, similarity="cosine")
        for cls in ["car", "person", "bike"]:
            feats = np.random.randn(n_per_class, dim).astype(np.float32)
            bank.add(feats, [cls] * n_per_class, [0.9] * n_per_class)
        bank.build_index()
        return bank

    def test_retrieve_shapes(self):
        bank = self._make_bank()
        retrieval = MemoryRetrieval(top_k=4, class_restricted=False)

        queries = torch.randn(5, 64)
        output = retrieval(queries, bank)

        assert output["retrieved_features"].shape == (5, 4, 64)
        assert output["similarity_scores"].shape == (5, 4)
        assert output["attention_weights"].shape == (5, 4)
        assert output["retrieved_indices"].shape == (5, 4)

    def test_attention_weights_sum_to_one(self):
        bank = self._make_bank()
        retrieval = MemoryRetrieval(top_k=4)

        queries = torch.randn(3, 64)
        output = retrieval(queries, bank, class_hints=["car", "person", "bike"])

        weight_sums = output["attention_weights"].sum(dim=-1)
        torch.testing.assert_close(
            weight_sums, torch.ones(3), atol=1e-5, rtol=1e-5
        )

    def test_class_restricted_retrieval(self):
        bank = self._make_bank()
        retrieval = MemoryRetrieval(top_k=4, class_restricted=True)

        queries = torch.randn(2, 64)
        output = retrieval(queries, bank, class_hints=["car", "car"])

        # All retrieved entries should be "car"
        for cls_name in output["retrieved_classes"]:
            assert cls_name == "car" or cls_name == "<pad>"

    def test_weighted_support(self):
        retrieval = MemoryRetrieval(top_k=4)
        features = torch.randn(3, 4, 64)
        weights = torch.softmax(torch.randn(3, 4), dim=-1)

        weighted = retrieval.compute_weighted_support(features, weights)
        assert weighted.shape == (3, 64)

    def test_temperature_is_learnable(self):
        retrieval = MemoryRetrieval(temperature=0.1)
        assert retrieval.log_temperature.requires_grad is True
        assert abs(retrieval.temperature - 0.1) < 0.01
