"""Tests for RefinementModule, DetectionHead, and DegradationPipeline."""

import numpy as np
import torch
import pytest

from src.models.refinement_module import RefinementModule
from src.models.detection_head import RefinedDetectionHead
from src.data.degradation_pipeline import (
    DegradationPipeline,
    AtmosphericScatteringModel,
    GammaAndNoiseDegradation,
    CompositeDegradation,
)


class TestRefinementModule:
    """Unit tests for the cross-attention refinement module."""

    def test_output_shape_batched(self):
        module = RefinementModule(num_layers=2, d_model=64, nhead=4)
        proposals = torch.randn(2, 10, 64)     # (B, N, D)
        supports = torch.randn(2, 10, 8, 64)   # (B, N, K, D)

        refined = module(proposals, supports)
        assert refined.shape == (2, 10, 64)

    def test_output_shape_unbatched(self):
        module = RefinementModule(num_layers=2, d_model=64, nhead=4)
        proposals = torch.randn(10, 64)        # (N, D)
        supports = torch.randn(10, 8, 64)      # (N, K, D)

        refined = module(proposals, supports)
        assert refined.shape == (10, 64)

    def test_gradient_flow(self):
        module = RefinementModule(num_layers=1, d_model=32, nhead=4)
        proposals = torch.randn(2, 5, 32, requires_grad=True)
        supports = torch.randn(2, 5, 4, 32)

        refined = module(proposals, supports)
        loss = refined.sum()
        loss.backward()

        assert proposals.grad is not None
        # Check that module parameters have gradients
        for p in module.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_gated_residual(self):
        module = RefinementModule(num_layers=1, d_model=32, nhead=4)
        proposals = torch.randn(5, 32)
        supports = torch.randn(5, 4, 32)

        # At init, gate ≈ 0 → sigmoid(0) = 0.5
        # Output should be between proposals and refined
        refined = module(proposals, supports)
        # Just ensure it doesn't crash and output is valid
        assert not torch.isnan(refined).any()
        assert not torch.isinf(refined).any()

    def test_single_layer(self):
        module = RefinementModule(num_layers=1, d_model=64, nhead=8)
        proposals = torch.randn(3, 64)
        supports = torch.randn(3, 4, 64)
        refined = module(proposals, supports)
        assert refined.shape == (3, 64)


class TestRefinedDetectionHead:
    """Unit tests for the detection head."""

    def test_output_keys(self):
        head = RefinedDetectionHead(input_dim=64, hidden_dim=64)
        features = torch.randn(10, 64)

        output = head(features)
        assert "box_deltas" in output
        assert "score_logits" in output
        assert output["box_deltas"].shape == (10, 4)
        assert output["score_logits"].shape == (10,)

    def test_with_initial_predictions(self):
        head = RefinedDetectionHead(input_dim=64)
        features = torch.randn(5, 64)
        boxes = torch.rand(5, 4) * 100
        scores = torch.rand(5) * 0.5 + 0.1

        output = head(features, initial_boxes=boxes, initial_scores=scores)
        assert "refined_boxes" in output
        assert "refined_scores" in output
        assert output["refined_boxes"].shape == (5, 4)
        assert output["refined_scores"].shape == (5,)

    def test_initial_near_identity(self):
        """At initialization, head should produce near-zero deltas."""
        head = RefinedDetectionHead(input_dim=64)
        features = torch.randn(5, 64)

        output = head(features)
        # Box deltas should be near zero
        assert output["box_deltas"].abs().max() < 0.5
        # Score logits should be near zero
        assert output["score_logits"].abs().max() < 0.5


class TestDegradationPipeline:
    """Unit tests for the degradation pipeline."""

    def _clean_image(self, h=256, w=256) -> np.ndarray:
        return np.random.rand(h, w, 3).astype(np.float32)

    def test_haze_output_range(self):
        model = AtmosphericScatteringModel()
        img = self._clean_image()
        hazy = model(img)

        assert hazy.shape == img.shape
        assert hazy.dtype == np.float32
        assert hazy.min() >= 0.0
        assert hazy.max() <= 1.0

    def test_night_output_range(self):
        model = GammaAndNoiseDegradation()
        img = self._clean_image()
        night = model(img)

        assert night.shape == img.shape
        assert night.dtype == np.float32
        assert night.min() >= 0.0
        assert night.max() <= 1.0

    def test_night_is_darker(self):
        model = GammaAndNoiseDegradation(gamma_range=(3.0, 3.0))
        img = self._clean_image()
        night = model(img, rng=np.random.default_rng(42))

        # Night image should be darker on average
        assert night.mean() < img.mean()

    def test_combined_degradation(self):
        model = CompositeDegradation(haze_prob=1.0, night_prob=1.0)
        img = self._clean_image()
        degraded = model(img)

        assert degraded.shape == img.shape
        assert degraded.min() >= 0.0
        assert degraded.max() <= 1.0

    def test_pipeline_modes(self):
        for mode in DegradationPipeline.available_modes():
            pipeline = DegradationPipeline(mode=mode, seed=42)
            img = self._clean_image(128, 128)
            degraded = pipeline(img)
            assert degraded.shape == img.shape

    def test_pipeline_generate_pair(self):
        pipeline = DegradationPipeline(mode="haze", seed=42)
        img = self._clean_image(128, 128)
        clean, degraded = pipeline.generate_pair(img, seed=42)

        assert clean.shape == degraded.shape
        # They should be different (degradation applied)
        assert not np.allclose(clean, degraded)

    def test_pipeline_uint8_input(self):
        pipeline = DegradationPipeline(mode="haze", seed=42)
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        degraded = pipeline(img)

        assert degraded.dtype == np.float32
        assert degraded.min() >= 0.0
        assert degraded.max() <= 1.0

    def test_reproducibility(self):
        pipeline1 = DegradationPipeline(mode="haze")
        pipeline2 = DegradationPipeline(mode="haze")
        img = self._clean_image(64, 64)

        d1 = pipeline1(img, seed=42)
        d2 = pipeline2(img, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            DegradationPipeline(mode="invalid_mode")
