"""
MODD Detector — Full pipeline composition.

Orchestrates all 5 modules into a single end-to-end pipeline:

  Degraded Image
    → Frozen G-DINO (detect + extract features)
    → Confidence Split (high → direct output, low → retrieval path)
    → Proposal Encoder (project into retrieval space)
    → Memory Retrieval (FAISS k-NN from memory bank)
    → Refinement Module (cross-attention fusion)
    → Refined Detection Head (updated boxes + scores)
    → Merge with high-confidence detections
    → (Optional) Memory Update (EMA for target-domain adaptation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image

from src.models.detection_head import RefinedDetectionHead
from src.models.gdino_wrapper import DetectionResult, GroundingDINOWrapper
from src.models.memory_bank import MemoryBank
from src.models.memory_retrieval import MemoryRetrieval
from src.models.memory_update import MemoryUpdater
from src.models.proposal_encoder import ProposalEncoder
from src.models.refinement_module import RefinementModule

logger = logging.getLogger(__name__)


@dataclass
class MODDConfig:
    """Configuration for the full MODD pipeline."""
    # Backbone
    model_id: str = "IDEA-Research/grounding-dino-base"
    dtype: str = "float16"
    box_threshold: float = 0.25
    text_threshold: float = 0.20

    # Confidence gating
    high_conf_threshold: float = 0.5
    low_conf_threshold: float = 0.15

    # Proposal encoder
    encoder_input_dim: int = 256
    encoder_hidden_dim: int = 512
    encoder_output_dim: int = 256
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1

    # Retrieval
    retrieval_top_k: int = 8
    retrieval_temperature: float = 0.07
    retrieval_class_restricted: bool = True

    # Refinement
    refinement_num_layers: int = 2
    refinement_d_model: int = 256
    refinement_nhead: int = 8
    refinement_dim_feedforward: int = 1024
    refinement_dropout: float = 0.1

    # Detection head
    det_head_hidden_dim: int = 256

    # Memory update
    memory_update_enabled: bool = True
    memory_update_threshold: float = 0.7
    memory_update_ema: float = 0.99


class MODDDetector(nn.Module):
    """
    Memory-Augmented Open-World Detector for Degraded Conditions.

    Composes all 5 modules into a single pipeline:
      - MODULE 1: Memory Bank (external, FAISS-backed)
      - MODULE 2: Proposal Feature Encoder (trainable)
      - MODULE 3: Memory Retrieval (FAISS k-NN + learnable temperature)
      - MODULE 4: Refinement Module (trainable cross-attention decoder)
      - MODULE 5: Memory Update (online, no learnable params)
      + Frozen Grounding DINO backbone
      + Refined Detection Head (trainable)

    Trainable parameters: ~8-10M (Encoder + Refinement + Head + Temperature)
    Frozen parameters: ~232M (G-DINO base)

    Usage:
        config = MODDConfig()
        detector = MODDDetector(config)
        memory_bank = MemoryBank.load("./memory_bank/")

        result = detector.detect(image, "person . car .", memory_bank)
    """

    def __init__(self, config: MODDConfig | None = None) -> None:
        super().__init__()

        self.config = config or MODDConfig()
        c = self.config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Frozen backbone (not registered as submodule to avoid saving its params)
        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        self._backbone = GroundingDINOWrapper(
            model_id=c.model_id,
            device=self.device,
            dtype=dtype_map.get(c.dtype, torch.float16),
            box_threshold=c.box_threshold,
            text_threshold=c.text_threshold,
        )

        # MODULE 2: Proposal Encoder
        self.proposal_encoder = ProposalEncoder(
            input_dim=c.encoder_input_dim,
            hidden_dim=c.encoder_hidden_dim,
            output_dim=c.encoder_output_dim,
            num_layers=c.encoder_num_layers,
            dropout=c.encoder_dropout,
        )

        # MODULE 3: Memory Retrieval
        self.memory_retrieval = MemoryRetrieval(
            top_k=c.retrieval_top_k,
            temperature=c.retrieval_temperature,
            class_restricted=c.retrieval_class_restricted,
        )

        # MODULE 4: Refinement Module
        self.refinement = RefinementModule(
            num_layers=c.refinement_num_layers,
            d_model=c.refinement_d_model,
            nhead=c.refinement_nhead,
            dim_feedforward=c.refinement_dim_feedforward,
            dropout=c.refinement_dropout,
        )

        # Detection Head
        self.detection_head = RefinedDetectionHead(
            input_dim=c.refinement_d_model,
            hidden_dim=c.det_head_hidden_dim,
        )

        # MODULE 5: Memory Updater (no learnable params)
        self.memory_updater = MemoryUpdater(
            confidence_threshold=c.memory_update_threshold,
            ema_momentum=c.memory_update_ema,
            enabled=c.memory_update_enabled,
        )

        # Log parameter count
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = trainable + sum(p.numel() for p in self._backbone.parameters())
        logger.info(
            f"MODDDetector: {trainable / 1e6:.2f}M trainable params, "
            f"{total / 1e6:.1f}M total (including frozen backbone)"
        )

    @property
    def backbone(self) -> GroundingDINOWrapper:
        """Access the frozen backbone."""
        return self._backbone

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        memory_bank: MemoryBank,
        update_memory: bool = False,
    ) -> DetectionResult:
        """
        Full MODD inference pipeline.

        Args:
            image: PIL Image to detect objects in.
            text_prompt: Detection class prompt (e.g., "person . car .").
            memory_bank: Populated memory bank for retrieval.
            update_memory: Whether to update memory bank with high-conf results.

        Returns:
            DetectionResult with refined boxes and scores.
        """
        c = self.config

        # Step 1: Frozen G-DINO detection + feature extraction
        proposal_data = self._backbone.extract_features(
            image, text_prompt,
            box_threshold=c.low_conf_threshold,  # use low threshold to get more proposals
        )

        if len(proposal_data.detection) == 0:
            return proposal_data.detection

        # Step 2: Confidence split
        scores = proposal_data.detection.scores
        high_mask = scores >= c.high_conf_threshold
        low_mask = (scores >= c.low_conf_threshold) & (scores < c.high_conf_threshold)

        # High-confidence detections go directly to output
        high_conf_result = DetectionResult(
            boxes=proposal_data.detection.boxes[high_mask],
            scores=proposal_data.detection.scores[high_mask],
            labels=[l for l, m in zip(proposal_data.detection.labels, high_mask) if m],
        )

        # Low-to-mid confidence proposals go through memory retrieval
        if not low_mask.any():
            return high_conf_result

        low_features = proposal_data.query_features[low_mask].to(self.device)
        low_boxes = proposal_data.detection.boxes[low_mask].to(self.device)
        low_scores = proposal_data.detection.scores[low_mask].to(self.device)
        low_labels = [l for l, m in zip(proposal_data.detection.labels, low_mask) if m]

        # Step 3: Encode proposals
        encoded = self.proposal_encoder(low_features)

        # Step 4: Retrieve from memory bank
        retrieval_output = self.memory_retrieval(
            encoded, memory_bank, class_hints=low_labels
        )

        # Step 5: Refinement via cross-attention
        retrieved = retrieval_output["retrieved_features"]  # (N, K, D)
        refined_features = self.refinement(
            proposal_features=low_features.unsqueeze(0),
            retrieved_features=retrieved.unsqueeze(0),
        ).squeeze(0)

        # Step 6: Refined detection head
        head_output = self.detection_head(
            refined_features,
            initial_boxes=low_boxes,
            initial_scores=low_scores,
        )

        # Build refined detection result
        refined_result = DetectionResult(
            boxes=head_output.get("refined_boxes", low_boxes).cpu(),
            scores=head_output.get("refined_scores", low_scores).cpu(),
            labels=low_labels,
        )

        # Filter refined results by confidence
        refined_result = refined_result.filter_by_score(c.low_conf_threshold)

        # Step 7: Merge high-conf and refined results
        merged = self._merge_results(high_conf_result, refined_result)

        # Step 8: Optional memory update
        if update_memory and self.memory_updater.enabled:
            high_score_mask = merged.scores >= c.memory_update_threshold
            if high_score_mask.any():
                # Use encoded features for high-confidence refined detections
                update_features = self.proposal_encoder(
                    proposal_data.query_features[: len(merged)].to(self.device)
                )
                self.memory_updater.update(
                    memory_bank=memory_bank,
                    features=update_features[:len(merged.scores)],
                    class_names=merged.labels,
                    scores=merged.scores,
                )

        return merged

    def forward_train(
        self,
        query_features: torch.Tensor,
        initial_boxes: torch.Tensor,
        initial_scores: torch.Tensor,
        class_labels: list[str],
        memory_bank: MemoryBank,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass (differentiable).

        Used during Phase B/C training with paired clean-degraded data.
        Assumes features are already extracted from frozen backbone.

        Args:
            query_features: (B, N, D) proposal features from G-DINO.
            initial_boxes: (B, N, 4) initial box predictions.
            initial_scores: (B, N) initial scores.
            class_labels: (N,) per-proposal class labels.
            memory_bank: Populated memory bank.

        Returns:
            Dictionary with refined predictions and intermediate outputs
            needed for loss computation.
        """
        B, N, D = query_features.shape

        # Encode proposals
        encoded = self.proposal_encoder(query_features)  # (B, N, D_out)

        # Retrieve from memory (per-batch processing)
        all_retrieved = []
        all_similarities = []
        all_attn_weights = []

        for b in range(B):
            retrieval_out = self.memory_retrieval(
                encoded[b],  # (N, D)
                memory_bank,
                class_hints=class_labels,
            )
            all_retrieved.append(retrieval_out["retrieved_features"])
            all_similarities.append(retrieval_out["similarity_scores"])
            all_attn_weights.append(retrieval_out["attention_weights"])

        retrieved_features = torch.stack(all_retrieved)      # (B, N, K, D)
        similarity_scores = torch.stack(all_similarities)    # (B, N, K)
        attention_weights = torch.stack(all_attn_weights)    # (B, N, K)

        # Refinement
        refined_features = self.refinement(
            proposal_features=query_features,
            retrieved_features=retrieved_features,
        )  # (B, N, D)

        # Detection head
        head_output = self.detection_head(
            refined_features,
            initial_boxes=initial_boxes,
            initial_scores=initial_scores,
        )

        return {
            "encoded_features": encoded,
            "retrieved_features": retrieved_features,
            "similarity_scores": similarity_scores,
            "attention_weights": attention_weights,
            "refined_features": refined_features,
            **head_output,
        }

    @staticmethod
    def _merge_results(
        high_conf: DetectionResult,
        refined: DetectionResult,
    ) -> DetectionResult:
        """Merge high-confidence direct outputs with refined outputs."""
        if len(high_conf) == 0:
            return refined
        if len(refined) == 0:
            return high_conf

        return DetectionResult(
            boxes=torch.cat([high_conf.boxes, refined.boxes], dim=0),
            scores=torch.cat([high_conf.scores, refined.scores], dim=0),
            labels=high_conf.labels + refined.labels,
        )

    def trainable_parameters(self) -> list[dict]:
        """Return parameter groups for optimizer configuration."""
        return [
            {
                "params": self.proposal_encoder.parameters(),
                "lr_scale": 1.0,
                "name": "proposal_encoder",
            },
            {
                "params": self.memory_retrieval.parameters(),
                "lr_scale": 0.1,  # temperature learns slowly
                "name": "memory_retrieval",
            },
            {
                "params": self.refinement.parameters(),
                "lr_scale": 1.0,
                "name": "refinement",
            },
            {
                "params": self.detection_head.parameters(),
                "lr_scale": 1.0,
                "name": "detection_head",
            },
        ]
