"""
Frozen Grounding DINO wrapper with feature extraction hooks.

Loads the HuggingFace Grounding DINO model, freezes all parameters, and
registers forward hooks to capture intermediate features at three levels:
  - Hook 1 (backbone): Multi-scale visual features from Swin Transformer
  - Hook 2 (encoder): Enhanced vision-language fused features (PRIMARY for memory bank)
  - Hook 3 (decoder): Per-query embeddings from the cross-modality decoder

The wrapper provides a unified interface for:
  1. Standard inference (boxes, scores, labels)
  2. Feature extraction for memory bank construction
  3. Proposal-level feature access for retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.utils.feature_hooks import FeatureHookManager

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Container for a single image's detection output."""
    boxes: torch.Tensor          # (N, 4) in xyxy format, pixel coords
    scores: torch.Tensor         # (N,)
    labels: list[str]            # (N,) text labels

    def filter_by_score(self, threshold: float) -> "DetectionResult":
        """Return a new DetectionResult with only detections above threshold."""
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=[l for l, m in zip(self.labels, mask) if m],
        )

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class ProposalFeatures:
    """Container for extracted per-proposal features + associated detections."""
    query_features: torch.Tensor      # (N, D_query) per-proposal decoder query embeddings
    encoder_features: torch.Tensor    # (HW, D_enc) flattened enhanced features
    detection: DetectionResult         # associated detection results
    raw_outputs: dict[str, Any] = field(default_factory=dict)  # raw model outputs


# ──────────────────────────────────────────────────────────────────────
# Main wrapper
# ──────────────────────────────────────────────────────────────────────

class GroundingDINOWrapper(nn.Module):
    """
    Frozen Grounding DINO model with feature extraction capabilities.

    Args:
        model_id: HuggingFace model identifier.
            - "IDEA-Research/grounding-dino-tiny" (SwinT backbone, ~172M params)
            - "IDEA-Research/grounding-dino-base" (SwinB backbone, ~232M params)
        device: Target device. If None, auto-selects cuda/cpu.
        dtype: Model precision. Default torch.float16 for memory efficiency.
        box_threshold: Minimum score to keep a detection.
        text_threshold: Minimum text-matching score.
    """

    # Default hook targets within the HuggingFace GroundingDINO architecture
    # These may need adjustment based on exact transformers version
    HOOK_TARGETS = {
        "backbone": "model.backbone",           # Swin Transformer output
        "encoder": "model.encoder",              # Feature Enhancer output
        "decoder": "model.decoder",              # Cross-modality decoder output
    }

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> None:
        super().__init__()

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_id = model_id

        logger.info(f"Loading Grounding DINO: {model_id} on {self.device} ({dtype})")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device)

        # Freeze everything
        self._freeze()

        # Setup feature hooks
        self.hook_manager = FeatureHookManager()
        self._register_hooks()

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"GroundingDINO loaded: {total_params / 1e6:.1f}M params, "
            f"all frozen, {len(self.hook_manager)} hooks registered"
        )

    def _freeze(self) -> None:
        """Freeze all model parameters."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("All model parameters frozen")

    def _register_hooks(self) -> None:
        """Register forward hooks at backbone, encoder, and decoder."""
        available_modules = FeatureHookManager.list_modules(self.model, max_depth=3)
        logger.debug(f"Available modules (depth<=3): {available_modules[:20]}...")

        for hook_name, module_path in self.HOOK_TARGETS.items():
            try:
                self.hook_manager.register(
                    model=self.model,
                    module_path=module_path,
                    name=hook_name,
                )
            except AttributeError as e:
                logger.warning(
                    f"Could not register hook '{hook_name}' at '{module_path}': {e}. "
                    f"Will attempt auto-discovery."
                )
                self._auto_discover_hook(hook_name)

    def _auto_discover_hook(self, hook_name: str) -> None:
        """
        Attempt to find and register a hook by scanning available modules.

        Fallback for when the default HOOK_TARGETS paths don't match
        the exact HuggingFace model structure.
        """
        keywords = {
            "backbone": ["backbone", "swin"],
            "encoder": ["encoder", "enhancer", "feature_enhancer"],
            "decoder": ["decoder"],
        }

        target_keywords = keywords.get(hook_name, [hook_name])
        all_modules = FeatureHookManager.list_modules(self.model, max_depth=4)

        for module_path in all_modules:
            module_name = module_path.split(".")[-1].lower()
            if any(kw in module_name for kw in target_keywords):
                try:
                    self.hook_manager.register(
                        model=self.model,
                        module_path=module_path,
                        name=hook_name,
                    )
                    logger.info(f"Auto-discovered hook '{hook_name}' at '{module_path}'")
                    return
                except AttributeError:
                    continue

        logger.error(
            f"Failed to auto-discover module for hook '{hook_name}'. "
            f"Available modules: {all_modules[:30]}"
        )

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> DetectionResult:
        """
        Run object detection on a single image.

        Args:
            image: PIL Image.
            text_prompt: Detection categories separated by periods.
                         Example: "person . car . bicycle ."
            box_threshold: Override default box threshold.
            text_threshold: Override default text threshold.

        Returns:
            DetectionResult with boxes in xyxy pixel coordinates.
        """
        box_thr = box_threshold or self.box_threshold
        text_thr = text_threshold or self.text_threshold

        # Preprocess
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        # Forward pass (also triggers hooks)
        self.hook_manager.clear()
        outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor(
            [[image.height, image.width]], device=self.device
        )
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=box_thr,
            text_threshold=text_thr,
        )[0]

        return DetectionResult(
            boxes=results["boxes"].cpu(),
            scores=results["scores"].cpu(),
            labels=results["labels"],
        )

    @torch.no_grad()
    def extract_features(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> ProposalFeatures:
        """
        Run detection AND extract intermediate features for memory bank / retrieval.

        Returns:
            ProposalFeatures containing:
              - query_features: per-proposal embeddings from decoder
              - encoder_features: enhanced vision-language features
              - detection: boxes, scores, labels
        """
        box_thr = box_threshold or self.box_threshold
        text_thr = text_threshold or self.text_threshold

        # Preprocess
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        # Forward pass
        self.hook_manager.clear()
        outputs = self.model(**inputs)

        # Post-process detections
        target_sizes = torch.tensor(
            [[image.height, image.width]], device=self.device
        )
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=box_thr,
            text_threshold=text_thr,
        )[0]

        detection = DetectionResult(
            boxes=results["boxes"].cpu(),
            scores=results["scores"].cpu(),
            labels=results["labels"],
        )

        # Extract hooked features
        hooked = self.hook_manager.get_features()

        # Process decoder output to get per-query features
        query_features = self._extract_query_features(outputs, hooked)

        # Process encoder output
        encoder_features = self._extract_encoder_features(hooked)

        return ProposalFeatures(
            query_features=query_features,
            encoder_features=encoder_features,
            detection=detection,
            raw_outputs={
                "logits": outputs.logits.cpu(),
                "pred_boxes": outputs.pred_boxes.cpu(),
            },
        )

    def _extract_query_features(
        self,
        outputs: Any,
        hooked_features: dict[str, Any],
    ) -> torch.Tensor:
        """
        Extract per-proposal query features from decoder outputs.

        Attempts multiple strategies:
        1. Use the last hidden state from the decoder hook
        2. Fall back to the output's last_hidden_state if available
        3. Use pred_boxes shape to infer query dimension from logits
        """
        # Strategy 1: decoder hook
        if "decoder" in hooked_features:
            decoder_out = hooked_features["decoder"]
            if isinstance(decoder_out, tuple):
                # DETR-style: (hidden_states, ...) — take last layer
                query_feats = decoder_out[0]
                if query_feats.dim() == 4:
                    # (num_layers, batch, num_queries, dim) → take last layer
                    query_feats = query_feats[-1]
                elif query_feats.dim() == 3:
                    # (batch, num_queries, dim)
                    pass
                return query_feats.squeeze(0).float()
            elif isinstance(decoder_out, torch.Tensor):
                if decoder_out.dim() == 3:
                    return decoder_out.squeeze(0).float()
                return decoder_out.float()

        # Strategy 2: use model output attributes
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.squeeze(0).float()

        # Strategy 3: construct from logits (fallback)
        logger.warning(
            "Could not extract query features from decoder hook or model outputs. "
            "Using logits as proxy features."
        )
        return outputs.logits.squeeze(0).float()

    def _extract_encoder_features(
        self,
        hooked_features: dict[str, Any],
    ) -> torch.Tensor:
        """
        Extract enhanced vision-language features from the encoder hook.

        These are the PRIMARY features for the memory bank — they encode
        both visual appearance and text-alignment information.
        """
        if "encoder" not in hooked_features:
            logger.warning("Encoder hook not triggered, returning empty tensor")
            return torch.empty(0)

        encoder_out = hooked_features["encoder"]

        if isinstance(encoder_out, tuple):
            # Take the vision features (typically first element)
            vision_feats = encoder_out[0]
            if isinstance(vision_feats, torch.Tensor):
                # Flatten spatial dims: (B, C, H, W) → (B, HW, C) or already (B, N, C)
                if vision_feats.dim() == 4:
                    b, c, h, w = vision_feats.shape
                    vision_feats = vision_feats.reshape(b, c, h * w).permute(0, 2, 1)
                return vision_feats.squeeze(0).float()
            elif isinstance(vision_feats, (list, tuple)):
                # Multi-scale features — concatenate
                all_feats = []
                for feat in vision_feats:
                    if isinstance(feat, torch.Tensor):
                        if feat.dim() == 4:
                            b, c, h, w = feat.shape
                            feat = feat.reshape(b, c, h * w).permute(0, 2, 1)
                        all_feats.append(feat.squeeze(0))
                if all_feats:
                    return torch.cat(all_feats, dim=0).float()

        elif isinstance(encoder_out, torch.Tensor):
            if encoder_out.dim() == 4:
                b, c, h, w = encoder_out.shape
                encoder_out = encoder_out.reshape(b, c, h * w).permute(0, 2, 1)
            return encoder_out.squeeze(0).float()

        logger.warning("Could not parse encoder features, returning empty tensor")
        return torch.empty(0)

    def get_feature_dim(self) -> dict[str, int | None]:
        """
        Return the feature dimensions at each hook point.

        Must be called after at least one forward pass.
        """
        dims = {}
        for name in self.hook_manager.feature_names:
            try:
                feat = self.hook_manager.get(name)
                if isinstance(feat, torch.Tensor):
                    dims[name] = feat.shape[-1]
                elif isinstance(feat, tuple) and len(feat) > 0:
                    first = feat[0]
                    if isinstance(first, torch.Tensor):
                        dims[name] = first.shape[-1]
                    else:
                        dims[name] = None
                else:
                    dims[name] = None
            except KeyError:
                dims[name] = None
        return dims

    def list_available_modules(self, max_depth: int = 4) -> list[str]:
        """List all submodule paths available for hooking."""
        return FeatureHookManager.list_modules(self.model, max_depth=max_depth)

    def forward(self, *args, **kwargs):
        """Direct forward pass through the underlying model."""
        return self.model(*args, **kwargs)
