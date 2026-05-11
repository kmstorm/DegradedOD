"""
Phase B Trainer — Adaptation Module Training.

Trains the ProposalEncoder, RefinementModule, and RefinedDetectionHead
using paired clean↔degraded data with memory retrieval.

Training loop features:
  - AMP (mixed precision) training
  - Gradient accumulation
  - Cosine LR scheduler with warmup
  - Checkpoint save/load with best-model tracking
  - WandB logging (optional)
  - Periodic evaluation
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.training.losses import MODDLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    MODD Phase B trainer.

    Orchestrates the training loop for adaptation modules while keeping
    the G-DINO backbone frozen.

    Args:
        model: MODDDetector instance (or the trainable sub-modules).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (optional).
        config: Training configuration dict.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict | None = None,
        output_dir: str = "./outputs/checkpoints",
    ) -> None:
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training config
        self.epochs = self.config.get("epochs", 50)
        self.lr = self.config.get("learning_rate", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.warmup_epochs = self.config.get("warmup_epochs", 5)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 2)
        self.log_interval = self.config.get("log_interval", 50)
        self.eval_interval = self.config.get("eval_interval", 1)
        self.use_amp = self.config.get("mixed_precision", True)

        # Loss
        loss_cfg = self.config.get("loss", {})
        self.criterion = MODDLoss(
            detection_weight=loss_cfg.get("detection_weight", 1.0),
            contrastive_weight=loss_cfg.get("contrastive_weight", 0.5),
            retrieval_weight=loss_cfg.get("retrieval_weight", 0.3),
        )

        # Optimizer — only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {num_trainable:,}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # LR Scheduler: warmup + cosine decay
        warmup_steps = self.warmup_epochs * len(train_loader)
        total_steps = self.epochs * len(train_loader)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.lr * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # AMP
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.best_epoch = 0

        # WandB
        self.use_wandb = self.config.get("use_wandb", False)
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.get("wandb_project", "modd"),
                    config=self.config,
                    name=self.config.get("wandb_name", None),
                )
            except Exception as e:
                logger.warning(f"WandB init failed: {e}, continuing without logging")

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dict with final training statistics.
        """
        logger.info("=" * 60)
        logger.info("Starting Phase B Training")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.train_loader.batch_size}")
        logger.info(f"  Grad accum: {self.grad_accum_steps}")
        logger.info(f"  LR: {self.lr}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 60)

        total_start = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Evaluate
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._validate(epoch)

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Checkpoint
            monitor_metric = val_metrics.get("mAP_50", train_metrics.get("loss", 0.0))
            is_best = monitor_metric > self.best_metric
            if is_best:
                self.best_metric = monitor_metric
                self.best_epoch = epoch
            self._save_checkpoint(epoch, is_best=is_best)

        total_time = time.time() - total_start
        logger.info(
            f"Training complete: {self.epochs} epochs in {total_time / 3600:.1f}h, "
            f"best mAP@50={self.best_metric:.4f} at epoch {self.best_epoch}"
        )

        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
        }

    def _train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {}
        num_batches = 0
        epoch_start = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            loss_dict = self._train_step(batch, batch_idx)

            # Accumulate losses for logging
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            self.global_step += 1

            # Log interval
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_losses.get("total", 0.0) / num_batches
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / elapsed
                logger.info(
                    f"[Epoch {epoch}/{self.epochs}] "
                    f"[{batch_idx + 1}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"speed={speed:.1f} batch/s"
                )

        # Average losses
        if num_batches > 0:
            epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return epoch_losses

    def _train_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Single training step.

        Flow:
          1. degraded_features → model.proposal_encoder → encoded features
          2. clean_features (detached GT) stay as targets
          3. Compute losses between encoded outputs and clean targets
          4. Backward through proposal_encoder

        Expected batch keys:
            pred_boxes, gt_boxes, pred_scores,
            degraded_features, clean_features,
            anchor_features, positive_features, negative_features
        """
        # Move to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            # ── Pass features through trainable modules ──
            # Degraded features → ProposalEncoder (creates grad graph)
            degraded_raw = batch.get("degraded_features")
            clean_raw = batch.get("clean_features")

            if degraded_raw is not None and degraded_raw.numel() > 0:
                encoded_degraded = self.model.proposal_encoder(degraded_raw)
                # Clean features are targets — encode them but detach
                encoded_clean = self.model.proposal_encoder(clean_raw).detach()
            else:
                encoded_degraded = degraded_raw
                encoded_clean = clean_raw

            # For retrieval loss, also encode anchor/positive/negative
            anchor_raw = batch.get("anchor_features")
            positive_raw = batch.get("positive_features")
            negative_raw = batch.get("negative_features")

            if anchor_raw is not None and anchor_raw.numel() > 0:
                encoded_anchor = self.model.proposal_encoder(anchor_raw)
                encoded_positive = self.model.proposal_encoder(positive_raw).detach()
                encoded_negative = self.model.proposal_encoder(negative_raw).detach()
            else:
                encoded_anchor = anchor_raw
                encoded_positive = positive_raw
                encoded_negative = negative_raw

            # ── Compute losses on encoded features ──
            loss_dict = self.criterion(
                pred_boxes=batch.get("pred_boxes", torch.empty(0, 4, device=self.device)),
                gt_boxes=batch.get("gt_boxes", torch.empty(0, 4, device=self.device)),
                pred_scores=batch.get("pred_scores"),
                gt_scores=batch.get("gt_scores"),
                degraded_features=encoded_degraded,
                clean_features=encoded_clean,
                anchor_features=encoded_anchor,
                positive_features=encoded_positive,
                negative_features=encoded_negative,
            )

            # Scale loss for gradient accumulation
            loss = loss_dict["total"] / self.grad_accum_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """Run validation and compute metrics."""
        self.model.eval()
        val_losses = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                loss_dict = self.criterion(
                    pred_boxes=batch.get("pred_boxes", torch.empty(0, 4, device=self.device)),
                    gt_boxes=batch.get("gt_boxes", torch.empty(0, 4, device=self.device)),
                    pred_scores=batch.get("pred_scores"),
                    gt_scores=batch.get("gt_scores"),
                    degraded_features=batch.get("degraded_features"),
                    clean_features=batch.get("clean_features"),
                    anchor_features=batch.get("anchor_features"),
                    positive_features=batch.get("positive_features"),
                    negative_features=batch.get("negative_features"),
                )

            for k, v in loss_dict.items():
                val_losses[k] = val_losses.get(k, 0.0) + v.item()
            num_batches += 1

        if num_batches > 0:
            val_losses = {k: v / num_batches for k, v in val_losses.items()}

        logger.info(
            f"[Epoch {epoch}] Val: "
            + " ".join(f"{k}={v:.4f}" for k, v in val_losses.items())
        )

        return val_losses

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Log metrics to console and WandB."""
        # Console
        train_str = " ".join(f"train/{k}={v:.4f}" for k, v in train_metrics.items())
        val_str = " ".join(f"val/{k}={v:.4f}" for k, v in val_metrics.items())
        logger.info(f"[Epoch {epoch}] {train_str} {val_str}")

        # WandB
        if self.wandb_run:
            import wandb
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["epoch"] = epoch
            log_dict["lr"] = self.scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=self.global_step)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        # Save latest
        latest_path = self.output_dir / "latest.ckpt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / "best_phase_b.ckpt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: epoch={epoch}, metric={self.best_metric:.4f}")

        # Save periodic
        save_top_k = self.config.get("save_top_k", 3)
        if (epoch + 1) % max(1, self.epochs // save_top_k) == 0:
            epoch_path = self.output_dir / f"epoch_{epoch:03d}.ckpt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric", 0.0)

        logger.info(
            f"Resumed from epoch {self.current_epoch}, "
            f"step {self.global_step}, best={self.best_metric:.4f}"
        )
