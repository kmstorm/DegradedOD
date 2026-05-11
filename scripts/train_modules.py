"""
Phase B: MODD Adaptation Training on ExDark.

Two modes:
  1. --cache-features: Pre-extract G-DINO features for all ExDark images (one-time)
  2. (default): Train on cached features — no G-DINO in loop → ~30s/epoch

Usage:
    # Step 1: Cache features (one-time, ~2h)
    CUDA_VISIBLE_DEVICES=6 python scripts/train_modules.py --cache-features

    # Step 2: Train on cached features (~15 min for 30 epochs)
    CUDA_VISIBLE_DEVICES=6 python scripts/train_modules.py \
        --epochs 30 --lr 1e-4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.exdark_dataset import ExDarkDataset, EXDARK_TO_COCO
from src.models.memory_bank import MemoryBank
from src.models.modd_detector import MODDConfig, MODDDetector
from src.training.losses import MODDLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_modules")


# ──────────────────────────────────────────────────────────────────────
# IoU Greedy Matching
# ──────────────────────────────────────────────────────────────────────

def iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes. (N, M)."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


def greedy_iou_match(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy IoU matching between predictions and GT."""
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
        )

    iou_mat = iou_matrix(pred_boxes, gt_boxes)  # (N_pred, N_gt)

    pred_indices = []
    gt_indices = []
    ious = []
    gt_matched = set()
    pred_matched = set()

    flat_ious, flat_idx = iou_mat.flatten().sort(descending=True)
    n_gt = iou_mat.shape[1]
    for k in range(len(flat_ious)):
        if flat_ious[k] < iou_threshold:
            break
        pi = (flat_idx[k] // n_gt).item()
        gi = (flat_idx[k] % n_gt).item()
        if pi not in pred_matched and gi not in gt_matched:
            pred_indices.append(pi)
            gt_indices.append(gi)
            ious.append(flat_ious[k].item())
            pred_matched.add(pi)
            gt_matched.add(gi)

    if not pred_indices:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
        )

    return (
        torch.tensor(pred_indices, dtype=torch.long),
        torch.tensor(gt_indices, dtype=torch.long),
        torch.tensor(ious),
    )


# ──────────────────────────────────────────────────────────────────────
# Feature Caching
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cache_gdino_features(
    data_dir: str,
    model_id: str,
    cache_dir: str,
    device: str = "cuda",
) -> None:
    """
    Pre-extract G-DINO features for all ExDark images (train + val).
    Saves per-image tensors to cache_dir/{split}/{idx}.pt
    """
    from src.models.gdino_wrapper import GroundingDINOWrapper

    cache_dir = Path(cache_dir)
    gdino = GroundingDINOWrapper(model_id=model_id, device=device, dtype=torch.float16)

    coco_names = sorted(set(EXDARK_TO_COCO.values()))
    text_prompt = " . ".join(coco_names) + " ."

    for split in ["train", "val"]:
        dataset = ExDarkDataset(data_dir, split=split, val_ratio=0.2)
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Caching {split}: {len(dataset)} images → {split_dir}")
        t0 = time.time()
        cached = 0
        skipped = 0

        for idx in range(len(dataset)):
            out_path = split_dir / f"{idx}.pt"
            if out_path.exists():
                cached += 1
                continue

            sample = dataset[idx]
            image = sample["image"]
            gt_boxes = sample["gt_boxes"]
            gt_labels = sample["gt_labels"]

            if gt_boxes.shape[0] == 0:
                skipped += 1
                continue

            try:
                proposal_data = gdino.extract_features(
                    image, text_prompt, box_threshold=0.15,
                )
            except Exception as e:
                logger.warning(f"[{idx}] G-DINO failed: {e}")
                skipped += 1
                continue

            n_det = len(proposal_data.detection)
            if n_det == 0:
                skipped += 1
                continue

            # Filter query features to detected proposals only
            raw_logits = proposal_data.raw_outputs["logits"].squeeze(0)
            max_scores = raw_logits.sigmoid().max(dim=-1).values
            det_mask = max_scores >= 0.15
            detected_query_feats = proposal_data.query_features[det_mask]

            if detected_query_feats.shape[0] != n_det:
                detected_query_feats = proposal_data.query_features[:n_det]

            # Save cached features (CPU tensors)
            torch.save({
                "query_features": detected_query_feats.cpu(),  # (N, 256)
                "boxes": proposal_data.detection.boxes.cpu(),   # (N, 4)
                "scores": proposal_data.detection.scores.cpu(), # (N,)
                "labels": proposal_data.detection.labels,       # list[str]
                "gt_boxes": gt_boxes,                            # (M, 4)
                "gt_labels": gt_labels,                          # list[str]
                "image_id": sample["image_id"],
            }, out_path)

            cached += 1
            if cached % 200 == 0:
                elapsed = time.time() - t0
                speed = cached / elapsed
                logger.info(
                    f"  [{split}] {cached}/{len(dataset)} cached, "
                    f"{skipped} skipped, {speed:.1f} img/s"
                )

        elapsed = time.time() - t0
        logger.info(
            f"  [{split}] Done: {cached} cached, {skipped} skipped, "
            f"{elapsed:.0f}s ({cached/max(elapsed,1):.1f} img/s)"
        )

    logger.info("Feature caching complete!")


class CachedExDarkDataset(Dataset):
    """Loads pre-cached G-DINO features. Preloads all into RAM for speed."""

    def __init__(self, cache_dir: str, split: str = "train", preload: bool = True) -> None:
        self.cache_dir = Path(cache_dir) / split
        self.files = sorted(self.cache_dir.glob("*.pt"))
        self.data = None
        if preload:
            logger.info(f"Preloading {len(self.files)} cached samples into RAM...")
            self.data = [torch.load(f, weights_only=False) for f in self.files]
        logger.info(f"CachedExDarkDataset [{split}]: {len(self.files)} samples (preload={preload})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        if self.data is not None:
            return self.data[idx]
        return torch.load(self.files[idx], weights_only=False)


def collate_batch(batch):
    """Return list of samples for batched processing."""
    return batch


# ──────────────────────────────────────────────────────────────────────
# Trainer (Cached Mode — no G-DINO in loop)
# ──────────────────────────────────────────────────────────────────────

class MODDTrainer:
    """
    MODD training on cached ExDark features.

    Each step:
      1. Load cached G-DINO features (no G-DINO inference!)
      2. forward_train() → Encoder → Retrieve → Refine → DetHead
      3. Match refined proposals to ExDark GT (IoU greedy)
      4. Compute losses and backprop
    """

    def __init__(
        self,
        detector: MODDDetector,
        memory_bank: MemoryBank,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict | None = None,
        output_dir: str = "./outputs/checkpoints",
    ) -> None:
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = detector.to(self.device)
        self.memory_bank = memory_bank
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training params
        self.epochs = self.config.get("epochs", 30)
        self.lr = self.config.get("learning_rate", 1e-4)
        self.grad_accum = self.config.get("gradient_accumulation_steps", 1)
        self.log_interval = self.config.get("log_interval", 10)
        self.use_amp = self.config.get("mixed_precision", True)
        self.iou_threshold = self.config.get("iou_match_threshold", 0.3)
        self.batch_size = self.config.get("batch_size", 32)

        # Loss
        loss_cfg = self.config.get("loss", {})
        self.criterion = MODDLoss(
            detection_weight=loss_cfg.get("detection_weight", 1.0),
            contrastive_weight=loss_cfg.get("contrastive_weight", 0.5),
            retrieval_weight=loss_cfg.get("retrieval_weight", 0.3),
        )

        # Optimizer
        param_groups = []
        for group in self.detector.trainable_parameters():
            param_groups.append({
                "params": list(group["params"]),
                "lr": self.lr * group.get("lr_scale", 1.0),
                "name": group.get("name", "unknown"),
            })
        total_trainable = sum(
            p.numel() for g in param_groups for p in g["params"] if p.requires_grad
        )
        logger.info(f"Trainable parameters: {total_trainable:,}")
        for g in param_groups:
            n = sum(p.numel() for p in g["params"] if p.requires_grad)
            logger.info(f"  {g['name']}: {n:,} params, lr={g['lr']:.2e}")

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.config.get("weight_decay", 1e-4),
        )

        # Scheduler
        total_steps = self.epochs * len(train_loader) // self.grad_accum
        warmup_steps = self.config.get("warmup_epochs", 3) * len(train_loader) // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=max(total_steps, 1),
            pct_start=min(warmup_steps / max(total_steps, 1), 0.3),
            anneal_strategy="cos",
        )

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Tracking
        self.global_step = 0
        self.best_metric = 0.0
        self.best_epoch = 0

    def train(self) -> dict:
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("Phase B: MODD Training on Cached ExDark Features")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Grad accum: {self.grad_accum}")
        logger.info(f"  LR: {self.lr}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  Memory bank: {self.memory_bank.num_entries} entries")
        logger.info("=" * 60)

        t_start = time.time()
        for epoch in range(self.epochs):
            metrics = self._train_epoch(epoch)

            # Validation every 5 epochs
            if self.val_loader and (epoch + 1) % 5 == 0:
                val_metrics = self._validate(epoch)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            # Checkpoint
            loss = metrics.get("total", float("inf"))
            is_best = loss < self.best_metric or self.best_metric == 0.0
            if is_best:
                self.best_metric = loss
                self.best_epoch = epoch
            self._save_checkpoint(epoch, is_best)

        total_time = time.time() - t_start
        logger.info(
            f"Training complete: {self.epochs} epochs in {total_time/60:.1f}min, "
            f"best loss={self.best_metric:.4f} at epoch {self.best_epoch}"
        )
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch}

    def _train_epoch(self, epoch: int) -> dict:
        """Train one epoch on cached features (batched)."""
        self.detector.train()
        epoch_losses = {}
        num_valid = 0
        t0 = time.time()
        self.optimizer.zero_grad()

        for batch_idx, samples in enumerate(self.train_loader):
            loss_dict = self._train_step_batched(samples)

            if loss_dict is not None:
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                num_valid += 1

            # Optimizer step every grad_accum batches
            if (batch_idx + 1) % self.grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            self.global_step += 1

            # Log
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_losses.get("total", 0) / max(num_valid, 1)
                elapsed = time.time() - t0
                speed = (batch_idx + 1) * self.batch_size / elapsed
                lr = self.optimizer.param_groups[0]["lr"]
                vram = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"[E{epoch}/{self.epochs}] [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"{speed:.0f} img/s VRAM={vram:.1f}GB"
                )

        # Flush remaining grads
        if (batch_idx + 1) % self.grad_accum != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        if num_valid > 0:
            epoch_losses = {k: v / num_valid for k, v in epoch_losses.items()}

        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items())
        elapsed = time.time() - t0
        logger.info(f"[Epoch {epoch}] {metrics_str} ({elapsed:.0f}s, {num_valid} valid batches)")

        return epoch_losses

    def _train_step_batched(self, samples: list[dict]) -> dict | None:
        """
        Batched training step: concatenate proposals from all images in batch,
        run single forward pass (global FAISS search), split back for per-image loss.
        """
        # ── Collect valid samples and concatenate proposals ──
        all_feats, all_boxes, all_scores = [], [], []
        gt_boxes_list = []
        offsets = [0]

        for s in samples:
            qf = s["query_features"]
            gt = s["gt_boxes"]
            if qf.shape[0] == 0 or gt.shape[0] == 0:
                continue
            all_feats.append(qf)
            all_boxes.append(s["boxes"])
            all_scores.append(s["scores"])
            gt_boxes_list.append(gt)
            offsets.append(offsets[-1] + qf.shape[0])

        if len(all_feats) == 0:
            return None

        # Concatenate into single tensors: (1, sum_N, D)
        cat_feats = torch.cat(all_feats).unsqueeze(0).to(self.device)
        cat_boxes = torch.cat(all_boxes).unsqueeze(0).to(self.device)
        cat_scores = torch.cat(all_scores).unsqueeze(0).to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            # Single forward pass — class_labels=None triggers global FAISS (FAST)
            outputs = self.detector.forward_train(
                query_features=cat_feats,
                initial_boxes=cat_boxes,
                initial_scores=cat_scores,
                class_labels=None,  # Global search, no per-query loop
                memory_bank=self.memory_bank,
            )

            refined_boxes_all = outputs["refined_boxes"].squeeze(0)
            refined_scores_all = outputs["refined_scores"].squeeze(0)
            encoded_all = outputs["encoded_features"].squeeze(0)
            retrieved_all = outputs["retrieved_features"].squeeze(0)

            # ── Per-image loss computation ──
            batch_loss_dicts = []

            for i in range(len(gt_boxes_list)):
                start, end = offsets[i], offsets[i + 1]
                gt_boxes = gt_boxes_list[i].to(self.device)

                r_boxes = refined_boxes_all[start:end]
                r_scores = refined_scores_all[start:end]
                enc_feats = encoded_all[start:end]
                ret_feats = retrieved_all[start:end]

                # Match refined proposals to GT
                pred_idx, gt_idx, match_ious = greedy_iou_match(
                    r_boxes.detach(), gt_boxes, iou_threshold=self.iou_threshold,
                )
                if len(pred_idx) == 0:
                    continue

                # Positive pairs
                m_pred_boxes = r_boxes[pred_idx]
                m_gt_boxes = gt_boxes[gt_idx]
                m_pred_scores = r_scores[pred_idx]
                m_gt_scores = match_ious.to(self.device)

                # Negative proposals (unmatched → target score = 0)
                matched_set = set(pred_idx.tolist())
                unmatched = [j for j in range(r_boxes.shape[0]) if j not in matched_set]
                n_neg = min(len(unmatched), len(pred_idx) * 2)
                if n_neg > 0:
                    neg_idx = torch.tensor(unmatched[:n_neg], dtype=torch.long)
                    all_p_scores = torch.cat([m_pred_scores, r_scores[neg_idx]])
                    all_g_scores = torch.cat([m_gt_scores, torch.zeros(n_neg, device=self.device)])
                else:
                    all_p_scores = m_pred_scores
                    all_g_scores = m_gt_scores

                # Contrastive
                m_enc = enc_feats[pred_idx]
                m_ret_top1 = ret_feats[pred_idx, 0, :]
                neg_roll = torch.roll(m_ret_top1, shifts=1, dims=0)

                loss_dict = self.criterion(
                    pred_boxes=m_pred_boxes, gt_boxes=m_gt_boxes,
                    pred_scores=all_p_scores, gt_scores=all_g_scores,
                    degraded_features=m_enc,
                    clean_features=m_ret_top1.detach(),
                    anchor_features=m_enc,
                    positive_features=m_ret_top1.detach(),
                    negative_features=neg_roll.detach(),
                )
                batch_loss_dicts.append(loss_dict)

            if not batch_loss_dicts:
                return None

            # Average loss across images in batch
            avg_loss_dict = {}
            for k in batch_loss_dicts[0]:
                avg_loss_dict[k] = sum(d[k] for d in batch_loss_dicts) / len(batch_loss_dicts)

            loss = avg_loss_dict["total"] / self.grad_accum

        self.scaler.scale(loss).backward()

        return {k: v.item() for k, v in avg_loss_dict.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """Validation on cached features."""
        self.detector.eval()
        val_losses = {}
        n = 0

        for samples in self.val_loader:
            sample = samples[0]  
            query_feats = sample["query_features"].to(self.device)
            init_boxes = sample["boxes"].to(self.device)
            init_scores = sample["scores"].to(self.device)
            gt_boxes = sample["gt_boxes"].to(self.device)

            if gt_boxes.shape[0] == 0 or query_feats.shape[0] == 0:
                continue

            outputs = self.detector.forward_train(
                query_features=query_feats.unsqueeze(0),
                initial_boxes=init_boxes.unsqueeze(0),
                initial_scores=init_scores.unsqueeze(0),
                class_labels=None,  
                memory_bank=self.memory_bank,
            )

            refined_boxes = outputs["refined_boxes"].squeeze(0)
            refined_scores = outputs["refined_scores"].squeeze(0)
            pred_idx, gt_idx, match_ious = greedy_iou_match(
                refined_boxes, gt_boxes, iou_threshold=self.iou_threshold,
            )
            if len(pred_idx) == 0:
                continue

            loss_dict = self.criterion(
                pred_boxes=refined_boxes[pred_idx],
                gt_boxes=gt_boxes[gt_idx],
                pred_scores=refined_scores[pred_idx],
                gt_scores=match_ious.to(self.device),
            )

            for k, v in loss_dict.items():
                val_losses[k] = val_losses.get(k, 0.0) + v.item()
            n += 1

        if n > 0:
            val_losses = {k: v / n for k, v in val_losses.items()}
        val_str = " ".join(f"val/{k}={v:.4f}" for k, v in val_losses.items())
        logger.info(f"[Epoch {epoch}] {val_str} ({n} val samples)")
        return val_losses

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.detector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        torch.save(checkpoint, self.output_dir / "latest.ckpt")
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_phase_b.ckpt")
            logger.info(f"  ★ New best: epoch={epoch}, loss={self.best_metric:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.output_dir / f"epoch_{epoch:03d}.ckpt")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase B: MODD training on ExDark")
    parser.add_argument("--data-dir", type=str, default="./data/exdark")
    parser.add_argument("--memory-bank", type=str, default="./outputs/memory_bank_final")
    parser.add_argument("--output-dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--cache-dir", type=str, default="./outputs/gdino_cache")
    parser.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-features", action="store_true",
                        help="Pre-extract G-DINO features (run once before training)")
    args = parser.parse_args()

    # ── Mode 1: Cache features ──
    if args.cache_features:
        logger.info("=" * 60)
        logger.info("Caching G-DINO features for ExDark")
        logger.info("=" * 60)
        cache_gdino_features(
            data_dir=args.data_dir,
            model_id=args.model_id,
            cache_dir=args.cache_dir,
        )
        return

    # ── Mode 2: Train on cached features ──
    cache_path = Path(args.cache_dir) / "train"
    if not cache_path.exists() or len(list(cache_path.glob("*.pt"))) == 0:
        logger.error(
            f"No cached features found at {cache_path}!\n"
            f"Run first: python scripts/train_modules.py --cache-features"
        )
        return

    logger.info("=" * 60)
    logger.info("Phase B: MODD Training on Cached ExDark Features")
    logger.info("=" * 60)

    # Load memory bank
    logger.info(f"Loading memory bank from {args.memory_bank}...")
    memory_bank = MemoryBank.load(args.memory_bank)
    logger.info(f"Memory bank: {memory_bank.num_entries} entries")

    # Create MODDDetector (no G-DINO needed for cached training,
    # but forward_train needs the trainable modules which are inside MODDDetector)
    logger.info("Creating MODDDetector...")
    modd_config = MODDConfig(model_id=args.model_id, dtype="float16")
    detector = MODDDetector(modd_config)

    # Cached datasets 
    train_dataset = CachedExDarkDataset(args.cache_dir, split="train", preload=True)
    val_dataset = CachedExDarkDataset(args.cache_dir, split="val", preload=True)

    if args.max_samples:
        train_dataset.data = train_dataset.data[:args.max_samples]
        train_dataset.files = train_dataset.files[:args.max_samples]

    batch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_batch,  
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )

    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "gradient_accumulation_steps": args.grad_accum,
        "mixed_precision": True,
        "log_interval": 10,
        "iou_match_threshold": 0.3,
        "batch_size": batch_size,
        "loss": {
            "detection_weight": 1.0,
            "contrastive_weight": 0.5,
            "retrieval_weight": 0.3,
        },
    }

    trainer = MODDTrainer(
        detector=detector,
        memory_bank=memory_bank,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir,
    )

    results = trainer.train()
    logger.info(f"Done: {results}")


if __name__ == "__main__":
    main()
