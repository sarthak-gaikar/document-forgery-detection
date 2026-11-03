"""
src/training/trainer.py
Full training loop for segmentation and multi-task forgery detection models.

Supports:
- Mixed precision (torch.cuda.amp)
- Cosine annealing LR schedule with warmup
- Gradient clipping
- Early stopping on validation Dice
- Checkpoint saving (best + last)
- TensorBoard + CSV logging
- Both segmentation-only and multi-task (seg+cls) models
"""
from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import time
import shutil
import gc
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import get_logger, ensure_dir
from src.metrics.segmentation import SegmentationMetrics
from src.metrics.classification import ClassificationMetrics

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

logger = get_logger("trainer")


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr)
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_epochs:
            scale = self._step / max(1, self.warmup_epochs)
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base_lr * scale
        else:
            self._cosine.step()

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


class Trainer:
    """
    Universal trainer supporting:
    - Segmentation models (output dict has 'seg_logits')
    - Multi-task models (output dict has 'seg_logits' + 'cls_logits')
    
    Args:
        model: nn.Module
        loss_fn: callable(outputs_dict, batch) -> (loss_tensor, loss_dict)
        optimizer: torch.optim.Optimizer
        device: torch.device
        cfg: OmegaConf config
        checkpoint_dir: directory for saving checkpoints
        log_dir: directory for TensorBoard + CSV logs
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)

        # Training config
        tc = cfg.training
        self.epochs = int(tc.epochs)
        self.grad_clip = float(getattr(tc, "grad_clip", 1.0))
        self.amp = bool(getattr(tc, "amp", True)) and device.type == "cuda"
        self.early_stop_patience = int(getattr(tc, "early_stopping_patience", 10))
        self.log_interval = int(getattr(cfg.logging, "log_interval", 20))
        self.threshold = float(getattr(cfg.metrics, "threshold", 0.5))

        # Scheduler
        warmup = int(getattr(tc, "warmup_epochs", 5))
        self.scheduler = WarmupCosineScheduler(optimizer, warmup, self.epochs)

        # Mixed precision
        if self.amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except Exception:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # TensorBoard
        self.writer = None
        if TB_AVAILABLE and getattr(cfg.logging, "tensorboard", True):
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # CSV log
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None

        # State
        self.best_dice = 0.0
        self._no_improve = 0
        self.global_step = 0

    def _compute_loss(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        """Routes to the loss function for seg-only or multitask outputs."""
        seg_targets = batch["mask"].to(self.device)
        seg_logits = outputs.get("seg_logits")

        if "cls_logits" in outputs and "label" in batch:
            cls_logits = outputs["cls_logits"]
            cls_targets = batch["label"].to(self.device)
            return self.loss_fn(seg_logits, cls_logits, seg_targets, cls_targets)
        elif seg_logits is not None:
            loss = self.loss_fn(seg_logits, seg_targets)
            return loss, {"loss_seg": loss.item()}
        elif "cls_logits" in outputs and "label" in batch:
            cls_logits = outputs["cls_logits"].view(-1)
            cls_targets = batch["label"].to(self.device).view(-1)
            loss = self.loss_fn(cls_logits, cls_targets)
            return loss, {"loss_cls": loss.item()}
        else:
            raise ValueError(f"Cannot compute loss for output keys: {list(outputs.keys())}")

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        seg_metrics = SegmentationMetrics(self.threshold)
        cls_metrics = ClassificationMetrics(self.threshold)
        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad()

            if self.amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_dict = self._compute_loss(outputs, batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, loss_dict = self._compute_loss(outputs, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            with torch.no_grad():
                if "seg_logits" in outputs:
                    seg_metrics.update(outputs["seg_logits"].detach(), batch["mask"].to(self.device))
                if "cls_logits" in outputs and "label" in batch:
                    cls_metrics.update(outputs["cls_logits"].detach(), batch["label"].to(self.device))

            if (i + 1) % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"  Epoch {epoch} step {i+1}/{len(loader)} | loss={loss.item():.4f} | lr={lr:.2e}")

            if self.writer:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)

        seg_results = seg_metrics.compute() if seg_metrics._n > 0 else {}
        cls_results = cls_metrics.compute() if len(cls_metrics._labels) > 0 else {}

        return {"loss": total_loss / max(n_batches, 1), **seg_results, **cls_results}

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.eval()
        seg_metrics = SegmentationMetrics(self.threshold)
        cls_metrics = ClassificationMetrics(self.threshold)
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            if self.amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, _ = self._compute_loss(outputs, batch)
            else:
                outputs = self.model(images)
                loss, _ = self._compute_loss(outputs, batch)

            total_loss += loss.item()
            n_batches += 1
            if "seg_logits" in outputs:
                seg_metrics.update(outputs["seg_logits"].detach(), batch["mask"].to(self.device))
            if "cls_logits" in outputs and "label" in batch:
                cls_metrics.update(outputs["cls_logits"].detach(), batch["label"].to(self.device))

        seg_results = seg_metrics.compute() if seg_metrics._n > 0 else {}
        cls_results = cls_metrics.compute() if len(cls_metrics._labels) > 0 else {}

        if self.writer:
            for k, v in {**seg_results, **cls_results}.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

        return {"loss": total_loss / max(n_batches, 1), **seg_results, **cls_results}

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
            "metrics": metrics,
        }
        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(state, last_path)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(last_path, best_path)
            score_key = "dice" if "dice" in metrics else "cls_acc"
            score_val = metrics.get(score_key, 0.0)
            logger.info(f"  Best model ({score_key}={score_val:.4f}) saved -> {best_path}")

    def load_checkpoint(self, path: str | Path) -> int:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.best_dice = state.get("best_dice", 0.0)
        return state.get("epoch", 0) + 1

    def _log_csv(self, epoch: int, split: str, metrics: dict):
        if self._csv_writer is None:
            fieldnames = ["epoch", "split"] + sorted(metrics.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        row = {"epoch": epoch, "split": split, **metrics}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0) -> dict:
        logger.info(f"Starting training: {self.epochs} epochs, device={self.device}, amp={self.amp}")
        best_metrics = {}

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()
            logger.info(f"\n{'='*60}\nEpoch [{epoch+1}/{self.epochs}]")

            train_metrics = self.train_epoch(train_loader, epoch)
            self._log_csv(epoch + 1, "train", train_metrics)
            d_tr = train_metrics.get("dice", train_metrics.get("cls_acc", 0))
            logger.info(f"  Train | loss={train_metrics['loss']:.4f} | metric={d_tr:.4f}")

            # Free GPU and host memory before validation
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            val_metrics = self.val_epoch(val_loader, epoch)
            self._log_csv(epoch + 1, "val", val_metrics)
            d_val = val_metrics.get("dice", val_metrics.get("cls_acc", 0))
            logger.info(f"  Val   | loss={val_metrics['loss']:.4f} | metric={d_val:.4f} | time={time.time()-t0:.1f}s")

            # Free memory after validation
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.scheduler.step()

            is_best = d_val > self.best_dice
            if is_best:
                self.best_dice = d_val
                best_metrics = val_metrics.copy()
                self._no_improve = 0
            else:
                self._no_improve += 1
                logger.info(f"  No improvement for {self._no_improve}/{self.early_stop_patience} epochs")

            self._save_checkpoint(epoch, val_metrics, is_best)

            if self._no_improve >= self.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        if self._csv_file:
            self._csv_file.close()
        if self.writer:
            self.writer.close()

        logger.info(f"\nTraining complete. Best val score: {self.best_dice:.4f}")
        return best_metrics
