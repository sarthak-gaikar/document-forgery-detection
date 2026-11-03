"""
scripts/finetune.py
Fine-tune a pretrained model on RealText-V2 with Focal-Dice loss.

Usage:
    python scripts/finetune.py --config configs/realtext_unet_finetune.yaml --weights outputs/checkpoints/realtext/unet_resnet34/best_model.pth --epochs 15
"""
from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.utils import load_config, get_logger, set_seed, resolve_device
from src.data.dataloader import build_dataloaders
from src.losses.combined import build_seg_loss, MultiTaskLoss
from src.training.trainer import Trainer

logger = get_logger("finetune")


def build_model(cfg):
    from scripts.train import build_model as _build_model
    return _build_model(cfg)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune document forgery detection model")
    parser.add_argument("--config", default="configs/realtext_unet_finetune.yaml", help="Path to config YAML")
    parser.add_argument("--weights", default="outputs/checkpoints/realtext/unet_resnet34/best_model.pth", help="Pretrained weights to initialize from")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    cfg.training.epochs = args.epochs
    cfg.training.lr = args.lr
    cfg.training.batch_size = args.batch_size

    set_seed(int(cfg.training.seed))
    device = resolve_device(str(cfg.training.device))
    logger.info(f"Fine-tuning on device: {device}")

    data_dir = ROOT / getattr(cfg.paths, "data_root", "data/realtext_v2")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.training.batch_size),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    logger.info(f"Dataset: Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")

    # Build model and load weights
    model = build_model(cfg)
    weights_path = ROOT / args.weights
    if weights_path.exists():
        logger.info(f"Loading pretrained weights from: {weights_path}")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        logger.warning(f"Weights file not found at {weights_path}; starting from scratch")

    model_type = getattr(cfg.model, "type", "segmentation").lower()
    if model_type == "multitask":
        loss_fn = MultiTaskLoss(
            alpha_cls=float(getattr(cfg.training, "alpha_cls", 0.5)),
            seg_loss=getattr(cfg.loss, "seg_loss", "focal_dice"),
        )
    else:
        loss_fn = build_seg_loss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    model_name = getattr(cfg.model, "name", "model")
    checkpoint_dir = ROOT / cfg.paths.checkpoint_dir / model_name
    log_dir = ROOT / cfg.paths.log_dir / model_name

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # If the checkpoint had a previous best dice, seed it so only improvements are marked best
    if weights_path.exists() and "best_dice" in state:
        trainer.best_dice = float(state["best_dice"])
        logger.info(f"Baseline best Dice: {trainer.best_dice:.4f}")

    logger.info(f"Starting {args.epochs} fine-tuning epochs with Focal-Dice loss (lr={args.lr})...")
    best_metrics = trainer.fit(train_loader, val_loader, start_epoch=0)
    logger.info(f"Fine-tuning complete. Best metrics: {best_metrics}")

    logger.info("\nEvaluating fine-tuned model on test set...")
    test_metrics = trainer.val_epoch(test_loader, epoch=cfg.training.epochs)
    logger.info(f"Test Set Metrics: {test_metrics}")


if __name__ == "__main__":
    main()

