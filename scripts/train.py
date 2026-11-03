"""
scripts/train.py
Training entry point for document forgery detection models.

Usage:
  python scripts/train.py --config configs/dual_branch.yaml
  python scripts/train.py --config configs/unet_resnet34.yaml --resume
  python scripts/train.py --config configs/dual_branch.yaml --data-dir data/synthetic --epochs 5 --batch-size 4
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
from src.utils import load_config, get_logger, set_seed, resolve_device, ensure_dir
from src.data.dataloader import build_dataloaders
from src.losses.combined import build_seg_loss, MultiTaskLoss
from src.training.trainer import Trainer

logger = get_logger("train")


def build_model(cfg):
    """Build model from config model.name or model.arch."""
    model_cfg = cfg.model
    name = getattr(model_cfg, "name", getattr(model_cfg, "arch", "")).lower()

    if "dual_branch" in name:
        from src.models.dual_branch import build_dual_branch
        return build_dual_branch(cfg)
    elif "unet" in name or getattr(model_cfg, "arch", "") == "unet":
        from src.models.unet import build_unet_resnet34
        return build_unet_resnet34(cfg)
    elif "deeplabv3plus" in name or "deeplab" in name:
        from src.models.deeplab import build_deeplab
        return build_deeplab(cfg)
    elif "segformer" in name:
        from src.models.segformer import build_segformer
        return build_segformer(cfg)
    elif "custom_cnn" in name:
        from src.models.classifiers import CustomCNN
        return CustomCNN(num_classes=getattr(model_cfg, "num_classes", 1))
    elif "resnet50" in name:
        from src.models.classifiers import ResNet50Classifier
        return ResNet50Classifier(num_classes=getattr(model_cfg, "num_classes", 1))
    elif "efficientnet_b0" in name:
        from src.models.classifiers import EfficientNetB0Classifier
        return EfficientNetB0Classifier(num_classes=getattr(model_cfg, "num_classes", 1))
    else:
        raise ValueError(f"Unknown model name in config: {name}")


def build_loss(cfg, model):
    """Build the appropriate loss function for the model."""
    model_type = getattr(cfg.model, "type", "segmentation").lower()
    if model_type == "multitask":
        alpha_cls = float(getattr(cfg.training, "alpha_cls", 0.5))
        seg_loss_name = getattr(cfg.loss, "seg_loss", "bce_dice")
        smooth = float(getattr(cfg.loss, "dice_smooth", 1.0))
        return MultiTaskLoss(alpha_cls=alpha_cls, seg_loss=seg_loss_name, smooth=smooth)
    elif model_type == "classification":
        return torch.nn.BCEWithLogitsLoss()
    else:
        return build_seg_loss(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train document forgery detection model")
    parser.add_argument("--config", default="configs/dual_branch.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", default=None, help="Dataset directory (defaults to config paths.data_root)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    # Load config
    cfg = load_config(ROOT / args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr

    # Setup
    set_seed(int(cfg.training.seed))
    device = resolve_device(str(cfg.training.device))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {ROOT / args.config}")

    # Data
    if args.data_dir is not None:
        data_dir = ROOT / args.data_dir
    else:
        cfg_data = getattr(cfg.paths, "data_root", None)
        if cfg_data and (ROOT / cfg_data / "manifest.json").exists():
            data_dir = ROOT / cfg_data
        elif (ROOT / "data/realtext_v2/manifest.json").exists():
            data_dir = ROOT / "data/realtext_v2"
        elif (ROOT / "data/synthetic/manifest.json").exists():
            data_dir = ROOT / "data/synthetic"
        else:
            data_dir = ROOT / "data/realtext_v2"

    if not (data_dir / "manifest.json").exists():
        logger.error(f"Dataset manifest not found at {data_dir / 'manifest.json'}.")
        logger.error("Run: python scripts/download_realtext.py to prepare RealText-V2.")
        sys.exit(1)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.training.batch_size),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {getattr(cfg.model, 'name', 'unknown')} | Params: {n_params/1e6:.1f}M")

    # Loss
    loss_fn = build_loss(cfg, model)

    # Optimizer
    lr = float(cfg.training.lr)
    wd = float(cfg.training.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Paths
    model_name = getattr(cfg.model, "name", "model")
    checkpoint_dir = ROOT / cfg.paths.checkpoint_dir / model_name
    log_dir = ROOT / cfg.paths.log_dir / model_name

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt_path = checkpoint_dir / "last_model.pth"
        if ckpt_path.exists():
            start_epoch = trainer.load_checkpoint(ckpt_path)
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, starting fresh")

    # Train
    best_metrics = trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    logger.info(f"Best val metrics: {best_metrics}")

    # Final evaluation on test set
    logger.info("\nFinal evaluation on test set...")
    test_metrics = trainer.val_epoch(test_loader, epoch=cfg.training.epochs)
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
