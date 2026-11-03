"""
src/losses/combined.py
Loss functions for segmentation + classification in document forgery detection.

Losses implemented:
- DiceLoss: standard Dice (F1) loss for segmentation
- BCEDiceLoss: Binary Cross-Entropy + Dice (for segmentation models A/B/C)
- FocalDiceLoss: Focal Loss + Dice (alternative for heavy class imbalance)
- MultiTaskLoss: Combined seg + cls loss for Model D (Dual-Branch)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1)
        intersection = (flat_p * flat_t).sum()
        dice = (2.0 * intersection + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """BCE + Dice combined loss. Default for segmentation models A, B, C."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice combined loss."""
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for Model D (Dual-Branch):
        L_total = L_seg + alpha_cls * L_cls
    where L_seg is BCEDiceLoss on the pixel mask,
    and L_cls is BCEWithLogitsLoss on the document label.
    """
    def __init__(self, alpha_cls: float = 0.5, seg_loss: str = "bce_dice", smooth: float = 1.0):
        super().__init__()
        self.alpha_cls = alpha_cls
        if seg_loss == "bce_dice":
            self.seg_loss_fn = BCEDiceLoss(smooth=smooth)
        elif seg_loss == "focal_dice":
            self.seg_loss_fn = FocalDiceLoss(smooth=smooth)
        else:
            raise ValueError(f"Unknown seg_loss: {seg_loss}")
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,    # (B, 1, H, W)
        cls_logits: torch.Tensor,    # (B, 1) or (B,)
        seg_targets: torch.Tensor,   # (B, 1, H, W)
        cls_targets: torch.Tensor,   # (B,) float
    ) -> tuple[torch.Tensor, dict]:
        l_seg = self.seg_loss_fn(seg_logits, seg_targets)
        cls_logits_flat = cls_logits.view(-1)
        cls_targets_flat = cls_targets.view(-1)
        l_cls = self.cls_loss_fn(cls_logits_flat, cls_targets_flat)
        total = l_seg + self.alpha_cls * l_cls
        return total, {"loss_seg": l_seg.item(), "loss_cls": l_cls.item()}


def build_seg_loss(cfg) -> nn.Module:
    """Build segmentation loss from config."""
    seg_loss_name = getattr(cfg.loss, "seg_loss", "bce_dice")
    smooth = float(getattr(cfg.loss, "dice_smooth", 1.0))
    if seg_loss_name == "bce_dice":
        return BCEDiceLoss(smooth=smooth)
    elif seg_loss_name == "focal_dice":
        gamma = float(getattr(cfg.loss, "focal_gamma", 2.0))
        return FocalDiceLoss(gamma=gamma, smooth=smooth)
    else:
        raise ValueError(f"Unknown seg_loss in config: {seg_loss_name}")
