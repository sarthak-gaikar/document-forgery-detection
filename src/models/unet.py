"""
src/models/unet.py
Model A: U-Net with ResNet-34 encoder via segmentation_models_pytorch.
"""
from __future__ import annotations
import torch
import torch.nn as nn
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


class UNetResNet34(nn.Module):
    """
    Model A: U-Net with ImageNet-pretrained ResNet-34 encoder.
    Input: (B, 3, H, W) — RGB document image
    Output: dict with key "seg_logits": (B, 1, H, W)
    """
    def __init__(self, encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1):
        super().__init__()
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch is not installed. Run: pip install segmentation-models-pytorch")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # raw logits
        )

    def forward(self, x: torch.Tensor) -> dict:
        seg_logits = self.model(x)
        return {"seg_logits": seg_logits}


def build_unet_resnet34(cfg) -> UNetResNet34:
    model_cfg = cfg.model
    weights = getattr(model_cfg, "encoder_weights", "imagenet")
    if str(weights).lower() in ("none", "null", ""):
        weights = None
    return UNetResNet34(
        encoder_weights=weights,
        in_channels=getattr(model_cfg, "in_channels", 3),
        classes=getattr(model_cfg, "classes", 1),
    )
