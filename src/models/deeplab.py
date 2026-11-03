"""
src/models/deeplab.py
Model B: DeepLabV3+ with EfficientNet-B3 encoder via segmentation_models_pytorch.
"""
from __future__ import annotations
import torch
import torch.nn as nn
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


class DeepLabV3PlusEfficientNet(nn.Module):
    """
    Model B: DeepLabV3+ with ImageNet-pretrained EfficientNet-B3 encoder.
    Input: (B, 3, H, W)
    Output: dict with "seg_logits": (B, 1, H, W)
    """
    def __init__(self, encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1):
        super().__init__()
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch is not installed.")
        self.model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {"seg_logits": self.model(x)}


def build_deeplab(cfg) -> DeepLabV3PlusEfficientNet:
    model_cfg = cfg.model
    weights = getattr(model_cfg, "encoder_weights", "imagenet")
    if str(weights).lower() in ("none", "null", ""):
        weights = None
    return DeepLabV3PlusEfficientNet(
        encoder_weights=weights,
        in_channels=getattr(model_cfg, "in_channels", 3),
        classes=getattr(model_cfg, "classes", 1),
    )
