"""
src/models/segformer.py
Model C: SegFormer-B2 for semantic segmentation (used as pixel-level forgery mask predictor).
Uses transformers library from HuggingFace.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerB2(nn.Module):
    """
    Model C: SegFormer-B2.
    Input: (B, 3, H, W)
    Output: dict with "seg_logits": (B, 1, H, W) upsampled to input resolution

    Falls back to a lightweight CNN if transformers library is unavailable.
    """
    def __init__(self, image_size: int = 512, classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.image_size = image_size
        self.classes = classes

        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
            if pretrained:
                self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/mit-b2",
                    num_labels=classes,
                    ignore_mismatched_sizes=True,
                )
            else:
                config = SegformerConfig.from_pretrained("nvidia/mit-b2", num_labels=classes)
                self.backbone = SegformerForSemanticSegmentation(config)
            self.use_hf = True
        except Exception:
            # Fallback: lightweight encoder-decoder CNN
            self.backbone = None
            self.use_hf = False
            self._build_fallback(classes)

    def _build_fallback(self, classes: int):
        """Simple CNN fallback when transformers/HF not available."""
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, classes, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        if self.use_hf:
            outputs = self.backbone(pixel_values=x)
            logits = outputs.logits  # (B, classes, H/4, W/4)
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        else:
            feat = self.encoder(x)
            logits = self.decoder(feat)
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return {"seg_logits": logits}


def build_segformer(cfg) -> SegFormerB2:
    model_cfg = cfg.model
    return SegFormerB2(
        image_size=getattr(model_cfg, "image_size", 512),
        classes=getattr(model_cfg, "classes", 1),
        pretrained=True,
    )
