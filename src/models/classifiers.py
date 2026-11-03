"""
src/models/classifiers.py
Models E, F, G: image-level document forgery classifiers.
  E: Custom CNN from scratch
  F: ResNet-50 (pretrained)
  G: EfficientNet-B0 (pretrained)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """Model E: Custom CNN from scratch."""
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.features(x)
        x = self.pool(x)
        return {"cls_logits": self.classifier(x)}


class ResNet50Classifier(nn.Module):
    """Model F: ResNet-50 with pretrained ImageNet weights."""
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            if pretrained:
                try:
                    weights = ResNet50_Weights.IMAGENET1K_V2
                    backbone = resnet50(weights=weights)
                except Exception:
                    backbone = resnet50(weights=None)
            else:
                backbone = resnet50(weights=None)
        except Exception:
            backbone = None
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> dict:
        return {"cls_logits": self.model(x)}


class EfficientNetB0Classifier(nn.Module):
    """Model G: EfficientNet-B0 with pretrained ImageNet weights."""
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            if pretrained:
                try:
                    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                    backbone = efficientnet_b0(weights=weights)
                except Exception:
                    backbone = efficientnet_b0(weights=None)
            else:
                backbone = efficientnet_b0(weights=None)
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_features, num_classes)
            self.model = backbone
            self.use_tv = True
        except Exception:
            # Fallback to timm
            import timm
            self.model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
            self.use_tv = False

    def forward(self, x: torch.Tensor) -> dict:
        return {"cls_logits": self.model(x)}


def build_classifier(model_name: str, num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """Factory function for classifier models."""
    name = model_name.lower()
    if name in ("custom_cnn", "e"):
        return CustomCNN(num_classes=num_classes)
    elif name in ("resnet50", "f"):
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    elif name in ("efficientnet_b0", "g"):
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown classifier model name: {model_name}")
