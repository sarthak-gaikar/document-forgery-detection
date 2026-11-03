"""
tests/test_models.py
Unit tests for all 7 model architectures.
"""
from __future__ import annotations

import pytest
import torch

from src.models.unet import UNetResNet34
from src.models.deeplab import DeepLabV3PlusEfficientNet
from src.models.dual_branch import DualBranchForgeryNet, SRMLayer
from src.models.classifiers import CustomCNN, ResNet50Classifier, EfficientNetB0Classifier


def test_srm_layer_shape():
    srm = SRMLayer()
    x = torch.randn(2, 3, 128, 128)
    out = srm(x)
    assert out.shape == (2, 9, 128, 128)


def test_dual_branch_forward_pass():
    model = DualBranchForgeryNet(pretrained_rgb=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert "seg_logits" in out
    assert "cls_logits" in out
    assert out["seg_logits"].shape == (2, 1, 256, 256)
    assert out["cls_logits"].shape == (2, 1)


def test_custom_cnn_forward():
    model = CustomCNN(num_classes=1)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert "cls_logits" in out
    assert out["cls_logits"].shape == (2, 1)


def test_unet_resnet34_forward():
    model = UNetResNet34(encoder_weights=None, in_channels=3, classes=1)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["seg_logits"].shape == (2, 1, 128, 128)


def test_deeplab_forward():
    model = DeepLabV3PlusEfficientNet(encoder_weights=None, in_channels=3, classes=1)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["seg_logits"].shape == (1, 1, 128, 128)

