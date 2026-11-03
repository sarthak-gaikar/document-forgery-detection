"""
tests/test_losses_metrics.py
Unit tests for losses and metrics.
"""
from __future__ import annotations

import torch
import pytest

from src.losses.combined import BCEDiceLoss, FocalDiceLoss, MultiTaskLoss, DiceLoss
from src.metrics.segmentation import SegmentationMetrics, dice_score, iou_score
from src.metrics.classification import ClassificationMetrics


def test_bce_dice_loss():
    loss_fn = BCEDiceLoss()
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_multitask_loss():
    loss_fn = MultiTaskLoss(alpha_cls=0.5)
    seg_logits = torch.randn(2, 1, 64, 64)
    cls_logits = torch.randn(2, 1)
    seg_targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    cls_targets = torch.tensor([0.0, 1.0])
    total, d = loss_fn(seg_logits, cls_logits, seg_targets, cls_targets)
    assert total.dim() == 0
    assert "loss_seg" in d
    assert "loss_cls" in d


def test_dice_score_perfect():
    mask = torch.ones(2, 1, 32, 32)
    score = dice_score(mask * 10.0, mask)  # large positive logits -> sigmoid ~ 1
    assert score > 0.99


def test_segmentation_metrics():
    metrics = SegmentationMetrics(threshold=0.5)
    preds = torch.randn(2, 1, 32, 32)
    trues = torch.randint(0, 2, (2, 1, 32, 32)).float()
    metrics.update(preds, trues)
    res = metrics.compute()
    assert "dice" in res
    assert "iou" in res
    assert 0.0 <= res["dice"] <= 1.0


def test_classification_metrics():
    metrics = ClassificationMetrics(threshold=0.5)
    preds = torch.tensor([[2.0], [-2.0]])
    labels = torch.tensor([1, 0])
    metrics.update(preds, labels)
    res = metrics.compute()
    assert "cls_acc" in res
    assert res["cls_acc"] == 1.0

