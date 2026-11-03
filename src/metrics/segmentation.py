"""
src/metrics/segmentation.py
Pixel-level segmentation metrics for forgery mask evaluation.
All metrics operate on torch tensors or numpy arrays.
"""
from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Dice (F1) score for binary segmentation.
    pred_mask: logits or probabilities (B, 1, H, W) or (B, H, W)
    true_mask: binary ground truth same shape
    """
    if pred_mask.shape != true_mask.shape:
        pred_mask = pred_mask.squeeze(1)
        true_mask = true_mask.squeeze(1)
    probs = torch.sigmoid(pred_mask) if pred_mask.min() < 0 or pred_mask.max() > 1 else pred_mask
    preds = (probs > threshold).float()
    flat_p = preds.view(-1)
    flat_t = true_mask.view(-1).float()
    intersection = (flat_p * flat_t).sum().item()
    return (2.0 * intersection + smooth) / (flat_p.sum().item() + flat_t.sum().item() + smooth)


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard) for binary segmentation."""
    if pred_mask.shape != true_mask.shape:
        pred_mask = pred_mask.squeeze(1)
        true_mask = true_mask.squeeze(1)
    probs = torch.sigmoid(pred_mask) if pred_mask.min() < 0 or pred_mask.max() > 1 else pred_mask
    preds = (probs > threshold).float()
    flat_p = preds.view(-1)
    flat_t = true_mask.view(-1).float()
    intersection = (flat_p * flat_t).sum().item()
    union = (flat_p + flat_t - flat_p * flat_t).sum().item()
    return (intersection + smooth) / (union + smooth)


def pixel_precision_recall(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    """Returns (precision, recall) at pixel level."""
    if pred_mask.shape != true_mask.shape:
        pred_mask = pred_mask.squeeze(1)
        true_mask = true_mask.squeeze(1)
    probs = torch.sigmoid(pred_mask) if pred_mask.min() < 0 or pred_mask.max() > 1 else pred_mask
    preds = (probs > threshold).float().view(-1)
    trues = true_mask.view(-1).float()
    tp = (preds * trues).sum().item()
    fp = (preds * (1 - trues)).sum().item()
    fn = ((1 - preds) * trues).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall


def pixel_auc(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    """AUC-ROC at pixel level. Falls back to 0.5 if only one class present."""
    probs = torch.sigmoid(pred_mask).detach().cpu().numpy().ravel()
    trues = true_mask.detach().cpu().numpy().ravel().astype(int)
    if len(np.unique(trues)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(trues, probs))
    except Exception:
        return 0.5


class SegmentationMetrics:
    """Accumulates segmentation metrics across batches."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._dice_sum = 0.0
        self._iou_sum = 0.0
        self._prec_sum = 0.0
        self._rec_sum = 0.0
        self._n = 0

    def update(self, pred_mask: torch.Tensor, true_mask: torch.Tensor):
        with torch.no_grad():
            self._dice_sum += dice_score(pred_mask, true_mask, self.threshold)
            self._iou_sum += iou_score(pred_mask, true_mask, self.threshold)
            p, r = pixel_precision_recall(pred_mask, true_mask, self.threshold)
            self._prec_sum += p
            self._rec_sum += r
            self._n += 1

    def compute(self) -> dict:
        n = max(self._n, 1)
        return {
            "dice": self._dice_sum / n,
            "iou": self._iou_sum / n,
            "precision": self._prec_sum / n,
            "recall": self._rec_sum / n,
        }
