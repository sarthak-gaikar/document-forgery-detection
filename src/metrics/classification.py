"""
src/metrics/classification.py
Document-level classification metrics for forgery detection.
"""
from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


class ClassificationMetrics:
    """Accumulates document-level classification metrics across batches."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._probs: list[float] = []
        self._labels: list[int] = []

    def update(self, cls_logits: torch.Tensor, labels: torch.Tensor):
        probs = torch.sigmoid(cls_logits).detach().cpu().view(-1).tolist()
        labs = labels.detach().cpu().view(-1).long().tolist()
        self._probs.extend(probs)
        self._labels.extend(labs)

    def compute(self) -> dict:
        if len(self._labels) == 0:
            return {"cls_acc": 0.0, "cls_f1": 0.0, "cls_auc": 0.5}
        preds = [1 if p > self.threshold else 0 for p in self._probs]
        acc = accuracy_score(self._labels, preds)
        f1 = f1_score(self._labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(self._labels, self._probs) if len(set(self._labels)) > 1 else 0.5
        except Exception:
            auc = 0.5
        return {"cls_acc": float(acc), "cls_f1": float(f1), "cls_auc": float(auc)}
