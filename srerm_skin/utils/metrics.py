from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1: float
    auc: float | None
    confusion_matrix: list[list[int]]


def _confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    # y in {0,1}
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def compute_binary_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> ClassificationMetrics:
    """Compute accuracy/F1/AUC for binary classification. logits shape: [N,2] or [N]."""
    with torch.no_grad():
        targets_np = targets.detach().cpu().numpy().astype(int)

        if logits.ndim == 2 and logits.shape[1] == 2:
            probs_pos = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs_pos = torch.sigmoid(logits.view(-1))
        probs_np = probs_pos.detach().cpu().numpy()

        pred_np = (probs_np >= 0.5).astype(int)

        acc = float((pred_np == targets_np).mean())

        # F1
        tp = float(((pred_np == 1) & (targets_np == 1)).sum())
        fp = float(((pred_np == 1) & (targets_np == 0)).sum())
        fn = float(((pred_np == 0) & (targets_np == 1)).sum())
        f1 = float(0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn))

        # AUC (guard against single-class edge cases)
        auc = None
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(targets_np)) == 2:
                auc = float(roc_auc_score(targets_np, probs_np))
        except Exception:
            auc = None

        cm = _confusion_binary(targets_np, pred_np)

        return ClassificationMetrics(accuracy=acc, f1=f1, auc=auc, confusion_matrix=cm)
