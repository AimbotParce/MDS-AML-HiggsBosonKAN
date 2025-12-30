"""Evaluation helpers (printing metrics and AMS used in the notebook)."""

from typing import Optional

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .ams import ams_score


def get_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, y_proba: torch.Tensor, weights: Optional[torch.Tensor] = None
):
    """
    Return common metrics as a dictionary. y_true / y_pred should be 0/1.
    y_proba is probability for class 1.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    for row in zip(["background", "signal"], p, r, f1, s):
        metrics[f"precision_{row[0]}"] = row[1]
        metrics[f"recall_{row[0]}"] = row[2]
        metrics[f"f1_{row[0]}"] = row[3]
        metrics[f"support_{row[0]}"] = row[4]

    # Compute averages
    for avg in ("micro", "macro", "weighted"):
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=avg)
        metrics[f"precision_{avg}"] = p
        metrics[f"recall_{avg}"] = r
        metrics[f"f1_{avg}"] = f1
        metrics[f"support_{avg}"] = s

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        pass
    if weights is not None:
        metrics["ams"] = ams_score(y_true, y_pred, weights)
    return metrics


def report_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, y_proba: torch.Tensor, weights: Optional[torch.Tensor] = None
):
    """
    Print common metrics. y_true / y_pred should be 0/1.
    y_proba is probability for class 1.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, target_names=["background", "signal"]))
    try:
        print("ROC AUC:", roc_auc_score(y_true, y_proba))
    except Exception:
        pass
    if weights is not None:
        print("AMS:", ams_score(y_true, y_pred, weights))
