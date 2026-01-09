from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

from ..evaluate import ams_score


class Metric(ABC):
    name: str

    @abstractmethod
    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
    ) -> float:
        pass


class Accuracy(Metric):
    name = "accuracy"

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
    ) -> float:
        preds = (y_pred >= 0.5).int()
        return torch.mean((preds == y_true).float()).float().item()


class AUC(Metric):
    name = "auc"

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
    ) -> float:

        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        return float(roc_auc_score(y_true_np, y_pred_np))


class AMS(Metric):
    name = "ams"

    def __init__(self, br: float = 10.0):
        self.br = br

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
    ) -> float:
        if w is None:
            raise ValueError("Weights tensor 'w' must be provided for AMS calculation.")
        return ams_score(
            y_true=y_true.detach().int(),
            y_pred=(y_pred.detach() >= 0.5).int(),
            weights=w.detach(),
            br=self.br,
        )


class History(BaseModel):
    train_loss: List[float] = []
    val_loss: List[float] = []
    val_metrics: Dict[str, List[float]] = {}
    val_metrics: Dict[str, List[float]] = {}
