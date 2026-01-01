from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from pydantic import BaseModel


class Metric(ABC):
    name: str

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        pass


class Accuracy(Metric):
    name = "accuracy"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        preds = (y_pred >= 0.5).int()
        return torch.mean((preds == y_true).float()).float().item()


class History(BaseModel):
    train_loss: List[float] = []
    val_loss: List[float] = []
    val_metrics: Dict[str, List[float]] = {}
