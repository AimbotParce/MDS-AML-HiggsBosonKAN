from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Dataset:
    train_input: torch.Tensor
    train_label: torch.Tensor
    train_weight: torch.Tensor
    val_input: torch.Tensor
    val_label: torch.Tensor
    val_weight: torch.Tensor


from .cross_validation import prepare_dataset_cv
from .playground import create_dataset

__all__ = ["create_dataset", "prepare_dataset_cv", "Dataset"]
