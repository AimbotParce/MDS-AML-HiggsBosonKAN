from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import mlflow
import torch
from kan import KAN
from torch import nn
from tqdm.auto import tqdm

from ..datasets import Dataset
from .metrics import History, Metric


def fit_kan(
    model: KAN,
    dataset: Dataset,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: Optional[int] = None,
    epochs: int = 1,
    shuffle: bool = True,
    lamb: float = 0.0,
    lamb_l1: float = 1.0,
    lamb_entropy: float = 2.0,
    lamb_coef: float = 0.0,
    lamb_coefdiff: float = 0.0,
    metrics: Sequence[Metric] = [],
    singularity_avoiding: bool = False,
    y_th: float = 1000.0,
    reg_metric: str = "edge_forward_spline_n",
    random_state: Optional[int] = None,
):
    """
    Custom fit function for a KAN, inspired by the pykan library and modified to log metrics to mlflow, and to not
    choose items randomly from the dataset but rather use mini-batches, shuffling the dataset at each epoch.
    """
    mlflow.log_param("loss_fn", loss_fn._get_name())
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("shuffle", shuffle)
    mlflow.log_param("lamb", lamb)
    mlflow.log_param("lamb_l1", lamb_l1)
    mlflow.log_param("lamb_entropy", lamb_entropy)
    mlflow.log_param("lamb_coef", lamb_coef)
    mlflow.log_param("lamb_coefdiff", lamb_coefdiff)
    mlflow.log_param("singularity_avoiding", singularity_avoiding)
    mlflow.log_param("y_th", y_th)
    mlflow.log_param("reg_metric", reg_metric)
    mlflow.log_param("metrics", [metric.name for metric in metrics])
    mlflow.log_param("random_state", random_state)

    old_save_act, old_symbolic_enabled = model.disable_symbolic_in_fit(lamb)

    random_generator = torch.Generator()
    if random_state is not None:
        random_generator.manual_seed(random_state)

    history = History(val_metrics={metric.name: [] for metric in metrics})

    if batch_size is None or batch_size > dataset.train_input.shape[0]:
        batch_size = dataset.train_input.shape[0]

    global_training_step = 0

    for epoch in range(1, epochs + 1):
        if shuffle:
            permutation = torch.randperm(dataset.train_input.size()[0], generator=random_generator)
        else:
            permutation = torch.arange(dataset.train_input.size()[0])

        model.train()
        train_bar = tqdm(
            range(0, dataset.train_input.size()[0], batch_size), desc=f"Epoch {epoch}/{epochs}", leave=False
        )
        train_loss_sum = 0.0
        train_count = 0
        for i in train_bar:
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = dataset.train_input[indices], dataset.train_label[indices]

            optimizer.zero_grad()
            pred = model.forward(batch_x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, batch_y)
            regularization_term = model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            loss = train_loss + lamb * regularization_term
            loss.backward()
            optimizer.step()
            mlflow.log_metric("train_loss_batch", train_loss.item(), step=global_training_step)
            train_bar.set_postfix(train_loss=train_loss.item())
            train_loss_sum += train_loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)
            global_training_step += 1
        avg_train_loss = train_loss_sum / train_count
        history.train_loss.append(avg_train_loss)
        mlflow.log_metric("train_loss_epoch", avg_train_loss, step=epoch)

        model.eval()
        val_bar = tqdm(range(0, dataset.val_input.shape[0], batch_size), desc="Validating", leave=False)
        val_loss_sum = 0.0
        val_count = 0
        metric_sums = {metric.name: 0.0 for metric in metrics}
        for i in val_bar:
            batch_x, batch_y = (
                dataset.val_input[i : i + batch_size],
                dataset.val_label[i : i + batch_size],
            )
            preds = model.forward(batch_x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            val_loss = loss_fn(preds, batch_y)
            val_loss_sum += val_loss.item() * batch_x.size(0)
            val_count += batch_x.size(0)
            bar_postfix = {"val_loss": val_loss.item()}
            for metric in metrics:
                metric_value = metric(preds, batch_y)
                metric_sums[metric.name] += metric_value * batch_x.size(0)
                bar_postfix[metric.name] = metric_sums[metric.name] / val_count
            val_bar.set_postfix(bar_postfix)
        avg_val_loss = val_loss_sum / val_count
        history.val_loss.append(avg_val_loss)
        mlflow.log_metric("val_loss_epoch", avg_val_loss, step=epoch)
        for metric in metrics:
            avg_metric = metric_sums[metric.name] / val_count
            history.val_metrics[metric.name].append(avg_metric)
            mlflow.log_metric(f"val_{metric.name}_epoch", avg_metric, step=epoch)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - "
            + " - ".join([f"Val {metric.name}: {history.val_metrics[metric.name][-1]:.4f}" for metric in metrics])
        )

    # revert back to original state
    model.symbolic_enabled = old_symbolic_enabled
    return history
