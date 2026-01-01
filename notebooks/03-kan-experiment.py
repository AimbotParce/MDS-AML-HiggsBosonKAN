from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import mlflow
import pandas as pd
import torch
from kan import KAN
from mlflow import pytorch as mlflow_pytorch
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm.auto import tqdm


class KANClassifier(KAN):
    def forward(self, x, *args, **kwargs):
        out = super().forward(x, *args, **kwargs)
        return torch.sigmoid(out)


@dataclass(slots=True)
class Dataset:
    train_input: torch.Tensor
    train_label: torch.Tensor
    train_weight: torch.Tensor
    val_input: torch.Tensor
    val_label: torch.Tensor
    val_weight: torch.Tensor


def prepare_dataset(
    data_path: str,
    cv_folds: int = 5,
    cv_fold_index: int = 0,
    random_state: int = 42,
    device: torch.device = torch.device("cpu"),
):
    df = pd.read_parquet(data_path)

    X = torch.tensor(df.drop(columns=["Label", "Weight"]).values, dtype=torch.float32)
    y = torch.tensor(df["Label"].values, dtype=torch.float32)
    w = torch.tensor(df["Weight"].values, dtype=torch.float32)

    permutation = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(random_state))
    folds = torch.chunk(permutation, cv_folds)

    val_indices = folds[cv_fold_index]
    train_indices = torch.cat([folds[i] for i in range(cv_folds) if i != cv_fold_index])

    X_train, y_train, w_train = X[train_indices], y[train_indices], w[train_indices]
    X_val, y_val, w_val = X[val_indices], y[val_indices], w[val_indices]

    # Encoding label as float, but we'll use binary cross-entropy loss regardless
    dataset = Dataset(
        train_input=X_train.to(device),
        train_label=y_train.reshape(-1, 1).to(device),
        train_weight=w_train.reshape(-1, 1).to(device),
        val_input=X_val.to(device),
        val_label=y_val.reshape(-1, 1).to(device),
        val_weight=w_val.reshape(-1, 1).to(device),
    )
    return dataset


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="KAN Higgs Experiment")
    parser.add_argument("--data-path", type=str, default="data/processed/higgs-challenge.parquet")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-fold-index", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs: int = args.epochs
    batch_size: int = args.batch_size
    seed: int = 42
    hidden_dim: int = args.hidden_dim
    grid: int = 3
    k: int = 3
    data_path: str = args.data_path
    cv_folds: int = args.cv_folds
    cv_fold_index: int = args.cv_fold_index

    if not (cv_folds > 1):
        parser.error("cv-folds must be greater than 1")
    if not (0 <= cv_fold_index < cv_folds):
        parser.error("cv-fold-index must be in the range [0, cv-folds)")

    mlflow_pytorch.autolog(log_models=False)
    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param("seed", seed)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("grid", grid)
        mlflow.log_param("k", k)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("cv_fold_index", cv_fold_index)

        dataset = prepare_dataset(data_path, cv_folds=5, cv_fold_index=0, random_state=seed, device=device)
        num_features = dataset.train_input.shape[1]
        mlflow.log_param("num_features", num_features)

        print("Dataset prepared.")
        print(f"Train samples: {dataset.train_input.shape[0]}, Validation samples: {dataset.val_input.shape[0]}")
        print(f"Number of features: {num_features}")

        model = KANClassifier(width=[num_features, hidden_dim, 1], grid=grid, k=k, seed=seed, auto_save=False).to(
            device
        )
        fit_kan(
            model,
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            loss_fn=nn.BCELoss(),
            lamb=0.001,
            lamb_entropy=5.0,
            metrics=(Accuracy(),),
        )
