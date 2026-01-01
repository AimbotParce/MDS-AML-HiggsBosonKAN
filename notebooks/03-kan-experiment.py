import os
import sys

import mlflow
import torch
from kan import KAN
from mlflow import pytorch as mlflow_pytorch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import prepare_dataset_cv
from src.trainers import Accuracy
from src.trainers.kan_trainer import fit_kan


class KANClassifier(KAN):
    def forward(self, x, *args, **kwargs):
        out = super().forward(x, *args, **kwargs)
        return torch.sigmoid(out)


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
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"KAN-hidden{hidden_dim}-fold{cv_fold_index}/{cv_folds}"
    ):
        mlflow.log_param("seed", seed)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("grid", grid)
        mlflow.log_param("k", k)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("cv_fold_index", cv_fold_index)

        dataset = prepare_dataset_cv(
            data_path, cv_folds=cv_folds, cv_fold_index=cv_fold_index, random_state=seed, device=device
        )
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
