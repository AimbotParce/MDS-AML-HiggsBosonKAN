import os
import sys

import mlflow
import torch
from kan import KAN
from mlflow import pytorch as mlflow_pytorch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional

from torch import nn

from src.datasets import prepare_dataset_cv
from src.trainers import Accuracy
from src.trainers.mlp_trainer import fit_mlp


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, random_state: Optional[int] = None):
        super(MLPClassifier, self).__init__()
        random_generator = torch.Generator()
        if random_state is not None:
            random_generator.manual_seed(random_state)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            # According to ChatGPT, Xavier initialization is usually not preferred for ReLU activations,
            # but for this toy example, we will use it.
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"), generator=random_generator)
            layers.append(layer)
            layers.append(nn.ReLU())  # Using ReLU activation for hidden layers
            prev_dim = hidden_dim
        output_layer = nn.Linear(prev_dim, output_dim)
        # For sigmoid output, Xavier initialization is appropriate
        nn.init.xavier_uniform_(output_layer.weight, gain=nn.init.calculate_gain("sigmoid"), generator=random_generator)
        layers.append(output_layer)
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="MLP Higgs Experiment")
    parser.add_argument("--data-path", type=str, default="data/processed/higgs-challenge.parquet")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-fold-index", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--l1-reg", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs: int = args.epochs
    batch_size: int = args.batch_size
    seed: int = 42
    hidden_dim: int = args.hidden_dim
    data_path: str = args.data_path
    cv_folds: int = args.cv_folds
    cv_fold_index: int = args.cv_fold_index
    lr: float = args.learning_rate
    l1_reg: float = args.l1_reg

    if not (cv_folds > 1):
        parser.error("cv-folds must be greater than 1")
    if not (0 <= cv_fold_index < cv_folds):
        parser.error("cv-fold-index must be in the range [0, cv-folds)")

    mlflow_pytorch.autolog(log_models=False)
    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"MLP-hidden{hidden_dim}-fold{cv_fold_index}/{cv_folds}"
    ):
        mlflow.log_param("seed", seed)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("cv_fold_index", cv_fold_index)
        mlflow.log_param("learning_rate", lr)

        dataset = prepare_dataset_cv(
            data_path, cv_folds=cv_folds, cv_fold_index=cv_fold_index, random_state=seed, device=device
        )
        num_features = dataset.train_input.shape[1]
        mlflow.log_param("num_features", num_features)

        print("Dataset prepared.")
        print(f"Train samples: {dataset.train_input.shape[0]}, Validation samples: {dataset.val_input.shape[0]}")
        print(f"Number of features: {num_features}")

        model = MLPClassifier(input_dim=num_features, hidden_dims=[hidden_dim], output_dim=1).to(device)
        fit_mlp(
            model,
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=nn.BCELoss(),
            l1_reg=l1_reg,
            metrics=(Accuracy(),),
            random_state=seed,
        )
