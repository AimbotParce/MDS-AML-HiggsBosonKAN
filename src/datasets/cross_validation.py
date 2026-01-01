import pandas as pd
import torch

from . import Dataset


def prepare_dataset_cv(
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
    return dataset
    return dataset
