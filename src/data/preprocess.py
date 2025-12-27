import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 1


def main(source: os.PathLike, destination: os.PathLike):
    # --- Reproducibility setup ---
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # --- Load Data ---
    df = pd.read_csv(source, compression="gzip")
    print(f"Data loaded from {source} with shape {df.shape}")

    # --- Indexing and re-coding NaNs ---
    # set EventId as index
    df.set_index("EventId", inplace=True)
    # Drop uninformative columns
    df = df.drop(["KaggleSet", "KaggleWeight"], axis=1)
    df.replace(-999.0, np.nan, inplace=True)

    # --- Train-test split with stratification ---
    train_NB, test_NB = train_test_split(df, test_size=0.15, stratify=df["Label"], random_state=42)

    # Join the split again for preprocess (flag train individuals in column is_train)
    train_NB["is_train"] = 1
    test_NB["is_train"] = 0

    print(f"Train set shape: {train_NB.shape}, Test set shape: {test_NB.shape}")

    df = pd.concat([train_NB, test_NB], ignore_index=True)

    # --- Handling missing values ---

    # Count number of missing values per column
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    # Print columns with missing values and their percentages
    print("Columns with missing values and their percentages:")
    for col, perc in missing_values_percentage.items():
        if perc > 0:
            print(f"Column: {col}, Missing Values: {perc:.2f}%")

    # Drop columns with NA except DER_mass_MMC
    cols_to_drop = [col for col in missing_values.index if missing_values[col] > 0 and col != "DER_mass_MMC"]
    df.drop(columns=cols_to_drop, inplace=True)

    print(f"Dropped columns with missing values except 'DER_mass_MMC': {cols_to_drop}")

    # Drop rows with any missing values in DER_mass_MMC
    df.dropna(subset=["DER_mass_MMC"], inplace=True)
    print(f"After dropping rows with missing 'DER_mass_MMC', new shape: {df.shape}")
    df = df.dropna()
    print(f"After dropping all remaining rows with missing values, new shape: {df.shape}")
    df.reset_index(drop=True, inplace=True)

    # --- Removing highly correlated features ---

    corr_matrix = df.drop(columns=["Label"]).corr().abs()

    # Keep only the upper triangle of the correlation matrix (to avoid duplicates)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns to drop
    to_drop = []
    drop_reasons = []

    for column in upper.columns:
        # Find correlations above threshold
        high_corr = upper[column][upper[column] > 0.85]
        if not high_corr.empty:
            to_drop.append(column)
            for correlated_col, corr_value in high_corr.items():
                drop_reasons.append((column, correlated_col, corr_value))

    # Drop the columns
    df.drop(columns=to_drop, inplace=True)

    # Print results
    print(f"Dropped {len(to_drop)} highly correlated (redundant) features:\n")
    for col, corr_col, corr_val in drop_reasons:
        print(f" - Dropped '{col}' (correlated with '{corr_col}' at {corr_val:.2f})")

    # --- Normalization ---

    # Select numeric features
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude columns 'is_train', 'Weight', 'Label'
    exclude_cols = ["is_train", "Weight", "Label"]
    num_features = [f for f in num_features if f not in exclude_cols]

    # Initialize scaler
    scaler = StandardScaler()

    # Fit only on training data
    scaler.fit(df.loc[df["is_train"] == 1, num_features])

    # Transform all data
    df[num_features] = scaler.transform(df[num_features])

    print("Standardized numeric features using only training data to fit the scaler.")

    # --- Re-coding Label ---

    df["Label"] = df["Label"].map({"b": 1, "s": 0}).astype(int)

    # --- Save preprocessed data ---

    # Split based on 'is_train' again
    train_preprocess = df[df["is_train"] == 1].drop(columns=["is_train"])
    test_preprocess = df[df["is_train"] == 0].drop(columns=["is_train"])
    df = df.drop(columns=["is_train"])

    # Save to CSV
    train_preprocess.to_csv(os.path.join(destination, "train.csv"), index=False)
    test_preprocess.to_csv(os.path.join(destination, "test.csv"), index=False)

    print("Preprocessed train and test sets saved without 'is_train' column.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Preprocessing for Higgs Boson Dataset")
    parser.add_argument("source", type=str, help="Path to the raw data file")
    parser.add_argument("destination", type=str, help="Directory to save the preprocessed data files")
    args = parser.parse_args()

    main(source=args.source, destination=args.destination)
