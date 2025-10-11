from __future__ import annotations

import pandas as pd


def load_train_test(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    return X_train, X_test


def split_features_target(df: pd.DataFrame, target: str = "y"):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

