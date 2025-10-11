from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .data_prep import split_features_target
from .config import PATHS


def _train_logreg(X, y):
    # For S-curve %complete you might regress a transformed variable; placeholder here
    m = LogisticRegression(max_iter=500)
    m.fit(X, y)
    return m, "logreg.pkl"


def _train_rf(X, y):
    m = RandomForestRegressor(n_estimators=300, random_state=42)
    m.fit(X, y)
    return m, "rf.pkl"


def _train_xgb(X, y):
    m = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    m.fit(X, y)
    return m, "xgb.pkl"


def train_entrypoint(algo: str = "logreg", data_path: str = "data/processed/train.csv"):
    df = pd.read_csv(data_path)
    X, y = split_features_target(df, target="y")

    trainers = {
        "logreg": _train_logreg,
        "rf": _train_rf,
        "xgb": _train_xgb,
    }

    Path(PATHS.models).mkdir(parents=True, exist_ok=True)

    if algo == "all":
        for key, fn in trainers.items():
            model, fname = fn(X, y)
            joblib.dump(model, f"{PATHS.models}/{fname}")
        print("Saved: logreg.pkl, rf.pkl, xgb.pkl")
    else:
        model, fname = trainers[algo](X, y)
        joblib.dump(model, f"{PATHS.models}/{fname}")
        print(f"Saved: {fname}")
