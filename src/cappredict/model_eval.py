from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .data_prep import split_features_target
from .config import PATHS
from .visualization import plot_feature_importance


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }


def evaluate_entrypoint(model: str = "rf.pkl", test_path: str = "data/processed/test.csv"):
    Path(PATHS.reports).mkdir(parents=True, exist_ok=True)
    m = joblib.load(f"{PATHS.models}/{model}")
    df = pd.read_csv(test_path)
    X, y = split_features_target(df, target="y")
    yhat = m.predict(X)
    metrics = _metrics(y, yhat)

    # append/overwrite a lightweight report
    report = Path(f"{PATHS.reports}/validation_report.md")
    lines = [
        "# Validation Report\n",
        f"**Model:** {model}\n",
        f"**Rows:** {len(df)}\n",
        f"- R2: {metrics['r2']:.4f}\n- MAE: {metrics['mae']:.4f}\n- RMSE: {metrics['rmse']:.4f}\n",
    ]
    report.write_text("\n".join(lines))
    print("Wrote reports/validation_report.md")


def explain_entrypoint(model: str = "rf.pkl", out_dir: str = "reports/visuals"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    m = joblib.load(f"{PATHS.models}/{model}")
    if hasattr(m, "feature_importances_"):
        out = Path(out_dir) / f"{model}_feature_importance.png"
        saved = plot_feature_importance(m.feature_importances_, str(out))
        print(f"Wrote {saved}")
    else:
        print("Model has no feature_importances_; skip.")

