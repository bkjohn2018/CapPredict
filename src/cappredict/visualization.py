from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_feature_importance(importances: np.ndarray, out_path: str) -> str:
    """Deterministic PNG; returns path."""
    rng = np.random.default_rng(0)  # guard any random downstream calls
    _ = rng.random(1)  # touches the seed to stabilize state

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(importances)), importances)
    ax.set_title("Feature Importance")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def animate_scurve(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    fps: int = 15,
    seconds: int = 3,
) -> str:
    """Make a simple line-draw GIF of observed vs predicted; returns path."""
    assert t.ndim == y_true.ndim == y_pred.ndim == 1
    nframes = max(5, fps * seconds)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(t.min(), t.max())
    ymin = float(min(y_true.min(), y_pred.min()))
    ymax = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1)
    ax.set_ylim(ymin - pad, ymax + pad)
    true_line, = ax.plot([], [], lw=2, label="Observed")
    pred_line, = ax.plot([], [], lw=2, linestyle="--", label="Predicted")
    ax.legend(loc="best")
    ax.set_title("S-curve Fit (Observed vs Predicted)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Completion / Spend")

    def init():
        true_line.set_data([], [])
        pred_line.set_data([], [])
        return true_line, pred_line

    def update(frame):
        k = int((frame + 1) / nframes * len(t))
        true_line.set_data(t[:k], y_true[:k])
        pred_line.set_data(t[:k], y_pred[:k])
        return true_line, pred_line

    anim = FuncAnimation(
        fig, update, init_func=init, frames=nframes, blit=True, interval=1000 / fps
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path)
    plt.close(fig)
    return out_path

