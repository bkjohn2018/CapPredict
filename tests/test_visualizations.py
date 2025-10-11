from pathlib import Path
import numpy as np

from cappredict.visualization import plot_feature_importance, animate_scurve


OUT_DIR = Path("reports/visuals")


def _file_ok(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 500


def test_feature_importance_png(tmp_path):
    importances = np.array([0.4, 0.3, 0.2, 0.1])
    out_path = tmp_path / "fi.png"
    saved = plot_feature_importance(importances, str(out_path))
    assert _file_ok(Path(saved))


def test_scurve_animation_gif(tmp_path):
    # synthetic logistic-ish S curve
    t = np.linspace(0, 10, 200)
    y_true = 1 / (1 + np.exp(-(t - 5)))
    y_pred = 1 / (1 + np.exp(-0.9 * (t - 5)))  # slightly different slope

    out_path = tmp_path / "scurve.gif"
    saved = animate_scurve(t, y_true, y_pred, str(out_path), fps=10, seconds=2)
    assert _file_ok(Path(saved))

