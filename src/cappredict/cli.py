from __future__ import annotations

import typer

from . import model_train, model_eval


app = typer.Typer(help="CapPredict: S-curve based capex forecasting.")


@app.command()
def train(
    algo: str = typer.Option("logreg", help="logreg|rf|xgb|all"),
    data_path: str = typer.Option("data/processed/train.csv"),
):
    """Train model(s) and persist to /models."""
    model_train.train_entrypoint(algo=algo, data_path=data_path)


@app.command()
def evaluate(
    model: str = typer.Option("logreg.pkl"),
    test_path: str = typer.Option("data/processed/test.csv"),
):
    """Evaluate a saved model and write metrics to /reports/validation_report.md."""
    model_eval.evaluate_entrypoint(model=model, test_path=test_path)


@app.command()
def explain(
    model: str = typer.Option("rf.pkl"),
    out_dir: str = typer.Option("reports/visuals"),
):
    """Export feature importance plots/animations to /reports/visuals."""
    model_eval.explain_entrypoint(model=model, out_dir=out_dir)


if __name__ == "__main__":
    app()

