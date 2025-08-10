# Main Pipeline

Module: `CapPredict.main`

## run_pipeline
Run the full demo pipeline: generate curves, extract + normalize features, split data, train RF and XGB, evaluate, and show plots.

Signature:
```python
def run_pipeline() -> None
```

Example CLI:
```bash
python -m CapPredict.main
```