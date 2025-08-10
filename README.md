# CapPredict

CapPredict generates synthetic S-curves, extracts features, and trains predictive models (Random Forest, XGBoost) to classify projects that are likely to hit high final completion.

- Docs: See `docs/index.md`
- Quickstart: `docs/quickstart.md`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m CapPredict.main
```

## Modules
- `CapPredict.data_preprocessing`: data generation and feature engineering
- `CapPredict.predictive_model`: model training and evaluation
- `CapPredict.main`: end-to-end demo pipeline