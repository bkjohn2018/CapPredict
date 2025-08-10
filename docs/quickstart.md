# Getting Started

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo pipeline

```bash
python -m CapPredict.main
```

This will generate synthetic S-curves, extract features, train Random Forest and XGBoost models, print evaluation metrics, and display feature importance plots.

## Minimal usage example

```python
from CapPredict.data_preprocessing import generate_curves, extract_features, normalize_features, split_data
from CapPredict.predictive_model import train_random_forest, evaluate_model

curves_df = generate_curves(num_curves=50)
features_df = extract_features(curves_df)
normalized_df = normalize_features(features_df)
train_df, test_df = split_data(normalized_df)

model = train_random_forest(train_df)
evaluate_model(model, test_df, model_name="Random Forest")
```