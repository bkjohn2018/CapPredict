# Predictive Model API

Module: `CapPredict.predictive_model`

## train_logistic_regression
Train logistic regression classifier using thresholded `Final_Value` label.

Signature:
```python
def train_logistic_regression(train_df: pd.DataFrame, threshold: float = 0.90) -> LogisticRegression
```
- Features used: `Inflection_Point, Growth_Rate, Final_Value, Initial_Growth_Rate, Time_to_50_Completion`.

Example:
```python
from CapPredict.predictive_model import train_logistic_regression
logreg = train_logistic_regression(train_df, threshold=0.9)
```

## train_random_forest
Train a RandomForest classifier without using `Final_Value` as a feature.

Signature:
```python
def train_random_forest(train_df: pd.DataFrame) -> RandomForestClassifier
```
- Features used: `Inflection_Point, Growth_Rate, Initial_Growth_Rate, Time_to_50_Completion`.

Example:
```python
from CapPredict.predictive_model import train_random_forest
rf = train_random_forest(train_df)
```

## train_xgboost
Train an XGBoost classifier without `Final_Value` as a feature.

Signature:
```python
def train_xgboost(train_df: pd.DataFrame) -> XGBClassifier
```
- Features used: `Inflection_Point, Growth_Rate, Initial_Growth_Rate, Time_to_50_Completion`.

Example:
```python
from CapPredict.predictive_model import train_xgboost
xgb = train_xgboost(train_df)
```

## evaluate_model
Evaluate a trained model on a test set and display feature importance bar chart.

Signature:
```python
def evaluate_model(model, test_df: pd.DataFrame, model_name: str = "Model") -> None
```
- Label on test set is computed as `Final_Value >= 0.90`.
- Plots non-blocking feature importances for tree-based models.

Example:
```python
from CapPredict.predictive_model import evaluate_model
evaluate_model(rf, test_df, model_name="Random Forest")
```