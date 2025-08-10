# Data Preprocessing API

Module: `CapPredict.data_preprocessing`

## generate_curves
Generate random logistic S-curves to simulate project progress.

Signature:
```python
def generate_curves(num_curves: int = 100, time_start: float = 0, time_end: float = 10, num_points: int = 500, seed: int = 42) -> pd.DataFrame
```
Parameters:
- num_curves: number of S-curves to generate
- time_start: start of time range
- time_end: end of time range
- num_points: time points per curve
- seed: random seed

Returns: `pd.DataFrame` with columns `Project_1..N` and `Time`.

Example:
```python
from CapPredict.data_preprocessing import generate_curves
curves_df = generate_curves(num_curves=20)
```

## extract_features
Extract features from generated curves.

Signature:
```python
def extract_features(curves_df: pd.DataFrame) -> pd.DataFrame
```
Returns a DataFrame with columns: `Project, Inflection_Point, Growth_Rate, Final_Value, Initial_Growth_Rate, Time_to_50_Completion`.

Example:
```python
from CapPredict.data_preprocessing import extract_features
features_df = extract_features(curves_df)
```

## normalize_features
Min-max normalize feature columns.

Signature:
```python
def normalize_features(features_df: pd.DataFrame) -> pd.DataFrame
```
Normalizes: `Inflection_Point, Growth_Rate, Final_Value, Initial_Growth_Rate, Time_to_50_Completion` and preserves `Project`.

Example:
```python
from CapPredict.data_preprocessing import normalize_features
normalized_df = normalize_features(features_df)
```

## split_data
Split into train/test and balance training labels using SMOTE.

Signature:
```python
def split_data(features_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]
```
- Uses feature columns: `Inflection_Point, Growth_Rate, Initial_Growth_Rate, Time_to_50_Completion`.
- Label: `Final_Value >= 0.90`.

Example:
```python
from CapPredict.data_preprocessing import split_data
train_df, test_df = split_data(normalized_df)
```