# Data Loader API

Module: `CapPredict.data_loader`

Currently provides a template for loading data from CSV or Microsoft Fabric Dataflows. The functions are placeholders and commented out in `data_loader.py`. To enable them, implement and uncomment as needed.

## Planned `load_data`
```python
def load_data(source: str = "csv", filepath: str | None = None) -> pd.DataFrame
```
- source: "csv" or "fabric"
- filepath: path to CSV when `source="csv"`

Returns: `pd.DataFrame` with project-level features.

Example (CSV):
```python
# from CapPredict.data_loader import load_data
# df = load_data(source="csv", filepath="/path/to/data.csv")
```