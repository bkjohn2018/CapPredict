# CapPredict Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment manager (venv, conda, etc.)

## Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd CapPredict

# Or download and extract the project files
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv cap_predict_env
source cap_predict_env/bin/activate  # On Windows: cap_predict_env\Scripts\activate

# Using conda
conda create -n cap_predict_env python=3.9
conda activate cap_predict_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Test import of key libraries
python -c "import pandas, numpy, sklearn, xgboost, matplotlib; print('All dependencies installed successfully')"
```

## Quick Start

### Run the Complete Pipeline
```bash
cd CapPredict
python main.py
```

This will execute the full machine learning pipeline:
1. Generate 100 synthetic S-curves
2. Extract features from the curves
3. Normalize and split the data
4. Train Random Forest and XGBoost models
5. Display evaluation metrics and feature importance plots

### Expected Output
```
Extracted Features (Sample):
     Project  Inflection_Point  Growth_Rate  Final_Value  Initial_Growth_Rate  Time_to_50_Completion
0  Project_1              5.12         0.45         0.98                 0.02                   5.08
...

Normalized Features (Sample):
     Project  Inflection_Point  Growth_Rate  Final_Value  Initial_Growth_Rate  Time_to_50_Completion
0  Project_1              0.42         0.73         0.95                 0.35                   0.41
...

Balanced Training Set:
False    40
True     20
Name: Final_Value, dtype: int64

🚀 Training Random Forest...
🚀 Random Forest Accuracy: 0.85

Classification Report (Random Forest):
              precision    recall  f1-score   support
           0       0.88      0.93      0.90        14
           1       0.80      0.67      0.73         6
    accuracy                           0.85        20
   macro avg       0.84      0.80      0.82        20
weighted avg       0.85      0.85      0.85        20

🚀 Training XGBoost...
🚀 XGBoost Accuracy: 0.80

[Feature importance plots will be displayed]
```

## Project Structure

```
CapPredict/
├── README.md                 # Project overview and documentation
├── ARCHITECTURE.md           # Technical architecture documentation
├── DATA_MODEL.md            # Data structures and feature definitions
├── SETUP.md                 # This setup guide
├── requirements.txt         # Python dependencies
├── CapPredict.sln          # Visual Studio solution file
└── CapPredict/             # Main source code directory
    ├── main.py             # Pipeline orchestration
    ├── data_preprocessing.py # Data generation and feature extraction
    ├── predictive_model.py  # Model training and evaluation
    ├── data_loader.py      # (Unused) Data loading utilities
    ├── CapPredict.py       # (Empty) Main module
    └── CapPredict.pyproj   # Visual Studio Python project file
```

## Development Environment

### Visual Studio Integration
The project includes Visual Studio solution and project files for development in Visual Studio with Python Tools.

1. Open `CapPredict.sln` in Visual Studio
2. Ensure the Python environment is set to `cap_predict_env`
3. Set `main.py` as the startup file
4. Run with F5 or Ctrl+F5

### Command Line Development
For development without Visual Studio:
```bash
# Navigate to source directory
cd CapPredict

# Run individual modules for testing
python data_preprocessing.py  # (if modified to include __main__ block)
python predictive_model.py    # (if modified to include __main__ block)
```

## Configuration

### Key Parameters (Currently Hardcoded)
- **Number of curves**: 100 (in `main.py`, line 7)
- **Success threshold**: 0.90 (90% completion)
- **Train/test split**: 80/20
- **Random seed**: 42
- **Time range**: 0 to 10 with 500 points

### Modifying Parameters
To change key parameters, edit the following files:

**Data Generation** (`main.py`):
```python
curves_df = generate_curves(num_curves=100)  # Change num_curves here
```

**Success Threshold** (`predictive_model.py`, multiple locations):
```python
y_train = (train_df["Final_Value"] >= 0.90).astype(int)  # Change 0.90 here
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Visual Studio Environment Issues**
   ```
   Python environment not found
   ```
   **Solution**: 
   - Check that the virtual environment is created
   - In Visual Studio, go to Python Environments and add the environment
   - Ensure the project is using the correct environment

3. **Memory Issues with Large Datasets**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce the number of curves or time points:
   ```python
   curves_df = generate_curves(num_curves=50)  # Reduce from 100
   ```

4. **Plotting Issues**
   ```
   No display name and no $DISPLAY environment variable
   ```
   **Solution**: If running on a headless server, use matplotlib backend:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Add before importing pyplot
   ```

### Performance Optimization

For better performance with larger datasets:
1. **Reduce curve resolution**: Decrease `num_points` in `generate_curves()`
2. **Limit model complexity**: Reduce `n_estimators` in model training
3. **Use fewer curves**: Start with smaller `num_curves` for testing

### Getting Help

1. **Check Documentation**: Review `README.md`, `ARCHITECTURE.md`, and `DATA_MODEL.md`
2. **Verify Dependencies**: Ensure all required packages are installed
3. **Test with Minimal Data**: Try with fewer curves first
4. **Check Python Version**: Ensure Python 3.8+ is being used

## Next Steps

After successful setup:
1. **Explore the Code**: Review the source files to understand the implementation
2. **Modify Parameters**: Experiment with different configurations
3. **Add Features**: Consider extending the feature engineering
4. **Real Data**: Plan integration with actual project data
5. **Improvements**: Refer to the recommendations in the main README for enhancement ideas