# CapPredict Data Model Documentation

## Overview

This document describes the data structures, transformations, and feature definitions used throughout the CapPredict machine learning pipeline.

## Data Structures

### 1. Raw Curve Data

**Source**: `generate_curves()` function
**Structure**: 
```
DataFrame[num_points x (num_curves + 1)]
Columns: ["Project_1", "Project_2", ..., "Project_N", "Time"]
```

**Example**:
```
   Project_1  Project_2  Project_3  ...  Time
0      0.002      0.001      0.003  ...   0.0
1      0.003      0.002      0.004  ...   0.02
2      0.005      0.003      0.006  ...   0.04
...      ...        ...        ...  ...   ...
499    0.998      0.996      0.999  ...  10.0
```

**Data Properties**:
- **Time Range**: 0 to 10 (configurable)
- **Curve Values**: 0 to 1 (logistic function output)
- **Points per Curve**: 500 (configurable)
- **Mathematical Model**: `y = 1 / (1 + e^(-k(x - x0)))`

### 2. Feature Data

**Source**: `extract_features()` function
**Structure**:
```
DataFrame[num_curves x 6]
Columns: ["Project", "Inflection_Point", "Growth_Rate", "Final_Value", 
          "Initial_Growth_Rate", "Time_to_50_Completion"]
```

**Example**:
```
     Project  Inflection_Point  Growth_Rate  Final_Value  Initial_Growth_Rate  Time_to_50_Completion
0  Project_1              5.12         0.45         0.98                 0.02                   5.08
1  Project_2              6.24         0.38         0.96                 0.01                   6.20
2  Project_3              4.86         0.52         0.99                 0.03                   4.82
```

### 3. Normalized Feature Data

**Source**: `normalize_features()` function
**Structure**: Same as Feature Data but with Min-Max scaled values (0-1 range)
**Transformation**: `X_norm = (X - X_min) / (X_max - X_min)`

### 4. Training/Testing Data

**Source**: `split_data()` function
**Structure**: Separate train/test DataFrames with SMOTE-balanced training data

## Feature Definitions

### 1. Inflection Point
- **Definition**: Time at which the S-curve has maximum slope (steepest growth)
- **Calculation**: `time[argmax(gradient(curve))]`
- **Business Meaning**: Point of maximum project acceleration
- **Typical Range**: 3-7 time units
- **Data Type**: Float

### 2. Growth Rate
- **Definition**: Maximum slope value of the S-curve
- **Calculation**: `max(gradient(curve))`
- **Business Meaning**: Peak velocity of project progress
- **Typical Range**: 0.1-0.6
- **Data Type**: Float

### 3. Final Value
- **Definition**: Completion percentage at project end
- **Calculation**: `curve.iloc[-1]`
- **Business Meaning**: Ultimate project completion level
- **Range**: 0.0-1.0 (0-100%)
- **Data Type**: Float
- **Note**: Used as target variable (success if ≥ 0.90)

### 4. Initial Growth Rate
- **Definition**: Slope of the curve at project start
- **Calculation**: `gradient(curve)[0]`
- **Business Meaning**: Early project momentum
- **Typical Range**: 0.005-0.05
- **Data Type**: Float

### 5. Time to 50% Completion
- **Definition**: Time required to reach 50% project completion
- **Calculation**: `time[first_index_where(curve >= 0.5)]`
- **Business Meaning**: Midpoint achievement timeline
- **Typical Range**: 3-7 time units
- **Data Type**: Float
- **Special Cases**: NaN if curve never reaches 50%

## Data Transformations

### 1. Curve Generation Parameters

**Randomization Strategy**:
- **Steepness (k)**: Uniform distribution [0.5, 2.0]
- **Midpoint (x0)**: Uniform distribution [3.0, 7.0]
- **Random Seed**: 42 (for reproducibility)

**Mathematical Properties**:
- **Function**: Logistic/Sigmoid curve
- **Asymptotes**: y → 0 as x → -∞, y → 1 as x → +∞
- **Symmetry**: Point symmetric around (x0, 0.5)

### 2. Feature Extraction Process

**Gradient Calculation**:
```python
gradient = np.gradient(curve)  # Numerical differentiation
inflection_point = time[np.argmax(gradient)]
growth_rate = np.max(gradient)
```

**Edge Case Handling**:
- **Missing 50% Completion**: Set to NaN, handled in normalization
- **Flat Curves**: Minimum gradient values preserved
- **Noisy Data**: Numerical gradient smooths minor variations

### 3. Normalization Strategy

**Method**: Min-Max Scaling
**Formula**: `X_scaled = (X - X_min) / (X_max - X_min)`
**Result Range**: [0, 1] for all features
**Preservation**: Relative relationships maintained

### 4. Class Balancing (SMOTE)

**Target Definition**: `success = (Final_Value >= 0.90)`
**Balancing Strategy**: SMOTE with sampling_strategy=0.5
**Result**: Minority class increased to 50% of majority class size
**Feature Set**: Excludes Final_Value to prevent data leakage

## Data Quality Considerations

### 1. Synthetic Data Limitations
- **Idealized Curves**: Perfect logistic functions without noise
- **Parameter Constraints**: Limited randomization ranges
- **No Missing Data**: Complete time series for all projects

### 2. Feature Interdependencies
- **Correlation**: Inflection_Point ≈ Time_to_50_Completion
- **Mathematical Relationship**: Growth_Rate derived from same gradient
- **Redundancy**: Some features may be linearly dependent

### 3. Target Variable Characteristics
- **Threshold Sensitivity**: 90% cutoff creates class imbalance
- **Distribution**: Majority of synthetic curves achieve high completion
- **Business Alignment**: Reflects real-world success criteria

## Data Validation Requirements

### 1. Input Validation (Not Currently Implemented)
- **Curve Monotonicity**: Ensure S-curves are non-decreasing
- **Value Bounds**: Verify 0 ≤ curve_values ≤ 1
- **Time Consistency**: Validate time series ordering

### 2. Feature Validation
- **Range Checks**: Ensure features within expected bounds
- **NaN Handling**: Explicit strategy for missing values
- **Outlier Detection**: Identify anomalous feature values

### 3. Model Input Validation
- **Feature Completeness**: All required features present
- **Data Types**: Correct numeric types for ML algorithms
- **Shape Consistency**: Matching dimensions for train/test sets

## Future Data Enhancements

### 1. Real Data Integration
- **CSV Support**: File-based project data loading
- **Database Connectivity**: Direct database integration
- **API Integration**: Real-time data streaming

### 2. Advanced Feature Engineering
- **Temporal Features**: Seasonality, trends, cycles
- **Statistical Features**: Variance, skewness, kurtosis
- **Domain Features**: Project-specific characteristics

### 3. Data Quality Framework
- **Automated Validation**: Input data quality checks
- **Data Profiling**: Statistical summaries and distributions
- **Anomaly Detection**: Outlier identification and handling