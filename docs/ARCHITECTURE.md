# CapPredict Architecture Documentation

## System Overview

CapPredict implements a machine learning pipeline for project success prediction using S-curve analysis. The architecture follows a functional programming approach with clear separation of data processing, feature engineering, and model training concerns.

## Component Architecture

### 1. Data Pipeline (`data_preprocessing.py`)

#### Data Generation
- **Function**: `generate_curves()`
- **Purpose**: Creates synthetic logistic S-curves simulating project progress
- **Input**: Configuration parameters (num_curves, time_range, etc.)
- **Output**: DataFrame with curves and time series data
- **Mathematical Model**: Logistic function with randomized parameters

#### Feature Extraction
- **Function**: `extract_features()`
- **Purpose**: Converts raw S-curve data into meaningful ML features
- **Input**: Curves DataFrame
- **Output**: Features DataFrame with 5 engineered features
- **Key Features**:
  - Inflection Point (time of steepest slope)
  - Growth Rate (maximum slope value)
  - Final Value (completion percentage)
  - Initial Growth Rate (starting slope)
  - Time to 50% Completion

#### Data Preprocessing
- **Normalization**: `normalize_features()` - Min-Max scaling
- **Data Splitting**: `split_data()` - Train/test split with SMOTE balancing

### 2. Model Pipeline (`predictive_model.py`)

#### Model Training
- **Random Forest**: `train_random_forest()`
  - 200 estimators, max_depth=5
  - Class balancing enabled
  - Prevents overfitting with min_samples constraints
  
- **XGBoost**: `train_xgboost()`
  - 200 estimators, learning_rate=0.1
  - scale_pos_weight=2 for imbalance handling
  - Gradient boosting approach

#### Model Evaluation
- **Function**: `evaluate_model()`
- **Metrics**: Accuracy, Classification Report
- **Visualization**: Feature importance plots
- **Output**: Performance metrics and visual analysis

### 3. Pipeline Orchestration (`main.py`)

#### Workflow Execution
- **Function**: `run_pipeline()`
- **Steps**:
  1. Data generation (100 curves)
  2. Feature extraction
  3. Data normalization
  4. Train/test splitting
  5. Model training (RF + XGBoost)
  6. Model evaluation and comparison

## Data Flow

```
Synthetic Data Generation
         ↓
Feature Extraction (5 features)
         ↓
Normalization (Min-Max)
         ↓
Train/Test Split + SMOTE
         ↓
Model Training (RF & XGBoost)
         ↓
Evaluation & Visualization
```

## Key Design Decisions

### 1. Synthetic Data Approach
- **Rationale**: Enables controlled experimentation and validation
- **Implementation**: Logistic function with randomized parameters
- **Benefits**: Reproducible results, known ground truth

### 2. Feature Engineering Strategy
- **Focus**: Time-series curve characteristics
- **Approach**: Mathematical derivatives and statistical measures
- **Validation**: Features exclude target variable to prevent leakage

### 3. Model Selection
- **Ensemble Approach**: Random Forest + XGBoost
- **Rationale**: Different algorithmic approaches for robustness
- **Class Imbalance**: SMOTE + class_weight/scale_pos_weight

### 4. Success Definition
- **Threshold**: 90% completion rate
- **Binary Classification**: Success/Failure prediction
- **Business Logic**: Reflects real-world project success criteria

## Configuration Management

### Current State
- Hardcoded parameters throughout codebase
- Magic numbers in multiple locations
- No centralized configuration

### Parameters
- **Data Generation**: 100 curves, 500 time points, seed=42
- **Success Threshold**: 0.90 (90% completion)
- **Train/Test Split**: 80/20
- **Model Hyperparameters**: Embedded in training functions

## Dependencies and Integration

### External Libraries
- **pandas**: DataFrame operations and data manipulation
- **numpy**: Numerical computations and array operations
- **scikit-learn**: ML algorithms, preprocessing, evaluation
- **xgboost**: Gradient boosting implementation
- **matplotlib**: Visualization and plotting
- **imbalanced-learn**: SMOTE implementation

### Integration Points
- **Data Sources**: Currently synthetic, designed for CSV/Fabric integration
- **Output**: Console metrics, matplotlib visualizations
- **Persistence**: No model saving/loading implemented

## Performance Characteristics

### Computational Complexity
- **Data Generation**: O(n*m) where n=curves, m=time_points
- **Feature Extraction**: O(n*m) for gradient calculations
- **Model Training**: Depends on algorithm (RF: O(n*log(n)*trees), XGB: iterative)

### Memory Usage
- **Data Storage**: In-memory DataFrames
- **Model Storage**: Scikit-learn/XGBoost objects
- **Scalability**: Limited by available RAM

### Execution Time
- **Pipeline Runtime**: ~seconds for 100 curves
- **Bottlenecks**: Model training, feature extraction
- **Optimization Opportunities**: Vectorization, parallel processing

## Error Handling and Robustness

### Current Implementation
- **Basic Exception Handling**: try/catch in feature extraction
- **Silent Failures**: NaN values for missing data
- **No Validation**: Input parameter validation missing

### Failure Modes
- **Data Quality**: Empty curves, invalid time series
- **Feature Extraction**: Mathematical edge cases
- **Model Training**: Convergence issues, data quality problems

## Extension Points

### Planned Enhancements
1. **Configuration System**: Centralized parameter management
2. **Data Abstraction**: Support for multiple data sources
3. **Model Registry**: Save/load trained models
4. **Validation Framework**: Comprehensive input validation
5. **Monitoring**: Performance and data quality monitoring