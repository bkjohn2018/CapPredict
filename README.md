# CapPredict

## Project Overview

CapPredict is a machine learning project designed to predict project completion success by analyzing S-curve patterns. The system generates synthetic logistic S-curves that simulate project progress over time, extracts meaningful features from these curves, and uses ensemble machine learning models to classify whether a project will achieve successful completion (≥90% completion rate).

## Business Problem

The project addresses the challenge of predicting project success early in the project lifecycle by analyzing progress curve patterns. This can help project managers:
- Identify at-risk projects before they fail
- Allocate resources more effectively
- Improve project planning and execution strategies

## Technical Approach

### Data Generation
- Creates synthetic S-curves using logistic functions: `y = 1 / (1 + e^(-k(x - x0)))`
- Simulates realistic project progression patterns
- Generates configurable numbers of curves for training and testing

### Feature Engineering
The system extracts five key features from each S-curve:
1. **Inflection Point**: Time at which the curve has the steepest slope (maximum acceleration)
2. **Growth Rate**: Maximum slope value (peak velocity of progress)
3. **Final Value**: End completion percentage
4. **Initial Growth Rate**: Slope at project start
5. **Time to 50% Completion**: Time required to reach halfway point

### Machine Learning Pipeline
1. **Data Preprocessing**: Normalization using Min-Max scaling
2. **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique) for handling imbalanced datasets
3. **Model Training**: Ensemble approach using Random Forest and XGBoost classifiers
4. **Evaluation**: Accuracy metrics and feature importance analysis

### Success Classification
Projects are classified as "successful" if they achieve ≥90% completion rate, based on the extracted features (excluding the final value to prevent data leakage).

## Current Architecture

```
CapPredict/
├── main.py              # Pipeline orchestration and execution
├── data_preprocessing.py # Data generation, feature extraction, normalization
├── predictive_model.py   # Model training, evaluation, and prediction
├── data_loader.py        # (Currently unused) Data loading utilities
└── CapPredict.py         # (Empty) Main module placeholder
```

## Usage

### ML Pipeline
Run the complete ML pipeline:
```bash
python main.py
```

This will:
1. Generate 100 synthetic S-curves
2. Extract features from each curve
3. Normalize the feature data
4. Split into training/testing sets with class balancing
5. Train Random Forest and XGBoost models
6. Evaluate both models and display feature importance plots

### Animation
Generate logistic curve fitting animations for presentations:
```bash
python -m cappredict.viz.animate_fit --seed 42 --frames 150 --fps 24
```

**CapPredict prefers MP4 (FFmpeg). If FFmpeg isn't available, it falls back to GIF automatically.**

**Platform-specific FFmpeg installation:**
- **Windows**: Install FFmpeg and ensure `ffmpeg` is on PATH (e.g., via chocolatey or manual install)
- **macOS**: `brew install ffmpeg`
- **Linux/Conda**: `conda install -c conda-forge ffmpeg`

## Dependencies

The project requires the following Python packages:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms and preprocessing
- xgboost: Gradient boosting framework
- matplotlib: Data visualization
- imbalanced-learn: Handling imbalanced datasets

## Model Performance

The system trains two models:
- **Random Forest**: Ensemble of decision trees with class balancing
- **XGBoost**: Gradient boosting with scale_pos_weight for imbalance handling

Feature importance analysis reveals which curve characteristics are most predictive of project success.

## Development Status

This is a proof-of-concept implementation focusing on synthetic data generation and model validation. The system demonstrates the feasibility of using S-curve analysis for project success prediction.

## Future Enhancements

- Integration with real project data sources
- Additional feature engineering techniques
- Model hyperparameter optimization
- Web-based dashboard for predictions
- Real-time project monitoring capabilities