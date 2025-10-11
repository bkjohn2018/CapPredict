from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from config import config

def train_logistic_regression(train_df, threshold=None):
    """
    Train a logistic regression model with configurable classification threshold.
    
    Args:
        train_df: Training DataFrame
        threshold: Success threshold (uses config default if None)
        
    Returns:
        LogisticRegression: Trained model
    """
    threshold = threshold if threshold is not None else config.models.success_threshold
    feature_cols = config.features.model_features

    if config.logging.show_debug_prints:
        print("\nTraining Data Columns:")
        print(train_df.columns.tolist())

    X_train = train_df[feature_cols]
    y_train = (train_df[config.features.target_feature] >= threshold).astype(int)

    model = LogisticRegression(random_state=config.data_processing.random_state)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, test_df, model_name="Model"):
    """
    Evaluate the model on test data and display feature importance with configurable visualization.
    
    Args:
        model: Trained model with feature_importances_ attribute
        test_df: Test DataFrame
        model_name: Name for display purposes
    """
    feature_cols = config.features.model_features
    X_test = test_df[feature_cols]
    y_test = (test_df[config.features.target_feature] >= config.models.success_threshold).astype(int)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance Metrics
    emoji = "🚀" if config.logging.use_emojis else ""
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{emoji} {model_name} Accuracy: {accuracy:.{config.logging.decimal_places}f}")
    print(f"\nClassification Report ({model_name}):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature Importance (only if model has this attribute)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]

        if config.visualization.show_plots:
            plt.figure(figsize=(config.visualization.figure_width, config.visualization.figure_height))
            plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
            plt.xticks(range(len(feature_importance)), 
                      np.array(feature_cols)[sorted_idx], 
                      rotation=config.visualization.rotation_angle)
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.title(f"Feature Importance in {model_name}")
            
            # Use configured display settings
            plt.tight_layout()
            plt.show(block=config.visualization.block_plots)
            
            # Save plot if configured
            if config.visualization.save_plots:
                import os
                os.makedirs(config.visualization.plot_output_dir, exist_ok=True)
                filename = f"{config.visualization.plot_output_dir}/{model_name.lower().replace(' ', '_')}_importance.{config.visualization.plot_format}"
                plt.savefig(filename, dpi=config.visualization.plot_dpi, bbox_inches='tight')
                if config.logging.show_debug_prints:
                    print(f"Plot saved: {filename}")
    else:
        if config.logging.show_debug_prints:
            print(f"Model {model_name} does not have feature importance attribute")




def train_random_forest(train_df):
    """
    Train a Random Forest model using configured hyperparameters.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        RandomForestClassifier: Trained Random Forest model
    """
    feature_cols = config.features.model_features
    X_train = train_df[feature_cols]
    y_train = (train_df[config.features.target_feature] >= config.models.success_threshold).astype(int)

    model = RandomForestClassifier(
        n_estimators=config.models.rf_n_estimators,
        max_depth=config.models.rf_max_depth,
        min_samples_split=config.models.rf_min_samples_split,
        min_samples_leaf=config.models.rf_min_samples_leaf,
        class_weight=config.models.rf_class_weight,
        random_state=config.models.rf_random_state
    )

    model.fit(X_train, y_train)
    return model

def train_xgboost(train_df):
    """
    Train an XGBoost model using configured hyperparameters.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        XGBClassifier: Trained XGBoost model
    """
    feature_cols = config.features.model_features
    X_train = train_df[feature_cols]
    y_train = (train_df[config.features.target_feature] >= config.models.success_threshold).astype(int)

    model = XGBClassifier(
        n_estimators=config.models.xgb_n_estimators,
        max_depth=config.models.xgb_max_depth,
        learning_rate=config.models.xgb_learning_rate,
        scale_pos_weight=config.models.xgb_scale_pos_weight,
        random_state=config.models.xgb_random_state,
        eval_metric='logloss'  # Suppress warnings
    )

    model.fit(X_train, y_train)
    return model