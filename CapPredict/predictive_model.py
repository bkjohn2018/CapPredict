from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

def train_logistic_regression(train_df, threshold=0.90):
    """
    Train a logistic regression model with a tunable classification threshold.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Final_Value", "Initial_Growth_Rate", "Time_to_50_Completion"]

    # Ensure all features exist
    print("\nTraining Data Columns:")
    print(train_df.columns)

    X_train = train_df[feature_cols]  # Check if columns exist
    y_train = (train_df["Final_Value"] >= threshold).astype(int)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, test_df, model_name="Model"):
    """
    Evaluate the model on test data and display feature importance without blocking execution.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Initial_Growth_Rate", "Time_to_50_Completion"]
    X_test = test_df[feature_cols]
    y_test = (test_df["Final_Value"] >= 0.90).astype(int)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance Metrics
    print(f"\n🚀 {model_name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"\nClassification Report ({model_name}):\n", classification_report(y_test, y_pred, zero_division=0))

    # Feature Importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
    plt.xticks(range(len(feature_importance)), np.array(feature_cols)[sorted_idx], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance in {model_name}")
    
    # Use non-blocking display
    plt.show(block=False)




def train_random_forest(train_df):
    """
    Train a Random Forest model without Final_Value to balance feature importance.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Initial_Growth_Rate", "Time_to_50_Completion"]  # Removed Final_Value
    X_train = train_df[feature_cols]
    y_train = (train_df["Final_Value"] >= 0.90).astype(int)  # Keep it for labeling but not as a predictor

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def train_xgboost(train_df):
    """
    Train an XGBoost model without Final_Value.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Initial_Growth_Rate", "Time_to_50_Completion"]  # Removed Final_Value
    X_train = train_df[feature_cols]
    y_train = (train_df["Final_Value"] >= 0.90).astype(int)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=2,  # Helps with class imbalance
        random_state=42
    )

    model.fit(X_train, y_train)
    return model