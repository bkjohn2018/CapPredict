from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report
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

def evaluate_model(model, test_df):
    """
    Evaluate the Random Forest model on test data with zero-division handling.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Final_Value", "Initial_Growth_Rate", "Time_to_50_Completion"]
    X_test = test_df[feature_cols]
    y_test = (test_df["Final_Value"] >= 0.90).astype(int)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance Metrics
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Feature Importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
    plt.xticks(range(len(feature_importance)), np.array(feature_cols)[sorted_idx], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance in Random Forest")
    plt.show()


def train_random_forest(train_df):
    """
    Train a Random Forest model for project success prediction with optimized parameters.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Final_Value", "Initial_Growth_Rate", "Time_to_50_Completion"]
    X_train = train_df[feature_cols]
    y_train = (train_df["Final_Value"] >= 0.90).astype(int)  # Binary success indicator

    # Optimize Random Forest
    model = RandomForestClassifier(
        n_estimators=200,  # Increase trees for better stability
        max_depth=5,  # Prevent overfitting by limiting tree depth
        min_samples_split=10,  # Require at least 10 samples to split nodes
        min_samples_leaf=5,  # Minimum samples per leaf to avoid overfitting
        class_weight="balanced",  # Handle class imbalance
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

