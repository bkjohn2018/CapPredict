from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_logistic_regression(train_df):
    """
    Train a logistic regression model to predict project success.

    Parameters:
        train_df (pd.DataFrame): Training dataset containing features.

    Returns:
        LogisticRegression: Trained model.
    """
    # Define features and target variable
    feature_cols = ["Inflection_Point", "Growth_Rate", "Final_Value"]
    X_train = train_df[feature_cols]
    y_train = (train_df["Final_Value"] >= 0.95).astype(int)  # Binary success indicator

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, test_df):
    """
    Evaluate the logistic regression model on test data.

    Parameters:
        model (LogisticRegression): Trained logistic regression model.
        test_df (pd.DataFrame): Test dataset containing features.

    Returns:
        None: Prints model accuracy and classification report.
    """
    feature_cols = ["Inflection_Point", "Growth_Rate", "Final_Value"]
    X_test = test_df[feature_cols]
    y_test = (test_df["Final_Value"] >= 0.95).astype(int)  # True labels

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
