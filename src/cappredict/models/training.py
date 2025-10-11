from typing import Any
import pandas as pd
from config import config


def train_model_generic(model: Any, train_df: pd.DataFrame) -> Any:
    """
    Train any sklearn-like classifier using configured feature/target.

    Args:
        model: Any object supporting fit(X, y)
        train_df: Training DataFrame

    Returns:
        The same model instance, fitted
    """
    feature_cols = config.features.model_features
    X_train = train_df[feature_cols]
    y_train = (train_df[config.features.target_feature] >= config.models.success_threshold).astype(int)
    model.fit(X_train, y_train)
    return model


