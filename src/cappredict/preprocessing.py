import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from .config import config as _global_config

log = logging.getLogger(__name__)


def ensure_no_nans(df: pd.DataFrame, ctx: str):
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"[{ctx}] NaNs after preprocessing in {cols}. "
                         "Set config.preprocessing.impute_strategy or fix upstream.")


def normalize_features(features_df: pd.DataFrame, *, config=None) -> pd.DataFrame:
    scaler = MinMaxScaler()
    cfg = config or _global_config
    feature_columns = cfg.features.all_features
    normalized_values = scaler.fit_transform(features_df[feature_columns])
    normalized_df = pd.DataFrame(normalized_values, columns=feature_columns)
    normalized_df.insert(0, cfg.features.project_id_column, features_df[cfg.features.project_id_column])
    ensure_no_nans(normalized_df, ctx="normalize_features")
    if cfg.logging.show_debug_prints:
        log.info("normalized features: rows=%s cols=%s", normalized_df.shape[0], normalized_df.shape[1])
    return normalized_df


def split_data(features_df: pd.DataFrame, test_size=None, random_state=None, *, config=None):
    cfg = config or _global_config
    test_size = test_size if test_size is not None else cfg.data_processing.test_size
    random_state = random_state if random_state is not None else cfg.data_processing.random_state

    train_df, test_df = train_test_split(features_df, test_size=test_size, random_state=random_state)

    feature_cols = cfg.features.model_features
    target_variable = train_df[cfg.features.target_feature] >= cfg.data_generation.success_threshold
    
    # Determine appropriate k_neighbors for SMOTE based on minority class size
    minority_class_count = min(target_variable.sum(), (~target_variable).sum())
    # SMOTE requires k_neighbors < minority_class_count
    k_neighbors = min(5, max(1, minority_class_count - 1))
    
    smote = SMOTE(
        sampling_strategy=cfg.data_processing.smote_sampling_strategy,
        random_state=cfg.data_processing.smote_random_state,
        k_neighbors=k_neighbors,
    )

    X_train, y_train = smote.fit_resample(train_df[feature_cols], target_variable)

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[cfg.features.target_feature] = y_train.astype(int)
    test_df = test_df[feature_cols + [cfg.features.target_feature]]

    ensure_no_nans(train_df, ctx="split_data.train")
    ensure_no_nans(test_df, ctx="split_data.test")

    if cfg.logging.show_debug_prints:
        counts = train_df[cfg.features.target_feature].value_counts().to_dict()
        log.info("balanced training labels: %s", counts)

    return train_df, test_df


