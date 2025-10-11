import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from config import config

log = logging.getLogger(__name__)


def ensure_no_nans(df: pd.DataFrame, ctx: str):
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"[{ctx}] NaNs after preprocessing in {cols}. "
                         "Set config.preprocessing.impute_strategy or fix upstream.")


def normalize_features(features_df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    feature_columns = config.features.all_features
    normalized_values = scaler.fit_transform(features_df[feature_columns])
    normalized_df = pd.DataFrame(normalized_values, columns=feature_columns)
    normalized_df.insert(0, config.features.project_id_column, features_df[config.features.project_id_column])
    ensure_no_nans(normalized_df, ctx="normalize_features")
    if config.logging.show_debug_prints:
        log.info("normalized features: rows=%s cols=%s", normalized_df.shape[0], normalized_df.shape[1])
    return normalized_df


def split_data(features_df: pd.DataFrame, test_size=None, random_state=None):
    test_size = test_size if test_size is not None else config.data_processing.test_size
    random_state = random_state if random_state is not None else config.data_processing.random_state

    train_df, test_df = train_test_split(features_df, test_size=test_size, random_state=random_state)

    feature_cols = config.features.model_features
    smote = SMOTE(
        sampling_strategy=config.data_processing.smote_sampling_strategy,
        random_state=config.data_processing.smote_random_state,
    )

    target_variable = train_df[config.features.target_feature] >= config.data_generation.success_threshold
    X_train, y_train = smote.fit_resample(train_df[feature_cols], target_variable)

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[config.features.target_feature] = y_train.astype(int)
    test_df = test_df[feature_cols + [config.features.target_feature]]

    ensure_no_nans(train_df, ctx="split_data.train")
    ensure_no_nans(test_df, ctx="split_data.test")

    if config.logging.show_debug_prints:
        counts = train_df[config.features.target_feature].value_counts().to_dict()
        log.info("balanced training labels: %s", counts)

    return train_df, test_df


