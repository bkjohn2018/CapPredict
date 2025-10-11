import pandas as pd
from data_preprocessing import generate_curves, extract_features, normalize_features, split_data
from config import config


def test_generate_curves_shape():
    curves = generate_curves()
    # Expect num_points rows and num_curves+1 columns (projects + time)
    assert curves.shape[0] == config.data_generation.num_points
    assert curves.shape[1] == config.data_generation.num_curves + 1
    assert config.features.time_column in curves.columns


def test_extract_and_normalize_features():
    curves = generate_curves()
    features = extract_features(curves)
    assert set(["Project", "Inflection_Point", "Growth_Rate", "Final_Value", "Initial_Growth_Rate", "Time_to_50_Completion"]).issubset(features.columns)

    normalized = normalize_features(features)
    for col in config.features.all_features:
        assert col in normalized.columns


def test_split_data_shapes():
    curves = generate_curves()
    features = extract_features(curves)
    normalized = normalize_features(features)
    train_df, test_df = split_data(normalized)

    # Ensure required columns are present
    for df in (train_df, test_df):
        for col in config.features.model_features + [config.features.target_feature]:
            assert col in df.columns


