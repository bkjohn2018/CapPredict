import pytest
from cappredict.models.registry import build_default_registry
from cappredict.models.training import train_model_generic
from cappredict.data_loading import generate_curves
from cappredict.feature_engineering import extract_features
from cappredict.preprocessing import normalize_features, split_data
from cappredict.config import config


def test_registry_has_default_models():
    registry = build_default_registry()
    names = registry.list_models()
    assert "Random Forest" in names
    assert "XGBoost" in names


@pytest.mark.parametrize("model_name", ["Random Forest", "XGBoost"])
def test_train_and_predict(model_name):
    registry = build_default_registry()
    model = registry.create(model_name)

    curves = generate_curves()
    features = extract_features(curves)
    normalized = normalize_features(features)
    train_df, test_df = split_data(normalized)

    trained = train_model_generic(model, train_df)
    # Basic predict sanity
    X = test_df[config.features.model_features]
    preds = trained.predict(X)
    assert len(preds) == len(X)


