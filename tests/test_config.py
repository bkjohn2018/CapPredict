from config import config


def test_config_validate():
    assert config.validate() is True


def test_feature_columns_non_empty():
    assert len(config.features.model_features) > 0
    assert config.features.target_feature not in config.features.model_features


