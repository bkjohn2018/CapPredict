from typing import Callable, Dict, List
from config import config
from sklearn.ensemble import RandomForestClassifier


class ModelRegistry:
    """
    Simple registry that maps model names to constructor callables.
    Extensible without modifying call sites (OCP), and injectable (DIP).
    """

    def __init__(self) -> None:
        self._name_to_factory: Dict[str, Callable[[], object]] = {}

    def register(self, name: str, factory: Callable[[], object]) -> None:
        self._name_to_factory[name] = factory

    def create(self, name: str):
        if name not in self._name_to_factory:
            raise KeyError(f"Model '{name}' not registered")
        return self._name_to_factory[name]()

    def list_models(self) -> List[str]:
        return list(self._name_to_factory.keys())


def build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()

    registry.register(
        "Random Forest",
        lambda: RandomForestClassifier(
            n_estimators=config.models.rf_n_estimators,
            max_depth=config.models.rf_max_depth,
            min_samples_split=config.models.rf_min_samples_split,
            min_samples_leaf=config.models.rf_min_samples_leaf,
            class_weight=config.models.rf_class_weight,
            random_state=config.models.rf_random_state,
        ),
    )

    # Lazy import XGBClassifier to avoid hard dependency at import time
    def _build_xgb():
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier(
            n_estimators=config.models.xgb_n_estimators,
            max_depth=config.models.xgb_max_depth,
            learning_rate=config.models.xgb_learning_rate,
            scale_pos_weight=config.models.xgb_scale_pos_weight,
            random_state=config.models.xgb_random_state,
            eval_metric="logloss",
        )

    registry.register("XGBoost", _build_xgb)

    return registry


