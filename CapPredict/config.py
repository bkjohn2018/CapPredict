"""
CapPredict Configuration Management

This module centralizes all configuration parameters used throughout the CapPredict
machine learning pipeline, eliminating hardcoded values and improving maintainability.
"""

from dataclasses import dataclass
from typing import List, Tuple
import os


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic S-curve data generation."""
    
    # Curve generation parameters
    num_curves: int = 100
    time_start: float = 0.0
    time_end: float = 10.0
    num_points: int = 500
    random_seed: int = 42
    
    # S-curve shape parameters
    steepness_min: float = 0.5
    steepness_max: float = 2.0
    midpoint_min: float = 3.0
    midpoint_max: float = 7.0
    
    # Success threshold for classification
    success_threshold: float = 0.90
    half_completion_threshold: float = 0.5


@dataclass
class FeatureConfig:
    """Configuration for feature engineering and data processing."""
    
    # Feature column definitions
    all_features: List[str] = None
    model_features: List[str] = None  # Features used for training (excludes target)
    target_feature: str = "Final_Value"
    project_id_column: str = "Project"
    time_column: str = "Time"
    
    def __post_init__(self):
        """Initialize feature lists after dataclass creation."""
        if self.all_features is None:
            self.all_features = [
                "Inflection_Point",
                "Growth_Rate", 
                "Final_Value",
                "Initial_Growth_Rate",
                "Time_to_50_Completion"
            ]
        
        if self.model_features is None:
            # Model features exclude the target variable to prevent data leakage
            self.model_features = [
                "Inflection_Point",
                "Growth_Rate",
                "Initial_Growth_Rate", 
                "Time_to_50_Completion"
            ]


@dataclass
class DataProcessingConfig:
    """Configuration for data preprocessing and splitting."""
    
    # Train/test split parameters
    test_size: float = 0.2
    random_state: int = 42
    
    # SMOTE parameters for class balancing
    smote_sampling_strategy: float = 0.5
    smote_random_state: int = 42
    
    # Normalization parameters
    normalization_method: str = "minmax"  # Future: support for other methods


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    
    # Random Forest parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 5
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 5
    rf_class_weight: str = "balanced"
    rf_random_state: int = 42
    
    # XGBoost parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1
    xgb_scale_pos_weight: float = 2.0
    xgb_random_state: int = 42
    
    # Evaluation parameters
    success_threshold: float = 0.90  # Same as DataGenerationConfig.success_threshold


@dataclass
class VisualizationConfig:
    """Configuration for plots and visualizations."""
    
    # Figure parameters
    figure_width: int = 8
    figure_height: int = 5
    
    # Plot display parameters
    show_plots: bool = True
    block_plots: bool = False  # Non-blocking display
    rotation_angle: int = 45   # X-axis label rotation
    
    # Output parameters
    save_plots: bool = False
    plot_output_dir: str = "plots"
    plot_format: str = "png"
    plot_dpi: int = 300


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""
    
    # Core logging parameters (aligned with requested schema)
    level: str = "INFO"              # DEBUG/INFO/WARNING/ERROR
    show_debug_prints: bool = False   # keep verbose previews behind this flag
    propagate: bool = False
    json: bool = False

    # Back-compat and extras used elsewhere in app
    log_level: str = "INFO"          # legacy alias; not used by new setup
    show_progress_bars: bool = True
    
    # Output formatting (non-logging core)
    use_emojis: bool = True
    decimal_places: int = 2


class Config:
    """
    Main configuration class that aggregates all configuration sections.
    
    This class provides a single point of access to all configuration parameters
    and can be easily extended or modified for different environments.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration with optional config file override.
        
        Args:
            config_file: Optional path to configuration file (future enhancement)
        """
        self.data_generation = DataGenerationConfig()
        self.features = FeatureConfig()
        self.data_processing = DataProcessingConfig()
        self.models = ModelConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        
        # Load from file if provided (future enhancement)
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_from_file(self, config_file: str):
        """
        Load configuration from file (placeholder for future implementation).
        
        Args:
            config_file: Path to configuration file (JSON/YAML/TOML)
        """
        # TODO: Implement configuration file loading
        # This could support JSON, YAML, or TOML formats
        pass
    
    def validate(self) -> bool:
        """
        Validate configuration parameters for consistency and correctness.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        errors = []
        
        # Validate data generation parameters
        if self.data_generation.num_curves <= 0:
            errors.append("num_curves must be positive")
        
        if self.data_generation.time_start >= self.data_generation.time_end:
            errors.append("time_start must be less than time_end")
        
        if self.data_generation.num_points <= 1:
            errors.append("num_points must be greater than 1")
        
        if not (0 < self.data_generation.success_threshold < 1):
            errors.append("success_threshold must be between 0 and 1")
        
        # Validate data processing parameters
        if not (0 < self.data_processing.test_size < 1):
            errors.append("test_size must be between 0 and 1")
        
        # Validate model parameters
        if self.models.rf_n_estimators <= 0:
            errors.append("rf_n_estimators must be positive")
        
        if self.models.xgb_n_estimators <= 0:
            errors.append("xgb_n_estimators must be positive")
        
        if self.models.xgb_learning_rate <= 0:
            errors.append("xgb_learning_rate must be positive")
        
        # Validate feature consistency
        if self.features.target_feature in self.features.model_features:
            errors.append("target_feature should not be in model_features")
        
        if errors:
            import logging
            log = logging.getLogger(__name__)
            for error in errors:
                log.error("config validation error: %s", error)
            return False
        
        return True
    
    def summary(self) -> str:
        """
        Generate a summary of current configuration settings.
        
        Returns:
            str: Formatted configuration summary
        """
        summary = []
        summary.append("=== CapPredict Configuration Summary ===")
        summary.append(f"Data Generation: {self.data_generation.num_curves} curves, "
                      f"seed={self.data_generation.random_seed}")
        summary.append(f"Features: {len(self.features.model_features)} model features")
        summary.append(f"Data Split: {int((1-self.data_processing.test_size)*100)}% train, "
                      f"{int(self.data_processing.test_size*100)}% test")
        summary.append(f"Models: RF({self.models.rf_n_estimators} trees), "
                      f"XGB({self.models.xgb_n_estimators} estimators)")
        summary.append(f"Success Threshold: {self.data_generation.success_threshold}")
        return "\n".join(summary)


# Global configuration instance
# This can be imported and used throughout the application
config = Config()


# Convenience functions for backward compatibility and ease of use
def get_feature_columns() -> List[str]:
    """Get the list of feature columns used for model training."""
    return config.features.model_features.copy()


def get_all_feature_columns() -> List[str]:
    """Get the list of all feature columns including target."""
    return config.features.all_features.copy()


def get_success_threshold() -> float:
    """Get the success threshold for classification."""
    return config.data_generation.success_threshold


def get_random_seed() -> int:
    """Get the random seed for reproducibility."""
    return config.data_generation.random_seed


# Development and testing configurations
class DevelopmentConfig(Config):
    """Configuration optimized for development and testing."""
    
    def __init__(self):
        super().__init__()
        # Smaller datasets for faster development
        self.data_generation.num_curves = 50
        self.data_generation.num_points = 100
        
        # Simpler models for faster training
        self.models.rf_n_estimators = 50
        self.models.xgb_n_estimators = 50
        
        # Enhanced debugging
        self.logging.show_debug_prints = True
        self.logging.show_progress_bars = True


class ProductionConfig(Config):
    """Configuration optimized for production use."""
    
    def __init__(self):
        super().__init__()
        # Larger datasets for better model performance
        self.data_generation.num_curves = 1000
        self.data_generation.num_points = 1000
        
        # More robust models
        self.models.rf_n_estimators = 500
        self.models.xgb_n_estimators = 500
        
        # Minimal logging for production
        self.logging.show_debug_prints = False
        self.logging.use_emojis = False
        
        # Save plots for analysis
        self.visualization.save_plots = True
        self.visualization.show_plots = False


if __name__ == "__main__":
    # Configuration validation and testing (using centralized logging)
    from logging_setup import setup_logging
    setup_logging()
    import logging
    log = logging.getLogger(__name__)

    default_config = Config()
    log.info("default config valid: %s", default_config.validate())
    log.info("%s", default_config.summary())

    dev_config = DevelopmentConfig()
    log.info("development config valid: %s", dev_config.validate())
    log.info("%s", dev_config.summary())

    prod_config = ProductionConfig()
    log.info("production config valid: %s", prod_config.validate())
    log.info("%s", prod_config.summary())
