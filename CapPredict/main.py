from data_loading import generate_curves
from feature_engineering import extract_features
from preprocessing import normalize_features, split_data
from predictive_model import evaluate_model
from models.registry import build_default_registry
from models.training import train_model_generic
from config import config
from logging_setup import setup_logging
import logging
import matplotlib.pyplot as plt

def run_pipeline():
    """
    Execute the complete CapPredict machine learning pipeline.
    
    Uses centralized configuration for all parameters to ensure consistency
    and maintainability across the pipeline execution.
    """
    # Display configuration summary
    log = logging.getLogger(__name__)
    if config.logging.show_debug_prints:
        log.info("%s", config.summary())
    
    # Step 1: Generate S-Curves using configured parameters
    if config.logging.show_debug_prints:
        log.info("Generating curves: num_curves=%s, points=%s", config.data_generation.num_curves, config.data_generation.num_points)
    
    curves_df = generate_curves(
        num_curves=config.data_generation.num_curves,
        time_start=config.data_generation.time_start,
        time_end=config.data_generation.time_end,
        num_points=config.data_generation.num_points,
        seed=config.data_generation.random_seed
    )

    # Step 2: Extract Enhanced Features
    if config.logging.show_debug_prints:
        log.info("Extracting features from curves...")
    features_df = extract_features(curves_df)

    # Step 3: Normalize Data
    if config.logging.show_debug_prints:
        log.info("Normalizing feature data...")
    normalized_df = normalize_features(features_df)

    # Step 4: Split Data with configured parameters
    if config.logging.show_debug_prints:
        log.info("Splitting data: train=%s%% test=%s%%", int((1-config.data_processing.test_size)*100), int(config.data_processing.test_size*100))
    train_df, test_df = split_data(
        normalized_df, 
        test_size=config.data_processing.test_size, 
        random_state=config.data_processing.random_state
    )

    # Step 5: Train and evaluate all registered models (OCP/DIP)
    emoji = "🚀" if config.logging.use_emojis else ""
    registry = build_default_registry()
    for model_name in registry.list_models():
        log.info("Training model: %s", model_name)
        model = registry.create(model_name)
        model = train_model_generic(model, train_df)
        evaluate_model(model, test_df, model_name=model_name)

    # Handle plot display based on configuration
    if config.visualization.show_plots:
        plt.show(block=config.visualization.block_plots)

if __name__ == "__main__":
    # Initialize centralized logging early
    setup_logging(level=config.logging.level, json_mode=config.logging.json, propagate=config.logging.propagate)
    run_pipeline()
