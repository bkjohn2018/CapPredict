from data_preprocessing import generate_curves, extract_features, normalize_features, split_data
from predictive_model import train_random_forest, train_xgboost, evaluate_model
from config import config
import matplotlib.pyplot as plt

def run_pipeline():
    """
    Execute the complete CapPredict machine learning pipeline.
    
    Uses centralized configuration for all parameters to ensure consistency
    and maintainability across the pipeline execution.
    """
    # Display configuration summary
    if config.logging.show_debug_prints:
        print(config.summary())
        print()
    
    # Step 1: Generate S-Curves using configured parameters
    if config.logging.show_debug_prints:
        print(f"🔄 Generating {config.data_generation.num_curves} synthetic S-curves...")
    
    curves_df = generate_curves(
        num_curves=config.data_generation.num_curves,
        time_start=config.data_generation.time_start,
        time_end=config.data_generation.time_end,
        num_points=config.data_generation.num_points,
        seed=config.data_generation.random_seed
    )

    # Step 2: Extract Enhanced Features
    if config.logging.show_debug_prints:
        print("🔍 Extracting features from S-curves...")
    features_df = extract_features(curves_df)

    # Step 3: Normalize Data
    if config.logging.show_debug_prints:
        print("📏 Normalizing feature data...")
    normalized_df = normalize_features(features_df)

    # Step 4: Split Data with configured parameters
    if config.logging.show_debug_prints:
        print(f"✂️ Splitting data ({int((1-config.data_processing.test_size)*100)}% train, {int(config.data_processing.test_size*100)}% test)...")
    train_df, test_df = split_data(
        normalized_df, 
        test_size=config.data_processing.test_size, 
        random_state=config.data_processing.random_state
    )

    # Step 5: Train and Evaluate Random Forest
    emoji = "🚀" if config.logging.use_emojis else ""
    print(f"\n{emoji} Training Random Forest...")
    rf_model = train_random_forest(train_df)
    evaluate_model(rf_model, test_df, model_name="Random Forest")

    # Step 6: Train and Evaluate XGBoost
    print(f"\n{emoji} Training XGBoost...")
    xgb_model = train_xgboost(train_df)
    evaluate_model(xgb_model, test_df, model_name="XGBoost")

    # Handle plot display based on configuration
    if config.visualization.show_plots:
        plt.show(block=config.visualization.block_plots)

if __name__ == "__main__":
    run_pipeline()
