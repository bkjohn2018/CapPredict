from .data_preprocessing import generate_curves, extract_features, normalize_features, split_data
from .predictive_model import train_random_forest, train_xgboost, evaluate_model
import matplotlib.pyplot as plt

def run_pipeline():
    # Step 1: Generate More Curves
    curves_df = generate_curves(num_curves=100)

    # Step 2: Extract Enhanced Features
    features_df = extract_features(curves_df)

    # Step 3: Normalize Data
    normalized_df = normalize_features(features_df)

    # Step 4: Split Data
    train_df, test_df = split_data(normalized_df)

    # Step 5: Train and Evaluate Random Forest
    print("\n🚀 Training Random Forest...")
    rf_model = train_random_forest(train_df)
    evaluate_model(rf_model, test_df, model_name="Random Forest")

    # Step 6: Train and Evaluate XGBoost
    print("\n🚀 Training XGBoost...")
    xgb_model = train_xgboost(train_df)
    evaluate_model(xgb_model, test_df, model_name="XGBoost")

    # Keep the plots open
    plt.show()

if __name__ == "__main__":
    run_pipeline()
