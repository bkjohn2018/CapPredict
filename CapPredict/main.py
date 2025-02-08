from data_preprocessing import generate_curves, extract_features, normalize_features, split_data
from predictive_model import train_random_forest, evaluate_model
import matplotlib.pyplot as plt

def run_pipeline():
    # Step 1: Generate More Curves
    curves_df = generate_curves(num_curves=100)

    # Step 2: Extract Enhanced Features
    features_df = extract_features(curves_df)

    # Step 3: Normalize Data
    normalized_df = normalize_features(features_df)

    # Step 4: Split Data (with SMOTE balancing)
    train_df, test_df = split_data(normalized_df)

    # Step 5: Train Model (Random Forest)
    model = train_random_forest(train_df)

    # Step 6: Evaluate Performance
    evaluate_model(model, test_df)

    print("\nRandom Forest Model Training Complete!")

if __name__ == "__main__":
    run_pipeline()
