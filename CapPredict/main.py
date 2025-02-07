from data_loader import load_data
from data_preprocessing import extract_features, normalize_features, split_data
from predictive_model import train_logistic_regression, evaluate_model
import matplotlib.pyplot as plt

def run_pipeline(source="csv", filepath="data/project_data.csv"):
    # Step 1: Load Data (from CSV for Palantir or OneLake for Fabric)
    data_df = load_data(source=source, filepath=filepath)

    # Step 2: Extract Features
    features_df = extract_features(data_df)

    # Step 3: Normalize Features
    normalized_df = normalize_features(features_df)

    # Step 4: Split Data
    train_df, test_df = split_data(normalized_df)

    # Step 5: Train Model
    model = train_logistic_regression(train_df)

    # Step 6: Evaluate Model
    evaluate_model(model, test_df)

    print("\nForecasting Pipeline Completed.")

if __name__ == "__main__":
    # Run pipeline using CSV for Palantir (default)
    run_pipeline(source="csv", filepath="data/project_data.csv")

    # Uncomment for Fabric deployment
    # run_pipeline(source="fabric")
