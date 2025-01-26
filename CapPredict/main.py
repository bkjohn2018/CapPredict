from data_preprocessing import generate_curves, extract_features
import matplotlib.pyplot as plt

def run_pipeline():
    # Step 1: Generate S-Curves
    curves_df = generate_curves()

    # Step 2: Extract Features
    features_df = extract_features(curves_df)
    print("Extracted Features:")
    print(features_df)

    # Plot a sample of 5 curves
    plt.figure(figsize=(10, 6))
    for col in curves_df.columns[:5]:  # Plot only the first 5 projects
        plt.plot(curves_df["Time"], curves_df[col], label=col)

    plt.title("Sample of Random S-Curves")
    plt.xlabel("Time")
    plt.ylabel("Progress")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_pipeline()
