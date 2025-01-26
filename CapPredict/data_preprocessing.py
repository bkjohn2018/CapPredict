import numpy as np
import pandas as pd

def generate_curves(num_curves=30, time_start=0, time_end=10, num_points=500, seed=42):
    """
    Generate random logistic S-curves to simulate project progress.

    Parameters:
        num_curves (int): Number of S-curves to generate.
        time_start (float): Start of the time range.
        time_end (float): End of the time range.
        num_points (int): Number of time points per curve.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing generated S-curves and a Time column.
    """
    np.random.seed(seed)  # Ensure reproducibility

    # Create the time range
    time = np.linspace(time_start, time_end, num_points)

    # Initialize storage for curves
    curves = []

    # Generate each curve
    for i in range(num_curves):
        # Randomize steepness (k) and midpoint (x0)
        k = np.random.uniform(0.5, 2.0)  # Steepness of the curve
        x0 = np.random.uniform(3, 7)     # Midpoint of the curve

        # Logistic function: y = 1 / (1 + e^(-k(x - x0)))
        y = 1 / (1 + np.exp(-k * (time - x0)))  # Sigmoid function
        curves.append(y)

    # Convert list of curves to a 2D array
    curves_array = np.array(curves)

    # Combine curves and time into a single DataFrame
    curves_df = pd.DataFrame(
        data=np.column_stack([curves_array.T, time]),  # Combine S-curves (columns) and time
        columns=[f"Project_{i+1}" for i in range(num_curves)] + ["Time"]  # Add "Time" as the last column
    )

    return curves_df

def extract_features(curves_df):
    """
    Extract features from S-curves for analysis and modeling.

    Parameters:
        curves_df (pd.DataFrame): DataFrame containing S-curves and a Time column.

    Returns:
        pd.DataFrame: A DataFrame of extracted features for each curve.
    """
    features = []
    time = curves_df["Time"]

    # Iterate through each project column
    for col in curves_df.columns[:-1]:  # Exclude the "Time" column
        curve = curves_df[col]

        # Calculate features
        inflection_point = time[np.argmax(np.gradient(curve))]  # Time at steepest change
        growth_rate = np.max(np.gradient(curve))               # Maximum rate of growth
        final_value = curve.iloc[-1]                           # Final value (asymptote)

        # Store the extracted features
        features.append({
            "Project": col,
            "Inflection_Point": inflection_point,
            "Growth_Rate": growth_rate,
            "Final_Value": final_value
        })

    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    return features_df