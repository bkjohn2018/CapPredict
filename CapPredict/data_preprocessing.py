from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def generate_curves(num_curves=100, time_start=0, time_end=10, num_points=500, seed=42):
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
    features = []
    time = curves_df["Time"]

    for col in curves_df.columns[:-1]:  # Exclude the "Time" column
        curve = curves_df[col]

        # Compute inflection point (steepest slope)
        inflection_point = time[np.argmax(np.gradient(curve))]

        # Growth rate (max slope)
        growth_rate = np.max(np.gradient(curve))

        # Final value (completion percentage)
        final_value = curve.iloc[-1]

        # Initial growth rate (slope at start)
        initial_growth_rate = np.gradient(curve)[0]

        # Time to reach 50% completion
        try:
            half_completion_index = np.where(curve >= 0.5)[0][0]
            time_to_half_completion = time[half_completion_index]
        except IndexError:
            # If curve never reaches 50%, set a high default value
            time_to_half_completion = np.nan  

        features.append({
            "Project": col,
            "Inflection_Point": inflection_point,
            "Growth_Rate": growth_rate,
            "Final_Value": final_value,
            "Initial_Growth_Rate": initial_growth_rate,
            "Time_to_50_Completion": time_to_half_completion
        })

    features_df = pd.DataFrame(features)
    print("\nExtracted Features (Sample):")
    print(features_df.head())  # Debugging print
    return features_df

def normalize_features(features_df):
    """
    Normalize feature values using Min-Max Scaling.
    """
    scaler = MinMaxScaler()
    
    # Selecting only numerical feature columns
    feature_columns = ["Inflection_Point", "Growth_Rate", "Final_Value", "Initial_Growth_Rate", "Time_to_50_Completion"]

    # Ensure only numerical columns are normalized
    normalized_values = scaler.fit_transform(features_df[feature_columns])
    
    # Convert back to DataFrame
    normalized_df = pd.DataFrame(normalized_values, columns=feature_columns)
    normalized_df.insert(0, "Project", features_df["Project"])  # Keep project names

    print("\nNormalized Features (Sample):")
    print(normalized_df.head())  # Debugging print

    return normalized_df

def split_data(features_df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets and ensure Final_Value is removed from feature set.
    """
    train_df, test_df = train_test_split(features_df, test_size=test_size, random_state=random_state)

    feature_cols = ["Inflection_Point", "Growth_Rate", "Initial_Growth_Rate", "Time_to_50_Completion"]

    # Apply SMOTE for class balancing
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(train_df[feature_cols], train_df["Final_Value"] >= 0.90)

    # Convert back to DataFrame
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["Final_Value"] = y_train.astype(int)  # Keep it for labels but not as a feature

    test_df = test_df[feature_cols + ["Final_Value"]]  # Ensure test_df has the correct structure

    print("\nBalanced Training Set:")
    print(train_df["Final_Value"].value_counts())  # Show class distribution

    return train_df, test_df
