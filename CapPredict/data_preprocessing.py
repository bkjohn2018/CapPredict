import warnings
warnings.warn(
    "data_preprocessing is deprecated; use cappredict.preprocessing instead.",
    DeprecationWarning, stacklevel=2
)

# Use absolute imports to work when this module is imported as a top-level module
from preprocessing import normalize_features, split_data, ensure_no_nans  # type: ignore
from data_loading import generate_curves  # type: ignore
from feature_engineering import extract_features  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from config import config
import logging

log = logging.getLogger(__name__)

def generate_curves(num_curves=None, time_start=None, time_end=None, num_points=None, seed=None):
    """
    Generate random logistic S-curves to simulate project progress.

    Parameters:
        num_curves (int, optional): Number of S-curves to generate. Uses config default if None.
        time_start (float, optional): Start of the time range. Uses config default if None.
        time_end (float, optional): End of the time range. Uses config default if None.
        num_points (int, optional): Number of time points per curve. Uses config default if None.
        seed (int, optional): Random seed for reproducibility. Uses config default if None.

    Returns:
        pd.DataFrame: A DataFrame containing generated S-curves and a Time column.
    """
    # Use configuration defaults if parameters not provided
    num_curves = num_curves if num_curves is not None else config.data_generation.num_curves
    time_start = time_start if time_start is not None else config.data_generation.time_start
    time_end = time_end if time_end is not None else config.data_generation.time_end
    num_points = num_points if num_points is not None else config.data_generation.num_points
    seed = seed if seed is not None else config.data_generation.random_seed
    np.random.seed(seed)  # Ensure reproducibility

    # Create the time range
    time = np.linspace(time_start, time_end, num_points)

    # Initialize storage for curves
    curves = []

    # Generate each curve
    for i in range(num_curves):
        # Randomize steepness (k) and midpoint (x0) using configured ranges
        k = np.random.uniform(config.data_generation.steepness_min, config.data_generation.steepness_max)
        x0 = np.random.uniform(config.data_generation.midpoint_min, config.data_generation.midpoint_max)

        # Logistic function: y = 1 / (1 + e^(-k(x - x0)))
        y = 1 / (1 + np.exp(-k * (time - x0)))  # Sigmoid function
        curves.append(y)

    # Convert list of curves to a 2D array
    curves_array = np.array(curves)

    # Combine curves and time into a single DataFrame
    curves_df = pd.DataFrame(
        data=np.column_stack([curves_array.T, time]),  # Combine S-curves (columns) and time
        columns=[f"Project_{i+1}" for i in range(num_curves)] + [config.features.time_column]
    )

    return curves_df

def extract_features(curves_df):
    """
    Extract meaningful features from S-curves for machine learning.
    
    Args:
        curves_df: DataFrame containing S-curves and time column
        
    Returns:
        pd.DataFrame: Features DataFrame with extracted characteristics
    """
    features = []
    time = curves_df[config.features.time_column]

    for col in curves_df.columns[:-1]:  # Exclude the time column
        curve = curves_df[col]

        # Compute inflection point (steepest slope)
        inflection_point = time[np.argmax(np.gradient(curve))]

        # Growth rate (max slope)
        growth_rate = np.max(np.gradient(curve))

        # Final value (completion percentage)
        final_value = curve.iloc[-1]

        # Initial growth rate (slope at start)
        initial_growth_rate = np.gradient(curve)[0]

        # Time to reach 50% completion (using configured threshold)
        try:
            threshold_indices = np.where(curve >= config.data_generation.half_completion_threshold)[0]
            if len(threshold_indices) > 0:
                time_to_half_completion = time[threshold_indices[0]]
            else:
                time_to_half_completion = np.nan
        except IndexError:
            # If curve never reaches threshold, set to NaN
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
    
    if config.logging.show_debug_prints:
        log.info("features extracted: rows=%s cols=%s", features_df.shape[0], features_df.shape[1])
    
    return features_df

def normalize_features(features_df):
    """
    Normalize feature values using configured normalization method.
    
    Args:
        features_df: DataFrame with extracted features
        
    Returns:
        pd.DataFrame: Normalized features DataFrame
    """
    scaler = MinMaxScaler()  # Currently only Min-Max, but configurable for future
    
    # Use configured feature columns for normalization
    feature_columns = config.features.all_features

    # Ensure only numerical columns are normalized
    normalized_values = scaler.fit_transform(features_df[feature_columns])
    
    # Convert back to DataFrame
    normalized_df = pd.DataFrame(normalized_values, columns=feature_columns)
    normalized_df.insert(0, config.features.project_id_column, features_df[config.features.project_id_column])

    if config.logging.show_debug_prints:
        log.info("normalized features: rows=%s cols=%s", normalized_df.shape[0], normalized_df.shape[1])

    return normalized_df

def split_data(features_df, test_size=None, random_state=None):
    """
    Split the dataset into training and testing sets with configured class balancing.
    
    Args:
        features_df: DataFrame with normalized features
        test_size: Fraction for test set (uses config default if None)
        random_state: Random seed (uses config default if None)
        
    Returns:
        tuple: (train_df, test_df) with balanced training data
    """
    # Use configuration defaults if parameters not provided
    test_size = test_size if test_size is not None else config.data_processing.test_size
    random_state = random_state if random_state is not None else config.data_processing.random_state
    
    train_df, test_df = train_test_split(features_df, test_size=test_size, random_state=random_state)

    # Use configured feature columns (excluding target)
    feature_cols = config.features.model_features

    # Apply SMOTE for class balancing with configured parameters
    smote = SMOTE(
        sampling_strategy=config.data_processing.smote_sampling_strategy, 
        random_state=config.data_processing.smote_random_state
    )
    
    # Create target variable using configured success threshold
    target_variable = train_df[config.features.target_feature] >= config.data_generation.success_threshold
    X_train, y_train = smote.fit_resample(train_df[feature_cols], target_variable)

    # Convert back to DataFrame
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[config.features.target_feature] = y_train.astype(int)

    # Ensure test_df has the correct structure
    test_df = test_df[feature_cols + [config.features.target_feature]]

    if config.logging.show_debug_prints:
        counts = train_df[config.features.target_feature].value_counts().to_dict()
        log.info("balanced training labels: %s", counts)

    return train_df, test_df
