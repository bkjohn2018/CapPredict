# import pandas as pd
# import os

# def load_data(source="csv", filepath=None):
#     """
#     Load project data dynamically based on the platform (Palantir vs. MS Fabric).
    
#     Parameters:
#         source (str): "csv" for local CSV files, "fabric" for MS Fabric Dataflows.
#         filepath (str, optional): Path to the dataset file.

#     Returns:
#         pd.DataFrame: Loaded dataset.
#     """
#     if source == "csv":
#         if filepath is None:
#             raise ValueError("Filepath must be provided for CSV source.")
#         print(f"Loading data from CSV: {filepath}")
#         return pd.read_csv(filepath)

#     elif source == "fabric":
#         print("Loading data from MS Fabric (OneLake)...")
#         # Placeholder: In Fabric, this would pull from a live Dataflow
#         # Simulating Fabric data for now
#         data = {
#             "Project": [f"Project_{i}" for i in range(1, 11)],
#             "Inflection_Point": [5.1, 6.0, 4.8, 5.5, 6.2, 5.0, 4.9, 5.7, 6.1, 4.5],
#             "Growth_Rate": [0.02, 0.03, 0.015, 0.018, 0.025, 0.019, 0.021, 0.022, 0.024, 0.017],
#             "Final_Value": [0.98, 0.96, 0.99, 1.0, 0.95, 0.97, 0.94, 0.93, 0.98, 0.96]
#         }
#         return pd.DataFrame(data)

#     else:
#         raise ValueError("Invalid source type. Choose 'csv' or 'fabric'.")
