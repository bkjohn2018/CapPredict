import numpy as np
import pandas as pd
import logging
from config import config

log = logging.getLogger(__name__)


def extract_features(curves_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw S-curves into domain-specific features for ML.
    """
    features = []
    time = curves_df[config.features.time_column]

    for col in curves_df.columns[:-1]:  # exclude time column
        curve = curves_df[col]

        inflection_point = time[np.argmax(np.gradient(curve))]
        growth_rate = np.max(np.gradient(curve))
        final_value = curve.iloc[-1]
        initial_growth_rate = np.gradient(curve)[0]

        try:
            threshold_indices = np.where(curve >= config.data_generation.half_completion_threshold)[0]
            if len(threshold_indices) > 0:
                time_to_half_completion = time[threshold_indices[0]]
            else:
                time_to_half_completion = np.nan
        except IndexError:
            time_to_half_completion = np.nan

        features.append({
            config.features.project_id_column: col,
            "Inflection_Point": inflection_point,
            "Growth_Rate": growth_rate,
            "Final_Value": final_value,
            "Initial_Growth_Rate": initial_growth_rate,
            "Time_to_50_Completion": time_to_half_completion,
        })

    features_df = pd.DataFrame(features)
    if config.logging.show_debug_prints:
        log.info("features extracted: rows=%s cols=%s", features_df.shape[0], features_df.shape[1])
    return features_df


