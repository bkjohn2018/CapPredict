import numpy as np
import pandas as pd
import logging
from config import config

log = logging.getLogger(__name__)


def generate_curves(num_curves=None, time_start=None, time_end=None, num_points=None, seed=None) -> pd.DataFrame:
    """
    Generate random logistic S-curves to simulate project progress.

    Acts as a data source provider for the pipeline.
    """
    # Use configuration defaults if parameters not provided
    num_curves = num_curves if num_curves is not None else config.data_generation.num_curves
    time_start = time_start if time_start is not None else config.data_generation.time_start
    time_end = time_end if time_end is not None else config.data_generation.time_end
    num_points = num_points if num_points is not None else config.data_generation.num_points
    seed = seed if seed is not None else config.data_generation.random_seed

    np.random.seed(seed)
    time = np.linspace(time_start, time_end, num_points)

    curves = []
    for _ in range(num_curves):
        k = np.random.uniform(config.data_generation.steepness_min, config.data_generation.steepness_max)
        x0 = np.random.uniform(config.data_generation.midpoint_min, config.data_generation.midpoint_max)
        y = 1 / (1 + np.exp(-k * (time - x0)))
        curves.append(y)

    curves_array = np.array(curves)
    curves_df = pd.DataFrame(
        data=np.column_stack([curves_array.T, time]),
        columns=[f"Project_{i+1}" for i in range(num_curves)] + [config.features.time_column],
    )

    if config.logging.show_debug_prints:
        log.info("curves generated: rows=%s cols=%s", curves_df.shape[0], curves_df.shape[1])

    return curves_df


