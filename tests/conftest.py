import os
import sys
import pytest


# Ensure modules under CapPredict/ can be imported as top-level
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "CapPredict")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import config  # noqa: E402


@pytest.fixture(autouse=True)
def use_dev_settings():
    """Apply small test-friendly settings and restore after each test."""
    # Snapshot
    orig = {
        "num_curves": config.data_generation.num_curves,
        "num_points": config.data_generation.num_points,
        "show_plots": config.visualization.show_plots,
        "block_plots": config.visualization.block_plots,
        "test_size": config.data_processing.test_size,
        "random_state": config.data_processing.random_state,
    }

    # Apply test-friendly settings
    config.data_generation.num_curves = 20
    config.data_generation.num_points = 100
    config.visualization.show_plots = False
    config.visualization.block_plots = False
    config.data_processing.test_size = 0.2
    config.data_processing.random_state = 0

    try:
        yield
    finally:
        # Restore
        config.data_generation.num_curves = orig["num_curves"]
        config.data_generation.num_points = orig["num_points"]
        config.visualization.show_plots = orig["show_plots"]
        config.visualization.block_plots = orig["block_plots"]
        config.data_processing.test_size = orig["test_size"]
        config.data_processing.random_state = orig["random_state"]


