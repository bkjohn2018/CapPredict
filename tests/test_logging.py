from logging_setup import setup_logging


def test_setup_logging_runs():
    setup_logging("DEBUG", json_mode=False, propagate=False)


