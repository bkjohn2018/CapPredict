import logging
import sys
import json
import time


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload)


def setup_logging(level: str = "INFO", json_mode: bool = False, propagate: bool = False) -> None:
    root = logging.getLogger()
    # Clear any existing handlers
    root.handlers.clear()
    # Configure level
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Stream handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    if json_mode:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)
    root.propagate = propagate


