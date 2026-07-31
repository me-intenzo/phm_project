"""Logging helpers."""

import logging
from datetime import date


def get_logger(name: str) -> logging.Logger:
    """Create and return a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def get_progress_output() -> str:
    """Return the progress output in the requested format."""
    today = date.today().strftime("%Y-%m-%d")
    return f"{today}\n\nLoading dataset...\n\nScaling...\n\nDone."
