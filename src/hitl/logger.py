"""Logging helpers for HITL interactions."""

from __future__ import annotations

import logging
from typing import Any


def log_feedback(logger: logging.Logger, feedback: dict[str, Any]) -> None:
    """Write a compact, auditable review event to the configured logger."""
    logger.info(
        "HITL review %s | status=%s | reviewer=%s | action=%s | final=%s",
        feedback.get("review_id", "unknown"),
        feedback.get("status", "unknown"),
        feedback.get("reviewer", "unknown"),
        feedback.get("review_action", "unknown"),
        feedback.get("final_action", "unknown"),
    )
