"""Centralised logging factory for PHM-XAI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "logs"

_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and optionally to *log_file*.

    Calling this multiple times with the same *name* returns the same logger
    without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_script_logger(category: str, stem: str) -> logging.Logger:
    """Convenience wrapper used by every script.

    Parameters
    ----------
    category:
        Sub-folder under ``outputs/logs/`` (e.g. ``"training"``).
    stem:
        Log-file stem, e.g. ``"train_FD001_lstm"`` → ``train_FD001_lstm.log``.
    """
    log_file = LOG_ROOT / category / f"{stem}.log"
    return get_logger(f"phm.{category}.{stem}", log_file)
