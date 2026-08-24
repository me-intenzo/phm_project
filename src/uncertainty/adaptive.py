"""Solution 1: normalized, adaptive conformal prediction.

The baseline implementation remains in :mod:`conformal` and
:mod:`calibration`.  This module holds only the learned-scale variant.
"""

from __future__ import annotations

import numpy as np

from .conformal import absolute_nonconformity, conformal_quantile


def normalized_nonconformity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scales: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return absolute residuals normalized by positive predicted scales."""
    scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    residuals = absolute_nonconformity(y_true, y_pred)
    if scales.shape != residuals.shape:
        raise ValueError("scales must have the same shape as predictions.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("scales must be finite and strictly positive.")
    return residuals / (scales + eps)


def adaptive_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scales: np.ndarray,
    alpha: float,
) -> float:
    """Calibrate a normalized split-conformal residual quantile."""
    return conformal_quantile(
        normalized_nonconformity(y_true, y_pred, scales),
        alpha,
    )


def prediction_interval(
    y_pred: np.ndarray,
    q_hat: float,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``prediction ± q_hat * scale`` intervals."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    q_hat = float(q_hat)
    if y_pred.shape != scales.shape:
        raise ValueError("y_pred and scales must have the same shape.")
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(scales)):
        raise ValueError("predictions and scales must be finite.")
    if np.any(scales <= 0.0) or not np.isfinite(q_hat) or q_hat < 0.0:
        raise ValueError("scales must be positive and q_hat non-negative.")
    return y_pred - q_hat * scales, y_pred + q_hat * scales
