"""
Calibration utilities for EARA-Conformal uncertainty estimation.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np

from .conformal import (
    absolute_nonconformity,
    calibrate_regime_quantiles,
    conformal_quantile,
    normalised_nonconformity,
)


# ------------------------------------------------------------------
# EARA-Conformal calibration
# ------------------------------------------------------------------

def calibrate_eara_levels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
    regimes: np.ndarray,
    n_regimes: int,
    coverage_levels: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[float, dict[str, object]]:
    """
    Calibrate EARA-Conformal quantiles for multiple coverage levels.

    Returns
    -------
    dict  coverage -> {alpha, q_hats (shape n_regimes,)}
    """
    scores = normalised_nonconformity(
        y_true=y_true,
        y_pred=y_pred,
        scale=scale,
    )

    results: dict[float, dict[str, object]] = {}
    for coverage in coverage_levels:
        alpha = 1.0 - coverage
        q_hats = calibrate_regime_quantiles(
            scores=scores,
            regimes=regimes,
            alpha=alpha,
            n_regimes=n_regimes,
            min_regime_size=30,
        )
        results[float(coverage)] = {
            "alpha": float(alpha),
            "q_hats": q_hats,
        }
    return results


# ------------------------------------------------------------------
# Legacy baseline calibration (kept for backward compatibility)
# ------------------------------------------------------------------

def calculate_calibration_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    return absolute_nonconformity(y_true=y_true, y_pred=y_pred)


def calibrate_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    scores = calculate_calibration_scores(y_true, y_pred)
    return conformal_quantile(scores=scores, alpha=alpha)


def calibrate_multiple_levels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage_levels: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[float, dict[str, float]]:
    scores = calculate_calibration_scores(y_true, y_pred)
    results: dict[float, dict[str, float]] = {}
    for coverage in coverage_levels:
        alpha = 1.0 - coverage
        results[float(coverage)] = {
            "alpha": float(alpha),
            "q_hat": conformal_quantile(scores=scores, alpha=alpha),
        }
    return results
