<<<<<<< HEAD
"""
Calibration utilities for conformal RUL uncertainty estimation.

This module handles calibration-set predictions and residual
statistics. Model loading and dataset handling belong in the
uncertainty pipeline script.

author: me-intenzo 
"""

from __future__ import annotations

import numpy as np

from .conformal import (
    absolute_nonconformity,
    conformal_quantile,
)


=======
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

>>>>>>> 232c8a9acfc44f8cf76034ecb550fec335210471
def calculate_calibration_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
<<<<<<< HEAD
    """Calculate absolute residual nonconformity scores."""
    return absolute_nonconformity(
        y_true=y_true,
        y_pred=y_pred,
    )


def calibrate_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """
    Calculate the conformal quantile from calibration data.

    Parameters
    ----------
    y_true:
        Calibration-set ground-truth RUL.

    y_pred:
        Calibration-set GRU predictions.

    alpha:
        Miscoverage level.

        0.20 -> 80% interval
        0.10 -> 90% interval
        0.05 -> 95% interval

    Returns
    -------
    float
        Calibrated q_hat.
    """

    scores = calculate_calibration_scores(
        y_true=y_true,
        y_pred=y_pred,
    )

    return conformal_quantile(
        scores=scores,
        alpha=alpha,
    )

def calibrate_multiple_levels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage_levels: tuple[float, ...] = (
        0.80,
        0.90,
        0.95,
    ),
) -> dict[float, dict[str, float]]:
    """
    Calibrate conformal quantiles for multiple coverage levels.

    Parameters
    ----------
    y_true:
        Calibration-set ground-truth RUL.

    y_pred:
        Calibration-set GRU predictions.

    coverage_levels:
        Desired nominal coverage levels.

    Returns
    -------
    dict
        Mapping:

            coverage level
                -> alpha
                -> q_hat
    """
    scores = calculate_calibration_scores(
        y_true=y_true,
        y_pred=y_pred,
    )

    results: dict[float, dict[str, float]] = {}

    for coverage in coverage_levels:

        if not 0.0 < coverage < 1.0:
            raise ValueError(
                "Coverage levels must be strictly between 0 and 1."
            )

        alpha = 1.0 - coverage

        q_hat = conformal_quantile(
            scores=scores,
            alpha=alpha,
        )

        results[float(coverage)] = {
            "alpha": float(alpha),
            "q_hat": float(q_hat),
        }

=======
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
>>>>>>> 232c8a9acfc44f8cf76034ecb550fec335210471
    return results
