"""
Coverage and interval-width metrics for conformal prediction.

These metrics evaluate whether the prediction intervals produced
by the uncertainty layer achieve their intended coverage while
remaining reasonably narrow.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Calculate empirical prediction interval coverage.

    Coverage is the fraction of true targets that fall inside
    their corresponding prediction intervals.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.

    lower:
        Lower prediction bounds.

    upper:
        Upper prediction bounds.

    Returns
    -------
    float
        Empirical coverage in [0, 1].
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    lower = np.asarray(
        lower,
        dtype=np.float64,
    ).reshape(-1)

    upper = np.asarray(
        upper,
        dtype=np.float64,
    ).reshape(-1)

    _validate_interval_inputs(
        y_true,
        lower,
        upper,
    )

    covered = (
        (y_true >= lower)
        & (y_true <= upper)
    )

    return float(
        np.mean(covered)
    )


def mean_prediction_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Calculate Mean Prediction Interval Width (MPIW).

    MPIW = mean(upper - lower)
    """

    lower = np.asarray(
        lower,
        dtype=np.float64,
    ).reshape(-1)

    upper = np.asarray(
        upper,
        dtype=np.float64,
    ).reshape(-1)

    if lower.shape != upper.shape:
        raise ValueError(
            "lower and upper must have the same shape."
        )

    if lower.size == 0:
        raise ValueError(
            "At least one interval is required."
        )

    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
    ):
        raise ValueError(
            "Interval bounds must be finite."
        )

    if np.any(upper < lower):
        raise ValueError(
            "Upper bounds cannot be below lower bounds."
        )

    return float(
        np.mean(upper - lower)
    )


def coverage_error(
    empirical: float,
    nominal: float,
) -> float:
    """
    Calculate absolute coverage error.

    Coverage Error =
        |Empirical Coverage - Nominal Coverage|
    """

    if not 0.0 <= empirical <= 1.0:
        raise ValueError(
            "Empirical coverage must be between 0 and 1."
        )

    if not 0.0 <= nominal <= 1.0:
        raise ValueError(
            "Nominal coverage must be between 0 and 1."
        )

    return float(
        abs(empirical - nominal)
    )


def evaluate_interval(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
) -> dict[str, float]:
    """
    Evaluate a prediction interval.

    Returns
    -------
    dict
        Empirical coverage
        Nominal coverage
        Coverage error
        Mean prediction interval width
    """

    empirical = empirical_coverage(
        y_true=y_true,
        lower=lower,
        upper=upper,
    )

    mpiw = mean_prediction_interval_width(
        lower=lower,
        upper=upper,
    )

    error = coverage_error(
        empirical=empirical,
        nominal=nominal_coverage,
    )

    return {
        "nominal_coverage": float(
            nominal_coverage
        ),
        "empirical_coverage": empirical,
        "coverage_error": error,
        "mean_interval_width": mpiw,
    }


def _validate_interval_inputs(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """
    Validate interval inputs shared by coverage metrics.
    """

    if not (
        y_true.shape
        == lower.shape
        == upper.shape
    ):
        raise ValueError(
            "y_true, lower and upper must have "
            "the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "At least one prediction interval is required."
        )

    if not (
        np.all(np.isfinite(y_true))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
    ):
        raise ValueError(
            "y_true and interval bounds must be finite."
        )

    if np.any(upper < lower):
        raise ValueError(
            "Upper bounds cannot be below lower bounds."
        )