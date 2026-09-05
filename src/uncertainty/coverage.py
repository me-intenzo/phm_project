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


def normalised_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
    y_range: float = 125.0,
) -> float:
    """
    Prediction Interval Normalised Average Width (PINAW).

    PINAW = MPIW / y_range

    Lower is better; penalises unnecessarily wide intervals.
    """
    mpiw = mean_prediction_interval_width(lower=lower, upper=upper)
    return float(mpiw / y_range)


def per_regime_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    regimes: np.ndarray,
    n_regimes: int,
) -> dict[int, float]:
    """
    Empirical coverage broken down by operating regime.

    Returns
    -------
    dict  regime_id -> empirical_coverage
    """
    y_true  = np.asarray(y_true,   dtype=np.float64).reshape(-1)
    lower   = np.asarray(lower,    dtype=np.float64).reshape(-1)
    upper   = np.asarray(upper,    dtype=np.float64).reshape(-1)
    regimes = np.asarray(regimes,  dtype=np.int32).reshape(-1)

    result: dict[int, float] = {}
    for k in range(n_regimes):
        mask = regimes == k
        if mask.sum() == 0:
            result[k] = float("nan")
        else:
            result[k] = empirical_coverage(
                y_true=y_true[mask],
                lower=lower[mask],
                upper=upper[mask],
            )
    return result


def evaluate_interval(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
    regimes: np.ndarray | None = None,
    n_regimes: int | None = None,
) -> dict[str, object]:
    """
    Evaluate a prediction interval.

    Returns
    -------
    dict
        nominal_coverage, empirical_coverage, coverage_error,
        mean_interval_width, pinaw,
        regime_coverage (only when regimes and n_regimes are given)
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

    out: dict[str, object] = {
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": empirical,
        "coverage_error": error,
        "mean_interval_width": mpiw,
        "pinaw": normalised_interval_width(lower=lower, upper=upper),
    }

    if regimes is not None and n_regimes is not None:
        out["regime_coverage"] = per_regime_coverage(
            y_true=y_true,
            lower=lower,
            upper=upper,
            regimes=regimes,
            n_regimes=n_regimes,
        )

    return out


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