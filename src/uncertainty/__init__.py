"""Uncertainty estimation package — EARA-Conformal."""

from .conformal import (
    adaptive_prediction_interval,
    assign_regimes,
    calibrate_regime_quantiles,
    eara_conformal_predict,
    fit_regime_detector,
    normalised_nonconformity,
    # legacy baseline
    absolute_nonconformity,
    conformal_quantile,
    prediction_interval,
)
from .calibration import (
    calibrate_eara_levels,
    calibrate_multiple_levels,
)
from .coverage import (
    empirical_coverage,
    evaluate_interval,
    mean_prediction_interval_width,
    normalised_interval_width,
    per_regime_coverage,
)

__all__ = [
    "adaptive_prediction_interval",
    "assign_regimes",
    "calibrate_eara_levels",
    "calibrate_multiple_levels",
    "calibrate_regime_quantiles",
    "eara_conformal_predict",
    "empirical_coverage",
    "evaluate_interval",
    "fit_regime_detector",
    "mean_prediction_interval_width",
    "normalised_interval_width",
    "normalised_nonconformity",
    "per_regime_coverage",
    # legacy
    "absolute_nonconformity",
    "conformal_quantile",
    "prediction_interval",
]
