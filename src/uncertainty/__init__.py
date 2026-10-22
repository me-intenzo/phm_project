"""Uncertainty estimation package — EARA-Conformal."""

from .conformal import (
    # legacy baseline
    absolute_nonconformity,
    adaptive_conformal_interval,
    adaptive_prediction_interval,
    assign_regimes,
    calibrate_eara_levels,
    calibrate_multiple_levels,
    calibrate_regime_quantiles,
    conformal_interval,
    conformal_quantile,
    cqr_interval,
    eara_conformal_predict,
    engine_joint_interval,
    fit_regime_detector,
    normalised_nonconformity,
    prediction_interval,
    scale_conformalized_interval,
    signed_conformal_quantile,
)
from .coverage import (
    empirical_coverage,
    engine_level_coverage,
    evaluate_interval,
    mean_prediction_interval_width,
    normalised_interval_width,
    per_regime_coverage,
    winkler_score,
)

__all__ = [
    "adaptive_prediction_interval",
    "adaptive_conformal_interval",
    "assign_regimes",
    "calibrate_eara_levels",
    "calibrate_multiple_levels",
    "calibrate_regime_quantiles",
    "eara_conformal_predict",
    "empirical_coverage",
    "engine_level_coverage",
    "evaluate_interval",
    "fit_regime_detector",
    "mean_prediction_interval_width",
    "normalised_interval_width",
    "normalised_nonconformity",
    "per_regime_coverage",
    "winkler_score",
    # legacy
    "absolute_nonconformity",
    "conformal_quantile",
    "conformal_interval",
    "cqr_interval",
    "scale_conformalized_interval",
    "engine_joint_interval",
    "prediction_interval",
    "signed_conformal_quantile",
]
