"""
Conformal prediction for RUL estimation.

This module contains three layers:

  1. Split-conformal baseline -- absolute residual nonconformity scores
     and a finite-sample global quantile.
  2. Engine-Aware Regime-Adaptive Conformal Prediction (EARA-Conformal) --
     regime detection, scale-normalised scores, and per-regime quantiles.
  3. Named conformal variants -- global, scale-adaptive, CQR-style, and
     engine-block intervals used by the uncertainty pipeline and tests.

Calibration and prediction share the same scale floor (``_EPS``) so that
normalised scores and adaptive radii remain numerically consistent.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

# RUL domain bounds (C-MAPSS piecewise-linear cap)
RUL_MIN: float = 0.0
RUL_MAX: float = 125.0
# Shared scale floor. The same constant is applied when a predicted scale
# is used in a normalised score (calibration) or an interval radius
# (prediction), keeping the two aligned.
_EPS: float = 1e-6


# ------------------------------------------------------------------
# Baseline nonconformity scores
# ------------------------------------------------------------------

def absolute_nonconformity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate absolute residual nonconformity scores.

    score_i = |y_i - y_hat_i|
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains non-finite values."
        )

    return np.abs(y_true - y_pred)


def conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Calculate the finite-sample split-conformal quantile.

    Parameters
    ----------
    scores:
        Calibration nonconformity scores.

    alpha:
        Miscoverage level.

        Examples:
            0.20 -> 80% interval
            0.10 -> 90% interval
            0.05 -> 95% interval

    Returns
    -------
    float
        Conformal quantile q_hat.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if scores.size == 0:
        raise ValueError(
            "At least one calibration score is required."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be strictly between 0 and 1."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Calibration scores must be finite."
        )

    if np.any(scores < 0):
        raise ValueError(
            "Nonconformity scores must be non-negative."
        )

    sorted_scores = np.sort(scores)

    # Finite-sample conformal rank:
    #
    # k = ceil((n + 1) * (1 - alpha))
    #
    # where n is the number of calibration samples.

    n = len(sorted_scores)

    rank = int(
        np.ceil(
            (n + 1) * (1.0 - alpha)
        )
    )

    # Convert one-based conformal rank
    # to a valid zero-based NumPy index.

    rank = min(
        max(rank, 1),
        n,
    )

    return float(
        sorted_scores[rank - 1]
    )


# ------------------------------------------------------------------
# Regime detection
# ------------------------------------------------------------------

def fit_regime_detector(
    X: np.ndarray,
    n_regimes: int = 7,  # optimal per regime_sweep.py
    random_state: int = 42,
) -> KMeans:
    """
    Fit a k-means regime detector on operating-condition features.

    Parameters
    ----------
    X:
        Windows of shape (N, T, F).  The last timestep's features
        are used as the operating-condition descriptor.
    n_regimes:
        Number of operating regimes (clusters).
    random_state:
        Reproducibility seed.

    Returns
    -------
    KMeans
        Fitted detector.
    """
    descriptors = X[:, -1, :]          # (N, F) - last timestep
    km = KMeans(
        n_clusters=n_regimes,
        random_state=random_state,
        n_init=10,
    )
    km.fit(descriptors)
    return km


def assign_regimes(
    km: KMeans,
    X: np.ndarray,
) -> np.ndarray:
    """
    Assign regime labels to windows.

    Returns
    -------
    np.ndarray of int, shape (N,)
    """
    return km.predict(X[:, -1, :]).astype(np.int32)


# ------------------------------------------------------------------
# Normalised nonconformity scores
# ------------------------------------------------------------------

def normalised_nonconformity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Compute normalised nonconformity scores.

    r_i = |y_i - y_hat_i| / (s_i + eps)
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)

    if not (y_true.shape == y_pred.shape == scale.shape):
        raise ValueError("y_true, y_pred and scale must have the same length.")

    return np.abs(y_true - y_pred) / (scale + _EPS)


# ------------------------------------------------------------------
# Per-regime conformal quantile
# ------------------------------------------------------------------

def regime_conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Finite-sample conformal quantile for a single regime.

    k = ceil((n + 1)(1 - alpha))
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return float("inf")          # no calibration data -> infinite radius
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def calibrate_regime_quantiles(
    scores: np.ndarray,
    regimes: np.ndarray,
    alpha: float,
    n_regimes: int,
    min_regime_size: int = 30,
) -> np.ndarray:
    """
    Calibrate one q_hat per regime.

    Regimes with fewer than min_regime_size calibration samples
    fall back to the global quantile to avoid unreliable estimates.

    Returns
    -------
    np.ndarray of float, shape (n_regimes,)
    """
    global_q = regime_conformal_quantile(scores, alpha)
    q_hats = np.empty(n_regimes, dtype=np.float64)
    for k in range(n_regimes):
        mask = regimes == k
        if mask.sum() >= min_regime_size:
            q_hats[k] = regime_conformal_quantile(scores[mask], alpha)
        else:
            q_hats[k] = global_q
    return q_hats


# ------------------------------------------------------------------
# Adaptive prediction intervals
# ------------------------------------------------------------------

def adaptive_prediction_interval(
    y_pred: np.ndarray,
    scale: np.ndarray,
    q_hats: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct per-sample adaptive intervals.

    C(x) = [y_hat - q_k * s(x), y_hat + q_k * s(x)]  clipped to [0, 125]

    Parameters
    ----------
    y_pred:
        GRU RUL predictions, shape (N,).
    scale:
        GRU scale predictions, shape (N,).
    q_hats:
        Per-regime quantiles, shape (n_regimes,).
    regimes:
        Regime label for each test sample, shape (N,).

    Returns
    -------
    lower, upper : np.ndarray, shape (N,)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    regimes = np.asarray(regimes, dtype=np.int32).reshape(-1)

    radius = q_hats[regimes] * (scale + _EPS)
    lower = np.clip(y_pred - radius, RUL_MIN, RUL_MAX)
    upper = np.clip(y_pred + radius, RUL_MIN, RUL_MAX)
    return lower, upper


# ------------------------------------------------------------------
# Full EARA-Conformal pipeline (single alpha)
# ------------------------------------------------------------------

def eara_conformal_predict(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    regimes_cal: np.ndarray,
    y_pred_test: np.ndarray,
    scale_test: np.ndarray,
    regimes_test: np.ndarray,
    alpha: float,
    n_regimes: int,
) -> dict[str, object]:
    """
    End-to-end EARA-Conformal calibration and interval generation.

    Calibration targets are NEVER mixed with test targets.

    Returns
    -------
    dict with keys:
        alpha, nominal_coverage, q_hats, lower, prediction,
        upper, calibration_scores, regimes_test
    """
    cal_scores = normalised_nonconformity(
        y_true=y_true_cal,
        y_pred=y_pred_cal,
        scale=scale_cal,
    )

    q_hats = calibrate_regime_quantiles(
        scores=cal_scores,
        regimes=regimes_cal,
        alpha=alpha,
        n_regimes=n_regimes,
        min_regime_size=30,
    )

    lower, upper = adaptive_prediction_interval(
        y_pred=y_pred_test,
        scale=scale_test,
        q_hats=q_hats,
        regimes=regimes_test,
    )

    return {
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "q_hats": q_hats,
        "lower": lower,
        "prediction": np.asarray(y_pred_test, dtype=np.float64),
        "upper": upper,
        "calibration_scores": cal_scores,
        "regimes_test": regimes_test,
    }


# ------------------------------------------------------------------
# Symmetric prediction intervals
# ------------------------------------------------------------------

def prediction_interval(
    y_pred: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric split-conformal prediction intervals.

    ``q_hat`` is the calibration residual quantile, so it must be a
    finite, non-negative scalar.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    q_hat = float(q_hat)

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred must contain only finite values.")

    if not np.isfinite(q_hat) or q_hat < 0.0:
        raise ValueError(
            "q_hat must be a finite, non-negative scalar."
        )

    return y_pred - q_hat, y_pred + q_hat


def conformal_predict(
    y_true_calibration: np.ndarray,
    y_pred_calibration: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, object]:
    """
    Calibrate conformal prediction intervals and apply them
    to test predictions.

    IMPORTANT
    ---------
    y_true_calibration and y_pred_calibration must come from
    the calibration/validation set.

    Final test targets must NOT be used to calculate q_hat.
    """

    calibration_scores = absolute_nonconformity(
        y_true=y_true_calibration,
        y_pred=y_pred_calibration,
    )

    q_hat = conformal_quantile(
        scores=calibration_scores,
        alpha=alpha,
    )

    lower, upper = prediction_interval(
        y_pred=y_pred_test,
        q_hat=q_hat,
    )

    return {
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "q_hat": float(q_hat),
        "lower": lower,
        "prediction": np.asarray(
            y_pred_test,
            dtype=np.float64,
        ),
        "upper": upper,
        "calibration_scores": calibration_scores,
    }


# ------------------------------------------------------------------
# Named conformal variants
# ------------------------------------------------------------------

def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")


def _clip_interval(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.clip(np.asarray(lower, dtype=np.float64), RUL_MIN, RUL_MAX)
    upper = np.clip(np.asarray(upper, dtype=np.float64), RUL_MIN, RUL_MAX)
    return np.minimum(lower, upper), np.maximum(lower, upper)


def conformal_interval(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    """Global split-conformal interval using absolute residuals."""
    _validate_alpha(alpha)
    scores = absolute_nonconformity(y_true_cal, y_pred_cal)
    q_hat = conformal_quantile(scores, alpha)
    lower, upper = _clip_interval(
        np.asarray(y_pred_test) - q_hat,
        np.asarray(y_pred_test) + q_hat,
    )
    return {"q_hat": q_hat, "calibration_scores": scores, "lower": lower, "upper": upper}


def adaptive_conformal_interval(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    y_pred_test: np.ndarray,
    scale_test: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    """Global scale-adaptive split-conformal interval."""
    _validate_alpha(alpha)
    scores = normalised_nonconformity(y_true_cal, y_pred_cal, scale_cal)
    q_hat = conformal_quantile(scores, alpha)
    radius = q_hat * (np.asarray(scale_test, dtype=np.float64) + _EPS)
    lower, upper = _clip_interval(np.asarray(y_pred_test) - radius, np.asarray(y_pred_test) + radius)
    return {"q_hat": q_hat, "calibration_scores": scores, "lower": lower, "upper": upper}


def scale_conformalized_interval(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    y_pred_test: np.ndarray,
    scale_test: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    """Scale-based conformalized interval (CQR-style).

    Current checkpoints do not expose separately trained quantile heads, so
    the available conditional bounds are prediction +/- scale. The returned
    interval is still calibrated with the CQR one-sided violation score.
    """
    _validate_alpha(alpha)
    cal_lower = np.asarray(y_pred_cal) - np.asarray(scale_cal)
    cal_upper = np.asarray(y_pred_cal) + np.asarray(scale_cal)
    scores = np.maximum(cal_lower - np.asarray(y_true_cal), np.asarray(y_true_cal) - cal_upper)
    scores = np.maximum(scores, 0.0)
    q_hat = conformal_quantile(scores, alpha)
    lower, upper = _clip_interval(
        np.asarray(y_pred_test) - np.asarray(scale_test) - q_hat,
        np.asarray(y_pred_test) + np.asarray(scale_test) + q_hat,
    )
    return {"q_hat": q_hat, "calibration_scores": scores, "lower": lower, "upper": upper}


def cqr_interval(*args, **kwargs) -> dict[str, object]:
    """Backward-compatible alias for :func:`scale_conformalized_interval`.

    This is not conventional CQR because the checkpoint has no learned
    lower and upper quantile heads.
    """
    return scale_conformalized_interval(*args, **kwargs)


def engine_joint_interval(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    engine_ids_cal: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    """Joint conformal interval using one maximum score per calibration engine."""
    _validate_alpha(alpha)
    scores = absolute_nonconformity(y_true_cal, y_pred_cal)
    engine_ids_cal = np.asarray(engine_ids_cal).reshape(-1)
    if scores.shape != engine_ids_cal.shape:
        raise ValueError("Calibration scores and engine_ids_cal must have the same length.")
    engine_scores = np.asarray(
        [np.max(scores[engine_ids_cal == engine_id]) for engine_id in np.unique(engine_ids_cal)],
        dtype=np.float64,
    )
    q_hat = conformal_quantile(engine_scores, alpha)
    lower, upper = _clip_interval(np.asarray(y_pred_test) - q_hat, np.asarray(y_pred_test) + q_hat)
    return {
        "q_hat": q_hat,
        "calibration_scores": scores,
        "engine_scores": engine_scores,
        "lower": lower,
        "upper": upper,
    }