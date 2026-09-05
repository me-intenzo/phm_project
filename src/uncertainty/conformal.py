"""
Engine-Aware Regime-Adaptive Conformal Prediction (EARA-Conformal)
------------------------------------------------------------------

Novel method that replaces the single global radius with:

  1. Regime detection   — k-means on per-window operating-condition
                          features assigns each window to a regime.
  2. Scale estimation   — the GRU scale head predicts a per-sample
                          heteroscedastic error magnitude s(x).
  3. Normalised scores  — r_i = |y_i - ŷ_i| / (s(x_i) + ε)
  4. Regime-conditional calibration — one q̂_k per regime, estimated
                          on engine-disjoint calibration windows.
  5. Adaptive intervals — C(x) = [ŷ - q̂_k · s(x), ŷ + q̂_k · s(x)]
                          clipped to [0, 125].

The engine-disjoint evaluation principle is preserved throughout.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

# RUL domain bounds (C-MAPSS piecewise-linear cap)
RUL_MIN: float = 0.0
RUL_MAX: float = 125.0
_EPS: float = 1e-6


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
    descriptors = X[:, -1, :]          # (N, F) — last timestep
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

    r_i = |y_i - ŷ_i| / (s_i + ε)
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scale  = np.asarray(scale,  dtype=np.float64).reshape(-1)

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

    k = ceil((n + 1)(1 - α))
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return float("inf")          # no calibration data → infinite radius
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

    C(x) = [ŷ - q̂_k · s(x), ŷ + q̂_k · s(x)]  clipped to [0, 125]

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
    y_pred  = np.asarray(y_pred,  dtype=np.float64).reshape(-1)
    scale   = np.asarray(scale,   dtype=np.float64).reshape(-1)
    regimes = np.asarray(regimes, dtype=np.int32).reshape(-1)

    radius = q_hats[regimes] * (scale + _EPS)
    lower  = np.clip(y_pred - radius, RUL_MIN, RUL_MAX)
    upper  = np.clip(y_pred + radius, RUL_MIN, RUL_MAX)
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
# Legacy baseline helpers (kept for backward compatibility)
# ------------------------------------------------------------------

def absolute_nonconformity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Absolute residual nonconformity scores (baseline)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return np.abs(y_true - y_pred)


def conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Global split-conformal quantile (baseline)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def prediction_interval(
    y_pred: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric global interval (baseline)."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return y_pred - q_hat, y_pred + q_hat
