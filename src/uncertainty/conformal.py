"""
Conformal prediction for RUL estimation.

This module contains three layers:

  1. Split-conformal baseline -- absolute residual nonconformity scores
     and a finite-sample global quantile.
  2. Engine-Aware Regime-Adaptive Conformal Prediction (EARA-Conformal) --
     regime detection, scale-normalised scores, and per-regime quantiles.
  3. Named conformal variants -- global, scale-adaptive, CQR-style, and
     engine-block intervals used by the uncertainty pipeline and tests.

It is also the **single source of truth** for the three quantities that
used to be re-declared in half a dozen places:

* :data:`SCALE_FLOOR` -- the floor applied to a predicted scale.  The
  *same* constant and the *same* bounding operator (``max``) are used
  when a scale is normalised into a calibration score and when it is
  multiplied into an interval radius, so calibration and prediction can
  never drift apart.
* :func:`conformal_quantile` -- the finite-sample conformal quantile.
  :func:`signed_conformal_quantile` is a thin wrapper around the same
  implementation for the signed (CQR) scores.
* :func:`clip_interval` -- the single interval builder: clip to the RUL
  domain *and* order the bounds.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from src.utils.config import DEFAULT_CONFIG

# RUL domain bounds (C-MAPSS piecewise-linear cap).
#
# The cap is declared in ``configs/<subset>.yaml`` as ``max_rul`` and is
# applied by ``scripts/preprocess.py`` when the labels are generated.
# ``scripts/train.py`` refuses to train when the stored dataset metadata
# disagrees with the config, so the two can never drift apart silently.
RUL_MIN: float = 0.0
RUL_MAX: float = float(DEFAULT_CONFIG.get("max_rul", 125.0))

#: Shared scale floor.
#:
#: A predicted scale is bounded from below with ``max(scale, SCALE_FLOOR)``
#: — never with an *addition*.  An additive floor changes the scale that
#: the calibration score was built from (the score divides by
#: ``scale + eps``) and therefore makes the calibration quantile and the
#: prediction radius disagree.  A multiplicative floor keeps the two
#: paths exactly aligned.
SCALE_FLOOR: float = 1e-3


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


def conformal_rank(n: int, alpha: float) -> int:
    """
    The finite-sample conformal rank ``k = ceil((n + 1)(1 - alpha))``.

    Parameters
    ----------
    n:
        Number of calibration scores.

    alpha:
        Miscoverage level.

    Returns
    -------
    int
        The one-based conformal rank.
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    return int(np.ceil((n + 1) * (1.0 - alpha)))


def _finite_sample_quantile(
    scores: np.ndarray,
    alpha: float,
    *,
    allow_infinite: bool = True,
    require_non_negative: bool = True,
) -> float:
    """The one implementation of the finite-sample conformal quantile.

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

    allow_infinite:
        When the finite-sample rank ``k = ceil((n + 1)(1 - alpha))`` exceeds
        the number of calibration scores ``n``, the calibration set is too
        small to certify the requested coverage and the mathematically
        correct quantile is ``+inf``.

        With ``allow_infinite=True`` (the default) ``+inf`` is returned so
        the caller can see that no finite interval is justified.  With
        ``allow_infinite=False`` a ``ValueError`` is raised instead.

        Earlier revisions of this module clamped the rank to ``n``, which
        silently returned the largest observed score and voided the
        coverage guarantee.  The clamp is gone.

    require_non_negative:
        Whether the scores have to be non-negative.  Signed (CQR) scores
        are perfectly valid nonconformity scores, so that check is
        skipped for them.

    Returns
    -------
    float
        Conformal quantile q_hat, possibly ``+inf``.
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

    if require_non_negative and np.any(scores < 0):
        raise ValueError(
            "Nonconformity scores must be non-negative."
        )

    sorted_scores = np.sort(scores)

    n = len(sorted_scores)

    rank = conformal_rank(n, alpha)

    if rank > n:
        if not allow_infinite:
            raise ValueError(
                f"The finite-sample conformal rank k={rank} exceeds the "
                f"number of calibration scores n={n}; the requested "
                f"coverage {1.0 - alpha:.2%} is not attainable at this "
                f"calibration size."
            )
        return float("inf")

    # Convert one-based conformal rank
    # to a valid zero-based NumPy index.

    return float(
        sorted_scores[rank - 1]
    )


def conformal_quantile(
    scores: np.ndarray,
    alpha: float,
    allow_infinite: bool = True,
) -> float:
    """
    Calculate the finite-sample split-conformal quantile.

    This is the canonical implementation; every other module in the
    project delegates to it.

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

    allow_infinite:
        See :func:`_finite_sample_quantile`.

    Returns
    -------
    float
        Conformal quantile q_hat, possibly ``+inf``.
    """

    return _finite_sample_quantile(
        scores,
        alpha,
        allow_infinite=allow_infinite,
        require_non_negative=True,
    )


def signed_conformal_quantile(
    scores: np.ndarray,
    alpha: float,
    allow_infinite: bool = True,
) -> float:
    """
    Finite-sample conformal quantile for **signed** nonconformity scores.

    Parameters
    ----------
    scores:
        Calibration scores that may be negative.

    alpha:
        Miscoverage level.

    allow_infinite:
        See :func:`_finite_sample_quantile`.

    Notes
    -----
    Conformalized quantile regression (CQR) uses the signed score

        ``max(q_lo(x) - y, y - q_hi(x))``

    which is **negative** whenever the true value falls strictly inside the
    predicted quantile interval.  Such scores are perfectly valid
    nonconformity scores, so the non-negativity check of
    :func:`conformal_quantile` must not be applied to them.

    The rank is never clamped: clamping would silently return the largest
    observed score and void the coverage guarantee.

    Returns
    -------
    float
        Conformal quantile q_hat, possibly ``+inf``.
    """

    return _finite_sample_quantile(
        scores,
        alpha,
        allow_infinite=allow_infinite,
        require_non_negative=False,
    )


# ------------------------------------------------------------------
# Scale handling
# ------------------------------------------------------------------

def floor_scale(
    scale: np.ndarray,
    floor: float = SCALE_FLOOR,
) -> np.ndarray:
    """Bound a predicted scale from below with the shared scale floor.

    The *same* operation is used by the calibration path (normalised
    nonconformity) and by the prediction path (interval radius), which is
    what keeps the calibrated quantile and the emitted interval aligned.
    """

    return np.maximum(
        np.asarray(scale, dtype=np.float64),
        float(floor),
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

    r_i = |y_i - y_hat_i| / max(s_i, SCALE_FLOOR)

    The scale floor is the shared :data:`SCALE_FLOOR` and the bounding
    operator is ``max`` — the very same one the interval radius uses.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)

    if not (y_true.shape == y_pred.shape == scale.shape):
        raise ValueError("y_true, y_pred and scale must have the same length.")

    return np.abs(y_true - y_pred) / floor_scale(scale)


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

    Returns ``+inf`` when the regime has no calibration data or when the
    finite-sample rank is not attainable, so the caller can fall back to the
    global quantile or widen the interval instead of silently clamping.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return float("inf")          # no calibration data -> infinite radius
    return conformal_quantile(scores, alpha)


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
# Calibration entry points
# ------------------------------------------------------------------

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
    scores = absolute_nonconformity(
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

    return results


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

    C(x) = [y_hat - q_k * max(s(x), SCALE_FLOOR), y_hat + q_k * max(s(x), SCALE_FLOOR)]
    clipped to the RUL domain and ordered.

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

    radius = np.asarray(q_hats, dtype=np.float64)[regimes] * floor_scale(scale)
    return clip_interval(y_pred - radius, y_pred + radius)


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
    non-negative scalar.  ``+inf`` is accepted: it is what
    :func:`conformal_quantile` returns when the finite-sample rank is not
    attainable, and the resulting interval is the whole RUL domain — i.e.
    "no finite interval is justified at this calibration size".
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    q_hat = float(q_hat)

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred must contain only finite values.")

    if np.isnan(q_hat) or q_hat < 0.0:
        raise ValueError(
            "q_hat must be a non-negative scalar (inf is allowed)."
        )

    lower, upper = y_pred - q_hat, y_pred + q_hat
    return clip_interval(lower, upper)


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


def clip_interval(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip an interval to the RUL domain and order its bounds.

    This is the **only** interval builder in the project.  Every
    conformal variant routes through it, so an interval can never be
    emitted with ``lower > upper`` or with a bound outside
    ``[RUL_MIN, RUL_MAX]``.
    """
    lower = np.clip(np.asarray(lower, dtype=np.float64), RUL_MIN, RUL_MAX)
    upper = np.clip(np.asarray(upper, dtype=np.float64), RUL_MIN, RUL_MAX)
    return np.minimum(lower, upper), np.maximum(lower, upper)


#: Backwards-compatible alias of :func:`clip_interval`.
_clip_interval = clip_interval


def scale_adaptive_interval(
    y_pred: np.ndarray,
    q_hat: float,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``prediction ± q_hat * max(scale, SCALE_FLOOR)`` intervals.

    The floored scale mirrors :func:`normalised_nonconformity`, so the
    calibrated quantile and the prediction radius use the same scale
    convention.
    """

    y_pred = np.asarray(y_pred, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    q_hat = float(q_hat)

    if y_pred.shape != scale.shape:
        raise ValueError("y_pred and scale must have the same shape.")

    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(scale)):
        raise ValueError("predictions and scales must be finite.")

    if np.isnan(q_hat) or q_hat < 0.0:
        raise ValueError("q_hat must be a non-negative scalar (inf is allowed).")

    radius = q_hat * floor_scale(scale)

    return clip_interval(y_pred - radius, y_pred + radius)


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
    lower, upper = clip_interval(
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
    radius = q_hat * floor_scale(scale_test)
    lower, upper = clip_interval(np.asarray(y_pred_test) - radius, np.asarray(y_pred_test) + radius)
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
    lower, upper = clip_interval(
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
    lower, upper = clip_interval(np.asarray(y_pred_test) - q_hat, np.asarray(y_pred_test) + q_hat)
    return {
        "q_hat": q_hat,
        "calibration_scores": scores,
        "engine_scores": engine_scores,
        "lower": lower,
        "upper": upper,
    }


__all__ = [
    "RUL_MAX",
    "RUL_MIN",
    "SCALE_FLOOR",
    "absolute_nonconformity",
    "adaptive_conformal_interval",
    "adaptive_prediction_interval",
    "assign_regimes",
    "calibrate_eara_levels",
    "calibrate_multiple_levels",
    "calibrate_regime_quantiles",
    "clip_interval",
    "conformal_interval",
    "conformal_quantile",
    "conformal_rank",
    "cqr_interval",
    "eara_conformal_predict",
    "engine_joint_interval",
    "fit_regime_detector",
    "floor_scale",
    "normalised_nonconformity",
    "prediction_interval",
    "regime_conformal_quantile",
    "scale_adaptive_interval",
    "scale_conformalized_interval",
    "signed_conformal_quantile",
]
