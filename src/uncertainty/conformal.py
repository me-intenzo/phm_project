"""
Split-Conformal Prediction
--------------------------

Baseline uncertainty quantification for RUL prediction.

Method:
    Absolute residual nonconformity

Calibration:
    Split-conformal prediction

Output:
    Symmetric prediction intervals around the GRU point prediction.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np


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


def prediction_interval(
    y_pred: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct symmetric prediction intervals.

    lower = prediction - q_hat
    upper = prediction + q_hat
    """

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains non-finite values."
        )

    if not np.isfinite(q_hat):
        raise ValueError(
            "q_hat must be finite."
        )

    if q_hat < 0:
        raise ValueError(
            "q_hat must be non-negative."
        )

    lower = y_pred - q_hat
    upper = y_pred + q_hat

    return lower, upper


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