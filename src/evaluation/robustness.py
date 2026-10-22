"""
src/evaluation/robustness.py

Robustness evaluation for the PHM-XAI pipeline.

Two robustness axes
-------------------
1. Input robustness  — MAE degradation under Gaussian sensor noise and
                       random sensor dropout (wraps metrics.py functions).
2. Cross-subset consistency — coefficient of variation of RUL MAE across
                              FD001–FD004 for a given model; low CV means
                              the model generalises across operating conditions.

These are called from scripts/evaluate.py (--robustness flag) and from
demo.py after the full pipeline has run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import (
    predictive_accuracy,
    robustness_under_dropout,
    robustness_under_noise,
)


def evaluate_robustness(
    model,
    X_test: np.ndarray,
    y_rul_true: np.ndarray,
    device: str = "cpu",
    noise_levels: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
    dropout_fractions: tuple[float, ...] = (0.10, 0.25, 0.50),
    n_trials: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Full input-robustness evaluation for one (model, subset) combination.

    Parameters
    ----------
    model         : trained PyTorch model in eval mode
    X_test        : (n_engines, window_size, features) float32
    y_rul_true    : (n_engines,) ground-truth RUL
    device        : "cuda" or "cpu"
    noise_levels  : Gaussian noise σ values to sweep
    dropout_fractions : sensor dropout fractions to sweep
    n_trials      : realisations per level
    seed          : RNG seed

    Returns
    -------
    dict with ``noise`` and ``dropout`` sub-dicts from metrics.py, plus
    a ``robustness_score`` in [0, 1] (1 = perfectly robust).
    """
    noise_result   = robustness_under_noise(
        model, X_test, y_rul_true,
        noise_levels=noise_levels, n_trials=n_trials, seed=seed, device=device,
    )
    dropout_result = robustness_under_dropout(
        model, X_test, y_rul_true,
        dropout_fractions=dropout_fractions, n_trials=n_trials, seed=seed, device=device,
    )

    # Robustness score: 1 - mean relative degradation across all conditions
    rel_degs = []
    for v in noise_result["noise_levels"].values():
        rel_degs.append(v["relative_degradation"])
    for v in dropout_result["dropout_fractions"].values():
        rel_degs.append(v["relative_degradation"])

    robustness_score = float(max(0.0, 1.0 - np.mean(rel_degs))) if rel_degs else float("nan")

    return {
        "noise":            noise_result,
        "dropout":          dropout_result,
        "robustness_score": robustness_score,
    }


def cross_subset_consistency(
    results_dir: str | Path,
    model: str,
    subsets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Cross-subset consistency check.

    Loads prediction artifacts for each subset and computes the
    coefficient of variation (CV = std/mean) of RUL MAE across subsets.
    Low CV (< 0.20) indicates the model generalises well across
    operating conditions.

    Parameters
    ----------
    results_dir : ``outputs/results/``
    model       : model name, e.g. "gru"
    subsets     : defaults to all four C-MAPSS subsets

    Returns
    -------
    dict with per-subset MAE, mean, std, CV, and a consistency_band label.
    """
    if subsets is None:
        subsets = ["FD001", "FD002", "FD003", "FD004"]

    results_dir = Path(results_dir)
    maes: dict[str, float] = {}

    for subset in subsets:
        pred_path = results_dir / f"{subset}_{model}_predictions.npz"
        if not pred_path.exists():
            maes[subset] = float("nan")
            continue
        pred = np.load(pred_path)
        acc  = predictive_accuracy(
            pred["y_rul_true"], pred["y_rul_pred"],
            pred["y_hi_true"],  pred["y_hi_pred"],
        )
        maes[subset] = acc["rul"]["MAE"]

    valid = [v for v in maes.values() if not np.isnan(v)]
    mean_mae = float(np.mean(valid)) if valid else float("nan")
    std_mae  = float(np.std(valid))  if valid else float("nan")
    cv       = std_mae / mean_mae    if mean_mae > 0 else float("nan")

    return {
        "per_subset_rul_mae": maes,
        "mean_rul_mae":       mean_mae,
        "std_rul_mae":        std_mae,
        "cv_rul_mae":         cv,
        "consistency_band":   (
            "HIGH"     if cv < 0.20 else
            "MODERATE" if cv < 0.40 else
            "LOW"
        ),
    }
