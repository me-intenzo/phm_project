"""Agreement and model-faithfulness metrics for feature attributions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import torch
from scipy.integrate import trapezoid
from scipy.stats import pearsonr, spearmanr


def _as_vector(values: Any, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite attribution vector.")
    return vector


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(values), kind="stable")[:k]


def explanation_agreement(
    attributions: dict[str, np.ndarray],
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    """Compute top-k overlap, sign agreement, and full-vector Spearman rho."""
    if len(attributions) < 2:
        raise ValueError("At least two attribution methods are required.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    vectors = {name: _as_vector(values, name) for name, values in attributions.items()}
    lengths = {vector.size for vector in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("All attribution vectors must have the same length.")

    k = min(top_k, next(iter(lengths)))
    metrics: dict[str, dict[str, float]] = {}
    for first, second in combinations(vectors, 2):
        a, b = vectors[first], vectors[second]
        top_a, top_b = _top_indices(a, k), _top_indices(b, k)
        shared = np.intersect1d(top_a, top_b, assume_unique=True)
        rank = spearmanr(np.abs(a), np.abs(b)).statistic
        metrics[f"{first}__{second}"] = {
            "rank_agreement_top_k": float(shared.size / k),
            "sign_agreement_top_k": (
                float(np.mean(np.sign(a[shared]) == np.sign(b[shared])))
                if shared.size
                else float("nan")
            ),
            "rank_correlation_spearman": float(rank),
        }
    return metrics


def _predict(model: torch.nn.Module, x: torch.Tensor, target: str) -> float:
    prediction = model(x)
    if target == "rul":
        output = prediction[0]
    elif target == "hi":
        output = prediction[1]
    else:
        raise ValueError("target must be 'rul' or 'hi'.")
    return float(output.reshape(-1)[0].detach().cpu())


def feature_ablation_faithfulness(
    model: torch.nn.Module,
    sample: torch.Tensor,
    attribution: np.ndarray,
    target: str,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float]:
    """Measure prediction change when top-ranked sensor trajectories are masked.

    Zero is the feature baseline (the training pipeline standardizes inputs).
    Deletion-gap AUC integrates normalized absolute prediction change as top-ranked
    sensors are removed. Insertion AUC integrates fidelity to the full
    prediction as top-ranked sensors are added to the baseline input.
    """
    if sample.ndim != 2:
        raise ValueError("sample must have shape (timesteps, features).")
    scores = _as_vector(attribution, "attribution")
    if scores.size != sample.shape[1]:
        raise ValueError("Attribution length must match the sample feature count.")
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must contain values in (0, 1].")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("fractions must be strictly increasing.")

    model.eval()
    batch = sample.unsqueeze(0)
    baseline = torch.zeros_like(batch)
    with torch.no_grad():
        full_prediction = _predict(model, batch, target)
        baseline_prediction = _predict(model, baseline, target)

    scale = abs(full_prediction - baseline_prediction)
    has_prediction_contrast = scale > 1e-12
    denominator = scale if has_prediction_contrast else 1.0
    ranking = np.argsort(-np.abs(scores), kind="stable")
    removed_fractions = [0.0]
    deletion_gaps = [0.0]
    inserted_fractions = [0.0]
    insertion_fidelity = [
        float(
            np.clip(
                1.0 - abs(baseline_prediction - full_prediction) / denominator,
                0.0,
                1.0,
            )
        )
        if has_prediction_contrast
        else float("nan")
    ]
    prediction_gap_top_50 = 0.0

    with torch.no_grad():
        for fraction in fractions:
            count = max(1, int(np.ceil(fraction * scores.size)))
            selected = ranking[:count]

            deleted = batch.clone()
            deleted[:, :, selected] = 0
            deleted_prediction = _predict(model, deleted, target)
            gap = abs(full_prediction - deleted_prediction)
            removed_fractions.append(count / scores.size)
            deletion_gaps.append(
                float(np.clip(gap / denominator, 0.0, 1.0))
                if has_prediction_contrast
                else float("nan")
            )
            if fraction == 0.5:
                prediction_gap_top_50 = float(gap)

            inserted = baseline.clone()
            inserted[:, :, selected] = batch[:, :, selected]
            inserted_prediction = _predict(model, inserted, target)
            fidelity = 1.0 - abs(full_prediction - inserted_prediction) / denominator
            inserted_fractions.append(count / scores.size)
            insertion_fidelity.append(
                float(np.clip(fidelity, 0.0, 1.0))
                if has_prediction_contrast
                else float("nan")
            )

    return {
        "prediction_gap_top_50": prediction_gap_top_50,
        "deletion_gap_auc": (
            float(trapezoid(deletion_gaps, removed_fractions))
            if has_prediction_contrast
            else float("nan")
        ),
        "insertion_auc": (
            float(trapezoid(insertion_fidelity, inserted_fractions))
            if has_prediction_contrast
            else float("nan")
        ),
        "full_prediction": full_prediction,
        "baseline_prediction": baseline_prediction,
    }


def correlation_statistics(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float | int]:
    """Pearson and Spearman correlation coefficients and two-sided p-values."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Correlation inputs must have equal lengths.")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result: dict[str, float | int] = {"n": int(x.size)}
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            **result,
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        **result,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


__all__ = [
    "correlation_statistics",
    "explanation_agreement",
    "feature_ablation_faithfulness",
]
