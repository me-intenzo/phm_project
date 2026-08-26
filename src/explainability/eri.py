"""
Explainability Reliability Index (ERI)

Novel trustworthiness metric for PHM explainability.

ERI combines:
  1. IG–Attention rank correlation  — Spearman ρ between sensor
     importance rankings from Integrated Gradients and temporal
     gradient relevance (model-intrinsic agreement).
  2. IG–SHAP rank correlation       — Spearman ρ between IG and
     SHAP sensor importance (post-hoc method agreement).
  3. Cross-method consensus score   — fraction of top-K sensors
     that appear in ALL three method rankings simultaneously.

Final ERI ∈ [0, 1]:
    ERI = 0.4 * ρ(IG, Attn) + 0.4 * ρ(IG, SHAP) + 0.2 * consensus_K

Interpretation
--------------
ERI > 0.8  → High trustworthiness: methods agree on what matters.
ERI > 0.6  → Moderate: partial agreement, review top sensors.
ERI ≤ 0.6  → Low: explanations diverge, model may be unreliable.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np


def _to_numpy(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D importance vectors."""
    n = len(a)
    if n < 2:
        return 0.0
    rank_a = np.argsort(np.argsort(-a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(-b)).astype(np.float64)
    d2 = ((rank_a - rank_b) ** 2).sum()
    rho = 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))
    return float(np.clip(rho, -1.0, 1.0))


def _sensor_importance_from_ig(ig_explanation: dict) -> np.ndarray:
    attrs = _to_numpy(ig_explanation["attributions"])
    if attrs.ndim == 3:
        imp = np.abs(attrs).mean(axis=(0, 1))
    else:
        imp = np.abs(attrs).mean(axis=0)
    total = imp.sum()
    return imp / total if total > 1e-12 else imp


def _sensor_importance_from_shap(shap_explanation: dict) -> np.ndarray:
    vals = _to_numpy(shap_explanation["sensor_shap_values"])
    imp = np.abs(vals).mean(axis=0)
    total = imp.sum()
    return imp / total if total > 1e-12 else imp


def _sensor_importance_from_attention(
    attention_explanation: dict,
    ig_explanation: dict,
) -> np.ndarray:
    """
    Derive sensor-level importance from temporal relevance by weighting
    IG sensor magnitudes with the mean temporal attention weight.

    This bridges temporal (timestep) attention to sensor space.
    """
    relevance = _to_numpy(attention_explanation["temporal_relevance"])  # (B, T)
    mean_temporal = relevance.mean(axis=0)  # (T,)

    attrs = _to_numpy(ig_explanation["attributions"])  # (B, T, F)
    if attrs.ndim == 3:
        # Weight each timestep's sensor attribution by its temporal relevance
        weighted = np.abs(attrs) * mean_temporal[np.newaxis, :, np.newaxis]
        imp = weighted.mean(axis=(0, 1))
    else:
        imp = np.abs(attrs).mean(axis=0)

    total = imp.sum()
    return imp / total if total > 1e-12 else imp


def _consensus_top_k(
    imp_ig: np.ndarray,
    imp_shap: np.ndarray,
    imp_attn: np.ndarray,
    k: int,
) -> float:
    """Fraction of top-K sensors shared across all three methods."""
    k = max(1, min(k, len(imp_ig)))
    top_ig   = set(np.argsort(-imp_ig)[:k])
    top_shap = set(np.argsort(-imp_shap)[:k])
    top_attn = set(np.argsort(-imp_attn)[:k])
    intersection = top_ig & top_shap & top_attn
    return len(intersection) / k


def compute_eri(
    ig_explanation: dict,
    shap_explanation: dict,
    attention_explanation: dict,
    top_k: int = 5,
) -> dict:
    """
    Compute the Explainability Reliability Index.

    Parameters
    ----------
    ig_explanation : dict
        Output of explain_with_integrated_gradients.
    shap_explanation : dict
        Output of explain_with_shap.
    attention_explanation : dict
        Output of explain_attention.
    top_k : int
        Number of top sensors for consensus scoring.

    Returns
    -------
    dict with keys:
        eri           — scalar ERI ∈ [0, 1]
        rho_ig_attn   — Spearman ρ(IG, Attention)
        rho_ig_shap   — Spearman ρ(IG, SHAP)
        consensus_k   — top-K consensus fraction
        top_k         — K used
        sensor_ranks  — dict of per-method sensor rankings (0-indexed)
        interpretation — human-readable trustworthiness label
    """
    imp_ig   = _sensor_importance_from_ig(ig_explanation)
    imp_shap = _sensor_importance_from_shap(shap_explanation)
    imp_attn = _sensor_importance_from_attention(attention_explanation, ig_explanation)

    rho_ig_attn = _spearman(imp_ig, imp_attn)
    rho_ig_shap = _spearman(imp_ig, imp_shap)

    # Normalize negative correlations to [0, 1] for ERI weighting
    norm_ig_attn = (rho_ig_attn + 1.0) / 2.0
    norm_ig_shap = (rho_ig_shap + 1.0) / 2.0

    consensus = _consensus_top_k(imp_ig, imp_shap, imp_attn, k=top_k)

    eri = 0.4 * norm_ig_attn + 0.4 * norm_ig_shap + 0.2 * consensus

    if eri > 0.8:
        interpretation = "HIGH — methods strongly agree, explanations are trustworthy"
    elif eri > 0.6:
        interpretation = "MODERATE — partial agreement, review top sensors manually"
    else:
        interpretation = "LOW — methods diverge, treat explanations with caution"

    return {
        "eri": float(eri),
        "rho_ig_attn": float(rho_ig_attn),
        "rho_ig_shap": float(rho_ig_shap),
        "consensus_k": float(consensus),
        "top_k": top_k,
        "sensor_importance": {
            "ig":        imp_ig.tolist(),
            "shap":      imp_shap.tolist(),
            "attention": imp_attn.tolist(),
        },
        "sensor_ranks": {
            "ig":        np.argsort(-imp_ig).tolist(),
            "shap":      np.argsort(-imp_shap).tolist(),
            "attention": np.argsort(-imp_attn).tolist(),
        },
        "interpretation": interpretation,
    }
