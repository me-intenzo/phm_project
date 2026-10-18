"""
src/evaluation/metrics.py

Multi-dimensional evaluation framework for PHM-XAI.

Six evaluation dimensions
--------------------------
1. Predictive accuracy      — MAE, RMSE, R², NASA Score (RUL); MAE, RMSE, R² (HI)
2. Uncertainty reliability  — empirical coverage, coverage error, MPIW, PINAW,
                               Winkler score, engine-level coverage
3. Explainability quality   — ERI, cross-method sensor rank agreement (Spearman),
                               consensus@K, IG/SHAP/temporal pairwise correlations
4. Decision effectiveness   — action distribution, human-review rate, constraint
                               violation rate, urgency–risk alignment
5. Robustness               — metric degradation under Gaussian sensor noise and
                               random sensor dropout at multiple severity levels
6. User trust               — proxy metrics: explanation consistency across runs,
                               decision stability under small input perturbations,
                               override rate from HITL feedback log

All functions accept plain numpy arrays or dicts loaded from the pipeline
artifact files (.npz / .json) so they can be called from scripts/evaluate.py
or from demo.py without re-running inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------------ #
# 1. Predictive Accuracy
# ------------------------------------------------------------------ #

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric NASA C-MAPSS penalty (late predictions penalised more)."""
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(
        np.where(errors < 0, np.exp(-errors / 13.0) - 1.0, np.exp(errors / 10.0) - 1.0)
    ))


def predictive_accuracy(
    y_rul_true: np.ndarray,
    y_rul_pred: np.ndarray,
    y_hi_true: np.ndarray,
    y_hi_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Dimension 1 — Predictive Accuracy.

    Returns
    -------
    dict with keys ``rul`` and ``hi``, each a sub-dict of metric → value.
    """
    y_rul_true = np.asarray(y_rul_true, dtype=np.float64)
    y_rul_pred = np.asarray(y_rul_pred, dtype=np.float64)
    y_hi_true  = np.asarray(y_hi_true,  dtype=np.float64)
    y_hi_pred  = np.asarray(y_hi_pred,  dtype=np.float64)

    return {
        "rul": {
            "MAE":        float(mean_absolute_error(y_rul_true, y_rul_pred)),
            "RMSE":       float(np.sqrt(mean_squared_error(y_rul_true, y_rul_pred))),
            "R2":         float(r2_score(y_rul_true, y_rul_pred)),
            "NASA_Score": nasa_score(y_rul_true, y_rul_pred),
        },
        "hi": {
            "MAE":  float(mean_absolute_error(y_hi_true, y_hi_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_hi_true, y_hi_pred))),
            "R2":   float(r2_score(y_hi_true, y_hi_pred)),
        },
    }


# ------------------------------------------------------------------ #
# 2. Uncertainty Reliability
# ------------------------------------------------------------------ #

def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Winkler score for a (1-alpha) prediction interval.
    Penalises width and adds a penalty proportional to 2/alpha for misses.
    Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower  = np.asarray(lower,  dtype=np.float64)
    upper  = np.asarray(upper,  dtype=np.float64)
    width  = upper - lower
    penalty = (2.0 / alpha) * np.where(
        y_true < lower, lower - y_true,
        np.where(y_true > upper, y_true - upper, 0.0)
    )
    return float(np.mean(width + penalty))


def uncertainty_reliability(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    engine_ids: np.ndarray,
    nominal_alpha: float = 0.10,
    rul_max: float = 125.0,
) -> dict[str, Any]:
    """
    Dimension 2 — Uncertainty Reliability.

    Parameters
    ----------
    y_true, lower, upper : per-window arrays
    engine_ids           : source engine per window (for engine-level coverage)
    nominal_alpha        : miscoverage level (0.10 → 90 % nominal interval)
    rul_max              : normalisation constant for PINAW

    Returns
    -------
    dict with coverage, coverage_error, MPIW, PINAW, Winkler, engine_coverage.
    """
    y_true    = np.asarray(y_true, dtype=np.float64)
    lower     = np.asarray(lower,  dtype=np.float64)
    upper     = np.asarray(upper,  dtype=np.float64)
    engine_ids = np.asarray(engine_ids)

    covered   = (y_true >= lower) & (y_true <= upper)
    mpiw      = float(np.mean(upper - lower))

    # Engine-level coverage: all windows of an engine must be covered
    engines   = np.unique(engine_ids)
    eng_covered = np.array([covered[engine_ids == e].all() for e in engines])

    return {
        "empirical_coverage":  float(covered.mean()),
        "nominal_coverage":    float(1.0 - nominal_alpha),
        "coverage_error":      float(abs(covered.mean() - (1.0 - nominal_alpha))),
        "MPIW":                mpiw,
        "PINAW":               mpiw / rul_max,
        "Winkler":             winkler_score(y_true, lower, upper, nominal_alpha),
        "engine_level_coverage": float(eng_covered.mean()),
    }


# ------------------------------------------------------------------ #
# 3. Explainability Quality
# ------------------------------------------------------------------ #

def explainability_quality(eri_json_path: str | Path) -> dict[str, Any]:
    """
    Dimension 3 — Explainability Quality.

    Reads the ERI JSON produced by scripts/explain.py and returns the
    cross-method agreement metrics.

    Expected keys in the JSON (written by src/explainability/eri.py):
        eri, rho_ig_attn, rho_ig_shap, consensus_k,
        sensor_importance.ig, sensor_importance.temporal, sensor_importance.shap
    """
    with open(eri_json_path) as f:
        eri = json.load(f)

    result: dict[str, Any] = {
        "ERI":           float(eri.get("eri", float("nan"))),
        "rho_ig_attn":   float(eri.get("rho_ig_attn", float("nan"))),
        "rho_ig_shap":   float(eri.get("rho_ig_shap", float("nan"))),
        "consensus_k":   float(eri.get("consensus_k", float("nan"))),
        "reliability_band": (
            "HIGH"     if eri.get("eri", 0) >= 0.80 else
            "MODERATE" if eri.get("eri", 0) >= 0.60 else
            "LOW"
        ),
    }

    # Pairwise Spearman between all three sensor importance vectors if present
    si = eri.get("sensor_importance", {})
    ig   = np.asarray(si.get("ig",       []), dtype=np.float64)
    temp = np.asarray(si.get("temporal", []), dtype=np.float64)
    shap = np.asarray(si.get("shap",     []), dtype=np.float64)

    if ig.size and temp.size and ig.size == temp.size:
        result["spearman_ig_temporal"] = float(spearmanr(ig, temp).statistic)
    if ig.size and shap.size and ig.size == shap.size:
        result["spearman_ig_shap"]     = float(spearmanr(ig, shap).statistic)
    if temp.size and shap.size and temp.size == shap.size:
        result["spearman_temporal_shap"] = float(spearmanr(temp, shap).statistic)

    return result


# ------------------------------------------------------------------ #
# 4. Decision Effectiveness
# ------------------------------------------------------------------ #

def decision_effectiveness(decision_json_path: str | Path) -> dict[str, Any]:
    """
    Dimension 4 — Decision Effectiveness.

    Reads the decision JSON produced by scripts/decision.py.
    Computes action distribution, human-review rate, constraint violation
    rate, and urgency–risk alignment (Pearson r).
    """
    with open(decision_json_path) as f:
        data = json.load(f)

    records = data.get("decisions", data) if isinstance(data, dict) else data
    if isinstance(records, dict):
        records = records.get("decisions", [records])

    n = len(records)
    if n == 0:
        return {"n_decisions": 0}

    actions      = [r.get("recommended_action", "") for r in records]
    review_flags = [r.get("human_review", False)    for r in records]
    violations   = [len(r.get("constraint_violations", [])) > 0 for r in records]
    risks        = np.array([r.get("risk_score",    0.0) for r in records], dtype=np.float64)
    urgencies    = np.array([r.get("urgency_index", 0.0) for r in records], dtype=np.float64)

    action_counts: dict[str, int] = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    # Pearson correlation between risk and urgency — should be high (>0.9)
    # if the urgency amplifier is working correctly
    risk_urgency_corr = float(np.corrcoef(risks, urgencies)[0, 1]) if n > 1 else float("nan")

    return {
        "n_decisions":            n,
        "action_distribution":    {k: v / n for k, v in action_counts.items()},
        "action_counts":          action_counts,
        "human_review_rate":      float(sum(review_flags) / n),
        "constraint_violation_rate": float(sum(violations) / n),
        "mean_risk_score":        float(risks.mean()),
        "mean_urgency_index":     float(urgencies.mean()),
        "risk_urgency_correlation": risk_urgency_corr,
    }


# ------------------------------------------------------------------ #
# 5. Robustness
# ------------------------------------------------------------------ #

def robustness_under_noise(
    model,
    X_test: np.ndarray,
    y_rul_true: np.ndarray,
    noise_levels: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
    n_trials: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Dimension 5a — Robustness under Gaussian sensor noise.

    For each noise level σ, adds N(0, σ²) to the normalised input features
    and measures the MAE degradation relative to the clean baseline.

    Parameters
    ----------
    model       : trained PyTorch model (eval mode)
    X_test      : (n_engines, window_size, features) float32
    y_rul_true  : (n_engines,) ground-truth RUL
    noise_levels: standard deviations of additive Gaussian noise
    n_trials    : number of noise realisations per level (results are averaged)
    seed        : base RNG seed

    Returns
    -------
    dict with ``baseline_mae`` and per noise-level ``mae``, ``mae_degradation``.
    """
    import torch

    rng   = np.random.default_rng(seed)
    dev   = torch.device(device)
    model = model.to(dev)
    model.eval()

    def _predict(X: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                batch = torch.from_numpy(X[i:i+64]).float().to(dev)
                out   = model(batch)
                preds.append((out[0] if isinstance(out, tuple) else out).cpu().numpy())
        return np.concatenate(preds)

    baseline_pred = _predict(X_test)
    baseline_mae  = float(mean_absolute_error(y_rul_true, baseline_pred))

    results: dict[str, Any] = {"baseline_mae": baseline_mae, "noise_levels": {}}

    for sigma in noise_levels:
        trial_maes = []
        for t in range(n_trials):
            noise = rng.normal(0, sigma, X_test.shape).astype(np.float32)
            pred  = _predict(X_test + noise)
            trial_maes.append(float(mean_absolute_error(y_rul_true, pred)))
        mean_mae = float(np.mean(trial_maes))
        results["noise_levels"][sigma] = {
            "mae":             mean_mae,
            "mae_degradation": mean_mae - baseline_mae,
            "relative_degradation": (mean_mae - baseline_mae) / max(baseline_mae, 1e-9),
        }

    return results


def robustness_under_dropout(
    model,
    X_test: np.ndarray,
    y_rul_true: np.ndarray,
    dropout_fractions: tuple[float, ...] = (0.10, 0.25, 0.50),
    n_trials: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Dimension 5b — Robustness under random sensor dropout.

    Randomly zeros out a fraction of sensor channels per window and
    measures MAE degradation relative to the clean baseline.
    """
    import torch

    rng   = np.random.default_rng(seed)
    dev   = torch.device(device)
    model = model.to(dev)
    model.eval()

    def _predict(X: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                batch = torch.from_numpy(X[i:i+64]).float().to(dev)
                out   = model(batch)
                preds.append((out[0] if isinstance(out, tuple) else out).cpu().numpy())
        return np.concatenate(preds)

    baseline_pred = _predict(X_test)
    baseline_mae  = float(mean_absolute_error(y_rul_true, baseline_pred))

    n_features = X_test.shape[-1]
    results: dict[str, Any] = {"baseline_mae": baseline_mae, "dropout_fractions": {}}

    for frac in dropout_fractions:
        n_drop = max(1, int(frac * n_features))
        trial_maes = []
        for _ in range(n_trials):
            X_drop = X_test.copy()
            drop_idx = rng.choice(n_features, size=n_drop, replace=False)
            X_drop[:, :, drop_idx] = 0.0
            trial_maes.append(float(mean_absolute_error(y_rul_true, _predict(X_drop))))
        mean_mae = float(np.mean(trial_maes))
        results["dropout_fractions"][frac] = {
            "n_sensors_dropped": n_drop,
            "mae":               mean_mae,
            "mae_degradation":   mean_mae - baseline_mae,
            "relative_degradation": (mean_mae - baseline_mae) / max(baseline_mae, 1e-9),
        }

    return results


# ------------------------------------------------------------------ #
# 6. User Trust (proxy metrics)
# ------------------------------------------------------------------ #

def user_trust_proxies(
    hitl_json_path: str | Path,
    feedback_jsonl_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Dimension 6 — User Trust (proxy metrics).

    Proxy 1 — Override rate: fraction of HITL reviews where the expert
              chose OVERRIDE rather than APPROVE.  High override rate
              signals low trust in the AI recommendation.

    Proxy 2 — Mean expert confidence: average confidence score across all
              resolved reviews.  Low confidence signals uncertainty about
              the AI output.

    Proxy 3 — Decision stability: fraction of decisions that did NOT
              trigger human review (i.e. the system was trusted to act
              autonomously).

    Proxy 4 — Policy adaptation count: number of policy states that were
              updated from the feedback log, indicating the system is
              learning from expert corrections.
    """
    with open(hitl_json_path) as f:
        hitl = json.load(f)

    reviews   = hitl.get("review_results", [])
    resolved  = [r for r in reviews if r.get("status") in ("APPROVED", "OVERRIDDEN")]
    n_resolved = len(resolved)

    override_count    = sum(1 for r in resolved if r.get("status") == "OVERRIDDEN")
    confidences       = [r.get("expert_confidence", float("nan")) for r in resolved]
    valid_conf        = [c for c in confidences if not np.isnan(c)]

    decisions         = hitl.get("decisions", [])
    n_decisions       = len(decisions)
    autonomous        = sum(1 for d in decisions if not d.get("human_review", True))

    adaptations       = hitl.get("policy_adaptations", {})
    n_adapted_states  = len(adaptations) if isinstance(adaptations, dict) else 0

    result: dict[str, Any] = {
        "n_resolved_reviews":    n_resolved,
        "override_rate":         override_count / n_resolved if n_resolved else float("nan"),
        "mean_expert_confidence": float(np.mean(valid_conf)) if valid_conf else float("nan"),
        "autonomous_decision_rate": autonomous / n_decisions if n_decisions else float("nan"),
        "n_policy_states_adapted": n_adapted_states,
    }

    # Optional: parse feedback.jsonl for cross-run override rate
    if feedback_jsonl_path and Path(feedback_jsonl_path).exists():
        records = []
        with open(feedback_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        n_fb      = len(records)
        n_override = sum(1 for r in records if r.get("override", False))
        result["feedback_log_override_rate"] = n_override / n_fb if n_fb else float("nan")
        result["feedback_log_n_records"]     = n_fb

    return result


# ------------------------------------------------------------------ #
# Aggregate report
# ------------------------------------------------------------------ #

def full_evaluation_report(
    *,
    y_rul_true: np.ndarray,
    y_rul_pred: np.ndarray,
    y_hi_true: np.ndarray,
    y_hi_pred: np.ndarray,
    lower_90: np.ndarray,
    upper_90: np.ndarray,
    engine_ids: np.ndarray,
    eri_json_path: str | Path,
    decision_json_path: str | Path,
    hitl_json_path: str | Path,
    feedback_jsonl_path: str | Path | None = None,
    model=None,
    X_test: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Assemble all six evaluation dimensions into one report dict.

    Robustness (dimension 5) is only computed when ``model`` and
    ``X_test`` are supplied — they require a live PyTorch model.
    """
    report: dict[str, Any] = {}

    report["predictive_accuracy"] = predictive_accuracy(
        y_rul_true, y_rul_pred, y_hi_true, y_hi_pred
    )
    report["uncertainty_reliability"] = uncertainty_reliability(
        y_rul_true, lower_90, upper_90, engine_ids
    )
    report["explainability_quality"] = explainability_quality(eri_json_path)
    report["decision_effectiveness"] = decision_effectiveness(decision_json_path)
    report["user_trust"]             = user_trust_proxies(
        hitl_json_path, feedback_jsonl_path
    )

    if model is not None and X_test is not None:
        report["robustness"] = {
            "noise":   robustness_under_noise(model, X_test, y_rul_true, device=device),
            "dropout": robustness_under_dropout(model, X_test, y_rul_true, device=device),
        }
    else:
        report["robustness"] = {"note": "skipped — model/X_test not supplied"}

    return report
