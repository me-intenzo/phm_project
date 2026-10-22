"""
src/evaluation/ablation.py

Ablation study — measures the contribution of each pipeline component
by comparing the full system against variants with one component removed.

Every variant is produced by **re-running the decision engine** on a patched
input state (``src.decision_engine.recommender.recommend_action``), so each
variant goes through exactly the same code path as the full pipeline and the
reported deltas are real measurements, not relabelled copies of the baseline.

Ablation variants
-----------------
full            : complete pipeline (RUL + HI + uncertainty + XAI + constraints)
no_uncertainty  : decision engine receives a wide fixed interval instead of
                  the conformal interval (simulates no uncertainty quantification)
no_xai          : ERI set to 0.5 (random agreement baseline) — the decision no
                  longer depends on the measured explanation reliability
no_constraints  : operational constraints are bypassed (no constraint layer)
point_only      : only RUL point prediction, no HI, no uncertainty, no XAI

Predictive accuracy is read from the prediction artifacts; the decision
effectiveness metrics of every variant are recomputed from the re-run
decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.decision_engine.recommender import recommend_action
from src.evaluation.metrics import (
    decision_effectiveness_from_records,
    predictive_accuracy,
)

RUL_MAX = 125.0


# ------------------------------------------------------------------ #
# Input handling
# ------------------------------------------------------------------ #

def _decision_records(full_data: dict) -> list[dict]:
    """Extract the list of decision records from the decision JSON."""
    decisions = full_data.get("decisions", full_data)
    if isinstance(decisions, dict):
        decisions = [decisions]
    return list(decisions)


def _state_from_record(record: dict) -> dict[str, Any]:
    """
    Rebuild the flat decision state consumed by ``recommend_action``.

    The decision JSON nests the prognostic inputs under ``inputs``; the
    recommender consumes them flat.  Patching therefore has to happen on
    this flat state, otherwise the patched values are never read.
    """
    inputs = record.get("inputs") or {}

    def _get(key: str, default: float) -> float:
        value = inputs.get(key, record.get(key, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    state: dict[str, Any] = {
        "rul":       _get("rul", 0.0),
        "hi":        _get("hi", 1.0),
        "rul_lower": _get("rul_lower", 0.0),
        "rul_upper": _get("rul_upper", 0.0),
        "eri":       _get("eri", 0.5),
    }

    top_k = record.get("top_k_sensors") or inputs.get("top_k_sensors")
    if top_k:
        state["top_k_sensors"] = list(top_k)

    return state


# ------------------------------------------------------------------ #
# Patches
# ------------------------------------------------------------------ #

def _identity(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state)


def _fixed_interval(state: dict[str, Any], interval_width: float = 100.0) -> dict[str, Any]:
    """Simulate no-uncertainty variant: replace intervals with a wide fixed band."""
    patched = dict(state)
    rul = float(patched.get("rul", 0.0))
    lower = max(0.0, rul - interval_width / 2.0)
    upper = min(RUL_MAX, rul + interval_width / 2.0)
    patched["rul_lower"] = lower
    patched["rul_upper"] = max(lower, upper)
    return patched


def _neutral_eri(state: dict[str, Any]) -> dict[str, Any]:
    """Simulate no-XAI variant: set ERI to 0.5 (random agreement baseline)."""
    patched = dict(state)
    patched["eri"] = 0.5
    return patched


def _recompute(
    records: list[dict],
    patch: Callable[[dict[str, Any]], dict[str, Any]],
    constraints: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> list[dict]:
    """Re-run the decision engine on the patched inputs of every record."""
    decisions = []
    for record in records:
        state = patch(_state_from_record(record))
        decisions.append(
            recommend_action(
                state=state,
                constraints=constraints,
                thresholds=thresholds,
            )
        )
    return decisions


def _same_outcome(a: dict, b: dict) -> bool:
    return (
        a.get("recommended_action") == b.get("recommended_action")
        and bool(a.get("human_review")) == bool(b.get("human_review"))
    )


# ------------------------------------------------------------------ #
# Ablation runner
# ------------------------------------------------------------------ #

def run_ablation(
    results_dir: str | Path,
    subset: str,
    model: str,
) -> dict[str, Any]:
    """
    Run the ablation study for one (subset, model) combination.

    Loads the full-pipeline decision JSON and prediction NPZ, re-runs the
    decision engine on the patched inputs of every ablation variant and
    reports the resulting decision metrics plus the deltas against the full
    pipeline.

    Parameters
    ----------
    results_dir : ``outputs/results/``
    subset      : e.g. "FD001"
    model       : e.g. "gru"

    Returns
    -------
    dict keyed by variant name → metric sub-dicts.
    """
    results_dir = Path(results_dir)
    decision_path = results_dir / f"decision_{subset}_{model}.json"
    pred_path     = results_dir / f"{subset}_{model}_predictions.npz"

    if not decision_path.exists():
        return {"error": f"missing {decision_path.name}"}

    with open(decision_path) as f:
        full_data = json.load(f)

    records = _decision_records(full_data)
    if not records:
        return {"error": f"no decision records in {decision_path.name}"}

    # The operational constraints the original decisions were produced with.
    constraints = full_data.get("operational_constraints") or {}

    # The policy thresholds the original decisions were produced with.  They
    # are read back from the artifact so that the re-run is the *same* policy,
    # including the calibrated HI breakpoints of the sensor-derived index.
    thresholds = full_data.get("thresholds") or None

    report: dict[str, Any] = {
        "subset": subset,
        "model": model,
        "n_decisions": len(records),
    }

    # ── full pipeline ──────────────────────────────────────────────
    # Re-run with the recorded constraints so that every variant shares
    # one code path; the stored decisions are kept for the consistency check.
    full_decisions = _recompute(
        records, _identity, constraints=constraints, thresholds=thresholds
    )
    report["full"] = decision_effectiveness_from_records(full_decisions)
    report["full"]["note"] = (
        "decision engine re-run on the recorded inputs "
        "(RUL + HI + conformal interval + ERI + operational constraints)"
    )
    report["full"]["matches_stored_decisions"] = bool(
        len(full_decisions) == len(records)
        and all(
            _same_outcome(new, old)
            for new, old in zip(full_decisions, records)
        )
    )

    # ── no_uncertainty ─────────────────────────────────────────────
    # Wide fixed interval instead of the conformal one.
    unc_decisions = _recompute(
        records, _fixed_interval, constraints=constraints, thresholds=thresholds
    )
    report["no_uncertainty"] = decision_effectiveness_from_records(unc_decisions)
    report["no_uncertainty"]["note"] = (
        "fixed interval width=100 cycles applied; the uncertainty level is no "
        "longer derived from the conformal interval"
    )

    # ── no_xai ─────────────────────────────────────────────────────
    xai_decisions = _recompute(
        records, _neutral_eri, constraints=constraints, thresholds=thresholds
    )
    report["no_xai"] = decision_effectiveness_from_records(xai_decisions)
    report["no_xai"]["note"] = (
        "ERI set to 0.5 (random baseline); explanation_reliability forced LOW"
    )

    # ── no_constraints ─────────────────────────────────────────────
    # The constraint layer is bypassed entirely: the engine runs without
    # any operational constraint instead of having its violations cleared
    # after the fact.
    con_decisions = _recompute(
        records, _identity, constraints={}, thresholds=thresholds
    )
    report["no_constraints"] = decision_effectiveness_from_records(con_decisions)
    report["no_constraints"]["note"] = (
        "operational constraints bypassed (constraint layer not applied)"
    )

    # ── point_only (predictive accuracy only, no interval/XAI) ────
    if pred_path.exists():
        pred = np.load(pred_path)
        report["point_only"] = {
            "predictive_accuracy": predictive_accuracy(
                pred["y_rul_true"], pred["y_rul_pred"],
                pred["y_hi_true"],  pred["y_hi_pred"],
            ),
            "note": "RUL/HI point prediction only; no uncertainty, XAI, or constraints",
        }
    else:
        report["point_only"] = {"error": f"missing {pred_path.name}"}

    # ── delta summary ──────────────────────────────────────────────
    def _delta(variant: str, metric: str) -> float:
        return float(
            report[variant].get(metric, float("nan"))
            - report["full"].get(metric, float("nan"))
        )

    report["delta_human_review_rate"] = {
        "no_uncertainty_vs_full": _delta("no_uncertainty", "human_review_rate"),
        "no_xai_vs_full":         _delta("no_xai", "human_review_rate"),
        "no_constraints_vs_full": _delta("no_constraints", "human_review_rate"),
        "interpretation": (
            "positive delta = more reviews triggered without that component "
            "(component reduces unnecessary escalation)"
        ),
    }

    report["delta_constraint_violation_rate"] = {
        "no_constraints_vs_full": _delta("no_constraints", "constraint_violation_rate"),
        "interpretation": (
            "positive delta = the removed constraint layer was responsible "
            "for the recorded violations"
        ),
    }

    report["delta_mean_risk_score"] = {
        "no_uncertainty_vs_full": _delta("no_uncertainty", "mean_risk_score"),
        "no_xai_vs_full":         _delta("no_xai", "mean_risk_score"),
        "no_constraints_vs_full": _delta("no_constraints", "mean_risk_score"),
        "interpretation": (
            "positive delta = the removed component contributed to the risk score"
        ),
    }

    # Action-mix deltas: they stay informative even when a variant does not
    # change the human-review rate.
    full_counts = report["full"].get("action_counts", {})
    report["delta_action_counts"] = {
        variant: {
            action: int(
                report[variant].get("action_counts", {}).get(action, 0)
                - full_counts.get(action, 0)
            )
            for action in sorted(
                set(full_counts) | set(report[variant].get("action_counts", {}))
            )
        }
        for variant in ("no_uncertainty", "no_xai", "no_constraints")
    }

    return report
