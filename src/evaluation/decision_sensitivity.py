"""
src/evaluation/decision_sensitivity.py

Sensitivity of the maintenance decision policy to its risk-score weights.

Why this module exists
----------------------
The risk score fuses four normalised signals with four hand-set weights

    R = w_rul * r_rul + w_hi * r_hi + w_u * r_u + w_eri * r_eri,
    w_rul + w_hi + w_u + w_eri = 1,

and those weights were never tested: the repository published a decision mix
without ever asking how much of it is an artefact of ``w_rul = 0.40``.  This
module answers that question by re-running the decision engine over a
weight space and measuring how far the decisions move.

Two sweeps are used, because they answer different questions:

*One-at-a-time (OAT)* — each weight in turn is set to a grid of values and
the remaining mass is redistributed proportionally among the other three.
This answers "how sensitive is the policy to this particular weight, with
the others held in proportion?".

*Dirichlet* — weights are drawn from a symmetric Dirichlet over the simplex.
This answers "how much does the decision mix move under a plausible
re-weighting that respects the simplex constraint?".

Metrics
-------
For every perturbation the module reports

``action_tv``
    Total-variation distance between the action distribution and the
    baseline one (0 = identical mix, 1 = disjoint).
``action_switch_rate``
    Fraction of records whose recommended action changed.
``mean_severity_delta``
    Mean change of the action's position on the severity ladder.
``review_rate_delta``
    Change of the human-review rate.
``risk_rank_spearman``
    Spearman correlation between the baseline and the perturbed risk
    ranking of the records — i.e. whether the *ordering* of the engines by
    risk survives the re-weighting.
``mean_abs_risk_delta``
    Mean absolute change of the risk score.
``risk_band_switch_rate``
    Fraction of records that change their risk *band* — the grouping key of
    the HITL policy refinement, and therefore the part of the risk score
    that actually changes a downstream decision.
``override_switch_count``
    Number of records that cross the risk-score override (0.85) and are
    escalated to urgent maintenance.

The headline of the report is ``influence``: the largest total-variation
distance reached by a weight, i.e. the largest swing in the decision mix
that one weight can produce on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.decision_engine.recommender import recommend_action
from src.decision_engine.rules import DEFAULT_THRESHOLDS, SEVERITY_LADDER, severity_of
from src.evaluation.ablation import _state_from_record
from src.utils.config import RISK_BANDS

#: The four fusion weights, in the order used by the risk score.
WEIGHT_NAMES: tuple[str, ...] = ("w_rul", "w_hi", "w_u", "w_eri")

#: The risk score above which the policy always escalates to urgent
#: maintenance.  It is the only place where the *continuous* risk score
#: changes the action, so it is the only route by which the fusion weights
#: can move the action mix.
RISK_OVERRIDE = 0.85


# ------------------------------------------------------------------
# Input handling
# ------------------------------------------------------------------

def load_decision_records(path: str | Path) -> list[dict[str, Any]]:
    """Read the decision records of a decision JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    decisions = payload.get("decisions", payload)
    if isinstance(decisions, dict):
        decisions = [decisions]
    return list(decisions)


def _baseline_weights(thresholds: dict[str, float] | None) -> dict[str, float]:
    source = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        source.update(thresholds)
    return {name: float(source.get(name, 0.0)) for name in WEIGHT_NAMES}


# ------------------------------------------------------------------
# Re-running the policy
# ------------------------------------------------------------------

def _run(
    records: Sequence[dict[str, Any]],
    weights: dict[str, float],
    constraints: dict[str, Any] | None,
    thresholds: dict[str, float] | None,
) -> list[dict[str, Any]]:
    """Re-run the decision engine with a given weight vector."""

    merged = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        merged.update(thresholds)
    merged.update(weights)

    decisions = []
    for record in records:
        state = _state_from_record(record)
        decisions.append(
            recommend_action(
                state=state,
                constraints=constraints,
                thresholds=merged,
            )
        )
    return decisions


def _action_mix(decisions: Sequence[dict[str, Any]]) -> dict[str, int]:
    mix = {action: 0 for action in SEVERITY_LADDER}
    for decision in decisions:
        action = str(decision.get("recommended_action", ""))
        mix[action] = mix.get(action, 0) + 1
    return mix


def _risk_band(value: float) -> str:
    """Risk band of a continuous risk score (the HITL evidence grouping)."""

    for upper, label in RISK_BANDS:
        if value < upper:
            return label
    return "HIGH"


def _total_variation(a: dict[str, int], b: dict[str, int]) -> float:
    """Total-variation distance between two action distributions."""

    total = sum(a.values()) or 1
    keys = set(a) | set(b)
    return 0.5 * sum(
        abs(a.get(key, 0) / total - b.get(key, 0) / total) for key in keys
    )


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (tie-aware, no scipy dependency)."""

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) < 2:
        return float("nan")

    rank_x = _rank(x)
    rank_y = _rank(y)
    rank_x = rank_x - rank_x.mean()
    rank_y = rank_y - rank_y.mean()
    denominator = np.sqrt((rank_x ** 2).sum() * (rank_y ** 2).sum())
    if denominator <= 0:
        return float("nan")
    return float((rank_x * rank_y).sum() / denominator)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties are handled correctly."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        stop = index
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        ranks[order[index:stop + 1]] = average
        index = stop + 1
    return ranks


# ------------------------------------------------------------------
# The sweeps
# ------------------------------------------------------------------

def _redistribute(
    weights: dict[str, float],
    name: str,
    value: float,
) -> dict[str, float]:
    """Set one weight to *value* and share the rest proportionally."""

    others = [key for key in WEIGHT_NAMES if key != name]
    remaining = max(0.0, 1.0 - value)
    others_total = sum(weights[key] for key in others) or 1.0

    perturbed = dict(weights)
    perturbed[name] = float(value)
    for key in others:
        perturbed[key] = remaining * weights[key] / others_total
    return perturbed


class _Baseline:
    """The reference reading every perturbation is compared against."""

    def __init__(
        self,
        decisions: Sequence[dict[str, Any]],
        weights: dict[str, float],
    ) -> None:
        self.decisions = list(decisions)
        self.weights = weights
        self.mix = _action_mix(self.decisions)
        self.risk = [float(d["risk_score"]) for d in self.decisions]
        self.severity = [severity_of(d["recommended_action"]) for d in self.decisions]
        self.review_rate = (
            sum(1 for d in self.decisions if d.get("human_review")) / len(self.decisions)
            if self.decisions else 0.0
        )
        self.bands = [_risk_band(value) for value in self.risk]
        self.overrides = [value > RISK_OVERRIDE for value in self.risk]


def _evaluate(
    decisions: Sequence[dict[str, Any]],
    baseline: _Baseline,
) -> dict[str, Any]:
    """Movement of the decisions relative to the baseline reading."""

    mix = _action_mix(decisions)
    risk = [float(d["risk_score"]) for d in decisions]
    severity = [severity_of(d["recommended_action"]) for d in decisions]
    review_rate = (
        sum(1 for d in decisions if d.get("human_review")) / len(decisions)
        if decisions else 0.0
    )

    switches = sum(
        1
        for new, old in zip(decisions, baseline.decisions)
        if new.get("recommended_action") != old.get("recommended_action")
    )
    band_switches = sum(
        1
        for value, old in zip(risk, baseline.risk)
        if _risk_band(value) != _risk_band(old)
    )
    override_switches = sum(
        1
        for value, old in zip(risk, baseline.risk)
        if (value > RISK_OVERRIDE) != (old > RISK_OVERRIDE)
    )

    correlation = _spearman(baseline.risk, risk)

    return {
        "action_tv": round(_total_variation(baseline.mix, mix), 6),
        "action_switch_rate": round(switches / len(decisions), 6) if decisions else 0.0,
        "mean_severity_delta": round(
            float(np.mean([s - b for s, b in zip(severity, baseline.severity)]))
            if severity else 0.0,
            6,
        ),
        "review_rate_delta": round(review_rate - baseline.review_rate, 6),
        "risk_rank_spearman": (
            round(correlation, 6) if not np.isnan(correlation) else None
        ),
        "mean_abs_risk_delta": round(
            float(np.mean(np.abs(np.asarray(risk) - np.asarray(baseline.risk))))
            if risk else 0.0,
            6,
        ),
        "risk_band_switch_rate": (
            round(band_switches / len(decisions), 6) if decisions else 0.0
        ),
        "override_switch_count": int(override_switches),
        "action_distribution": mix,
    }


def oat_sweep(
    records: Sequence[dict[str, Any]],
    constraints: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
    grid: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> list[dict[str, Any]]:
    """One-at-a-time sweep over every fusion weight."""

    baseline_weights = _baseline_weights(thresholds)
    baseline = _Baseline(
        _run(records, baseline_weights, constraints, thresholds),
        baseline_weights,
    )

    rows: list[dict[str, Any]] = []

    for name in WEIGHT_NAMES:
        for value in grid:
            perturbed = _redistribute(baseline_weights, name, value)
            decisions = _run(records, perturbed, constraints, thresholds)
            row = {
                "sweep": "oat",
                "weight": name,
                "value": float(value),
                "weights": {k: round(float(v), 6) for k, v in perturbed.items()},
            }
            row.update(_evaluate(decisions, baseline))
            rows.append(row)

    return rows


def dirichlet_sweep(
    records: Sequence[dict[str, Any]],
    constraints: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
    n_samples: int = 200,
    concentration: float = 5.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Random sweep over the weight simplex."""

    baseline_weights = _baseline_weights(thresholds)
    baseline = _Baseline(
        _run(records, baseline_weights, constraints, thresholds),
        baseline_weights,
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for index in range(int(n_samples)):
        draw = rng.dirichlet(
            np.full(len(WEIGHT_NAMES), float(concentration))
        )
        perturbed = {
            name: float(value) for name, value in zip(WEIGHT_NAMES, draw)
        }
        decisions = _run(records, perturbed, constraints, thresholds)
        row = {
            "sweep": "dirichlet",
            "sample": index,
            "weight": "all",
            "value": None,
            "weights": {k: round(float(v), 6) for k, v in perturbed.items()},
        }
        row.update(_evaluate(decisions, baseline))
        rows.append(row)

    return rows


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Condense the sweep rows into the headline numbers."""

    oat_rows = [row for row in rows if row["sweep"] == "oat"]
    dirichlet_rows = [row for row in rows if row["sweep"] == "dirichlet"]

    influence: dict[str, float] = {}
    for name in WEIGHT_NAMES:
        values = [
            row["action_tv"] for row in oat_rows if row["weight"] == name
        ]
        influence[name] = round(max(values), 6) if values else 0.0

    band_influence: dict[str, float] = {}
    for name in WEIGHT_NAMES:
        values = [
            row["risk_band_switch_rate"] for row in oat_rows if row["weight"] == name
        ]
        band_influence[name] = round(max(values), 6) if values else 0.0

    def _stats(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {}
        tvs = [row["action_tv"] for row in subset]
        switches = [row["action_switch_rate"] for row in subset]
        band_switches = [row["risk_band_switch_rate"] for row in subset]
        override_switches = [row["override_switch_count"] for row in subset]
        correlations = [
            row["risk_rank_spearman"] for row in subset
            if row["risk_rank_spearman"] is not None
        ]
        return {
            "n": len(subset),
            "action_tv_mean": round(float(np.mean(tvs)), 6),
            "action_tv_max": round(float(np.max(tvs)), 6),
            "action_tv_p95": round(float(np.percentile(tvs, 95)), 6),
            "action_switch_rate_mean": round(float(np.mean(switches)), 6),
            "action_switch_rate_max": round(float(np.max(switches)), 6),
            "risk_band_switch_rate_mean": round(float(np.mean(band_switches)), 6),
            "risk_band_switch_rate_max": round(float(np.max(band_switches)), 6),
            "override_switch_count_max": int(max(override_switches)),
            "risk_rank_spearman_mean": (
                round(float(np.mean(correlations)), 6) if correlations else None
            ),
            "risk_rank_spearman_min": (
                round(float(np.min(correlations)), 6) if correlations else None
            ),
        }

    return {
        "influence_by_weight": influence,
        "risk_band_influence_by_weight": band_influence,
        "most_influential_weight": (
            max(influence, key=influence.get) if influence else None
        ),
        "oat": _stats(oat_rows),
        "dirichlet": _stats(dirichlet_rows),
    }


def run_sensitivity(
    decision_path: str | Path,
    n_dirichlet: int = 200,
    concentration: float = 5.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run both sweeps on a decision artifact and return the full report."""

    path = Path(decision_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("decisions", payload)
    if isinstance(records, dict):
        records = [records]
    records = list(records)

    if not records:
        return {"error": f"no decision records in {path.name}"}

    constraints = payload.get("operational_constraints") or {}
    thresholds = payload.get("thresholds") or None

    oat_rows = oat_sweep(records, constraints, thresholds)
    dirichlet_rows = dirichlet_sweep(
        records, constraints, thresholds,
        n_samples=n_dirichlet, concentration=concentration, seed=seed,
    )
    rows = oat_rows + dirichlet_rows

    return {
        "source": str(path),
        "subset": payload.get("subset"),
        "model": payload.get("model"),
        "tag": payload.get("tag"),
        "constraint_scenario": payload.get("constraint_scenario", "unconstrained"),
        "n_decisions": len(records),
        "baseline_weights": _baseline_weights(thresholds),
        "n_dirichlet_samples": int(n_dirichlet),
        "dirichlet_concentration": float(concentration),
        "seed": int(seed),
        "summary": summarise(rows),
        "rows": rows,
    }


__all__ = [
    "WEIGHT_NAMES",
    "dirichlet_sweep",
    "load_decision_records",
    "oat_sweep",
    "run_sensitivity",
    "summarise",
]
