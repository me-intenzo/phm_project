"""Final maintenance recommendation interface.

O4 pipeline
-----------
1. Determine health state from RUL + HI.
2. Classify uncertainty from the conformal interval.
3. Classify explanation reliability from ERI.
4. Compute continuous risk score and urgency index.
5. Select a candidate maintenance action.
6. Apply operational constraints.
7. Return an auditable structured recommendation.

author: me-intenzo
"""

from __future__ import annotations

from typing import Any

from .constraints import check_constraints
from .rules import apply_rules


def recommend_action(
    state: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate an O4 maintenance recommendation.

    Parameters
    ----------
    state:
        Must contain: rul, hi, rul_lower, rul_upper, eri.
        Optional: top_k_sensors (list[int]) from O3 XAI output.
    constraints:
        Optional operational constraints (see constraints.py).
    thresholds:
        Optional policy thresholds for calibration experiments.

    Returns
    -------
    dict
        Fully auditable maintenance decision including quantitative
        risk score, urgency index, and XAI context.
    """
    ruled      = apply_rules(state=state, thresholds=thresholds)
    constrained = check_constraints(decision=ruled, constraints=constraints)

    return {
        # --- Prognostic inputs ---
        "inputs": {
            "rul":       constrained["rul"],
            "hi":        constrained["hi"],
            "rul_lower": constrained["rul_lower"],
            "rul_upper": constrained["rul_upper"],
            "eri":       constrained["eri"],
        },
        # --- Health assessment ---
        "health_state":            constrained["health_state"],
        "uncertainty_level":       constrained["uncertainty_level"],
        "explanation_reliability": constrained["explanation_reliability"],
        # --- Quantitative risk ---
        "risk_score":              constrained["risk_score"],
        "urgency_index":           constrained["urgency_index"],
        "risk_components":         constrained["risk_components"],
        # --- Decision ---
        "recommended_action":      constrained["candidate_action"],
        "human_review":            constrained["human_review"],
        "reasons":                 constrained["reasons"],
        # --- Constraint audit ---
        "constraint_status":       constrained["constraint_status"],
        "constraint_violations":   constrained["constraint_violations"],
        "constraint_adjustments":  constrained["constraint_adjustments"],
        # --- XAI context (O3 passthrough) ---
        "top_k_sensors":           constrained.get("top_k_sensors", []),
    }
