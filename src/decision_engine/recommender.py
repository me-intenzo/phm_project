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

The recommendation is the **full payload** of the rule and constraint
layers.  This function used to project the result onto a hand-written
list of twenty keys, which silently dropped every field that the
upstream layers had added since the list was written.  Now the complete
payload is passed through and the few fields with a stable meaning
(``inputs``, ``recommended_action``) are added on top of it.

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
        risk score, urgency index, and XAI context.  Every field the
        rule and constraint layers produced is present, so a new field
        upstream never has to be added to a projection list here.
    """
    ruled      = apply_rules(state=state, thresholds=thresholds)
    constrained = check_constraints(decision=ruled, constraints=constraints)

    # ------------------------------------------------------------------
    # Pass the full payload through.
    # ------------------------------------------------------------------

    result: dict[str, Any] = dict(constrained)

    # The prognostic inputs are also exposed as a nested block: the
    # decision report, the ablation runner and the downstream consumers
    # read them from there.
    result["inputs"] = {
        "rul":       constrained["rul"],
        "hi":        constrained["hi"],
        "rul_lower": constrained["rul_lower"],
        "rul_upper": constrained["rul_upper"],
        "eri":       constrained["eri"],
    }

    # The rule layer names the action ``candidate_action`` (it may still
    # be changed by the constraint layer); the recommendation exposes it
    # under its final name.
    result["recommended_action"] = constrained["candidate_action"]

    # Stable defaults for the fields a consumer may rely on.
    result.setdefault("action_severity", 0)
    result.setdefault("severity_delta", 0)
    result.setdefault("human_review", False)
    result.setdefault("reasons", [])
    result.setdefault("risk_components", {})
    result.setdefault("top_k_sensors", [])
    result.setdefault("constraint_status", "SATISFIED")
    result.setdefault("constraint_violations", [])
    result.setdefault("constraint_adjustments", [])
    result.setdefault("constraint_substitutions", [])

    return result
