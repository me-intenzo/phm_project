"""Operational constraint handling for the O4 decision engine.

Constraints are a hard feasibility layer applied AFTER the rule-based
policy.  They can escalate but never de-escalate an action.

Supported constraints
---------------------
maintenance_window_available : bool
    Whether scheduled maintenance can currently be performed.
minimum_safe_rul : float
    Minimum RUL required to defer maintenance.  If RUL falls at or
    below this value while the action is MONITOR/CONTINUE, escalate
    to URGENT_MAINTENANCE.
maintenance_lead_time : float
    Cycles required before maintenance can begin.  If RUL ≤ lead time
    and action is SCHEDULE_MAINTENANCE, escalate to URGENT_MAINTENANCE.
max_allowable_delay : float
    Maximum number of cycles the operator is willing to wait before
    acting.  If RUL ≤ max_allowable_delay and action is passive
    (CONTINUE/MONITOR), escalate to INSPECT.
resource_availability : bool
    Whether maintenance crew / parts are available.  If False and
    action requires physical intervention, flag for human review.

author: me-intenzo
"""

from __future__ import annotations

from typing import Any

VALID_ACTIONS = frozenset({
    "CONTINUE_OPERATION",
    "MONITOR",
    "INSPECT",
    "SCHEDULE_MAINTENANCE",
    "URGENT_MAINTENANCE",
})

_PASSIVE_ACTIONS  = frozenset({"CONTINUE_OPERATION", "MONITOR"})
_ACTIVE_ACTIONS   = frozenset({"INSPECT", "SCHEDULE_MAINTENANCE", "URGENT_MAINTENANCE"})


def check_constraints(
    decision: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply operational constraints to a candidate decision.

    Constraints can only escalate the action, never downgrade it.
    """
    constraints = constraints or {}

    action = str(decision.get("candidate_action", "")).upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown candidate action: {action!r}")

    result = dict(decision)
    violations: list[str] = []
    adjustments: list[str] = []

    rul = float(decision.get("rul", 0.0))

    # ------------------------------------------------------------------
    # 1. Maintenance-window availability
    # ------------------------------------------------------------------
    window_available = bool(constraints.get("maintenance_window_available", True))

    if action == "SCHEDULE_MAINTENANCE" and not window_available:
        violations.append("Maintenance window is unavailable.")
        result["candidate_action"] = "INSPECT"
        result["human_review"] = True
        adjustments.append("Scheduled maintenance downgraded to inspection (no window).")

    # ------------------------------------------------------------------
    # 2. Minimum safe RUL
    # ------------------------------------------------------------------
    minimum_safe_rul = constraints.get("minimum_safe_rul")
    if minimum_safe_rul is not None:
        minimum_safe_rul = float(minimum_safe_rul)
        if result["candidate_action"] in _PASSIVE_ACTIONS and rul <= minimum_safe_rul:
            violations.append(
                f"RUL ({rul:.1f}) is at or below the minimum safe operating limit "
                f"({minimum_safe_rul:.1f})."
            )
            result["candidate_action"] = "URGENT_MAINTENANCE"
            result["human_review"] = True
            adjustments.append("Deferred operation escalated to urgent maintenance (min-safe-RUL).")

    # ------------------------------------------------------------------
    # 3. Maintenance lead time
    # ------------------------------------------------------------------
    maintenance_lead_time = constraints.get("maintenance_lead_time")
    if maintenance_lead_time is not None:
        maintenance_lead_time = float(maintenance_lead_time)
        if result["candidate_action"] == "SCHEDULE_MAINTENANCE" and rul <= maintenance_lead_time:
            violations.append(
                f"RUL ({rul:.1f}) is shorter than the required maintenance lead time "
                f"({maintenance_lead_time:.1f} cycles)."
            )
            result["candidate_action"] = "URGENT_MAINTENANCE"
            result["human_review"] = True
            adjustments.append(
                "Scheduled maintenance escalated to urgent (RUL < lead time)."
            )

    # ------------------------------------------------------------------
    # 4. Maximum allowable delay
    # ------------------------------------------------------------------
    max_allowable_delay = constraints.get("max_allowable_delay")
    if max_allowable_delay is not None:
        max_allowable_delay = float(max_allowable_delay)
        if result["candidate_action"] in _PASSIVE_ACTIONS and rul <= max_allowable_delay:
            violations.append(
                f"RUL ({rul:.1f}) is within the maximum allowable delay window "
                f"({max_allowable_delay:.1f} cycles) — passive action is insufficient."
            )
            result["candidate_action"] = "INSPECT"
            result["human_review"] = True
            adjustments.append(
                "Passive action escalated to inspection (RUL within allowable delay)."
            )

    # ------------------------------------------------------------------
    # 5. Resource availability
    # ------------------------------------------------------------------
    resource_available = constraints.get("resource_availability")
    if resource_available is not None and not bool(resource_available):
        if result["candidate_action"] in _ACTIVE_ACTIONS:
            violations.append("Maintenance resources (crew/parts) are unavailable.")
            result["human_review"] = True
            adjustments.append(
                "Human review flagged: active maintenance required but resources unavailable."
            )

    result["constraint_status"]      = "VIOLATION" if violations else "SATISFIED"
    result["constraint_violations"]  = violations
    result["constraint_adjustments"] = adjustments
    return result
