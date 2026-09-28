"""
Human-in-the-loop decision-policy refinement.

O5 does not retrain the prognostic model.

Instead:
    accumulated expert feedback
            ↓
    empirical policy profile
            ↓
    conservative consensus check
            ↓
    recommendation refinement
            ↓
    operational constraint re-check

author: me-intenzo
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from src.decision_engine.constraints import check_constraints
from src.hitl.feedback import VALID_ACTIONS
from src.hitl.logger import load_feedback


# ---------------------------------------------------------------------
# Conservative adaptation thresholds
# ---------------------------------------------------------------------

MIN_SUPPORT = 3

MIN_WEIGHTED_SUPPORT = 2.0

MIN_CONSENSUS_RATIO = 0.70

RISK_BANDS = ((0.33, "LOW"), (0.66, "MODERATE"))
URGENCY_BANDS = ((0.33, "LOW"), (0.66, "MODERATE"))


SAFETY_CRITICAL_ACTIONS = {
    "URGENT_MAINTENANCE",
}


def _feedback_weight(
    record: dict[str, Any],
) -> float:
    """Use expert confidence as feedback weight."""

    confidence = float(
        record.get(
            "expert_confidence",
            0.0,
        )
    )

    return max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


def _continuous_band(value: Any, bands: tuple[tuple[float, str], ...]) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    for upper, label in bands:
        if numeric < upper:
            return label
    return "HIGH"


def _rul_severity(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        rul = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if rul <= 10.0:
        return "CRITICAL"
    if rul <= 25.0:
        return "AT_RISK"
    if rul <= 50.0:
        return "DEGRADING"
    return "HEALTHY"


def policy_state(record: dict[str, Any]) -> dict[str, str]:
    """Return the categorical state used for policy evidence grouping."""
    return {
        "subset": str(record.get("subset", "UNKNOWN")).upper(),
        "model": str(record.get("model", "UNKNOWN")).lower(),
        "health_state": str(record.get("health_state", "UNKNOWN")).upper(),
        "uncertainty_level": str(record.get("uncertainty_level", "UNKNOWN")).upper(),
        "explanation_reliability": str(record.get("explanation_reliability", "UNKNOWN")).upper(),
        "risk_band": _continuous_band(record.get("risk_score"), RISK_BANDS),
        "urgency_band": _continuous_band(record.get("urgency_index"), URGENCY_BANDS),
        "rul_severity": _rul_severity(record.get("rul", record.get("rul_prediction"))),
    }


def _state_key(
    record: dict[str, Any],
) -> tuple[str, ...]:
    """
    Group feedback by decision context rather than engine identity.

    Context:
        subset
        model
        health state
        uncertainty level
        explanation reliability
        risk band
        urgency band
        RUL severity
    """
    state = policy_state(record)
    return tuple(state.values())


def build_policy_profile(
    feedback_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build an empirical O5 policy profile.

    The profile records action support, observation counts,
    weighted support and consensus ratio.
    """

    groups: dict[
        tuple[str, str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in feedback_records:
        groups[
            _state_key(record)
        ].append(record)

    profile: dict[str, Any] = {}

    for key, records in groups.items():

        action_support: dict[str, float] = (
            defaultdict(float)
        )

        action_count: dict[str, int] = (
            defaultdict(int)
        )

        for record in records:

            action = str(
                record.get(
                    "expert_action",
                    "",
                )
            ).upper()

            if action not in VALID_ACTIONS:
                continue

            weight = _feedback_weight(record)

            action_support[action] += weight
            action_count[action] += 1

        if not action_support:
            continue

        preferred_action = max(
            action_support,
            key=action_support.get,
        )

        total_weight = sum(
            action_support.values()
        )

        consensus_ratio = (
            action_support[preferred_action]
            / total_weight
            if total_weight > 0
            else 0.0
        )

        profile[str(key)] = {
            "policy_state": policy_state(records[0]),
            "preferred_action": preferred_action,
            "support": round(
                action_support[preferred_action],
                4,
            ),
            "count": action_count[
                preferred_action
            ],
            "total_observations": len(records),
            "total_weight": round(
                total_weight,
                4,
            ),
            "consensus_ratio": round(
                consensus_ratio,
                4,
            ),
            "action_support": {
                action: round(
                    support,
                    4,
                )
                for action, support
                in action_support.items()
            },
            "action_count": dict(
                action_count
            ),
        }

    return profile


def _apply_constraints_after_adaptation(
    decision: dict[str, Any],
    candidate_action: str,
    constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Re-apply O4 operational constraints after O5
    changes the recommendation.
    """

    result = dict(decision)

    result["candidate_action"] = (
        candidate_action
    )

    if not constraints:
        return result

    constrained = check_constraints(
        decision=result,
        constraints=constraints,
    )

    return constrained


def refine_recommendation(
    decision: dict[str, Any],
    feedback_records: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    min_support: int = MIN_SUPPORT,
    min_weighted_support: float = MIN_WEIGHTED_SUPPORT,
    min_consensus_ratio: float = MIN_CONSENSUS_RATIO,
) -> dict[str, Any]:
    """
    Apply conservative human-feedback adaptation.

    Adaptation occurs only when:
        1. An alternative action has repeated support.
        2. Confidence-weighted support is sufficient.
        3. Weighted consensus is sufficiently strong.

    Safety-critical O4 recommendations are never automatically
    weakened.
    """

    result = dict(decision)

    original_action = str(
        decision.get(
            "recommended_action",
            "",
        )
    ).upper()

    result["original_action"] = (
        original_action
    )

    result["adaptation_applied"] = False

    result["adaptation_reason"] = None

    result["adaptation_support"] = {
        "count": 0,
        "weighted_support": 0.0,
        "consensus_ratio": 0.0,
    }

    # ---------------------------------------------------------------
    # Validate current action
    # ---------------------------------------------------------------

    if original_action not in VALID_ACTIONS:
        result["adaptation_reason"] = (
            "Invalid original recommendation."
        )
        return result

    # ---------------------------------------------------------------
    # Safety protection
    # ---------------------------------------------------------------

    if (
        original_action
        in SAFETY_CRITICAL_ACTIONS
    ):
        result["adaptation_reason"] = (
            "Safety-critical action protected "
            "from automatic weakening."
        )
        return result

    # ---------------------------------------------------------------
    # Find relevant historical feedback
    # ---------------------------------------------------------------

    decision_for_matching = dict(decision)
    if feedback_records:
        decision_for_matching.setdefault("subset", feedback_records[0].get("subset", "UNKNOWN"))
        decision_for_matching.setdefault("model", feedback_records[0].get("model", "UNKNOWN"))
    key = _state_key(decision_for_matching)

    relevant = [
        record
        for record in feedback_records
        if _state_key(record) == key
    ]

    if not relevant:
        result["adaptation_reason"] = (
            "No historical expert feedback "
            "for this decision state."
        )
        return result

    # ---------------------------------------------------------------
    # Aggregate expert actions
    # ---------------------------------------------------------------

    action_support: dict[str, float] = (
        defaultdict(float)
    )

    action_count: dict[str, int] = (
        defaultdict(int)
    )

    for record in relevant:

        expert_action = str(
            record.get(
                "expert_action",
                "",
            )
        ).upper()

        if expert_action not in VALID_ACTIONS:
            continue

        weight = _feedback_weight(
            record
        )

        action_support[
            expert_action
        ] += weight

        action_count[
            expert_action
        ] += 1

    if not action_support:
        result["adaptation_reason"] = (
            "No valid expert actions available."
        )
        return result

    # ---------------------------------------------------------------
    # Select strongest action
    # ---------------------------------------------------------------

    preferred_action = max(
        action_support,
        key=action_support.get,
    )

    support = action_support[
        preferred_action
    ]

    count = action_count[
        preferred_action
    ]

    total_weight = sum(
        action_support.values()
    )

    consensus_ratio = (
        support / total_weight
        if total_weight > 0
        else 0.0
    )

    result["adaptation_support"] = {
        "count": count,
        "weighted_support": round(
            support,
            4,
        ),
        "consensus_ratio": round(
            consensus_ratio,
            4,
        ),
    }

    # ---------------------------------------------------------------
    # Existing policy already agrees
    # ---------------------------------------------------------------

    if preferred_action == original_action:
        result["adaptation_reason"] = (
            "Historical expert feedback "
            "supports the existing action."
        )
        return result

    # ---------------------------------------------------------------
    # Minimum observation requirement
    # ---------------------------------------------------------------

    if count < min_support:
        result["adaptation_reason"] = (
            f"Insufficient expert support: "
            f"{count}/{min_support}."
        )
        return result

    # ---------------------------------------------------------------
    # Minimum confidence-weighted support
    # ---------------------------------------------------------------

    if support < min_weighted_support:
        result["adaptation_reason"] = (
            "Insufficient confidence-weighted "
            "expert support."
        )
        return result

    # ---------------------------------------------------------------
    # Consensus requirement
    # ---------------------------------------------------------------

    if consensus_ratio < min_consensus_ratio:
        result["adaptation_reason"] = (
            f"Insufficient expert consensus: "
            f"{consensus_ratio:.3f} < "
            f"{min_consensus_ratio:.3f}."
        )
        return result

    # ---------------------------------------------------------------
    # Candidate adaptation
    # ---------------------------------------------------------------

    constrained = (
        _apply_constraints_after_adaptation(
            decision=result,
            candidate_action=preferred_action,
            constraints=constraints,
        )
    )

    final_action = str(
        constrained.get(
            "candidate_action",
            preferred_action,
        )
    ).upper()

    # Never allow the policy updater to weaken urgent action.
    if (
        original_action
        == "URGENT_MAINTENANCE"
        and final_action
        != "URGENT_MAINTENANCE"
    ):
        result["adaptation_reason"] = (
            "Constraint processing attempted "
            "to weaken a safety-critical action; "
            "adaptation rejected."
        )
        return result

    # ---------------------------------------------------------------
    # Apply adaptation
    # ---------------------------------------------------------------

    result["recommended_action"] = (
        final_action
    )

    result["adaptation_applied"] = (
        final_action != original_action
    )

    result["adaptation_reason"] = (
        f"Expert feedback refined "
        f"{original_action} to {final_action} "
        f"with {count} supporting observations, "
        f"weighted support={support:.3f}, "
        f"consensus={consensus_ratio:.3f}."
    )

    result["human_review"] = True

    # Preserve the constraint audit generated after adaptation.
    if constraints:
        result["constraint_status"] = (
            constrained.get(
                "constraint_status",
                result.get(
                    "constraint_status",
                    "SATISFIED",
                ),
            )
        )

        result["constraint_violations"] = (
            constrained.get(
                "constraint_violations",
                result.get(
                    "constraint_violations",
                    [],
                ),
            )
        )

        result["constraint_adjustments"] = (
            constrained.get(
                "constraint_adjustments",
                result.get(
                    "constraint_adjustments",
                    [],
                ),
            )
        )

    return result


def update_model(
    model: Any,
    feedback: Any,
) -> Any:
    """
    Backward-compatible entry point.

    Despite the historical name, this function does NOT modify
    prognostic model parameters.
    """

    if not isinstance(model, dict):
        return model

    if not isinstance(feedback, list):
        return model

    return refine_recommendation(
        decision=model,
        feedback_records=feedback,
    )


def update_from_log(
    decision: dict[str, Any],
    log_path: Optional[str] = None,
    *,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine one decision using accumulated HITL feedback."""

    records = load_feedback(log_path)

    return refine_recommendation(
        decision=decision,
        feedback_records=records,
        constraints=constraints,
    )