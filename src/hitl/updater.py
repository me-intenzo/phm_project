"""
Human-in-the-loop decision-policy refinement.

This does not retrain the prognostic model. Instead, accumulated expert
feedback is used to estimate systematic decision-policy corrections.

The updater is deliberately conservative:
    - insufficient feedback -> no adaptation
    - low-confidence feedback -> reduced influence
    - safety-critical actions are never weakened automatically
    - adaptation is based on repeated evidence rather than one override

author: me-intenzo
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from src.hitl.feedback import VALID_ACTIONS
from src.hitl.logger import load_feedback


# Minimum number of consistent expert observations required before
# automatically changing a recommendation.
MIN_SUPPORT = 3

# Minimum weighted expert support required for a refinement.
MIN_WEIGHTED_SUPPORT = 2.0

# Safety-critical actions.
SAFETY_CRITICAL_ACTIONS = {
    "URGENT_MAINTENANCE",
}


def _feedback_weight(record: dict[str, Any]) -> float:
    """
    Weight feedback according to expert confidence.
    """

    confidence = float(
        record.get("expert_confidence", 0.0)
    )

    return max(0.0, min(1.0, confidence))


def _state_key(
    record: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    """
    Build a coarse decision-state key.

    This avoids learning from engine identity alone and instead
    groups feedback by the decision context.
    """

    return (
        str(record.get("subset", "UNKNOWN")).upper(),
        str(record.get("model", "UNKNOWN")).lower(),
        str(record.get("health_state", "UNKNOWN")).upper(),
        str(record.get("uncertainty_level", "UNKNOWN")).upper(),
        str(record.get("explanation_reliability", "UNKNOWN")).upper(),
    )


def build_policy_profile(
    feedback_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build an empirical policy-refinement profile from expert feedback.

    For each decision-state group, estimate which expert action has
    the strongest support.
    """

    groups: dict[
        tuple[str, str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in feedback_records:
        groups[_state_key(record)].append(record)

    profile: dict[str, Any] = {}

    for key, records in groups.items():

        action_support: dict[str, float] = defaultdict(float)
        action_count: dict[str, int] = defaultdict(int)

        for record in records:
            action = str(
                record.get("expert_action", "")
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

        profile[str(key)] = {
            "preferred_action": preferred_action,
            "support": action_support[preferred_action],
            "count": action_count[preferred_action],
            "action_support": dict(action_support),
            "action_count": dict(action_count),
        }

    return profile


def refine_recommendation(
    decision: dict[str, Any],
    feedback_records: list[dict[str, Any]],
    *,
    min_support: int = MIN_SUPPORT,
    min_weighted_support: float = MIN_WEIGHTED_SUPPORT,
) -> dict[str, Any]:
    """
    Apply evidence-based human feedback to an O4 recommendation.

    The original O4 decision remains available in
    'original_action'.

    A refinement is applied only when sufficient expert evidence
    supports an alternative action.
    """

    result = dict(decision)

    original_action = str(
        decision.get("recommended_action", "")
    ).upper()

    result["original_action"] = original_action
    result["adaptation_applied"] = False
    result["adaptation_reason"] = None

    if original_action not in VALID_ACTIONS:
        return result

    # Never automatically weaken an urgent maintenance decision.
    if original_action in SAFETY_CRITICAL_ACTIONS:
        result["adaptation_reason"] = (
            "Safety-critical action protected from automatic weakening."
        )
        return result

    key = _state_key(decision)

    relevant = [
        record
        for record in feedback_records
        if _state_key(record) == key
    ]

    if not relevant:
        result["adaptation_reason"] = (
            "No historical expert feedback for this decision state."
        )
        return result

    action_support: dict[str, float] = defaultdict(float)
    action_count: dict[str, int] = defaultdict(int)

    for record in relevant:

        expert_action = str(
            record.get("expert_action", "")
        ).upper()

        if expert_action not in VALID_ACTIONS:
            continue

        action_support[expert_action] += _feedback_weight(record)
        action_count[expert_action] += 1

    if not action_support:
        result["adaptation_reason"] = (
            "No valid expert actions available."
        )
        return result

    preferred_action = max(
        action_support,
        key=action_support.get,
    )

    support = action_support[preferred_action]
    count = action_count[preferred_action]

    # No refinement when experts agree with the existing policy.
    if preferred_action == original_action:
        result["adaptation_reason"] = (
            "Historical expert feedback supports the existing action."
        )
        return result

    # Require repeated evidence.
    if count < min_support:
        result["adaptation_reason"] = (
            f"Insufficient expert support: {count}/{min_support}."
        )
        return result

    if support < min_weighted_support:
        result["adaptation_reason"] = (
            "Insufficient confidence-weighted expert support."
        )
        return result

    # Apply the refinement.
    result["recommended_action"] = preferred_action
    result["adaptation_applied"] = True
    result["adaptation_reason"] = (
        f"Expert feedback refined {original_action} "
        f"to {preferred_action} with "
        f"{count} supporting observations "
        f"(weighted support={support:.3f})."
    )

    result["human_review"] = True

    return result


def update_model(
    model: Any,
    feedback: Any,
) -> Any:
    """
    Backward-compatible updater entry point.

    This function intentionally does not modify the prognostic model.

    If `model` is a decision dictionary and `feedback` is a list of
    feedback records, the decision is refined.

    Otherwise the original model/object is returned unchanged.
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
) -> dict[str, Any]:
    """
    Refine one decision using the accumulated HITL log.
    """

    records = load_feedback(log_path)

    return refine_recommendation(
        decision=decision,
        feedback_records=records,
    )