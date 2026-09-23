"""
Human-in-the-loop review workflow.

O5 workflow:
    O4 recommendation
        ↓
    human review
        ↓
    approve / override
        ↓
    structured expert feedback
        ↓
    policy refinement

author: me-intenzo
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.hitl.feedback import VALID_ACTIONS, create_feedback


REVIEW_ACTIONS = {
    "APPROVE",
    "OVERRIDE",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_review_queue(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create pending review records for O4 decisions
    that require human review.
    """

    queue: list[dict[str, Any]] = []

    for index, decision in enumerate(decisions):

        if not decision.get("human_review", False):
            continue

        identifier = decision.get(
            "engine_id",
            decision.get("index", index),
        )

        queue.append({
            "review_id": f"decision-{identifier}-{index}",
            "status": "PENDING",
            "created_at": _timestamp(),

            "reviewer": None,
            "review_action": None,
            "override_action": None,

            "expert_confidence": None,
            "comment": None,

            "final_action": None,

            "decision": decision,
        })

    return queue


def resolve_review(
    review: dict[str, Any],
    reviewer: str,
    review_action: str,
    *,
    expert_confidence: float,
    comment: str = "",
    override_action: str | None = None,
) -> dict[str, Any]:
    """
    Resolve a pending review.

    APPROVE:
        expert_action = AI recommendation

    OVERRIDE:
        expert_action = override_action
    """

    action = str(review_action).upper()

    if action not in REVIEW_ACTIONS:
        raise ValueError(
            f"review_action must be one of "
            f"{sorted(REVIEW_ACTIONS)}"
        )

    reviewer = str(reviewer).strip()

    if not reviewer:
        raise ValueError(
            "reviewer must not be empty."
        )

    confidence = float(expert_confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "expert_confidence must be in [0, 1]."
        )

    decision = review.get("decision", {})

    ai_action = str(
        decision.get("recommended_action", "")
    ).upper()

    if ai_action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid AI recommendation: {ai_action}"
        )

    if action == "APPROVE":
        expert_action = ai_action
        final_action = ai_action

    else:
        if not override_action:
            raise ValueError(
                "override_action is required for OVERRIDE."
            )

        expert_action = str(
            override_action
        ).upper()

        if expert_action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid override action: {expert_action}"
            )

        if expert_action == ai_action:
            raise ValueError(
                "Override action must differ from the AI action."
            )

        final_action = expert_action

    review.update({
        "status": "RESOLVED",
        "reviewer": reviewer,
        "review_action": action,
        "override_action": (
            expert_action
            if action == "OVERRIDE"
            else None
        ),
        "expert_action": expert_action,
        "expert_confidence": confidence,
        "comment": str(comment).strip(),
        "final_action": final_action,
        "reviewed_at": _timestamp(),
    })

    return review


def review_to_feedback(
    review: dict[str, Any],
    *,
    subset: str,
    model: str,
    source_decision_output: str | None = None,
) -> dict[str, Any]:
    """
    Convert a resolved review into the canonical O5 feedback schema.
    """

    if review.get("status") != "RESOLVED":
        raise ValueError(
            "Only resolved reviews can become feedback."
        )

    decision = review.get("decision", {})

    return create_feedback(
        engine_id=decision.get(
            "engine_id",
            decision.get("index"),
        ),
        subset=subset,
        model=model,
        ai_action=decision["recommended_action"],
        expert_action=review["expert_action"],
        expert_confidence=review["expert_confidence"],
        reason=review.get("comment", ""),
        risk_score=decision.get("risk_score"),
        urgency_index=decision.get("urgency_index"),
        health_state=decision.get("health_state"),
        uncertainty_level=decision.get("uncertainty_level"),
        explanation_reliability=decision.get(
            "explanation_reliability"
        ),
        review_id=review.get("review_id"),
        source_decision_output=source_decision_output,
    )


def summarize_review_queue(
    decisions: list[dict[str, Any]],
    queue: list[dict[str, Any]],
) -> dict[str, int]:
    """Return review workflow counts."""

    return {
        "decision_count": len(decisions),

        "human_review_count": sum(
            1
            for decision in decisions
            if decision.get("human_review", False)
        ),

        "pending_review_count": sum(
            1
            for review in queue
            if review.get("status") == "PENDING"
        ),

        "resolved_review_count": sum(
            1
            for review in queue
            if review.get("status") == "RESOLVED"
        ),
    }