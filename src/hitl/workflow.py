"""Small, auditable human-in-the-loop review workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

REVIEW_ACTIONS = {"APPROVE", "OVERRIDE", "REJECT"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_review_queue(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create pending review records for decisions requiring a human."""
    queue = []
    for index, decision in enumerate(decisions):
        if not decision.get("human_review", False):
            continue
        identifier = decision.get("engine_id", decision.get("index", index))
        queue.append({
            "review_id": f"decision-{identifier}-{index}",
            "status": "PENDING",
            "created_at": _timestamp(),
            "reviewer": None,
            "review_action": None,
            "override_action": None,
            "comment": None,
            "final_action": None,
            "decision": decision,
        })
    return queue


def resolve_review(
    review: dict[str, Any],
    reviewer: str,
    review_action: str,
    comment: str = "",
    override_action: str | None = None,
) -> dict[str, Any]:
    """Resolve a pending review and calculate its final action."""
    action = review_action.upper()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"review_action must be one of {sorted(REVIEW_ACTIONS)}")
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    if action == "OVERRIDE" and not override_action:
        raise ValueError("override_action is required for OVERRIDE")

    decision = review.get("decision", {})
    if action == "OVERRIDE":
        final_action = override_action
    elif action == "APPROVE":
        final_action = decision.get("recommended_action")
    else:
        final_action = "REJECTED"

    review.update({
        "status": "RESOLVED",
        "reviewer": reviewer,
        "review_action": action,
        "override_action": override_action,
        "comment": comment,
        "final_action": final_action,
        "reviewed_at": _timestamp(),
    })
    return review


def summarize_review_queue(
    decisions: list[dict[str, Any]],
    queue: list[dict[str, Any]],
) -> dict[str, int]:
    """Return counts suitable for JSON output and HTML reporting."""
    return {
        "decision_count": len(decisions),
        "human_review_count": sum(
            1 for decision in decisions if decision.get("human_review", False)
        ),
        "pending_review_count": sum(
            1 for review in queue if review.get("status") == "PENDING"
        ),
        "resolved_review_count": sum(
            1 for review in queue if review.get("status") == "RESOLVED"
        ),
    }
