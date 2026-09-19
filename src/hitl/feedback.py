"""
Human-in-the-loop feedback representation and validation.

O5 captures expert validation/override decisions generated from O4.
The feedback is kept independent of the prognostic model so that
human feedback refines decision support rather than directly
modifying the GRU/RUL/HI model.

author: me-intenzo
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional


VALID_ACTIONS = {
    "CONTINUE_OPERATION",
    "MONITOR",
    "INSPECT",
    "SCHEDULE_MAINTENANCE",
    "URGENT_MAINTENANCE",
}


@dataclass
class ExpertFeedback:
    """
    Structured expert feedback for one O4 recommendation.
    """

    engine_id: Any
    subset: str
    model: str

    ai_action: str
    expert_action: str

    decision_valid: bool
    override: bool

    expert_confidence: float
    reason: str

    risk_score: Optional[float] = None
    urgency_index: Optional[float] = None
    health_state: Optional[str] = None
    uncertainty_level: Optional[str] = None
    explanation_reliability: Optional[str] = None

    timestamp: str = ""

    def __post_init__(self) -> None:
        self.subset = str(self.subset).upper()
        self.model = str(self.model).lower()

        if self.ai_action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid AI action: {self.ai_action}"
            )

        if self.expert_action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid expert action: {self.expert_action}"
            )

        if not 0.0 <= float(self.expert_confidence) <= 1.0:
            raise ValueError(
                "expert_confidence must be in [0, 1]."
            )

        self.expert_confidence = float(self.expert_confidence)

        # Override is determined from the actual actions rather than
        # trusting an independently supplied boolean.
        self.override = self.ai_action != self.expert_action

        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Return the feedback as a serializable dictionary."""
        return asdict(self)


def create_feedback(
    *,
    engine_id: Any,
    subset: str,
    model: str,
    ai_action: str,
    expert_action: str,
    expert_confidence: float,
    reason: str,
    risk_score: Optional[float] = None,
    urgency_index: Optional[float] = None,
    health_state: Optional[str] = None,
    uncertainty_level: Optional[str] = None,
    explanation_reliability: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create and validate a structured expert feedback record.

    decision_valid is automatically derived from whether the expert
    agrees with the AI recommendation.
    """

    ai_action = str(ai_action).upper()
    expert_action = str(expert_action).upper()

    feedback = ExpertFeedback(
        engine_id=engine_id,
        subset=subset,
        model=model,
        ai_action=ai_action,
        expert_action=expert_action,
        decision_valid=(ai_action == expert_action),
        override=(ai_action != expert_action),
        expert_confidence=expert_confidence,
        reason=str(reason).strip(),
        risk_score=risk_score,
        urgency_index=urgency_index,
        health_state=health_state,
        uncertainty_level=uncertainty_level,
        explanation_reliability=explanation_reliability,
    )

    return feedback.to_dict()


def validate_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """
    Validate an externally supplied feedback dictionary.

    Returns a normalized copy.
    """

    required = {
        "engine_id",
        "subset",
        "model",
        "ai_action",
        "expert_action",
        "expert_confidence",
        "reason",
    }

    missing = required.difference(feedback.keys())

    if missing:
        raise ValueError(
            f"Missing required feedback fields: {sorted(missing)}"
        )

    return create_feedback(
        engine_id=feedback["engine_id"],
        subset=feedback["subset"],
        model=feedback["model"],
        ai_action=feedback["ai_action"],
        expert_action=feedback["expert_action"],
        expert_confidence=feedback["expert_confidence"],
        reason=feedback["reason"],
        risk_score=feedback.get("risk_score"),
        urgency_index=feedback.get("urgency_index"),
        health_state=feedback.get("health_state"),
        uncertainty_level=feedback.get("uncertainty_level"),
        explanation_reliability=feedback.get(
            "explanation_reliability"
        ),
    )


def store_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize feedback.

    Persistence is handled by logger.py.
    """

    if isinstance(feedback, ExpertFeedback):
        feedback = feedback.to_dict()

    if not isinstance(feedback, dict):
        raise TypeError("feedback must be a dictionary.")

    return validate_feedback(feedback)