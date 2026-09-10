"""
Persistent logging for human-in-the-loop feedback.

Feedback is stored as JSON Lines (JSONL), where every line represents
one expert interaction. This provides an append-only, auditable record
that can later be used by the O5 policy refinement mechanism.

author: me-intenzo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.hitl.feedback import store_feedback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "outputs" / "hitl" / "feedback.jsonl"


def log_feedback(
    feedback: dict[str, Any],
    log_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Validate and append one feedback record to the HITL log.
    """

    validated = store_feedback(feedback)

    path = (
        Path(log_path)
        if log_path is not None
        else DEFAULT_LOG_PATH
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                validated,
                ensure_ascii=False,
            )
            + "\n"
        )

    return validated


def load_feedback(
    log_path: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    """
    Load all valid feedback records from the JSONL log.
    """

    path = (
        Path(log_path)
        if log_path is not None
        else DEFAULT_LOG_PATH
    )

    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {path}"
                ) from exc

            records.append(record)

    return records


def feedback_statistics(
    log_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Calculate basic O5 feedback statistics.
    """

    records = load_feedback(log_path)

    if not records:
        return {
            "total_feedback": 0,
            "accepted": 0,
            "overrides": 0,
            "acceptance_rate": 0.0,
            "override_rate": 0.0,
        }

    accepted = sum(
        bool(record.get("decision_valid", False))
        for record in records
    )

    overrides = sum(
        bool(record.get("override", False))
        for record in records
    )

    total = len(records)

    return {
        "total_feedback": total,
        "accepted": accepted,
        "overrides": overrides,
        "acceptance_rate": accepted / total,
        "override_rate": overrides / total,
    }