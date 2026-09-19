"""Persistence helpers for human-in-the-loop feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def store_feedback(feedback: dict[str, Any], path: str | Path) -> None:
    """Append one resolved review to a JSONL feedback file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(feedback, ensure_ascii=True) + "\n")
