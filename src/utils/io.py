"""I/O helpers."""

from __future__ import annotations


def ensure_parent_directory(path):
    """Create parent directories for a file path."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
