"""Integrated gradients helpers."""

from __future__ import annotations


def explain_with_integrated_gradients(model, data):
    """Return a placeholder explanation."""
    return {"model": model, "data": data}
