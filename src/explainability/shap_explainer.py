"""SHAP explanation helpers."""

from __future__ import annotations


def explain_with_shap(model, data):
    """Return a placeholder explanation."""
    return {"model": model, "data": data}
