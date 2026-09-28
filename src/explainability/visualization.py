"""
Visualization and aggregation utilities for XAI.

Supported explanations
----------------------
1. Integrated Gradients
2. SHAP
3. Temporal attention/relevance

This module is responsible for converting raw explanation
outputs into interpretable PHM visualizations.

The predictor architecture is not modified here.

author: me-intenzo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def _to_numpy(value) -> np.ndarray:
    """Convert PyTorch tensors or array-like values to NumPy."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def normalize_importance(
    values,
) -> np.ndarray:
    """
    Normalize absolute importance values to sum to one.
    """

    values = _to_numpy(values).astype(
        np.float64
    )

    values = np.abs(values)

    total = values.sum()

    if total <= 1e-12:
        return np.zeros_like(values)

    return values / total


def _ensure_output_dir(
    output_dir,
) -> Path:
    """Create and return the visualization directory."""

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ---------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------

def aggregate_ig_sensor_importance(
    explanation: dict,
) -> np.ndarray:
    """
    Aggregate IG attribution into sensor-level importance.

    Raw IG:
        (batch, timesteps, features)

    Output:
        (features,)
    """

    attributions = _to_numpy(
        explanation["attributions"]
    )

    if attributions.ndim != 3:
        raise ValueError(
            "IG attributions must have shape "
            "(batch, timesteps, features)."
        )

    importance = np.abs(
        attributions
    ).mean(axis=(0, 1))

    return normalize_importance(
        importance
    )


def aggregate_ig_temporal_importance(
    explanation: dict,
) -> np.ndarray:
    """
    Aggregate IG attribution into timestep-level importance.

    Output:
        (timesteps,)
    """

    attributions = _to_numpy(
        explanation["attributions"]
    )

    if attributions.ndim != 3:
        raise ValueError(
            "IG attributions must have shape "
            "(batch, timesteps, features)."
        )

    importance = np.abs(
        attributions
    ).mean(axis=(0, 2))

    return normalize_importance(
        importance
    )


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def aggregate_shap_sensor_importance(
    explanation: dict,
) -> np.ndarray:
    """
    Aggregate sensor-level SHAP contributions.

    Expected:
        (batch, features)
    """

    values = _to_numpy(
        explanation["sensor_shap_values"]
    )

    if values.ndim != 2:
        raise ValueError(
            "SHAP sensor values must have shape "
            "(batch, features)."
        )

    importance = np.abs(
        values
    ).mean(axis=0)

    return normalize_importance(
        importance
    )


def aggregate_shap_temporal_importance(
    explanation: dict,
) -> np.ndarray | None:
    """
    Aggregate SHAP temporal contributions.

    Expected:
        (batch, timesteps, features)

    Returns None when temporal SHAP values are unavailable
    (sensor-level SHAP only).
    """

    raw = explanation.get("temporal_shap_values")

    if raw is None:
        return None

    values = _to_numpy(raw)

    if values.ndim != 3:
        raise ValueError(
            "SHAP temporal values must have shape "
            "(batch, timesteps, features)."
        )

    importance = np.abs(values).mean(axis=2).mean(axis=0)

    return normalize_importance(importance)


# ---------------------------------------------------------------------
# Attention / temporal relevance
# ---------------------------------------------------------------------

def aggregate_attention_temporal_importance(
    explanation: dict,
) -> np.ndarray:
    """
    Aggregate temporal attention/relevance.

    Expected:
        (batch, timesteps)
    """

    relevance = _to_numpy(
        explanation["temporal_relevance"]
    )

    if relevance.ndim != 2:
        raise ValueError(
            "Temporal relevance must have shape "
            "(batch, timesteps)."
        )

    importance = relevance.mean(
        axis=0
    )

    return normalize_importance(
        importance
    )


# ---------------------------------------------------------------------
# Sensor importance plot
# ---------------------------------------------------------------------

def plot_sensor_importance(
    ig_explanation: dict,
    shap_explanation: dict,
    sensor_names: list[str] | None = None,
    output_dir="outputs/xai",
    target="rul",
    top_k: int | None = None,
):
    """
    Compare IG and SHAP sensor importance.

    Returns
    -------
    dict
        Normalized IG and SHAP importance.
    """

    ig = aggregate_ig_sensor_importance(
        ig_explanation
    )

    shap = aggregate_shap_sensor_importance(
        shap_explanation
    )

    if len(ig) != len(shap):
        raise ValueError(
            "IG and SHAP must contain the "
            "same number of sensors."
        )

    n_features = len(ig)

    if sensor_names is None:
        sensor_names = [
            f"S{i + 1}"
            for i in range(n_features)
        ]

    if len(sensor_names) != n_features:
        raise ValueError(
            "sensor_names length must match "
            "the number of features."
        )

    indices = np.arange(
        n_features
    )

    if top_k is not None:
        top_k = min(
            max(int(top_k), 1),
            n_features,
        )

        combined = (
            ig + shap
        ) / 2.0

        indices = np.argsort(
            combined
        )[::-1][:top_k]

    labels = [
        sensor_names[i]
        for i in indices
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    width = 0.38

    positions = np.arange(
        len(indices)
    )

    ax.bar(
        positions - width / 2,
        ig[indices],
        width,
        label="Integrated Gradients",
    )

    ax.bar(
        positions + width / 2,
        shap[indices],
        width,
        label="SHAP",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Normalized Importance"
    )

    ax.set_xlabel(
        "Sensor / Feature"
    )

    ax.set_title(
        f"Sensor Importance — {target.upper()}"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        _ensure_output_dir(
            output_dir
        )
        / f"sensor_importance_{target}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "ig": ig,
        "shap": shap,
        "output_path": str(
            output_path
        ),
    }


# ---------------------------------------------------------------------
# Temporal importance plot
# ---------------------------------------------------------------------

def plot_temporal_importance(
    ig_explanation: dict,
    shap_explanation: dict,
    attention_explanation: dict,
    output_dir="outputs/xai",
    target="rul",
):
    """
    Compare temporal importance from:

        IG
        SHAP
        Attention-style relevance
    """

    ig = aggregate_ig_temporal_importance(
        ig_explanation
    )

    shap = aggregate_shap_temporal_importance(
        shap_explanation
    )

    # Fall back to IG when temporal SHAP is unavailable
    # (sensor-level SHAP only).
    if shap is None:
        shap = ig

    attention = (
        aggregate_attention_temporal_importance(
            attention_explanation
        )
    )

    lengths = {
        len(ig),
        len(shap),
        len(attention),
    }

    if len(lengths) != 1:
        raise ValueError(
            "All temporal explanations must "
            "have the same number of timesteps."
        )

    timesteps = np.arange(
        1,
        len(ig) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        timesteps,
        shap,
        marker="s",
        label="SHAP",
        zorder=2,
    )

    ax.plot(
        timesteps,
        attention,
        marker="^",
        label="Temporal Relevance",
        zorder=3,
    )

    # Draw IG last so exact agreement with SHAP remains visible.
    ax.plot(
        timesteps,
        ig,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.2,
        label="Integrated Gradients",
        zorder=4,
    )

    ax.set_xlabel(
        "Timestep"
    )

    ax.set_ylabel(
        "Normalized Importance"
    )

    ax.set_title(
        f"Temporal Importance — {target.upper()}"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        _ensure_output_dir(
            output_dir
        )
        / f"temporal_importance_{target}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "ig": ig,
        "shap": shap,
        "temporal_relevance": attention,
        "output_path": str(
            output_path
        ),
    }


# ---------------------------------------------------------------------
# Sensor × timestep heatmap
# ---------------------------------------------------------------------

def plot_ig_heatmap(
    ig_explanation: dict,
    sensor_names: list[str] | None = None,
    output_dir="outputs/xai",
    target="rul",
):
    """
    Plot the raw IG attribution structure:

        timestep × sensor

    This is the most direct temporal-feature explanation
    produced by Integrated Gradients.
    """

    attributions = _to_numpy(
        ig_explanation["attributions"]
    )

    if attributions.ndim != 3:
        raise ValueError(
            "IG attributions must have shape "
            "(batch, timesteps, features)."
        )

    # Aggregate across samples.

    heatmap = np.mean(
        attributions,
        axis=0,
    )

    n_timesteps, n_features = (
        heatmap.shape
    )

    if sensor_names is None:
        sensor_names = [
            f"S{i + 1}"
            for i in range(n_features)
        ]

    if len(sensor_names) != n_features:
        raise ValueError(
            "sensor_names length must match "
            "the number of features."
        )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    image = ax.imshow(
        heatmap.T,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel(
        "Timestep"
    )

    ax.set_ylabel(
        "Sensor / Feature"
    )

    ax.set_title(
        f"Integrated Gradients Attribution "
        f"Heatmap — {target.upper()}"
    )

    ax.set_yticks(
        np.arange(n_features)
    )

    ax.set_yticklabels(
        sensor_names
    )

    ax.set_xticks(
        np.arange(n_timesteps)
    )

    ax.set_xticklabels(
        np.arange(
            1,
            n_timesteps + 1,
        )
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Attribution",
    )

    fig.tight_layout()

    output_path = (
        _ensure_output_dir(
            output_dir
        )
        / f"ig_heatmap_{target}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "heatmap": heatmap,
        "output_path": str(
            output_path
        ),
    }


# ---------------------------------------------------------------------
# Unified visualization
# ---------------------------------------------------------------------

def plot_explanation(
    explanation,
    output_dir="outputs/xai",
):
    """
    Backward-compatible generic entry point.

    If a complete explanation bundle is provided, generate the
    available visualizations.

    Otherwise return the explanation unchanged.
    """

    if not isinstance(
        explanation,
        dict,
    ):
        return explanation

    required = {
        "ig",
        "shap",
        "temporal_relevance",
    }

    if not required.issubset(
        explanation.keys()
    ):
        return explanation

    target = explanation.get(
        "target",
        "rul",
    )

    results = {}

    results["sensor"] = (
        plot_sensor_importance(
            ig_explanation=explanation["ig"],
            shap_explanation=explanation["shap"],
            output_dir=output_dir,
            target=target,
        )
    )

    results["temporal"] = (
        plot_temporal_importance(
            ig_explanation=explanation["ig"],
            shap_explanation=explanation["shap"],
            attention_explanation=explanation["temporal_relevance"],
            output_dir=output_dir,
            target=target,
        )
    )

    results["heatmap"] = (
        plot_ig_heatmap(
            ig_explanation=explanation["ig"],
            output_dir=output_dir,
            target=target,
        )
    )

    return results