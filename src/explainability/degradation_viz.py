"""
Temporal Degradation Visualization

Produces engine-level degradation trajectory plots combining:
  - Predicted HI over the engine's operational life
  - Temporal attention weight heatmap overlay
  - Predicted RUL vs true RUL curve
  - Multi-model comparison of degradation trajectories

These are the "temporal degradation visualization" outputs
required for the comprehensive XAI framework.

author: me-intenzo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _to_numpy(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


def plot_engine_degradation(
    hi_pred: np.ndarray,
    rul_pred: np.ndarray,
    rul_true: np.ndarray,
    temporal_relevance: np.ndarray,
    engine_id: int = 1,
    model_name: str = "model",
    output_dir: str = "outputs/xai",
    subset: str = "FD001",
) -> dict:
    """
    Plot a single engine's degradation trajectory.

    Layout (3-panel):
      Top    : HI prediction over time with degradation phases
      Middle : Temporal attention/relevance heatmap
      Bottom : RUL prediction vs true RUL

    Parameters
    ----------
    hi_pred           : (T,) predicted Health Index over engine life
    rul_pred          : (T,) predicted RUL over engine life
    rul_true          : (T,) true RUL over engine life
    temporal_relevance: (T,) or (T, F) mean temporal attention weights
    """
    hi_pred  = _to_numpy(hi_pred).ravel()
    rul_pred = _to_numpy(rul_pred).ravel()
    rul_true = _to_numpy(rul_true).ravel()
    rel      = _to_numpy(temporal_relevance)

    if rel.ndim == 2:
        rel = rel.mean(axis=1)
    rel = rel.ravel()

    T = len(hi_pred)
    cycles = np.arange(1, T + 1)

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45, height_ratios=[2, 0.8, 2])

    # ── Panel 1: HI trajectory ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cycles, hi_pred, color="#2196F3", lw=2, label="Predicted HI")
    ax1.axhline(0.5, color="#FF9800", ls="--", lw=1.2, label="Degradation threshold (0.5)")
    ax1.axhline(0.2, color="#F44336", ls="--", lw=1.2, label="Critical threshold (0.2)")
    ax1.fill_between(cycles, hi_pred, 0, alpha=0.12, color="#2196F3")
    ax1.set_ylabel("Health Index")
    ax1.set_title(
        f"Engine {engine_id} — Temporal Degradation Trajectory "
        f"[{model_name} | {subset}]",
        fontsize=11,
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Temporal relevance heatmap ─────────────────────────
    ax2 = fig.add_subplot(gs[1])
    rel_norm = rel / (rel.max() + 1e-12)
    ax2.imshow(
        rel_norm.reshape(1, -1),
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        extent=[1, T, 0, 1],
    )
    ax2.set_yticks([])
    ax2.set_xlabel("")
    ax2.set_ylabel("Attention", fontsize=8)
    ax2.set_title("Temporal Attention Relevance", fontsize=9)

    # ── Panel 3: RUL prediction vs true ─────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(cycles, rul_true, color="#4CAF50", lw=2, label="True RUL")
    ax3.plot(cycles, rul_pred, color="#E91E63", lw=2, ls="--", label="Predicted RUL")
    ax3.fill_between(
        cycles,
        rul_true,
        rul_pred,
        where=(rul_pred > rul_true),
        alpha=0.2,
        color="#F44336",
        label="Late prediction (unsafe)",
    )
    ax3.fill_between(
        cycles,
        rul_true,
        rul_pred,
        where=(rul_pred <= rul_true),
        alpha=0.15,
        color="#4CAF50",
        label="Early prediction (conservative)",
    )
    ax3.set_xlabel("Operational Cycle")
    ax3.set_ylabel("RUL (cycles)")
    ax3.set_title("RUL Prediction vs True RUL", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"degradation_engine{engine_id}_{model_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"output_path": str(out_path), "engine_id": engine_id}


def plot_multi_model_degradation(
    model_hi_dict: dict[str, np.ndarray],
    rul_true: np.ndarray,
    engine_id: int = 1,
    output_dir: str = "outputs/xai",
    subset: str = "FD001",
) -> dict:
    """
    Overlay HI trajectories from multiple models for one engine.

    Parameters
    ----------
    model_hi_dict : {model_name: hi_pred array (T,)}
    rul_true      : (T,) true RUL for secondary axis reference
    """
    rul_true = _to_numpy(rul_true).ravel()
    T = len(rul_true)
    cycles = np.arange(1, T + 1)

    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"hspace": 0.4})

    ax1 = axes[0]
    for idx, (mname, hi) in enumerate(model_hi_dict.items()):
        hi = _to_numpy(hi).ravel()
        ax1.plot(cycles[:len(hi)], hi, lw=2, color=colors[idx % len(colors)], label=mname)
    ax1.axhline(0.5, color="#FF9800", ls="--", lw=1, alpha=0.7)
    ax1.axhline(0.2, color="#F44336", ls="--", lw=1, alpha=0.7)
    ax1.set_ylabel("Health Index")
    ax1.set_title(
        f"Multi-Model HI Comparison — Engine {engine_id} [{subset}]",
        fontsize=11,
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.plot(cycles, rul_true, color="#4CAF50", lw=2.5, label="True RUL")
    ax2.set_xlabel("Operational Cycle")
    ax2.set_ylabel("RUL (cycles)")
    ax2.set_title("True RUL Reference", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multi_model_degradation_engine{engine_id}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"output_path": str(out_path), "engine_id": engine_id}


def plot_eri_radar(
    eri_results: dict[str, dict],
    output_dir: str = "outputs/xai",
    subset: str = "FD001",
    target: str = "rul",
) -> dict:
    """
    Radar chart comparing ERI components across models.

    Axes: ERI, ρ(IG,Attn), ρ(IG,SHAP), Consensus@K
    """
    model_names = list(eri_results.keys())
    metrics = ["ERI", "ρ(IG,Attn)", "ρ(IG,SHAP)", "Consensus@K"]
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    for idx, mname in enumerate(model_names):
        r = eri_results[mname]
        # Normalize rho from [-1,1] to [0,1]
        values = [
            r["eri"],
            (r["rho_ig_attn"] + 1) / 2,
            (r["rho_ig_shap"] + 1) / 2,
            r["consensus_k"],
        ]
        values += values[:1]
        ax.plot(angles, values, lw=2, color=colors[idx % len(colors)], label=mname)
        ax.fill(angles, values, alpha=0.08, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"ERI Radar — {subset} | {target.upper()}",
        fontsize=12,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eri_radar_{target}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"output_path": str(out_path)}
