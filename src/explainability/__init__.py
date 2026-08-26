"""Explainability package."""

from src.explainability.integrated_gradients import explain_with_integrated_gradients
from src.explainability.shap_explainer import explain_with_shap
from src.explainability.attention import explain_attention
from src.explainability.eri import compute_eri
from src.explainability.visualization import (
    plot_sensor_importance,
    plot_temporal_importance,
    plot_ig_heatmap,
    plot_explanation,
)
from src.explainability.degradation_viz import (
    plot_engine_degradation,
    plot_multi_model_degradation,
    plot_eri_radar,
)

__all__ = [
    "explain_with_integrated_gradients",
    "explain_with_shap",
    "explain_attention",
    "compute_eri",
    "plot_sensor_importance",
    "plot_temporal_importance",
    "plot_ig_heatmap",
    "plot_explanation",
    "plot_engine_degradation",
    "plot_multi_model_degradation",
    "plot_eri_radar",
]
