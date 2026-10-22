"""
src/evaluation

Multi-dimensional evaluation framework for PHM-XAI.

Six dimensions
--------------
1. Predictive accuracy      metrics.predictive_accuracy
2. Uncertainty reliability  metrics.uncertainty_reliability
3. Explainability quality   metrics.explainability_quality
4. Decision effectiveness   metrics.decision_effectiveness
5. Robustness               robustness.evaluate_robustness
                            robustness.cross_subset_consistency
6. User trust               metrics.user_trust_proxies

Aggregate
---------
metrics.full_evaluation_report   — all six dimensions in one call

Comparative / ablation
----------------------
benchmark.comparative_benchmark
ablation.run_ablation
"""

from src.evaluation.ablation import run_ablation
from src.evaluation.benchmark import comparative_benchmark
from src.evaluation.metrics import (
    decision_effectiveness,
    decision_effectiveness_from_records,
    explainability_quality,
    full_evaluation_report,
    mae,
    nasa_score,
    predictive_accuracy,
    r2,
    rmse,
    uncertainty_reliability,
    user_trust_proxies,
    winkler_score,
)
from src.evaluation.robustness import cross_subset_consistency, evaluate_robustness

__all__ = [
    "predictive_accuracy",
    "mae",
    "rmse",
    "r2",
    "nasa_score",
    "winkler_score",
    "uncertainty_reliability",
    "explainability_quality",
    "decision_effectiveness",
    "decision_effectiveness_from_records",
    "user_trust_proxies",
    "full_evaluation_report",
    "comparative_benchmark",
    "run_ablation",
    "evaluate_robustness",
    "cross_subset_consistency",
]
