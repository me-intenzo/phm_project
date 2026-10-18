import numpy as np
import pytest
import torch

from src.evaluation.explainability_metrics import (
    correlation_statistics,
    explanation_agreement,
    feature_ablation_faithfulness,
)
from scripts.benchmark_explainability import _summarize


class SensorSumModel(torch.nn.Module):
    def forward(self, x):
        prediction = x.sum(dim=(1, 2))
        return prediction, prediction, prediction


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        prediction = torch.ones(x.shape[0], device=x.device)
        return prediction, prediction, prediction


def test_agreement_metrics_find_matching_rank_and_sign():
    metrics = explanation_agreement(
        {"ig": np.array([4.0, -3.0, 1.0]), "shap": np.array([2.0, -1.0, 0.5])},
        top_k=2,
    )["ig__shap"]

    assert metrics["rank_agreement_top_k"] == 1.0
    assert metrics["sign_agreement_top_k"] == 1.0
    assert metrics["rank_correlation_spearman"] == pytest.approx(1.0)


def test_agreement_metrics_validate_dimensions():
    with pytest.raises(ValueError, match="same length"):
        explanation_agreement({"ig": np.ones(2), "shap": np.ones(3)})


def test_ablation_faithfulness_ranks_relevant_sensor_first():
    model = SensorSumModel()
    sample = torch.tensor([[1.0, 10.0], [1.0, 10.0]])
    result = feature_ablation_faithfulness(
        model, sample, np.array([2.0, 20.0]), target="rul"
    )

    assert result["prediction_gap_top_50"] == pytest.approx(20.0)
    assert result["deletion_gap_auc"] > 0
    assert result["insertion_auc"] > 0


def test_faithfulness_auc_is_undefined_without_prediction_contrast():
    result = feature_ablation_faithfulness(
        ConstantModel(), torch.ones(2, 2), np.ones(2), target="rul"
    )

    assert np.isnan(result["deletion_gap_auc"])
    assert np.isnan(result["insertion_auc"])


def test_correlations_return_coefficients_and_significance():
    result = correlation_statistics(np.arange(10), np.arange(10) * 2)

    assert result["n"] == 10
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["pearson_p"] < 0.05


def test_benchmark_summarizes_engine_means_per_model_seed():
    rows = []
    for seed in (42, 43, 44):
        for engine_id in range(4):
            row = {"model_seed": seed, "engine_id": engine_id, "eri": float(engine_id)}
            for explainer in ("ig", "shap", "lime"):
                for metric in (
                    "prediction_gap_top_50",
                    "deletion_gap_auc",
                    "insertion_auc",
                ):
                    row[f"{explainer}_{metric}"] = float(engine_id + seed)
            rows.append(row)

    summary = _summarize(rows)

    per_seed = [
        result
        for result in summary
        if result["analysis_level"].startswith("engine-mean-seed-")
    ]
    assert {result["analysis_level"] for result in per_seed} == {
        "engine-mean-seed-42",
        "engine-mean-seed-43",
        "engine-mean-seed-44",
    }
    assert all(result["n"] == 4 for result in per_seed)
