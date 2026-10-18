"""Explainability behavior tests."""

import pytest
from scipy.stats import spearmanr
import torch

from src.explainability.eri import compute_eri
from src.explainability.integrated_gradients import integrated_gradients


class BatchNormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        prediction = self.normalization(x.mean(dim=1)).sum(dim=1)
        return prediction, prediction, prediction


def test_integrated_gradients_handles_single_sample_and_restores_mode():
    model = BatchNormModel().train()
    sample = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    attributions = integrated_gradients(model, sample, target="rul", steps=4)

    assert attributions.shape == sample.shape
    assert torch.isfinite(attributions).all()
    assert model.training


def test_eri_spearman_uses_average_ranks_for_ties():
    ig_values = torch.tensor([[[1.0, 1.0, 3.0, 4.0]]])
    shap_values = [[1.0, 2.0, 3.0, 4.0]]
    eri = compute_eri(
        {"attributions": ig_values},
        {"sensor_shap_values": shap_values},
        {"temporal_relevance": torch.ones(1, 1)},
        top_k=2,
    )

    assert eri["rho_ig_shap"] == pytest.approx(
        spearmanr([1.0, 1.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]).statistic
    )
