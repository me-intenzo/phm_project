"""Regression tests for the explainability modules.

The important cases here are the ones that a placeholder test file cannot
catch: an attribution routine must not mutate the model it explains, and it
must be deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.integrated_gradients import integrated_gradients
from src.models.gru import GRUPrognosticsModel

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _model_with_batchnorm(input_size: int = 14, window: int = 40) -> nn.Module:
    """A GRU wrapper that also contains a BatchNorm layer.

    The point of the BatchNorm is that ``train()`` mode updates its running
    statistics, which would silently corrupt the model being explained.
    """

    class _WithBatchNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = GRUPrognosticsModel(
                input_size=input_size,
                hidden_size=16,
                num_layers=1,
                dropout=0.3,
            )
            self.bn = nn.BatchNorm1d(input_size)

        def forward(self, x: torch.Tensor) -> tuple:
            # BatchNorm1d expects (N, C, L); the windows are (N, L, C).
            normed = self.bn(x.transpose(1, 2)).transpose(1, 2)
            return self.gru(normed)

    return _WithBatchNorm()


def _batchnorm_of(model: nn.Module) -> nn.BatchNorm1d:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            return module
    raise AssertionError("model has no BatchNorm layer")


# ---------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------

def test_integrated_gradients_does_not_update_batchnorm_statistics():
    """Explaining a model must not change the model.

    Regression test: the attribution routine used to call
    ``model.train()``, which updates BatchNorm running statistics and
    permanently changes the weights that downstream stages then reuse.
    """

    torch.manual_seed(0)
    model = _model_with_batchnorm()
    model.eval()

    bn = _batchnorm_of(model)
    before = bn.running_var.clone()

    x = torch.randn(4, 40, 14)
    integrated_gradients(model, x, "rul", steps=5)

    assert torch.allclose(before, bn.running_var), (
        "BatchNorm running statistics were modified by the attribution call"
    )
    assert int(bn.num_batches_tracked) == 0, (
        "BatchNorm counted a training batch during an attribution call"
    )


def test_integrated_gradients_is_deterministic():
    """Two identical calls must return identical attributions.

    Regression test: with Dropout enabled (``train()`` mode) the two calls
    sampled different masks, so the attributions were not reproducible.
    """

    torch.manual_seed(0)
    model = GRUPrognosticsModel(
        input_size=14, hidden_size=16, num_layers=1, dropout=0.3
    )
    model.eval()

    x = torch.randn(4, 40, 14)

    first = integrated_gradients(model, x, "rul", steps=8)
    second = integrated_gradients(model, x, "rul", steps=8)

    assert torch.allclose(first, second, atol=1e-6), (
        "integrated_gradients is not deterministic in eval mode"
    )


def test_integrated_gradients_preserves_the_model_mode():
    """The caller's train/eval mode must survive the call."""

    torch.manual_seed(0)
    model = GRUPrognosticsModel(
        input_size=14, hidden_size=16, num_layers=1, dropout=0.3
    )
    x = torch.randn(2, 40, 14)

    model.eval()
    integrated_gradients(model, x, "rul", steps=3)
    assert model.training is False

    model.train()
    integrated_gradients(model, x, "rul", steps=3)
    assert model.training is True


def test_integrated_gradients_works_inside_no_grad():
    """The routine must be usable from a ``torch.no_grad()`` block."""

    torch.manual_seed(0)
    model = GRUPrognosticsModel(
        input_size=14, hidden_size=16, num_layers=1, dropout=0.3
    )
    model.eval()
    x = torch.randn(2, 40, 14)

    with torch.no_grad():
        attributions = integrated_gradients(model, x, "rul", steps=3)

    assert attributions.shape == x.shape
    assert torch.all(torch.isfinite(attributions))


def test_integrated_gradients_rejects_invalid_steps():
    model = GRUPrognosticsModel(
        input_size=14, hidden_size=16, num_layers=1, dropout=0.3
    )
    with pytest.raises(ValueError):
        integrated_gradients(model, torch.randn(1, 40, 14), "rul", steps=0)


def test_integrated_gradients_rejects_mismatched_baseline():
    model = GRUPrognosticsModel(
        input_size=14, hidden_size=16, num_layers=1, dropout=0.3
    )
    x = torch.randn(2, 40, 14)
    with pytest.raises(ValueError):
        integrated_gradients(model, x, "rul", steps=2, baseline=torch.zeros(1, 40, 14))


def test_integrated_gradients_is_zero_when_the_model_ignores_the_input():
    """A model with a zero gradient w.r.t. the input must give zero attributions."""

    class _IgnoresInput(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Keep the graph connected so autograd can run, but the gradient
            # w.r.t. x is exactly zero.
            return torch.zeros(x.shape[0], 3) + x[..., 0:1].sum() * 0.0

    model = _IgnoresInput()
    x = torch.randn(4, 40, 14)
    attributions = integrated_gradients(model, x, "rul", steps=4)

    assert torch.allclose(attributions, torch.zeros_like(attributions))


# ---------------------------------------------------------------------
# ERI
# ---------------------------------------------------------------------

def _eri_inputs(
    n_batch: int = 4,
    n_steps: int = 40,
    n_sensors: int = 14,
    reverse: bool = False,
) -> tuple[dict, dict, dict]:
    """Build (IG, SHAP, attention) explanation payloads.

    When *reverse* is set, SHAP and the attention-derived importances rank
    the sensors in the opposite order to IG.
    """

    ig = np.zeros((n_batch, n_steps, n_sensors), dtype=np.float32)
    shap = np.zeros((n_batch, n_sensors), dtype=np.float32)

    for sensor in range(n_sensors):
        ig[:, :, sensor] = float(sensor + 1)
        shap[:, sensor] = float(n_sensors - sensor if reverse else sensor + 1)

    # Constant temporal relevance keeps the attention-derived importance in
    # the same sensor order as IG (it is a per-timestep weighting).
    relevance = np.ones((n_batch, n_steps), dtype=np.float32)

    return (
        {"attributions": ig},
        {"sensor_shap_values": shap},
        {"temporal_relevance": relevance},
    )


def test_eri_is_one_when_the_three_methods_agree():
    from src.explainability.eri import compute_eri

    result = compute_eri(*_eri_inputs())

    assert 0.0 <= result["eri"] <= 1.0
    assert result["eri"] == pytest.approx(1.0, abs=1e-6)
    assert result["rho_ig_attn"] == pytest.approx(1.0, abs=1e-6)
    assert result["rho_ig_shap"] == pytest.approx(1.0, abs=1e-6)
    assert result["consensus_k"] == pytest.approx(1.0, abs=1e-6)


def test_eri_is_zero_when_the_methods_disagree():
    from src.explainability.eri import compute_eri

    result = compute_eri(*_eri_inputs(reverse=True))

    # SHAP ranks the sensors in the opposite order to IG, so that pair
    # contributes 0 (rho = -1 -> normalised 0) and the top-K sets do not
    # intersect.  The attention-derived importance still follows IG because
    # the temporal relevance is constant, so that pair contributes 0.4.
    assert result["rho_ig_shap"] == pytest.approx(-1.0, abs=1e-6)
    assert result["rho_ig_attn"] == pytest.approx(1.0, abs=1e-6)
    assert result["consensus_k"] == pytest.approx(0.0, abs=1e-6)
    assert result["eri"] == pytest.approx(0.4, abs=1e-6)


def test_eri_matches_its_formula():
    """ERI must equal 0.4·n1 + 0.4·n2 + 0.2·consensus for any input."""

    from src.explainability.eri import compute_eri

    result = compute_eri(*_eri_inputs(reverse=True))

    expected = (
        0.4 * (result["rho_ig_attn"] + 1.0) / 2.0
        + 0.4 * (result["rho_ig_shap"] + 1.0) / 2.0
        + 0.2 * result["consensus_k"]
    )
    assert result["eri"] == pytest.approx(expected, abs=1e-9)


def test_eri_drops_when_the_attention_derived_importance_disagrees():
    """The attention-derived importance can disagree even when SHAP agrees.

    The attention-derived importance weights the IG attributions by the mean
    temporal relevance, so concentrating the relevance on a timestep where IG
    ranks the sensors the other way round flips the pair correlation.
    """

    from src.explainability.eri import compute_eri

    n_batch, n_steps, n_sensors = 4, 40, 14

    ig = np.zeros((n_batch, n_steps, n_sensors), dtype=np.float32)
    for sensor in range(n_sensors):
        ig[:, 0, sensor] = float(n_sensors - sensor)      # reversed at t=0
        ig[:, 1:, sensor] = float(1000 * (sensor + 1))      # dominant elsewhere

    shap = np.zeros((n_batch, n_sensors), dtype=np.float32)
    for sensor in range(n_sensors):
        shap[:, sensor] = float(sensor + 1)

    relevance = np.zeros((n_batch, n_steps), dtype=np.float32)
    relevance[:, 0] = 1.0

    result = compute_eri(
        {"attributions": ig},
        {"sensor_shap_values": shap},
        {"temporal_relevance": relevance},
    )

    assert result["rho_ig_attn"] == pytest.approx(-1.0, abs=1e-6)
    assert result["eri"] < 0.5


def test_eri_reports_sensor_ranks_and_an_interpretation():
    from src.explainability.eri import compute_eri

    result = compute_eri(*_eri_inputs())

    assert set(result["sensor_ranks"]) == {"ig", "shap", "temporal_relevance"}
    assert len(result["sensor_ranks"]["ig"]) == 14
    assert result["interpretation"]
