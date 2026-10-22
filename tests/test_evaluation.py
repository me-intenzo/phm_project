"""Regression tests for the evaluation framework.

These cover the failure modes that made the evaluation layer report
meaningless numbers:

- the ablation runner has to *re-run* the decision engine, not relabel the
  stored decisions;
- the ERI sensor importances are stored under ``temporal_relevance``, not
  ``temporal``;
- there is exactly one NASA-score implementation;
- every interval metric validates its input through the shared validator;
- the dropout robustness mask is per window, not global;
- the attention attribution runs through the eval-guarded gradient path.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.ablation import run_ablation
from src.evaluation.metrics import (
    decision_effectiveness_from_records,
    explainability_quality,
    nasa_score,
    per_window_dropout_mask,
    predictive_accuracy,
)
from src.uncertainty.coverage import (
    empirical_coverage,
    engine_level_coverage,
    mean_prediction_interval_width,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _decision_record(rul, hi, lower, upper, eri, **extra):
    """A decision record in the shape written by scripts/decision.py."""
    inputs = {
        "rul": rul,
        "hi": hi,
        "rul_lower": lower,
        "rul_upper": upper,
        "eri": eri,
    }
    return {
        "index": 0,
        "inputs": inputs,
        "health_state": "HEALTHY",
        "uncertainty_level": "LOW",
        "explanation_reliability": "HIGH",
        "risk_score": 0.1,
        "urgency_index": 0.1,
        "recommended_action": "CONTINUE_OPERATION",
        "human_review": False,
        "constraint_status": "SATISFIED",
        "constraint_violations": [],
        "constraint_adjustments": [],
        "top_k_sensors": [1, 2, 3],
        **extra,
    }


def _engine_decision(rul, hi, lower, upper, eri, constraints=None):
    """A decision record produced by the real decision engine."""
    from src.decision_engine.recommender import recommend_action

    return recommend_action(
        state={
            "rul": rul,
            "hi": hi,
            "rul_lower": lower,
            "rul_upper": upper,
            "eri": eri,
            "top_k_sensors": [1, 2, 3],
        },
        constraints=constraints,
    )


def _write_decision_json(tmp_path, records, constraints=None):
    path = tmp_path / "decision_FD001_gru.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subset": "FD001",
                "model": "gru",
                "summary": {},
                "decisions": records,
                "operational_constraints": constraints or {},
            },
            f,
        )
    return path


# ---------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------

class TestAblation:
    def _records(self):
        return [
            _engine_decision(110.0, 0.90, 100.0, 120.0, 0.90),
            _engine_decision(60.0, 0.60, 55.0, 65.0, 0.85),
            _engine_decision(20.0, 0.30, 15.0, 25.0, 0.80),
            _engine_decision(5.0, 0.10, 0.0, 15.0, 0.75),
        ]

    def test_variants_are_recomputed_not_relabelled(self, tmp_path):
        """The no_xai variant must actually change the decisions."""

        _write_decision_json(tmp_path, self._records())
        report = run_ablation(tmp_path, "FD001", "gru")

        assert report["no_xai"]["human_review_rate"] > report["full"]["human_review_rate"]
        assert report["delta_human_review_rate"]["no_xai_vs_full"] > 0.0

    def test_no_uncertainty_variant_widens_the_interval(self, tmp_path):
        """The no_uncertainty variant must feed a wide interval to the engine."""

        _write_decision_json(tmp_path, self._records())
        report = run_ablation(tmp_path, "FD001", "gru")

        # A fixed 100-cycle band is much wider than the conformal interval,
        # so the risk score must grow.
        assert (
            report["no_uncertainty"]["mean_risk_score"]
            > report["full"]["mean_risk_score"]
        )

    def test_no_constraints_variant_drops_the_constraint_layer(self, tmp_path):
        records = [
            _engine_decision(110.0, 0.90, 100.0, 120.0, 0.90),
            _engine_decision(
                60.0, 0.60, 55.0, 65.0, 0.85,
                constraints={"minimum_safe_rul": 65.0},
            ),
        ]
        _write_decision_json(
            tmp_path,
            records,
            constraints={"minimum_safe_rul": 65.0},
        )
        report = run_ablation(tmp_path, "FD001", "gru")

        # With the constraint layer the passive action is escalated ...
        assert report["full"]["constraint_violation_rate"] > 0.0
        # ... without it the escalation disappears.
        assert report["no_constraints"]["constraint_violation_rate"] == 0.0
        assert (
            report["delta_constraint_violation_rate"]["no_constraints_vs_full"] < 0.0
        )

    def test_action_deltas_are_reported(self, tmp_path):
        _write_decision_json(tmp_path, self._records())
        report = run_ablation(tmp_path, "FD001", "gru")

        deltas = report["delta_action_counts"]["no_xai"]
        assert sum(deltas.values()) == 0
        assert any(count != 0 for count in deltas.values())

    def test_missing_artifact_is_reported(self, tmp_path):
        report = run_ablation(tmp_path, "FD001", "gru")
        assert "error" in report

    def test_full_variant_matches_the_stored_decisions(self, tmp_path):
        """The re-run has to reproduce the decisions that were stored."""

        _write_decision_json(tmp_path, self._records())
        report = run_ablation(tmp_path, "FD001", "gru")
        assert report["full"]["matches_stored_decisions"] is True


# ---------------------------------------------------------------------
# ERI sensor-importance keys
# ---------------------------------------------------------------------

class TestExplainabilityQuality:
    def _eri_json(self, tmp_path, sensor_importance):
        path = tmp_path / "eri_rul.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "eri": 0.77,
                    "rho_ig_attn": 0.5,
                    "rho_ig_shap": 0.6,
                    "consensus_k": 0.4,
                    "sensor_importance": sensor_importance,
                },
                f,
            )
        return path

    def test_temporal_relevance_key_is_read(self, tmp_path):
        """eri.py writes "temporal_relevance" — the reader must use that key."""

        from scipy.stats import spearmanr

        ig = [0.1, 0.2, 0.7]
        shap = [0.3, 0.2, 0.5]
        temporal = [0.9, 0.1, 0.4]

        path = self._eri_json(
            tmp_path,
            {
                "ig": ig,
                "shap": shap,
                "temporal_relevance": temporal,
            },
        )
        result = explainability_quality(path)

        assert "spearman_ig_temporal" in result
        assert "spearman_temporal_shap" in result
        assert result["spearman_ig_temporal"] == pytest.approx(
            float(spearmanr(ig, temporal).statistic)
        )
        assert result["spearman_temporal_shap"] == pytest.approx(
            float(spearmanr(temporal, shap).statistic)
        )

    def test_missing_key_does_not_raise(self, tmp_path):
        path = self._eri_json(tmp_path, {"ig": [0.1, 0.2], "shap": [0.3, 0.4]})
        result = explainability_quality(path)
        assert "spearman_ig_temporal" not in result


# ---------------------------------------------------------------------
# Single NASA-score implementation
# ---------------------------------------------------------------------

class TestNasaScore:
    def test_duplicate_evaluator_module_is_gone(self):
        assert importlib.util.find_spec("src.models.evaluator") is None

    def test_nasa_score_matches_the_published_formula(self):
        y_true = np.array([100.0, 50.0, 10.0])
        y_pred = np.array([90.0, 60.0, 5.0])

        errors = y_pred - y_true
        expected = float(np.sum(
            np.where(errors < 0, np.exp(-errors / 13.0) - 1.0, np.exp(errors / 10.0) - 1.0)
        ))
        assert nasa_score(y_true, y_pred) == pytest.approx(expected)

    def test_late_prediction_is_penalised_more_than_an_early_one(self):
        """The NASA score is asymmetric: the same error costs more when late."""

        early = nasa_score(np.array([100.0]), np.array([90.0]))
        late = nasa_score(np.array([100.0]), np.array([110.0]))
        assert late > early


# ---------------------------------------------------------------------
# Shared interval validation
# ---------------------------------------------------------------------

class TestIntervalValidation:
    def test_mpiw_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            mean_prediction_interval_width(np.array([0.0, 1.0]), np.array([1.0]))

    def test_mpiw_rejects_non_finite_bounds(self):
        with pytest.raises(ValueError, match="finite"):
            mean_prediction_interval_width(np.array([0.0, np.nan]), np.array([1.0, 2.0]))

    def test_mpiw_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="Upper bounds"):
            mean_prediction_interval_width(np.array([2.0]), np.array([1.0]))

    def test_mpiw_rejects_empty_input(self):
        with pytest.raises(ValueError, match="At least one"):
            mean_prediction_interval_width(np.array([]), np.array([]))

    def test_engine_level_coverage_rejects_mismatched_engine_ids(self):
        with pytest.raises(ValueError, match="engine_ids"):
            engine_level_coverage(
                y_true=np.array([1.0, 2.0]),
                lower=np.array([0.0, 0.0]),
                upper=np.array([3.0, 3.0]),
                engine_ids=np.array([1]),
            )

    def test_engine_level_coverage_rejects_invalid_bounds(self):
        with pytest.raises(ValueError, match="Upper bounds"):
            engine_level_coverage(
                y_true=np.array([1.0, 2.0]),
                lower=np.array([3.0, 0.0]),
                upper=np.array([1.0, 3.0]),
                engine_ids=np.array([1, 1]),
            )

    def test_empirical_coverage_and_engine_coverage_agree_on_valid_input(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.0, 1.0, 2.0, 5.0])
        upper = np.array([2.0, 3.0, 4.0, 6.0])
        engine_ids = np.array([1, 1, 2, 2])

        assert empirical_coverage(y_true, lower, upper) == pytest.approx(0.75)
        assert engine_level_coverage(y_true, lower, upper, engine_ids) == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Per-window dropout
# ---------------------------------------------------------------------

class TestDropoutMask:
    def test_every_window_drops_the_same_number_of_sensors(self):
        rng = np.random.default_rng(0)
        mask = per_window_dropout_mask((4, 10, 14), 3, rng)

        assert mask.shape == (4, 10, 14)
        assert np.all(mask.sum(axis=2) == 3)

    def test_mask_is_not_a_single_global_sensor_set(self):
        """The old implementation dropped the same sensors in every window."""

        rng = np.random.default_rng(1)
        mask = per_window_dropout_mask((8, 20, 14), 3, rng)

        dropped_per_window = [set(np.flatnonzero(row)) for row in mask[0]]
        assert len(set(map(frozenset, dropped_per_window))) > 1

    def test_at_least_one_sensor_stays_alive(self):
        rng = np.random.default_rng(2)
        mask = per_window_dropout_mask((3, 5, 4), 3, rng)

        assert np.all(mask.sum(axis=2) <= 3)
        assert np.all(~mask.all(axis=2))


# ---------------------------------------------------------------------
# Attention attribution (eval-guarded)
# ---------------------------------------------------------------------

class TestExplainAttention:
    def _model_with_batchnorm(self, input_size=14, window=40):
        from src.models.gru import GRUPrognosticsModel

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
                normed = self.bn(x.transpose(1, 2)).transpose(1, 2)
                return self.gru(normed)

        return _WithBatchNorm()

    def test_attention_does_not_update_batchnorm_statistics(self):
        from src.explainability.integrated_gradients import explain_attention

        torch.manual_seed(0)
        model = self._model_with_batchnorm()
        model.eval()

        bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm1d))
        before = bn.running_var.clone()

        explain_attention(model=model, data=torch.randn(4, 40, 14), target="rul")

        assert torch.allclose(before, bn.running_var)
        assert int(bn.num_batches_tracked) == 0

    def test_attention_is_deterministic(self):
        from src.explainability.integrated_gradients import explain_attention

        torch.manual_seed(0)
        model = self._model_with_batchnorm()
        model.eval()

        x = torch.randn(4, 40, 14)
        first = explain_attention(model=model, data=x, target="rul")["temporal_relevance"]
        second = explain_attention(model=model, data=x, target="rul")["temporal_relevance"]

        assert torch.allclose(first, second, atol=1e-6)

    def test_attention_preserves_the_model_mode(self):
        from src.explainability.integrated_gradients import explain_attention

        torch.manual_seed(0)
        model = self._model_with_batchnorm()
        x = torch.randn(2, 40, 14)

        model.eval()
        explain_attention(model=model, data=x, target="rul")
        assert model.training is False

        model.train()
        explain_attention(model=model, data=x, target="rul")
        assert model.training is True

    def test_temporal_relevance_is_normalised_per_sample(self):
        from src.explainability.integrated_gradients import explain_attention

        torch.manual_seed(0)
        model = self._model_with_batchnorm()
        model.eval()

        relevance = explain_attention(
            model=model, data=torch.randn(4, 40, 14), target="rul"
        )["temporal_relevance"]

        assert relevance.shape == (4, 40)
        assert torch.allclose(relevance.sum(dim=1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------
# Decision effectiveness from records
# ---------------------------------------------------------------------

def test_decision_effectiveness_from_records_counts_reviews():
    records = [
        _decision_record(110.0, 0.9, 100.0, 120.0, 0.9, human_review=True),
        _decision_record(5.0, 0.1, 0.0, 15.0, 0.9, human_review=False),
    ]
    result = decision_effectiveness_from_records(records)

    assert result["n_decisions"] == 2
    assert result["human_review_rate"] == pytest.approx(0.5)


def test_predictive_accuracy_reports_the_nasa_score():
    result = predictive_accuracy(
        np.array([100.0, 50.0]),
        np.array([95.0, 55.0]),
        np.array([1.0, 0.5]),
        np.array([0.9, 0.55]),
    )
    assert "NASA_Score" in result["rul"]
    assert result["rul"]["NASA_Score"] > 0.0
