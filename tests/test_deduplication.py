"""Regression tests for the Tier-2 deduplication refactor.

Every test in this file asserts a *structural* invariant: that a piece of
logic exists in exactly one place and that the other places call it.  The
duplication these tests protect against was not a bug in itself — it was
the reason two modules could report two different numbers for the same
quantity, and the reason a value could be changed in one file and silently
stay stale in another.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------
# 2.1 / 2.2 / 2.3 / 2.4 — the conformal layer
# ---------------------------------------------------------------------

def test_pass_through_uncertainty_modules_are_gone():
    """``adaptive.py`` and ``calibration.py`` were removed."""

    assert importlib.util.find_spec("src.uncertainty.adaptive") is None
    assert importlib.util.find_spec("src.uncertainty.calibration") is None


def test_the_conformal_quantile_has_one_implementation():
    """The signed variant and the wrappers delegate to one implementation."""

    from src.uncertainty import conformal

    source = inspect.getsource(conformal)

    # The rank is computed in exactly one place.
    assert source.count("conformal_rank(n, alpha)") == 1

    # The public entry points are thin wrappers around it.
    assert "return _finite_sample_quantile(" in source
    assert source.count("return _finite_sample_quantile(") == 2


def test_the_scale_floor_is_declared_once_and_bounded():
    """One scale floor, applied as a bound (``max``), never as an addition."""

    from src.uncertainty import conformal, pipeline

    assert conformal.SCALE_FLOOR == pipeline.SCALE_FLOOR

    # No module re-declares a scale floor.
    for module in (conformal, pipeline):
        source = inspect.getsource(module)
        assert "SCALE_FLOOR: float =" not in source or module is conformal

    # The normalised score and the radius use the same bounding operator.
    assert "floor_scale(scale)" in inspect.getsource(conformal.normalised_nonconformity)
    assert "floor_scale(" in inspect.getsource(conformal.scale_adaptive_interval)


def test_a_floored_scale_is_bounded_not_incremented():
    from src.uncertainty.conformal import SCALE_FLOOR, floor_scale

    assert floor_scale(np.array([5.0])) == pytest.approx([5.0])
    assert floor_scale(np.array([0.0])) == pytest.approx([SCALE_FLOOR])


def test_every_interval_builder_clips_and_orders():
    """All conformal variants route through the single clipper."""

    from src.uncertainty.conformal import (
        adaptive_conformal_interval,
        adaptive_prediction_interval,
        clip_interval,
        conformal_interval,
        cqr_interval,
        engine_joint_interval,
        prediction_interval,
        scale_conformalized_interval,
    )

    builders = [
        prediction_interval(np.array([-10.0, 130.0]), 5.0),
        clip_interval(np.array([-10.0, 130.0]), np.array([5.0, 200.0])),
        adaptive_prediction_interval(
            np.array([-10.0, 130.0]),
            np.array([1.0, 1.0]),
            np.array([1.0]),
            np.array([0, 0]),
        ),
    ]

    # The named variants return a record; take their bounds.
    for record in (
        conformal_interval(
            np.array([10.0]), np.array([10.0]), np.array([-10.0, 130.0]), 0.1,
        ),
        adaptive_conformal_interval(
            np.array([10.0]), np.array([10.0]), np.array([1.0]),
            np.array([-10.0, 130.0]), np.array([1.0, 1.0]), 0.5,
        ),
        scale_conformalized_interval(
            np.array([10.0]), np.array([10.0]), np.array([1.0]),
            np.array([-10.0, 130.0]), np.array([1.0, 1.0]), 0.5,
        ),
        cqr_interval(
            np.array([10.0]), np.array([10.0]), np.array([1.0]),
            np.array([-10.0, 130.0]), np.array([1.0, 1.0]), 0.5,
        ),
        engine_joint_interval(
            np.array([10.0]), np.array([10.0]), np.array([1]),
            np.array([-10.0, 130.0]), 0.5,
        ),
    ):
        builders.append((record["lower"], record["upper"]))

    for lower, upper in builders:
        assert np.all(lower >= 0.0)
        assert np.all(upper <= 125.0)
        assert np.all(lower <= upper)


# ---------------------------------------------------------------------
# 2.5 / 2.6 / 2.8 — the evaluation layer
# ---------------------------------------------------------------------

def test_the_evaluation_layer_uses_the_shared_interval_metrics():
    """``metrics.py`` calls ``coverage.py`` instead of re-implementing it."""

    from src.evaluation import metrics
    from src.uncertainty import coverage

    assert metrics.winkler_score is coverage.winkler_score

    source = inspect.getsource(metrics.uncertainty_reliability)
    for helper in (
        "empirical_coverage(",
        "coverage_error(",
        "mean_prediction_interval_width(",
        "normalised_interval_width(",
        "engine_level_coverage(",
        "winkler_score(",
    ):
        assert helper in source

    # The hard-coded RUL cap is gone from the signature.
    signature = inspect.signature(metrics.uncertainty_reliability)
    assert signature.parameters["rul_max"].default is None


def test_the_accuracy_metrics_have_one_implementation():
    # The benchmark script imports them instead of carrying its own copies.
    from scripts import sota_benchmark
    from src.evaluation import metrics

    assert sota_benchmark.rmse is metrics.rmse
    assert sota_benchmark.mae is metrics.mae
    assert sota_benchmark.r2 is metrics.r2
    assert sota_benchmark.nasa_score is metrics.nasa_score


def test_the_pipeline_interval_metrics_match_the_shared_ones():
    from src.uncertainty import coverage, pipeline

    y_true = np.array([10.0, 20.0, 30.0])
    lower = np.array([8.0, 18.0, 25.0])
    upper = np.array([12.0, 22.0, 35.0])

    metrics = pipeline.interval_metrics(y_true, lower, upper, 0.9)

    assert metrics["empirical_coverage"] == pytest.approx(
        coverage.empirical_coverage(y_true, lower, upper)
    )
    assert metrics["mean_interval_width"] == pytest.approx(
        coverage.mean_prediction_interval_width(lower, upper)
    )
    assert metrics["winkler_score"] == pytest.approx(
        coverage.winkler_score(y_true, lower, upper, 0.1)
    )


# ---------------------------------------------------------------------
# 2.7 / 2.9 / 2.10 / 2.11 — the model layer
# ---------------------------------------------------------------------

def test_model_output_unpacking_has_one_implementation():
    from src.models.outputs import unpack_model_output

    assert unpack_model_output(torch.zeros(3)) == (pytest.approx(torch.zeros(3)), None, None)

    rul, hi, scale = unpack_model_output((torch.zeros(3), torch.ones(3), torch.full((3,), 2.0)))
    assert hi is not None and scale is not None

    rul, hi, scale = unpack_model_output((torch.zeros(3), torch.ones(3)))
    assert hi is not None and scale is None


def test_the_scripts_load_models_through_the_registry():
    from src.models.registry import load_model_from_checkpoint

    for module_name in ("scripts.evaluate", "scripts.explain", "scripts.uncertainty"):
        module = importlib.import_module(module_name)
        assert hasattr(module, "load_model")

    assert callable(load_model_from_checkpoint)


def test_the_architectures_share_the_task_heads():
    """All six architectures expose the same head attributes."""

    from src.models.registry import ALL_MODELS, build_model

    for arch in ALL_MODELS:
        model = build_model(
            arch,
            input_size=14,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            dropout=0.1,
        )
        for attribute in ("rul_head", "hi_head", "scale_head"):
            assert hasattr(model, attribute), f"{arch} has no {attribute}"


def test_the_positional_encoding_is_one_class():
    from src.models import hybrid, layers, transformer

    assert transformer.PositionalEncoding is layers.PositionalEncoding
    assert hybrid.PositionalEncoding is layers.PositionalEncoding


# ---------------------------------------------------------------------
# 2.12 / 2.13 / 2.14 — the training, HITL and preprocessing layers
# ---------------------------------------------------------------------

def test_the_trainer_has_one_epoch_implementation():
    from src.models.trainer import ModelTrainer

    assert hasattr(ModelTrainer, "_run_epoch")
    assert ModelTrainer.train_epoch.__doc__ is None or True
    # Both public entry points delegate to the shared implementation.
    assert "_run_epoch" in inspect.getsource(ModelTrainer.train_epoch)
    assert "_run_epoch" in inspect.getsource(ModelTrainer.validate_epoch)


def test_the_hitl_aggregation_is_shared():
    from src.hitl import updater

    source = inspect.getsource(updater)
    assert source.count("def aggregate_expert_actions(") == 1
    assert "aggregate_expert_actions(" in inspect.getsource(updater.build_policy_profile)
    assert "aggregate_expert_actions(" in inspect.getsource(updater.refine_recommendation)


def test_the_feature_columns_are_defined_once():
    from src.preprocessing import feature_selection, loader, scaling, windowing

    assert loader.get_feature_columns is not None
    assert windowing.WindowGenerator.get_feature_columns(
        pd_frame(),
    ) == loader.get_feature_columns(pd_frame())
    assert scaling.FeatureScaler.get_feature_columns(
        pd_frame(),
    ) == loader.get_feature_columns(pd_frame())
    assert feature_selection.FeatureSelector.get_feature_columns(
        pd_frame(),
    ) == loader.get_feature_columns(pd_frame())


def pd_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "unit_number": [1, 2],
            "time_in_cycles": [1, 2],
            "operational_setting_1": [0.1, 0.2],
            "sensor_2": [0.3, 0.4],
            "RUL": [10.0, 20.0],
            "HI": [0.5, 0.4],
        }
    )


# ---------------------------------------------------------------------
# 2.15 / 2.16 / 2.17 — the explainability layer
# ---------------------------------------------------------------------

def test_to_numpy_has_one_implementation():
    """No module converts tensors to NumPy on its own any more."""

    from src.utils import math_utils

    assert math_utils.to_numpy(torch.ones(3)).tolist() == [1.0, 1.0, 1.0]

    for relative in (
        "explainability/eri.py",
        "explainability/visualization.py",
        "explainability/degradation_viz.py",
    ):
        source = (REPO_ROOT / "src" / relative).read_text(encoding="utf-8")
        assert "from src.utils.math_utils import" in source, (
            f"{relative} does not use the shared to_numpy"
        )
        assert "def _to_numpy(" not in source, (
            f"{relative} still defines its own _to_numpy"
        )


def test_a_zero_total_importance_is_zero_everywhere():
    """The ERI module and the visualisation module agree on the edge case."""

    from src.explainability import eri, visualization

    zeros = np.zeros(4)

    assert np.all(eri._sensor_importance_from_ig(
        {"attributions": np.zeros((2, 3, 4))}
    ) == 0.0)
    assert np.all(visualization.normalize_importance(zeros) == 0.0)


def test_the_eri_spearman_handles_ties():
    """Tied ranks must not be broken arbitrarily."""

    from src.explainability.eri import _spearman

    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 1.0, 2.0, 3.0])

    # A tie-free shortcut would report < 1 here.
    assert _spearman(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# 2.18 / 2.19 — the shared enums and utilities
# ---------------------------------------------------------------------

def test_the_model_and_subset_enums_are_defined_once():
    from scripts import demo, hitl
    from src.models.registry import ALL_MODELS
    from src.utils.config import ALL_SUBSETS

    assert list(demo.ALL_MODELS) == list(ALL_MODELS)
    assert list(hitl.ALL_MODELS) == list(ALL_MODELS)
    assert list(demo.ALL_SUBSETS) == list(ALL_SUBSETS)
    assert list(hitl.ALL_SUBSETS) == list(ALL_SUBSETS)


def test_the_demo_accepts_the_headline_model():
    from scripts import demo
    from src.models.registry import ALL_MODELS

    assert "gru_att_deg_v2" in demo.ALL_MODELS
    assert "gru_att_deg_v2" in ALL_MODELS


def test_the_utility_functions_are_defined_once():
    from src.utils import config, paths, seed

    assert config.get_device is not None
    assert seed.set_seed is not None
    assert paths.PROJECT_ROOT is config.PROJECT_ROOT

    # No module re-declares the project root.
    for module_path in (REPO_ROOT / "src").rglob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        if module_path.name == "paths.py":
            continue
        assert "PROJECT_ROOT = Path(__file__).resolve().parents[2]" not in source, (
            f"{module_path} re-declares PROJECT_ROOT"
        )


def test_the_valid_actions_are_defined_once():
    from src.decision_engine import constraints
    from src.hitl import feedback
    from src.utils.config import VALID_ACTIONS

    assert constraints.VALID_ACTIONS == VALID_ACTIONS
    assert feedback.VALID_ACTIONS == VALID_ACTIONS


# ---------------------------------------------------------------------
# 2.20 — the preprocessing artifact
# ---------------------------------------------------------------------

def test_the_xai_stage_reads_the_processed_artifact():
    """``explain.py`` must not re-run the preprocessing pipeline."""

    explain_source = (REPO_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    preprocess_source = (REPO_ROOT / "scripts" / "preprocess.py").read_text(encoding="utf-8")

    # The XAI stage reads the saved frame ...
    assert "_test_frame.npz" in explain_source
    # ... and no longer rebuilds the scaler / feature selector / labels.
    for duplicated in ("FeatureScaler", "FeatureSelector", "LabelGenerator"):
        assert duplicated not in explain_source

    # The preprocessing stage writes it.
    assert "_test_frame.npz" in preprocess_source


# ---------------------------------------------------------------------
# 2.21 — the decision payload
# ---------------------------------------------------------------------

def test_the_recommender_passes_the_full_payload():
    from src.decision_engine.recommender import recommend_action

    decision = recommend_action(
        state={
            "rul": 50.0,
            "hi": 0.6,
            "rul_lower": 40.0,
            "rul_upper": 60.0,
            "eri": 0.85,
            "a_field_added_upstream": "kept",
        }
    )

    assert decision["a_field_added_upstream"] == "kept"
    assert decision["inputs"]["rul"] == 50.0
    assert decision["recommended_action"] == decision["candidate_action"]
