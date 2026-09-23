"""Deterministic tests for conformal uncertainty methods."""

import numpy as np
import pytest

from src.uncertainty.conformal import (
    adaptive_conformal_interval,
    conformal_interval,
    conformal_quantile,
    cqr_interval,
    engine_joint_interval,
)
from src.uncertainty.coverage import engine_level_coverage


def test_conformal_quantile_uses_finite_sample_rank():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert conformal_quantile(scores, alpha=0.25) == 4.0


def test_conformal_interval_is_clipped_and_ordered():
    result = conformal_interval(
        y_true_cal=np.array([10.0, 20.0, 30.0]),
        y_pred_cal=np.array([10.0, 20.0, 40.0]),
        y_pred_test=np.array([-10.0, 130.0]),
        alpha=0.1,
    )
    assert np.all(result["lower"] >= 0.0)
    assert np.all(result["upper"] <= 125.0)
    assert np.all(result["lower"] <= result["upper"])


def test_adaptive_conformal_calibrates_normalised_scores():
    result = adaptive_conformal_interval(
        y_true_cal=np.array([10.0, 20.0]),
        y_pred_cal=np.array([8.0, 16.0]),
        scale_cal=np.array([2.0, 4.0]),
        y_pred_test=np.array([50.0]),
        scale_test=np.array([3.0]),
        alpha=0.5,
    )
    assert result["q_hat"] == pytest.approx(1.0)
    assert result["lower"][0] == pytest.approx(47.0)
    assert result["upper"][0] == pytest.approx(53.0)


def test_cqr_uses_one_sided_bound_violation_score():
    result = cqr_interval(
        y_true_cal=np.array([10.0, 30.0]),
        y_pred_cal=np.array([10.0, 25.0]),
        scale_cal=np.array([2.0, 2.0]),
        y_pred_test=np.array([40.0]),
        scale_test=np.array([3.0]),
        alpha=0.5,
    )
    assert result["q_hat"] == pytest.approx(3.0)
    assert result["lower"][0] == pytest.approx(34.0)
    assert result["upper"][0] == pytest.approx(46.0)


def test_engine_joint_uses_one_score_per_engine():
    result = engine_joint_interval(
        y_true_cal=np.array([10.0, 12.0, 30.0, 35.0]),
        y_pred_cal=np.array([9.0, 10.0, 30.0, 30.0]),
        engine_ids_cal=np.array([1, 1, 2, 2]),
        y_pred_test=np.array([50.0]),
        alpha=0.5,
    )
    assert np.array_equal(result["engine_scores"], np.array([2.0, 5.0]))
    assert result["q_hat"] == pytest.approx(5.0)


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        conformal_interval(np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.0)


def test_engine_level_coverage_requires_every_window_per_engine():
    assert engine_level_coverage(
        y_true=np.array([1.0, 2.0, 3.0, 4.0]),
        lower=np.array([0.0, 1.0, 2.0, 5.0]),
        upper=np.array([2.0, 3.0, 4.0, 6.0]),
        engine_ids=np.array([1, 1, 2, 2]),
    ) == 0.5


def test_placeholder():
    assert True
