"""Tests for the SOTA benchmark bootstrap utilities."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.sota_benchmark import (
    bootstrap_mean_rmse,
    paired_delta,
    r2,
    rmse,
)

# ------------------------------------------------------------------
# Basic metrics
# ------------------------------------------------------------------


def test_rmse_matches_manual_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 3.0])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_r2_is_one_for_perfect_predictions():
    y_true = np.array([1.0, 2.0, 4.0])

    assert r2(y_true, y_true.copy()) == pytest.approx(1.0)


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------


def _dataset(offset: float, n: int = 60) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    y_true = rng.uniform(10.0, 120.0, size=n)
    y_pred = y_true + offset
    return {"FD001": (y_true, y_pred)}


def test_bootstrap_ci_is_ordered_and_close_to_point_estimate():
    data = _dataset(offset=3.0)
    rng = np.random.default_rng(42)

    mean_est, lower, upper = bootstrap_mean_rmse(data, resamples=500, rng=rng)

    assert lower <= mean_est <= upper
    assert lower == pytest.approx(3.0, abs=1.0)
    assert upper == pytest.approx(3.0, abs=1.0)


def test_paired_delta_is_negative_when_candidate_is_better():
    reference = _dataset(offset=5.0)
    candidate = _dataset(offset=1.0)
    rng = np.random.default_rng(42)

    result = paired_delta(reference, candidate, resamples=500, rng=rng)

    assert result["delta"] == pytest.approx(-4.0, abs=1e-9)
    assert result["ci_high"] < 0.0
    assert result["p_boot"] < 0.01


def test_paired_delta_is_near_zero_for_identical_predictions():
    reference = _dataset(offset=2.0)
    candidate = {
        subset: (y_true, y_pred.copy())
        for subset, (y_true, y_pred) in reference.items()
    }
    rng = np.random.default_rng(42)

    result = paired_delta(reference, candidate, resamples=500, rng=rng)

    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
