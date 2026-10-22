"""Tests for the multi-seed certification machinery.

These cover the pieces that make a multi-seed claim defensible:

* the finite-sample conformal rank is never silently clamped;
* signed (CQR) nonconformity scores are accepted;
* the engine-level calibration unit reduces windows to one score per engine;
* the artifact paths carry the run tag, so different seeds of the same
  model never overwrite each other.
"""

import numpy as np
import pytest

from src.uncertainty.conformal import (
    conformal_quantile,
    conformal_rank,
    signed_conformal_quantile,
)
from src.uncertainty.pipeline import (
    engine_max,
    finite_sample_quantile,
    prequential_aci_intervals,
    reduce_calibration_scores,
    signed_finite_sample_quantile,
)
from src.utils.paths import (
    checkpoint_path,
    flat_stem,
    results_path,
    run_dir_name,
    structured_results_path,
)

# ------------------------------------------------------------------
# Finite-sample conformal rank
# ------------------------------------------------------------------


def test_conformal_rank_matches_the_textbook_formula():
    # ceil((n + 1) * (1 - alpha))
    assert conformal_rank(n=19, alpha=0.05) == 19
    assert conformal_rank(n=8, alpha=0.10) == 9
    assert conformal_rank(n=2, alpha=0.25) == 3
    assert conformal_rank(n=18, alpha=0.05) == 19


def test_conformal_rank_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        conformal_rank(n=0, alpha=0.1)
    with pytest.raises(ValueError):
        conformal_rank(n=10, alpha=0.0)


def test_engine_level_calibration_attains_95_percent_with_20_engines():
    """The engine-level unit is what makes 95% attainable at n = 20.

    With the 20% engine hold-out the trainer uses, the calibration set
    holds 20 engines, and ceil(21 * 0.95) = 20 <= 20, so the rank exists.
    """

    assert conformal_rank(n=20, alpha=0.05) == 20
    assert conformal_quantile(np.arange(1.0, 21.0), alpha=0.05) == 20.0


def test_conformal_quantile_returns_infinite_when_rank_is_unattainable():
    """A clamped rank would void the coverage guarantee.

    With n = 10 calibration engines the 95% level needs rank
    ceil(11 * 0.95) = 11, which does not exist.  The honest answer is
    ``+inf`` (no finite interval is justified), not the largest observed
    score.
    """

    scores = np.arange(1.0, 11.0)

    assert conformal_quantile(scores, alpha=0.05) == float("inf")
    assert finite_sample_quantile(scores, alpha=0.05) == float("inf")


def test_conformal_quantile_can_raise_instead_of_returning_infinite():
    scores = np.arange(1.0, 11.0)

    with pytest.raises(ValueError, match="not attainable"):
        conformal_quantile(scores, alpha=0.05, allow_infinite=False)


def test_conformal_quantile_still_works_when_the_rank_is_attainable():
    scores = np.arange(1.0, 20.0)

    # ceil(20 * 0.80) = 16 -> the 16th smallest score.
    assert conformal_quantile(scores, alpha=0.20) == 16.0


# ------------------------------------------------------------------
# Signed (CQR) scores
# ------------------------------------------------------------------


def test_signed_conformal_quantile_accepts_negative_scores():
    """CQR scores are negative when the truth is inside the quantile band."""

    scores = np.array([-5.0, -3.0, -1.0, 2.0, 4.0])

    with pytest.raises(ValueError, match="non-negative"):
        conformal_quantile(scores, alpha=0.2)

    assert signed_conformal_quantile(scores, alpha=0.2) == 4.0


def test_signed_conformal_quantile_returns_infinite_when_unattainable():
    scores = np.arange(1.0, 11.0)

    assert signed_conformal_quantile(scores, alpha=0.05) == float("inf")
    assert signed_finite_sample_quantile(scores, alpha=0.05) == float("inf")


def test_signed_conformal_quantile_can_be_negative():
    """A negative q_hat is exactly what allows CQR to tighten an interval."""

    scores = np.array([-8.0, -6.0, -4.0, -2.0])

    assert signed_conformal_quantile(scores, alpha=0.25) == -2.0


# ------------------------------------------------------------------
# Engine-level calibration unit
# ------------------------------------------------------------------


def test_engine_max_reduces_to_one_score_per_engine():
    scores = np.array([1.0, 5.0, 2.0, 3.0])
    engine_ids = np.array([1, 1, 2, 2])

    ids, maxima = engine_max(scores, engine_ids)

    assert list(ids) == [1, 2]
    assert maxima == pytest.approx([5.0, 3.0])


def test_engine_max_handles_negative_scores():
    scores = np.array([-4.0, -1.0, -7.0])
    engine_ids = np.array([7, 7, 7])

    _, maxima = engine_max(scores, engine_ids)

    assert maxima == pytest.approx([-1.0])


def test_engine_max_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        engine_max(np.array([1.0, 2.0]), np.array([1]))


# ------------------------------------------------------------------
# Tag-aware artifact paths
# ------------------------------------------------------------------


def test_run_dir_name_is_optional():
    assert run_dir_name("gru_att_deg_v2") == "gru_att_deg_v2"
    assert run_dir_name("gru_att_deg_v2", "ms42") == "gru_att_deg_v2_ms42"


def test_untagged_paths_are_unchanged():
    """Backwards compatibility: without a tag nothing moves."""

    assert flat_stem("FD001", "gru_att_deg_v2") == "FD001_gru_att_deg_v2"
    assert flat_stem("FD001", "gru_att_deg_v2", suffix="conformal") == (
        "FD001_gru_att_deg_v2_conformal"
    )


def test_tagged_paths_are_distinct_per_seed():
    a = results_path("FD001", "gru_att_deg_v2", "ms42", "conformal")
    b = results_path("FD001", "gru_att_deg_v2", "ms43", "conformal")

    assert a != b
    assert a.name == "FD001_gru_att_deg_v2_ms42_conformal.npz"
    assert b.name == "FD001_gru_att_deg_v2_ms43_conformal.npz"


def test_checkpoint_path_carries_the_tag():
    path = checkpoint_path("gru_att_deg_v2", "FD001", "ms42")

    assert path.parent.name == "FD001"
    assert path.parent.parent.name == "gru_att_deg_v2_ms42"
    assert path.name == "best_model.pt"


def test_untagged_checkpoint_path_is_unchanged():
    path = checkpoint_path("gru_att_deg_v2", "FD001")

    assert path.parent.parent.name == "gru_att_deg_v2"


def test_structured_results_path_carries_the_tag():
    path = structured_results_path("eara_conformal", "FD001", "gru_att_deg_v2", "ms42")

    assert path.name == "gru_att_deg_v2_ms42.npz"
    assert "eara_conformal" in path.parts
    assert "FD001" in path.parts


# ------------------------------------------------------------------
# Calibration units
# ------------------------------------------------------------------


def test_calibration_unit_window_keeps_every_window_score():
    scores = np.asarray([1.0, 5.0, 2.0, 9.0])
    engines = np.asarray([10, 10, 11, 11])

    window = reduce_calibration_scores(scores, engines, "window")

    assert window.tolist() == scores.tolist()


def test_calibration_unit_engine_keeps_one_max_per_engine():
    scores = np.asarray([1.0, 5.0, 2.0, 9.0])
    engines = np.asarray([10, 10, 11, 11])

    engine = reduce_calibration_scores(scores, engines, "engine")

    assert engine.tolist() == [5.0, 9.0]


def test_calibration_unit_rejects_unknown_unit():
    with pytest.raises(ValueError):
        reduce_calibration_scores(np.ones(3), np.ones(3), "regime")


def test_prequential_aci_keeps_alpha_strictly_positive():
    """A run of misses must not drive alpha to 0 (an undefined level)."""

    scores = np.linspace(0.0, 5.0, 100)
    y_pred = np.full(20, 60.0)
    scales = np.ones(20)
    y_true = np.full(20, 160.0)  # every interval misses

    lower, upper, stream = prequential_aci_intervals(
        scores, y_pred, scales, y_true, alpha=0.2, gamma=0.2
    )

    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert min(stream) >= 0.01
    assert max(stream) < 1.0

