"""Tests for artifact provenance and the artifact-pairing rules.

These cover the guarantees that make a published comparison trustworthy:

* a provenance stamp survives a ``.npz`` round trip **without** pickle;
* two artifacts that declare different checkpoints are reported as a
  mismatch rather than being compared as if they agreed;
* the comparative benchmark's summary resolves the *tagged* artifact key
  (it used to look up the untagged key and silently find nothing);
* the decision engine prefers the flat artifact over the legacy
  structured tree, so a stale file cannot shadow a fresh one.
"""

import numpy as np
import pytest

from src.evaluation.benchmark import comparative_benchmark
from src.utils.provenance import (
    META_KEY,
    STATUS_MISMATCH,
    STATUS_OK,
    STATUS_UNKNOWN,
    build_provenance,
    compare_provenance,
    pack_provenance,
    sha256_file,
    unpack_provenance,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_predictions(path, *, n=10, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.linspace(10.0, 100.0, n)
    y_pred = y_true + rng.normal(0.0, 1.0, n)
    hi_true = np.full(n, 0.5)
    hi_pred = np.full(n, 0.5)
    np.savez(
        path,
        y_rul_true=y_true,
        y_rul_pred=y_pred,
        y_hi_true=hi_true,
        y_hi_pred=hi_pred,
    )
    return y_true, y_pred


def _write_uncertainty(path, y_true, *, y_pred=None, stamp=None):
    y_pred = y_true if y_pred is None else y_pred
    payload = {
        "lower_90": y_pred - 5.0,
        "upper_90": y_pred + 5.0,
        "test_engine_ids": np.arange(len(y_true)),
        "y_rul_pred": y_pred,
        "y_rul_test": y_true,
    }
    if stamp is not None:
        payload[META_KEY] = pack_provenance(stamp)
    np.savez(path, **payload)


def _stamp(sha, **extra):
    document = {"checkpoint_sha256": sha}
    document.update(extra)
    return document


# ------------------------------------------------------------------
# The stamp itself
# ------------------------------------------------------------------

def test_provenance_round_trips_through_a_npz_without_pickle(tmp_path):
    """The stamp must be readable with the default ``allow_pickle=False``."""

    path = tmp_path / "artifact.npz"
    meta = build_provenance(subset="FD001", model="gru_att_deg_v2", tag="ms42")
    meta["checkpoint_sha256"] = "a" * 64

    np.savez(path, meta_json=pack_provenance(meta), y_rul_pred=np.zeros(3))

    # allow_pickle defaults to False -> a pickled stamp would raise here.
    with np.load(path) as data:
        recovered = unpack_provenance(data)

    assert recovered is not None
    assert recovered["checkpoint_sha256"] == "a" * 64
    assert recovered["subset"] == "FD001"
    assert recovered["tag"] == "ms42"


def test_unpack_provenance_returns_none_for_a_legacy_artifact(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, y_rul_pred=np.zeros(3))

    with np.load(path) as data:
        assert unpack_provenance(data) is None


def test_unpack_provenance_survives_a_corrupt_stamp(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, meta_json=np.array("{not json"))

    with np.load(path) as data:
        assert unpack_provenance(data) is None


def test_sha256_file_is_empty_for_a_missing_path(tmp_path):
    assert sha256_file(tmp_path / "nope.bin") == ""


# ------------------------------------------------------------------
# Comparing two stamps
# ------------------------------------------------------------------

def test_compare_provenance_reports_ok_for_the_same_checkpoint():
    result = compare_provenance(_stamp("a" * 64), _stamp("a" * 64))
    assert result["status"] == STATUS_OK


def test_compare_provenance_reports_a_mismatch_for_different_checkpoints():
    result = compare_provenance(_stamp("a" * 64), _stamp("b" * 64))
    assert result["status"] == STATUS_MISMATCH
    assert result["left_sha256"] != result["right_sha256"]


def test_compare_provenance_reports_unknown_when_a_stamp_is_missing():
    assert compare_provenance(None, _stamp("a" * 64))["status"] == STATUS_UNKNOWN
    assert compare_provenance(_stamp("a" * 64), None)["status"] == STATUS_UNKNOWN
    assert compare_provenance(None, None)["status"] == STATUS_UNKNOWN


# ------------------------------------------------------------------
# The comparative benchmark
# ------------------------------------------------------------------

def test_comparative_benchmark_summary_resolves_the_tagged_key(tmp_path):
    """A tagged run must still produce a summary.

    The summary used to look up ``{subset}_{model}`` while the report was
    keyed ``{subset}_{model}_{tag}``, so every tagged run reported
    ``best_model_by_rul_mae: None`` and ``best_rul_mae: inf``.
    """

    # Two models, same tag.  The summary has to pick the better one.
    _write_predictions(tmp_path / "FD001_gru_ms42_predictions.npz", seed=1)
    _write_predictions(tmp_path / "FD001_lstm_ms42_predictions.npz", seed=2)

    for model in ("gru", "lstm"):
        y_true, _ = _write_predictions(
            tmp_path / f"FD001_{model}_ms42_predictions.npz", seed=1
        )
        _write_uncertainty(
            tmp_path / f"FD001_{model}_ms42_engine_joint_conformal.npz",
            y_true,
        )

    report = comparative_benchmark(
        tmp_path,
        subsets=["FD001"],
        models=["gru", "lstm"],
        tag="ms42",
    )

    summary = report["summary"]["FD001"]
    assert summary["best_model_by_rul_mae"] in {"gru_ms42", "lstm_ms42"}
    assert np.isfinite(summary["best_rul_mae"])
    assert summary["best_rul_mae"] > 0.0


def test_comparative_benchmark_flags_two_different_checkpoints(tmp_path):
    _write_predictions(tmp_path / "FD001_gru_predictions.npz")
    y_true, y_pred = _write_predictions(tmp_path / "FD001_gru_predictions.npz")
    _write_uncertainty(
        tmp_path / "FD001_gru_engine_joint_conformal.npz",
        y_true,
        stamp=_stamp("b" * 64),
    )

    # Stamp the predictions with a *different* checkpoint.
    path = tmp_path / "FD001_gru_predictions.npz"
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files}
    payload[META_KEY] = pack_provenance(_stamp("a" * 64))
    np.savez(path, **payload)

    report = comparative_benchmark(
        tmp_path,
        subsets=["FD001"],
        models=["gru"],
    )

    entry = report["FD001_gru"]
    assert entry["provenance"]["status"] == STATUS_MISMATCH
    assert "warning" in entry["provenance"]


def test_comparative_benchmark_can_be_strict_about_provenance(tmp_path):
    y_true, _ = _write_predictions(tmp_path / "FD001_gru_predictions.npz")
    _write_uncertainty(
        tmp_path / "FD001_gru_engine_joint_conformal.npz",
        y_true,
        stamp=_stamp("b" * 64),
    )
    path = tmp_path / "FD001_gru_predictions.npz"
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files}
    payload[META_KEY] = pack_provenance(_stamp("a" * 64))
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="different checkpoints"):
        comparative_benchmark(
            tmp_path,
            subsets=["FD001"],
            models=["gru"],
            strict_provenance=True,
        )


def test_comparative_benchmark_reports_unknown_for_legacy_artifacts(tmp_path):
    y_true, _ = _write_predictions(tmp_path / "FD001_gru_predictions.npz")
    _write_uncertainty(tmp_path / "FD001_gru_engine_joint_conformal.npz", y_true)

    report = comparative_benchmark(
        tmp_path,
        subsets=["FD001"],
        models=["gru"],
    )

    entry = report["FD001_gru"]
    assert entry["provenance"]["status"] == STATUS_UNKNOWN
    assert "warning" not in entry["provenance"]


def test_comparative_benchmark_accepts_matching_checkpoints(tmp_path):
    y_true, _ = _write_predictions(tmp_path / "FD001_gru_predictions.npz")
    _write_uncertainty(
        tmp_path / "FD001_gru_engine_joint_conformal.npz",
        y_true,
        stamp=_stamp("a" * 64),
    )
    path = tmp_path / "FD001_gru_predictions.npz"
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files}
    payload[META_KEY] = pack_provenance(_stamp("a" * 64))
    np.savez(path, **payload)

    report = comparative_benchmark(
        tmp_path,
        subsets=["FD001"],
        models=["gru"],
        strict_provenance=True,
    )

    assert report["FD001_gru"]["provenance"]["status"] == STATUS_OK


# ------------------------------------------------------------------
# The decision engine's artifact resolution
# ------------------------------------------------------------------

def test_decision_prefers_the_flat_artifact_over_the_legacy_structured_one(
    tmp_path, monkeypatch
):
    """A stale structured file must not shadow the fresh flat artifact."""

    import scripts.decision as decision

    results_dir = tmp_path / "results"
    structured_dir = results_dir / "uncertainty" / "eara_conformal" / "FD001"
    structured_dir.mkdir(parents=True)

    flat = results_dir / "FD001_gru_eara_conformal.npz"
    structured = structured_dir / "gru.npz"
    np.savez(flat, y_rul_pred=np.arange(3.0))
    np.savez(structured, y_rul_pred=np.arange(90.0, 93.0))

    monkeypatch.setattr(decision, "RESULTS_DIR", results_dir)

    path, note = decision.resolve_uncertainty_path("FD001", "gru", "eara_conformal")

    assert path == flat
    assert "both" in note

    with np.load(path) as data:
        assert data["y_rul_pred"][0] == 0.0


def test_decision_falls_back_to_the_structured_artifact(tmp_path, monkeypatch):
    import scripts.decision as decision

    results_dir = tmp_path / "results"
    structured_dir = results_dir / "uncertainty" / "eara_conformal" / "FD001"
    structured_dir.mkdir(parents=True)
    structured = structured_dir / "gru.npz"
    np.savez(structured, y_rul_pred=np.arange(3.0))

    monkeypatch.setattr(decision, "RESULTS_DIR", results_dir)

    path, note = decision.resolve_uncertainty_path("FD001", "gru", "eara_conformal")

    assert path == structured
    assert "legacy structured file" in note


def test_decision_reports_a_missing_artifact(tmp_path, monkeypatch):
    import scripts.decision as decision

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    monkeypatch.setattr(decision, "RESULTS_DIR", results_dir)

    path, note = decision.resolve_uncertainty_path("FD001", "gru", "eara_conformal")

    assert not path.exists()
    assert note == ""
