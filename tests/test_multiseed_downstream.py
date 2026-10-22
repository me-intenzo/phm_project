"""Tests for the multi-seed downstream-chain summary."""

import json

import pytest

from scripts import multiseed_downstream as downstream


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_artifacts(tmp_path, monkeypatch, subset="FD001", seed=42, tag="ms42",
                    model="gru_att_deg_v2", eri=0.9, width=20.0, decisions=4):
    """Create a synthetic downstream artifact tree and point the module at it."""

    results_dir = tmp_path / "results"
    xai_root = tmp_path / "xai"
    monkeypatch.setattr(downstream, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(downstream, "XAI_DIR", xai_root)

    run_name = f"{model}_{tag}"
    _write(
        xai_root / subset / run_name / "eri_rul.json",
        {"eri": eri, "rho_ig_attn": 0.8, "rho_ig_shap": 0.7, "consensus_k": 0.6},
    )
    _write(xai_root / subset / run_name / "eri_hi.json", {"eri": eri - 0.1})

    decision_records = []
    for index in range(decisions):
        decision_records.append(
            {
                "index": index,
                "inputs": {
                    "rul": 100.0,
                    "rul_lower": 100.0 - width / 2.0,
                    "rul_upper": 100.0 + width / 2.0,
                    "eri": eri,
                },
                "recommended_action": "MONITOR",
                "human_review": index == 0,
            }
        )
    _write(
        results_dir / f"decision_{subset}_{run_name}.json",
        {
            "subset": subset,
            "tag": tag,
            "summary": {
                "num_decisions": decisions,
                "action_distribution": {"MONITOR": decisions},
                "health_state_distribution": {"HEALTHY": decisions},
                "uncertainty_distribution": {"MODERATE": decisions},
                "human_review_count": 1,
                "constraint_violation_count": 0,
                "risk_score_mean": 0.3,
                "urgency_index_mean": 0.4,
            },
            "decisions": decision_records,
        },
    )
    _write(
        results_dir / f"hitl_{subset}_{run_name}.json",
        {
            "review_summary": {
                "decision_count": decisions,
                "human_review_count": 1,
                "pending_review_count": 1,
                "resolved_review_count": 0,
            },
            "feedback_statistics": {
                "total_feedback": 2,
                "override_rate": 1.0,
                "acceptance_rate": 0.0,
            },
            "prognostic_model_updated": False,
        },
    )


def test_collect_run_reads_every_artifact_of_the_seed(tmp_path, monkeypatch):
    _make_artifacts(tmp_path, monkeypatch)

    row = downstream.collect_run("FD001", 42, "ms42", "gru_att_deg_v2")

    assert row["missing_artifacts"] == ""
    assert row["eri_rul"] == pytest.approx(0.9)
    assert row["eri_hi"] == pytest.approx(0.8)
    assert row["n_decisions"] == 4
    assert row["human_review_count"] == 1
    assert row["review_rate"] == pytest.approx(0.25)
    # The width is the one the decision engine actually consumed.
    assert row["mean_interval_width"] == pytest.approx(20.0)
    assert row["action_MONITOR"] == 4
    assert row["prognostic_model_updated"] == "False"


def test_collect_run_reports_missing_artifacts_instead_of_crashing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(downstream, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(downstream, "XAI_DIR", tmp_path / "xai")

    row = downstream.collect_run("FD001", 42, "ms42", "gru_att_deg_v2")

    assert row["missing_artifacts"]
    assert "eri_rul.json" in row["missing_artifacts"]
    assert row["n_decisions"] == 0
    assert row["eri_rul"] != row["eri_rul"]  # NaN


def test_summary_reports_mean_and_std_per_subset_and_aggregate(tmp_path, monkeypatch):
    _make_artifacts(tmp_path, monkeypatch, seed=42, tag="ms42", eri=0.8, width=10.0)
    _make_artifacts(tmp_path, monkeypatch, seed=43, tag="ms43", eri=0.9, width=20.0)

    rows = [
        downstream.collect_run("FD001", 42, "ms42", "gru_att_deg_v2"),
        downstream.collect_run("FD001", 43, "ms43", "gru_att_deg_v2"),
    ]

    summary = downstream.summarise(rows, ["FD001"])

    assert [entry["subset"] for entry in summary] == ["FD001", "ALL"]
    fd001 = summary[0]
    assert fd001["n_seeds"] == 2
    assert fd001["eri_rul_mean"] == pytest.approx(0.85)
    assert fd001["eri_rul_std"] == pytest.approx(0.07071067811865475)
    assert fd001["mean_interval_width_mean"] == pytest.approx(15.0)

    # The aggregate averages the subset means within a seed first, so a
    # single-subset grid reduces to the per-seed value and the std is 0.
    aggregate = summary[1]
    assert aggregate["n_seeds"] == 2
    assert aggregate["eri_rul_mean"] == pytest.approx(0.85)
    assert aggregate["eri_rul_std"] == pytest.approx(0.07071067811865475)
