"""Tests for the decision-weight sensitivity analysis."""

from __future__ import annotations

import json

import pytest

from src.evaluation.decision_sensitivity import (
    WEIGHT_NAMES,
    dirichlet_sweep,
    oat_sweep,
    run_sensitivity,
    summarise,
)


def _records(n: int = 40):
    """A small, varied set of decision records."""

    records = []
    for index in range(n):
        rul = 5.0 + (index % 10) * 12.0
        records.append({
            "inputs": {
                "rul": rul,
                "hi": max(0.05, 1.0 - rul / 125.0),
                "rul_lower": max(0.0, rul * 0.95),
                "rul_upper": rul * 1.05,
                "eri": 0.9,
            },
            "top_k_sensors": [1, 2, 3],
        })
    return records


def _artifact(tmp_path, records=None, thresholds=None, constraints=None):
    payload = {
        "subset": "FD001",
        "model": "gru_att_deg_v2",
        "tag": "ms42",
        "constraint_scenario": "unconstrained",
        "thresholds": thresholds or {},
        "operational_constraints": constraints or {},
        "decisions": records if records is not None else _records(),
    }
    path = tmp_path / "decision_FD001_gru_att_deg_v2_ms42.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

class TestOatSweep:
    def test_every_weight_is_swept(self):
        rows = oat_sweep(_records(), grid=(0.0, 0.5, 1.0))
        assert {row["weight"] for row in rows} == set(WEIGHT_NAMES)

    def test_weights_always_sum_to_one(self):
        for row in oat_sweep(_records(), grid=(0.0, 0.25, 0.75, 1.0)):
            assert sum(row["weights"].values()) == pytest.approx(1.0, abs=1e-5)

    def test_baseline_row_has_zero_movement(self):
        rows = oat_sweep(
            _records(),
            thresholds={"w_rul": 0.4, "w_hi": 0.3, "w_u": 0.15, "w_eri": 0.15},
            grid=(0.4,),
        )
        for row in rows:
            if row["weight"] == "w_rul":
                assert row["action_tv"] == 0.0

    def test_action_tv_is_a_probability(self):
        for row in oat_sweep(_records(), grid=(0.0, 0.5, 1.0)):
            assert 0.0 <= row["action_tv"] <= 1.0

    def test_switch_rate_is_a_fraction(self):
        for row in oat_sweep(_records(), grid=(0.0, 0.5, 1.0)):
            assert 0.0 <= row["action_switch_rate"] <= 1.0

    def test_extreme_weight_perturbs_the_decisions(self):
        """Putting everything on one weight must move at least some records."""

        rows = oat_sweep(_records(), grid=(1.0,))
        assert any(row["action_tv"] > 0.0 for row in rows)


class TestDirichletSweep:
    def test_sample_count(self):
        rows = dirichlet_sweep(_records(), n_samples=25, seed=0)
        assert len(rows) == 25

    def test_weights_sum_to_one(self):
        for row in dirichlet_sweep(_records(), n_samples=10, seed=0):
            assert sum(row["weights"].values()) == pytest.approx(1.0, abs=1e-5)

    def test_is_reproducible(self):
        first = dirichlet_sweep(_records(), n_samples=10, seed=7)
        second = dirichlet_sweep(_records(), n_samples=10, seed=7)
        assert [row["weights"] for row in first] == [row["weights"] for row in second]

    def test_different_seeds_differ(self):
        first = dirichlet_sweep(_records(), n_samples=10, seed=1)
        second = dirichlet_sweep(_records(), n_samples=10, seed=2)
        assert [row["weights"] for row in first] != [row["weights"] for row in second]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestSummary:
    def test_influence_is_reported_per_weight(self):
        rows = oat_sweep(_records(), grid=(0.0, 0.5, 1.0))
        rows += dirichlet_sweep(_records(), n_samples=5, seed=0)
        summary = summarise(rows)
        assert set(summary["influence_by_weight"]) == set(WEIGHT_NAMES)
        assert summary["most_influential_weight"] in WEIGHT_NAMES

    def test_influence_is_the_maximum_tv(self):
        rows = oat_sweep(_records(), grid=(0.0, 0.5, 1.0))
        summary = summarise(rows)
        for name in WEIGHT_NAMES:
            values = [row["action_tv"] for row in rows if row["weight"] == name]
            assert summary["influence_by_weight"][name] == pytest.approx(max(values))

    def test_stats_blocks_are_present(self):
        rows = oat_sweep(_records(), grid=(0.0, 1.0))
        rows += dirichlet_sweep(_records(), n_samples=5, seed=0)
        summary = summarise(rows)
        assert summary["oat"]["n"] == len(WEIGHT_NAMES) * 2
        assert summary["dirichlet"]["n"] == 5


class TestRunSensitivity:
    def test_report_structure(self, tmp_path):
        path = _artifact(tmp_path)
        report = run_sensitivity(path, n_dirichlet=10)
        assert report["n_decisions"] == 40
        assert report["subset"] == "FD001"
        assert report["tag"] == "ms42"
        assert "summary" in report
        assert len(report["rows"]) > 0

    def test_baseline_weights_are_reported(self, tmp_path):
        path = _artifact(
            tmp_path,
            thresholds={"w_rul": 0.5, "w_hi": 0.2, "w_u": 0.2, "w_eri": 0.1},
        )
        report = run_sensitivity(path, n_dirichlet=5)
        assert report["baseline_weights"]["w_rul"] == pytest.approx(0.5)

    def test_empty_artifact_is_reported(self, tmp_path):
        path = _artifact(tmp_path, records=[])
        report = run_sensitivity(path, n_dirichlet=5)
        assert "error" in report

    def test_thresholds_are_applied(self, tmp_path):
        """A different threshold set must change the decisions."""

        base = _artifact(tmp_path / "base")
        report_a = run_sensitivity(base, n_dirichlet=5)

        strict = _artifact(
            tmp_path / "strict",
            thresholds={"rul_critical": 40.0, "rul_at_risk": 60.0},
        )
        report_b = run_sensitivity(strict, n_dirichlet=5)

        assert report_a["summary"] != report_b["summary"]
