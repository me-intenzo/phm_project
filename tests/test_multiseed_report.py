"""Tests for the multi-seed aggregation and the certification verdict."""

import json

import pytest

from scripts.multiseed_report import (
    build_aggregate,
    build_summary,
    compare_with_literature,
    metric_block,
)


def _per_seed(rmse_by_seed_subset: dict[tuple[int, str], float]) -> list[dict]:
    rows = []
    for (seed, subset), rmse in sorted(rmse_by_seed_subset.items()):
        rows.append(
            {
                "subset": subset,
                "seed": seed,
                "tag": f"ms{seed}",
                "RMSE": rmse,
                "MAE": rmse / 2.0,
                "R2": 0.9,
                "NASA_Score": rmse * 10.0,
            }
        )
    return rows


def test_summary_reports_sample_std_across_seeds():
    rows = _per_seed(
        {
            (42, "FD001"): 10.0,
            (43, "FD001"): 12.0,
            (44, "FD001"): 14.0,
        }
    )

    summary = build_summary(rows)

    assert len(summary) == 1
    entry = summary[0]
    assert entry["subset"] == "FD001"
    assert entry["n_seeds"] == 3
    assert entry["RMSE_mean"] == pytest.approx(12.0)
    # ddof=1 sample standard deviation of {10, 12, 14} is 2.0
    assert entry["RMSE_std"] == pytest.approx(2.0)
    assert entry["RMSE_min"] == pytest.approx(10.0)
    assert entry["RMSE_max"] == pytest.approx(14.0)


def test_aggregate_uses_the_unweighted_four_subset_mean():
    rows = _per_seed(
        {
            (42, "FD001"): 10.0,
            (42, "FD002"): 14.0,
            (42, "FD003"): 12.0,
            (42, "FD004"): 16.0,
            (43, "FD001"): 11.0,
            (43, "FD002"): 15.0,
            (43, "FD003"): 13.0,
            (43, "FD004"): 17.0,
        }
    )

    aggregate = build_aggregate(rows, tag_prefix="ms")

    assert [row["seed"] for row in aggregate] == [42, 43]
    assert aggregate[0]["mean_RMSE"] == pytest.approx(13.0)
    assert aggregate[1]["mean_RMSE"] == pytest.approx(14.0)
    assert aggregate[0]["tag"] == "ms42"


def test_aggregate_skips_incomplete_seeds():
    """A seed with only some subsets is not a four-subset aggregate."""

    rows = _per_seed(
        {
            (42, "FD001"): 10.0,
            (42, "FD002"): 14.0,
            (42, "FD003"): 12.0,
            (42, "FD004"): 16.0,
            (43, "FD001"): 11.0,
        }
    )

    aggregate = build_aggregate(rows)

    assert [row["seed"] for row in aggregate] == [42]


def _reference(tmp_path, published: float) -> object:
    path = tmp_path / "literature_reference.json"
    path.write_text(
        json.dumps(
            {
                "references": [
                    {"method": "Best", "venue": "X", "mean_rmse": published},
                    {"method": "Worse", "venue": "Y", "mean_rmse": published + 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _aggregate_with_mean(value: float) -> list[dict]:
    return [{"seed": 42, "tag": "ms42", "mean_RMSE": value}]


def test_verdict_says_seed_noise_when_gap_is_inside_the_band(tmp_path):
    rows = _per_seed(
        {
            (seed, subset): value
            for seed, value in zip((42, 43, 44), (12.5, 13.0, 13.5))
            for subset in ("FD001", "FD002", "FD003", "FD004")
        }
    )
    aggregate = build_aggregate(rows)

    _, analysis = compare_with_literature(aggregate, _reference(tmp_path, 12.83))

    assert analysis["gap_exceeds_seed_std"] is False
    assert "not established" in analysis["verdict"]


def test_verdict_names_the_gap_when_it_exceeds_the_band(tmp_path):
    rows = _per_seed(
        {
            (seed, subset): value
            for seed, value in zip((42, 43, 44), (13.4, 13.5, 13.6))
            for subset in ("FD001", "FD002", "FD003", "FD004")
        }
    )
    aggregate = build_aggregate(rows)

    _, analysis = compare_with_literature(aggregate, _reference(tmp_path, 12.83))

    assert analysis["gap_exceeds_seed_std"] is True
    assert analysis["gap_to_best_published"] > 0
    assert "behind" in analysis["verdict"]
    assert analysis["best_published_method"] == "Best"


def test_verdict_reports_a_genuine_win_as_a_win(tmp_path):
    rows = _per_seed(
        {
            (seed, subset): value
            for seed, value in zip((42, 43, 44), (11.0, 11.1, 11.2))
            for subset in ("FD001", "FD002", "FD003", "FD004")
        }
    )
    aggregate = build_aggregate(rows)

    _, analysis = compare_with_literature(aggregate, _reference(tmp_path, 12.83))

    assert analysis["gap_to_best_published"] < 0
    assert "better than the best published" in analysis["verdict"]


def test_missing_reference_does_not_crash(tmp_path):
    rows = _per_seed(
        {
            (42, subset): 12.0
            for subset in ("FD001", "FD002", "FD003", "FD004")
        }
    )

    literature, analysis = compare_with_literature(
        build_aggregate(rows), tmp_path / "does_not_exist.json"
    )

    assert literature == []
    assert analysis["best_published_mean_rmse"] is None
    assert "No published reference" in analysis["verdict"]


def test_metric_block_accepts_the_upper_case_artifact_shape():
    """scripts/evaluate.py writes ``{"metrics": {"RUL": ..., "HI": ...}}``.

    The report must read that, not silently return an empty block (which
    would make every aggregate ``nan``).
    """

    payload = {
        "subset": "FD001",
        "arch": "gru_att_deg_v2",
        "metrics": {
            "RUL": {"RMSE": 11.9, "MAE": 8.8, "R2": 0.91, "NASA_Score": 224.0},
            "HI": {"RMSE": 0.105, "MAE": 0.081, "R2": 0.892},
        },
    }

    rul = metric_block(payload, "rul")
    hi = metric_block(payload, "hi")

    assert rul["RMSE"] == pytest.approx(11.9)
    assert hi["MAE"] == pytest.approx(0.081)
    assert metric_block(payload, "missing") == {}


def test_metric_block_accepts_lower_case_fixtures():
    payload = {"metrics": {"rul": {"RMSE": 1.0}}}

    assert metric_block(payload, "rul")["RMSE"] == pytest.approx(1.0)
    assert metric_block({}, "rul") == {}

