"""
decision_sensitivity.py

Two experiments on the *policy* half of the pipeline, run against the
decisions the repository actually produced:

1. **Weight sensitivity** — how much of the published decision mix is an
   artefact of the hand-set risk-score weights ``w_rul`` / ``w_hi`` /
   ``w_u`` / ``w_eri``?  The decision engine is re-run over a weight
   space (one-at-a-time plus a Dirichlet sweep over the simplex) and the
   movement of the action mix is measured.

2. **Constraint scenarios** — what does the decision engine do when the
   operational context is not the unconstrained one?  Every named scenario
   of ``src.decision_engine.scenarios`` is applied to the recorded states
   and the action mix, the violation count and the review rate are
   compared.  This is also what makes the constraint layer produce
   non-zero violation counts in a recorded run.

Usage
-----
    # Every certified (subset, seed) decision artifact
    python scripts/decision_sensitivity.py --subsets all --seeds 42 43 44 45 46

    # One artifact
    python scripts/decision_sensitivity.py \\
        --decision outputs/results/decision_FD001_gru_att_deg_v2_ms42.json

Outputs
-------
    outputs/reports/decision_sensitivity.json         full report
    outputs/reports/decision_sensitivity.csv          per-run summary
    outputs/reports/decision_weight_sweep.csv         per-weight detail
    outputs/reports/decision_constraint_scenarios.csv per-scenario detail

author: me-intenzo
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_engine.recommender import recommend_action
from src.decision_engine.rules import SEVERITY_LADDER, severity_of
from src.decision_engine.scenarios import (
    describe_scenario,
    get_scenario,
    scenario_names,
)
from src.evaluation.ablation import _state_from_record
from src.evaluation.decision_sensitivity import (
    run_sensitivity,
)
from src.utils.config import ALL_SUBSETS
from src.utils.paths import RESULTS_DIR

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

WEIGHT_FIELDS = (
    "subset",
    "seed",
    "tag",
    "sweep",
    "weight",
    "value",
    "w_rul",
    "w_hi",
    "w_u",
    "w_eri",
    "action_tv",
    "action_switch_rate",
    "mean_severity_delta",
    "review_rate_delta",
    "risk_rank_spearman",
    "mean_abs_risk_delta",
    "risk_band_switch_rate",
    "override_switch_count",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if v == v]
    return float(statistics.fmean(clean)) if clean else float("nan")


def find_decision_artifacts(
    subsets: list[str],
    seeds: list[int],
    model: str,
    tag_prefix: str,
) -> list[tuple[str, int, str, Path]]:
    """Locate the decision artifacts of the requested (subset, seed) pairs."""

    found: list[tuple[str, int, str, Path]] = []
    # Allow "all" as a shorthand for every subset.
    if "all" in subsets:
        subsets = list(ALL_SUBSETS)
    for subset in subsets:
        for seed in seeds:
            tag = f"{tag_prefix}{seed}"
            run_name = f"{model}_{tag}" if tag else model
            path = RESULTS_DIR / f"decision_{subset}_{run_name}.json"
            if path.exists():
                found.append((subset, seed, tag, path))
    return found


# ------------------------------------------------------------------
# Constraint scenarios
# ------------------------------------------------------------------

def run_constraint_scenarios(
    decision_path: Path,
) -> list[dict[str, Any]]:
    """Re-run the recorded states under every named constraint scenario."""

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    records = payload.get("decisions", payload)
    if isinstance(records, dict):
        records = [records]
    records = list(records)

    thresholds = payload.get("thresholds") or None
    states = [_state_from_record(record) for record in records]

    rows: list[dict[str, Any]] = []

    for scenario in scenario_names():
        constraints = get_scenario(scenario)
        decisions = [
            recommend_action(
                state=state,
                constraints=constraints,
                thresholds=thresholds,
            )
            for state in states
        ]

        mix = {action: 0 for action in SEVERITY_LADDER}
        violations = 0
        violated_records = 0
        reviews = 0
        severities: list[int] = []
        for decision in decisions:
            mix[str(decision["recommended_action"])] = (
                mix.get(str(decision["recommended_action"]), 0) + 1
            )
            record_violations = decision.get("constraint_violations", [])
            violations += len(record_violations)
            if record_violations:
                violated_records += 1
            if decision.get("human_review"):
                reviews += 1
            severities.append(severity_of(decision["recommended_action"]))

        rows.append({
            "subset": payload.get("subset"),
            "model": payload.get("model"),
            "tag": payload.get("tag"),
            "scenario": scenario,
            "constraints": describe_scenario(scenario),
            "n_decisions": len(decisions),
            "constraint_violation_count": violations,
            "constraint_violated_records": violated_records,
            "human_review_count": reviews,
            "review_rate": round(reviews / len(decisions), 6) if decisions else 0.0,
            "mean_severity": round(_mean(severities), 6),
            **{f"action_{action}": mix.get(action, 0) for action in SEVERITY_LADDER},
        })

    return rows


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="Subsets to analyse.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seeds to analyse.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help="Architecture whose decisions are analysed.",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="ms",
        help="Tag prefix of the multi-seed family (e.g. 'ms' for 'ms42').",
    )
    parser.add_argument(
        "--decision",
        type=str,
        default=None,
        help=(
            "Analyse a single decision artifact instead of the seed grid. "
            "The artifact path is used verbatim."
        ),
    )
    parser.add_argument(
        "--dirichlet-samples",
        type=int,
        default=200,
        help="Number of Dirichlet weight draws per run.",
    )
    parser.add_argument(
        "--dirichlet-concentration",
        type=float,
        default=5.0,
        help="Dirichlet concentration (higher = closer to uniform).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed of the Dirichlet sampler.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.decision:
        artifacts = [("?", 0, "", Path(args.decision))]
    else:
        artifacts = find_decision_artifacts(
            subsets=args.subsets,
            seeds=args.seeds,
            model=args.model,
            tag_prefix=args.tag_prefix,
        )

    if not artifacts:
        print("No decision artifacts found. Run scripts/multiseed_pipeline.py first.")
        return 1

    reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for subset, seed, tag, path in artifacts:
        report = run_sensitivity(
            decision_path=path,
            n_dirichlet=args.dirichlet_samples,
            concentration=args.dirichlet_concentration,
            seed=args.seed,
        )
        if "error" in report:
            print(f"  [WARN] {path.name}: {report['error']}")
            continue

        reports.append(report)

        summary = report["summary"]
        summary_rows.append({
            "subset": report.get("subset") or subset,
            "seed": seed,
            "tag": tag,
            "model": report.get("model"),
            "constraint_scenario": report.get("constraint_scenario"),
            "n_decisions": report["n_decisions"],
            "most_influential_weight": summary["most_influential_weight"],
            **{f"influence_{key}": value for key, value in summary["influence_by_weight"].items()},
            "oat_action_tv_max": summary["oat"].get("action_tv_max"),
            "oat_action_switch_rate_max": summary["oat"].get("action_switch_rate_max"),
            "oat_risk_rank_spearman_min": summary["oat"].get("risk_rank_spearman_min"),
            "oat_risk_band_switch_rate_max": summary["oat"].get("risk_band_switch_rate_max"),
            "dirichlet_action_tv_mean": summary["dirichlet"].get("action_tv_mean"),
            "dirichlet_action_tv_max": summary["dirichlet"].get("action_tv_max"),
            "dirichlet_action_switch_rate_mean": summary["dirichlet"].get("action_switch_rate_mean"),
            "dirichlet_risk_rank_spearman_min": summary["dirichlet"].get("risk_rank_spearman_min"),
            "dirichlet_risk_band_switch_rate_mean": summary["dirichlet"].get("risk_band_switch_rate_mean"),
        })

        for row in report["rows"]:
            weights = row.get("weights", {})
            weight_rows.append({
                "subset": report.get("subset") or subset,
                "seed": seed,
                "tag": tag,
                "sweep": row["sweep"],
                "weight": row["weight"],
                "value": row["value"],
                "w_rul": weights.get("w_rul"),
                "w_hi": weights.get("w_hi"),
                "w_u": weights.get("w_u"),
                "w_eri": weights.get("w_eri"),
                "action_tv": row["action_tv"],
                "action_switch_rate": row["action_switch_rate"],
                "mean_severity_delta": row["mean_severity_delta"],
                "review_rate_delta": row["review_rate_delta"],
                "risk_rank_spearman": row["risk_rank_spearman"],
                "mean_abs_risk_delta": row["mean_abs_risk_delta"],
                "risk_band_switch_rate": row.get("risk_band_switch_rate"),
                "override_switch_count": row.get("override_switch_count"),
            })

        scenario_rows.extend(run_constraint_scenarios(path))

        print(
            f"  {report.get('subset')} / {tag}: "
            f"most influential = {summary['most_influential_weight']} "
            f"(max action TV {summary['influence_by_weight'].get(summary['most_influential_weight'])})"
        )

    # ---- aggregate over the analysed runs --------------------------
    aggregate: dict[str, Any] = {}
    if summary_rows:
        influence_keys = [key for key in summary_rows[0] if key.startswith("influence_")]
        aggregate = {
            "n_runs": len(summary_rows),
            "mean_influence_by_weight": {
                key.replace("influence_", ""): round(
                    _mean([row[key] for row in summary_rows]), 6
                )
                for key in influence_keys
            },
            "max_influence_by_weight": {
                key.replace("influence_", ""): round(
                    max(row[key] for row in summary_rows if row[key] == row[key]), 6
                )
                for key in influence_keys
            },
            "mean_oat_action_tv_max": round(
                _mean([row["oat_action_tv_max"] for row in summary_rows]), 6
            ),
            "mean_dirichlet_action_tv_mean": round(
                _mean([row["dirichlet_action_tv_mean"] for row in summary_rows]), 6
            ),
            "mean_dirichlet_action_tv_max": round(
                _mean([row["dirichlet_action_tv_max"] for row in summary_rows]), 6
            ),
            "mean_dirichlet_risk_rank_spearman_min": round(
                _mean([row["dirichlet_risk_rank_spearman_min"] for row in summary_rows]), 6
            ),
            "mean_dirichlet_risk_band_switch_rate": round(
                _mean([row["dirichlet_risk_band_switch_rate_mean"] for row in summary_rows]), 6
            ),
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "decision_sensitivity.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "aggregate": aggregate,
                "summary": summary_rows,
                "runs": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_csv(
        REPORTS_DIR / "decision_sensitivity.csv",
        summary_rows,
        tuple(summary_rows[0].keys()) if summary_rows else (),
    )
    _write_csv(
        REPORTS_DIR / "decision_weight_sweep.csv",
        weight_rows,
        WEIGHT_FIELDS,
    )
    if scenario_rows:
        _write_csv(
            REPORTS_DIR / "decision_constraint_scenarios.csv",
            scenario_rows,
            tuple(scenario_rows[0].keys()),
        )

    print()
    print(f"Analysed {len(reports)} run(s).")
    print(f"Report: {json_path}")
    print(f"  {REPORTS_DIR / 'decision_sensitivity.csv'}")
    print(f"  {REPORTS_DIR / 'decision_weight_sweep.csv'}")
    print(f"  {REPORTS_DIR / 'decision_constraint_scenarios.csv'}")
    if aggregate:
        print()
        print("Aggregate influence of each fusion weight (max action-mix TV):")
        for name, value in aggregate["max_influence_by_weight"].items():
            print(f"  {name:<8} {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
