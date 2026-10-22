"""Downstream-chain summary for the multi-seed certification.

What this script produces
-------------------------
The certification harness proves that the *point prediction* of every seed is
reproducible.  A reviewer asking "and do the intervals, the explanations, the
maintenance decisions and the human-review workflow actually come from those
same weights?" needs a second kind of evidence: the artifacts of the
downstream chain, per seed, with the cross-stage consistency check that
``scripts/multiseed_pipeline.py`` performs.

This script reads the artifacts that the pipeline produced for every
(subset, seed) pair and writes

* ``results/multiseed/downstream_per_run.csv`` -- one row per (subset, seed):
  the ERI of both explainability targets, the decision-engine summary
  (action mix, health-state mix, review count, constraint violations, risk and
  urgency), the mean width of the interval the decision engine actually
  consumed, and the HITL review/feedback summary.
* ``results/multiseed/downstream_summary.csv`` -- the same quantities as
  mean ± std across seeds, per subset and for the four-subset aggregate.

Every number comes from a file the pipeline wrote for that seed.  Nothing is
recomputed from a proxy model, and a missing artifact is reported as missing
instead of being silently replaced by a default.

Usage
-----
    python scripts/multiseed_downstream.py --model gru_att_deg_v2

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

from src.utils.paths import RESULTS_DIR

MULTISEED_DIR = PROJECT_ROOT / "results" / "multiseed"
PIPELINE_MANIFEST = MULTISEED_DIR / "pipeline_manifest.json"
XAI_DIR = PROJECT_ROOT / "outputs" / "xai"

ACTIONS = (
    "CONTINUE_OPERATION",
    "MONITOR",
    "INSPECT",
    "SCHEDULE_MAINTENANCE",
    "URGENT_MAINTENANCE",
)

HEALTH_STATES = ("HEALTHY", "DEGRADING", "AT_RISK", "CRITICAL")

UNCERTAINTY_LEVELS = ("LOW", "MODERATE", "HIGH")

SUMMARY_METRICS = (
    "eri_rul",
    "eri_hi",
    "review_rate",
    "mean_interval_width",
    "risk_score_mean",
    "urgency_index_mean",
    "constraint_violation_count",
    "n_decisions",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON artifact, or return ``None`` when it is absent."""

    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* standard deviation (ddof=1)."""

    clean = [float(value) for value in values if value == value]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.fmean(clean)), float(statistics.stdev(clean))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------
# Collection
# ------------------------------------------------------------------

def collect_run(
    subset: str,
    seed: int,
    tag: str,
    model_name: str,
) -> dict[str, Any]:
    """Read the downstream artifacts of one (subset, seed) pair."""

    run_name = f"{model_name}_{tag}" if tag else model_name

    row: dict[str, Any] = {
        "subset": subset,
        "seed": seed,
        "tag": tag,
        "n_decisions": 0,
        "human_review_count": 0,
        "review_rate": float("nan"),
        "constraint_violation_count": 0,
        "risk_score_mean": float("nan"),
        "urgency_index_mean": float("nan"),
        "mean_interval_width": float("nan"),
        "eri_rul": float("nan"),
        "eri_hi": float("nan"),
        "rho_ig_attn": float("nan"),
        "rho_ig_shap": float("nan"),
        "consensus_k": float("nan"),
        "pending_review_count": 0,
        "resolved_review_count": 0,
        "total_feedback": 0,
        "override_rate": float("nan"),
        "acceptance_rate": float("nan"),
        "prognostic_model_updated": "",
    }
    for action in ACTIONS:
        row[f"action_{action}"] = 0
    for state in HEALTH_STATES:
        row[f"health_{state}"] = 0
    for level in UNCERTAINTY_LEVELS:
        row[f"uncertainty_{level}"] = 0

    missing: list[str] = []

    # ---- explainability -------------------------------------------
    xai_dir = XAI_DIR / subset / run_name
    eri_rul = read_json(xai_dir / "eri_rul.json")
    if eri_rul is None:
        missing.append("eri_rul.json")
    else:
        row["eri_rul"] = float(eri_rul.get("eri", float("nan")))
        row["rho_ig_attn"] = float(eri_rul.get("rho_ig_attn", float("nan")))
        row["rho_ig_shap"] = float(eri_rul.get("rho_ig_shap", float("nan")))
        row["consensus_k"] = float(eri_rul.get("consensus_k", float("nan")))

    eri_hi = read_json(xai_dir / "eri_hi.json")
    if eri_hi is None:
        missing.append("eri_hi.json")
    else:
        row["eri_hi"] = float(eri_hi.get("eri", float("nan")))

    # ---- decision engine ------------------------------------------
    decision_path = RESULTS_DIR / f"decision_{subset}_{run_name}.json"
    decision = read_json(decision_path)
    if decision is None:
        missing.append(decision_path.name)
    else:
        summary = decision.get("summary") or {}
        row["n_decisions"] = int(summary.get("num_decisions", 0) or 0)
        row["human_review_count"] = int(summary.get("human_review_count", 0) or 0)
        row["constraint_violation_count"] = int(
            summary.get("constraint_violation_count", 0) or 0
        )
        row["risk_score_mean"] = float(summary.get("risk_score_mean", float("nan")))
        row["urgency_index_mean"] = float(
            summary.get("urgency_index_mean", float("nan"))
        )
        if row["n_decisions"]:
            row["review_rate"] = row["human_review_count"] / row["n_decisions"]

        for action in ACTIONS:
            row[f"action_{action}"] = int(
                (summary.get("action_distribution") or {}).get(action, 0) or 0
            )
        for state in HEALTH_STATES:
            row[f"health_{state}"] = int(
                (summary.get("health_state_distribution") or {}).get(state, 0) or 0
            )
        for level in UNCERTAINTY_LEVELS:
            row[f"uncertainty_{level}"] = int(
                (summary.get("uncertainty_distribution") or {}).get(level, 0) or 0
            )

        # The interval the decision engine actually consumed.
        widths = []
        for decision_record in decision.get("decisions") or []:
            inputs = decision_record.get("inputs") or {}
            lower = inputs.get("rul_lower")
            upper = inputs.get("rul_upper")
            if lower is None or upper is None:
                continue
            widths.append(float(upper) - float(lower))
        if widths:
            row["mean_interval_width"] = float(statistics.fmean(widths))

    # ---- HITL ------------------------------------------------------
    hitl_path = RESULTS_DIR / f"hitl_{subset}_{run_name}.json"
    hitl = read_json(hitl_path)
    if hitl is None:
        missing.append(hitl_path.name)
    else:
        review = hitl.get("review_summary") or {}
        row["pending_review_count"] = int(review.get("pending_review_count", 0) or 0)
        row["resolved_review_count"] = int(review.get("resolved_review_count", 0) or 0)
        feedback = hitl.get("feedback_statistics") or {}
        row["total_feedback"] = int(feedback.get("total_feedback", 0) or 0)
        row["override_rate"] = float(feedback.get("override_rate", float("nan")))
        row["acceptance_rate"] = float(feedback.get("acceptance_rate", float("nan")))
        row["prognostic_model_updated"] = str(
            hitl.get("prognostic_model_updated", "")
        )

    row["missing_artifacts"] = ";".join(missing)
    return row


def summarise(rows: list[dict[str, Any]], subsets: list[str]) -> list[dict[str, Any]]:
    """Mean ± std across seeds, per subset and for the four-subset aggregate."""

    summary_rows: list[dict[str, Any]] = []

    for subset in subsets:
        subset_rows = [row for row in rows if row["subset"] == subset]
        if not subset_rows:
            continue
        entry: dict[str, Any] = {
            "subset": subset,
            "n_seeds": len(subset_rows),
        }
        for metric in SUMMARY_METRICS:
            mean, std = mean_and_std([row[metric] for row in subset_rows])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        summary_rows.append(entry)

    # Four-subset aggregate: average within a seed first, then across seeds.
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if by_seed:
        per_seed_means: dict[int, dict[str, float]] = {}
        for seed, seed_rows in by_seed.items():
            per_seed_means[seed] = {}
            for metric in SUMMARY_METRICS:
                mean, _ = mean_and_std([row[metric] for row in seed_rows])
                per_seed_means[seed][metric] = mean
        entry = {
            "subset": "ALL",
            "n_seeds": len(per_seed_means),
        }
        for metric in SUMMARY_METRICS:
            mean, std = mean_and_std(
                [values[metric] for values in per_seed_means.values()]
            )
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        summary_rows.append(entry)

    return summary_rows


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help="Architecture whose downstream artifacts are summarised.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subsets to include (default: every subset in the pipeline manifest).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to include (default: every seed in the pipeline manifest).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not PIPELINE_MANIFEST.exists():
        print(
            "Missing results/multiseed/pipeline_manifest.json - run "
            "scripts/multiseed_pipeline.py first."
        )
        return 1

    manifest = read_json(PIPELINE_MANIFEST) or {}
    runs = manifest.get("runs") or []

    subsets = args.subsets or list(dict.fromkeys(
        str(run.get("subset")) for run in runs if run.get("subset")
    ))
    seeds = args.seeds or list(dict.fromkeys(
        int(run.get("seed")) for run in runs if run.get("seed") is not None
    ))

    rows: list[dict[str, Any]] = []
    for run in runs:
        subset = str(run.get("subset"))
        seed = int(run.get("seed"))
        if subset not in subsets or seed not in seeds:
            continue
        rows.append(
            collect_run(subset, seed, str(run.get("tag") or ""), args.model)
        )

    if not rows:
        print("No downstream runs matched the requested subsets/seeds.")
        return 1

    per_run_path = MULTISEED_DIR / "downstream_per_run.csv"
    summary_path = MULTISEED_DIR / "downstream_summary.csv"
    write_csv(per_run_path, rows)
    write_csv(summary_path, summarise(rows, subsets))

    incomplete = [row for row in rows if row["missing_artifacts"]]
    print(f"Wrote {per_run_path}")
    print(f"Wrote {summary_path}")
    print(f"Runs summarised: {len(rows)}  incomplete: {len(incomplete)}")
    for row in incomplete:
        print(
            f"  {row['subset']} seed {row['seed']}: missing "
            f"{row['missing_artifacts']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
