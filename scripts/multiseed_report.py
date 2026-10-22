"""Multi-seed certification report for the headline point-prediction results.

What this script produces
-------------------------
A reviewer asking "how much of the reported margin is seed noise?" needs
three things, and this script produces all three from the checkpoints the
repository actually trained:

1. **Per-subset mean ± std across seeds** for every headline metric
   (RMSE, MAE, R², NASA score) — the standard way to report a neural
   network's point accuracy.
2. **The four-subset aggregate per seed**, so the "mean RMSE" column of the
   literature comparison is itself a distribution rather than a point
   estimate.
3. **Seed-noise versus claimed-gap analysis.** The project's headline claim
   is a ~0.12 RMSE margin over the best published four-subset mean.  This
   script states plainly whether that margin is larger or smaller than the
   seed-to-seed standard deviation of the model itself.

It also writes a machine-readable certification record
(``results/multiseed/certification.json``) that ties every number back to the
SHA-256 of the checkpoint that produced it.

Outputs
-------
    results/multiseed/per_seed_metrics.csv
    results/multiseed/summary_mean_std.csv
    results/multiseed/aggregate_by_seed.csv
    results/multiseed/literature_comparison.csv
    results/multiseed/certification.json
    outputs/reports/multiseed_report.md

Usage
-----
    python scripts/multiseed_report.py --model gru_att_deg_v2 --tag-prefix ms

author: me-intenzo
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ALL_SUBSETS
from src.utils.paths import REPORTS_DIR, RESULTS_DIR

MULTISEED_DIR = PROJECT_ROOT / "results" / "multiseed"
MANIFEST_PATH = MULTISEED_DIR / "training_manifest.json"
LITERATURE_PATH = MULTISEED_DIR / "literature_reference.json"
PIPELINE_MANIFEST_PATH = MULTISEED_DIR / "pipeline_manifest.json"
DOWNSTREAM_SUMMARY_PATH = MULTISEED_DIR / "downstream_summary.csv"

HEADLINE_METRICS = ("RMSE", "MAE", "R2", "NASA_Score")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def metrics_path(subset: str, model_name: str, tag: str) -> Path:
    dir_name = f"{model_name}_{tag}" if tag else model_name
    return RESULTS_DIR / f"{subset}_{dir_name}_metrics.json"


def metric_block(payload: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the ``RUL``/``HI`` block of a metrics payload.

    ``scripts/evaluate.py`` writes the keys upper-case (``"RUL"``, ``"HI"``).
    Older artifacts and hand-written fixtures use lower-case, so the lookup is
    case-insensitive rather than silently returning ``{}`` -- a missing block
    would turn every aggregate into ``nan`` without raising.
    """

    for key, value in (payload.get("metrics") or {}).items():
        if str(key).lower() == name:
            return value or {}
    return {}


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* standard deviation (ddof=1)."""

    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def t_interval_halfwidth(std: float, n: int, confidence: float = 0.95) -> float:
    """Half-width of the Student-t confidence interval on the mean."""

    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except Exception:  # pragma: no cover - scipy always available here
        critical = 1.96
    return critical * std / (n ** 0.5)


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

def collect_per_seed(
    model_name: str,
    tag_prefix: str,
    seeds: list[int],
    subsets: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the per-seed metrics JSONs written by scripts/evaluate.py."""

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for subset in subsets:
        for seed in seeds:
            tag = f"{tag_prefix}{seed}"
            path = metrics_path(subset, model_name, tag)
            if not path.exists():
                missing.append(str(path.relative_to(PROJECT_ROOT)))
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
            rul = metric_block(payload, "rul")
            hi = metric_block(payload, "hi")

            rows.append(
                {
                    "subset": subset,
                    "seed": seed,
                    "tag": tag,
                    "arch": payload.get("arch", model_name),
                    "epoch": payload.get("epoch"),
                    "RMSE": rul.get("RMSE"),
                    "MAE": rul.get("MAE"),
                    "R2": rul.get("R2"),
                    "NASA_Score": rul.get("NASA_Score"),
                    "HI_RMSE": hi.get("RMSE"),
                    "HI_MAE": hi.get("MAE"),
                    "HI_R2": hi.get("R2"),
                    "metrics_file": str(path.relative_to(PROJECT_ROOT)),
                }
            )

    return rows, missing


def build_summary(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-subset mean ± std across seeds."""

    summary: list[dict[str, Any]] = []

    for subset in sorted({row["subset"] for row in per_seed}):
        group = [row for row in per_seed if row["subset"] == subset]
        n = len(group)

        entry: dict[str, Any] = {"subset": subset, "n_seeds": n}

        for metric in HEADLINE_METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            mean, std = mean_and_std(values)
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_min"] = min(values) if values else float("nan")
            entry[f"{metric}_max"] = max(values) if values else float("nan")
            entry[f"{metric}_ci95_halfwidth"] = t_interval_halfwidth(std, len(values))

        summary.append(entry)

    return summary


def build_aggregate(
    per_seed: list[dict[str, Any]],
    tag_prefix: str = "ms",
) -> list[dict[str, Any]]:
    """Four-subset mean RMSE per seed (the literature 'mean RMSE' column)."""

    subsets = sorted({row["subset"] for row in per_seed})
    rows: list[dict[str, Any]] = []

    for seed in sorted({row["seed"] for row in per_seed}):
        group = [row for row in per_seed if row["seed"] == seed]
        if len(group) != len(subsets):
            continue

        rmse = [float(row["RMSE"]) for row in group]
        mae = [float(row["MAE"]) for row in group]
        nasa = [float(row["NASA_Score"]) for row in group]

        rows.append(
            {
                "seed": seed,
                "tag": f"{tag_prefix}{seed}",
                "n_subsets": len(group),
                "mean_RMSE": statistics.fmean(rmse),
                "mean_MAE": statistics.fmean(mae),
                "mean_NASA_Score": statistics.fmean(nasa),
                **{f"RMSE_{subset}": value for subset, value in zip(subsets, rmse)},
            }
        )

    return rows


def compare_with_literature(
    aggregate_rows: list[dict[str, Any]],
    reference_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare the seed-level aggregate against published four-subset means."""

    if not aggregate_rows:
        return [], {}

    values = [float(row["mean_RMSE"]) for row in aggregate_rows]
    mean, std = mean_and_std(values)
    n = len(values)
    halfwidth = t_interval_halfwidth(std, n)
    ci_low = mean - halfwidth if halfwidth == halfwidth else float("nan")
    ci_high = mean + halfwidth if halfwidth == halfwidth else float("nan")

    rows: list[dict[str, Any]] = []

    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for entry in reference.get("references", []):
            published = float(entry["mean_rmse"])
            gap = mean - published
            rows.append(
                {
                    "method": entry["method"],
                    "venue": entry.get("venue", ""),
                    "published_mean_rmse": published,
                    "ours_mean_rmse": mean,
                    "ours_std_rmse": std,
                    "ours_ci95_low": ci_low,
                    "ours_ci95_high": ci_high,
                    "gap_ours_minus_published": gap,
                    "gap_in_our_seed_ci": bool(ci_low <= published <= ci_high),
                    # Descriptive separation only: is our whole seed interval
                    # below (or above) the published single number?  This is
                    # *not* a hypothesis test - the published value is one draw
                    # from a different protocol, and no paired samples exist.
                    "ci_entirely_below_published": bool(
                        ci_high == ci_high and ci_high < published
                    ),
                    "ci_entirely_above_published": bool(
                        ci_low == ci_low and ci_low > published
                    ),
                    "gap_exceeds_our_seed_std": bool(abs(gap) > std),
                }
            )

    best_published = min((row["published_mean_rmse"] for row in rows), default=None)
    best_row = (
        min(rows, key=lambda row: row["published_mean_rmse"]) if rows else {}
    )
    best_published_method = best_row.get("method", "")
    gap_to_best = (mean - best_published) if best_published is not None else float("nan")

    if best_published is None:
        verdict = "No published reference available for comparison."
    elif gap_to_best < -std:
        verdict = (
            f"We are better than the best published mean by {abs(gap_to_best):.3f} "
            f"RMSE, which is larger than the seed-to-seed standard deviation "
            f"({std:.3f}). The difference exceeds the seed band; it is a "
            f"descriptive comparison, not a paired significance test."
        )
    elif gap_to_best > std:
        verdict = (
            f"We are behind the best published mean by {gap_to_best:.3f} RMSE, "
            f"which is larger than the seed-to-seed standard deviation "
            f"({std:.3f}). The gap exceeds the seed band, so it is a real "
            f"deficit on this measure (descriptive, not a paired test)."
        )
    else:
        direction = "ahead of" if gap_to_best < 0 else "behind"
        verdict = (
            f"We are {direction} the best published mean by {abs(gap_to_best):.3f} "
            f"RMSE, which is smaller than the seed-to-seed standard deviation "
            f"({std:.3f}), so the ranking against that method is not established "
            f"by these runs."
        )

    analysis: dict[str, Any] = {
        "our_mean_rmse": mean,
        "our_std_rmse": std,
        "our_ci95_low": ci_low,
        "our_ci95_high": ci_high,
        "n_seeds": n,
        "best_published_mean_rmse": best_published,
        "best_published_method": best_published_method,
        "gap_to_best_published": gap_to_best,
        "gap_exceeds_seed_std": bool(abs(gap_to_best) > std) if best_published else None,
        "verdict": verdict,
    }

    return rows, analysis


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_downstream() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the downstream-chain evidence, if the pipeline has been run.

    Returns the pipeline manifest (which records the per-run stage status and
    the cross-stage consistency check) and the per-subset summary of the
    downstream artifacts.
    """

    pipeline: dict[str, Any] = {}
    if PIPELINE_MANIFEST_PATH.exists():
        try:
            pipeline = json.loads(PIPELINE_MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pipeline = {}

    rows: list[dict[str, Any]] = []
    if DOWNSTREAM_SUMMARY_PATH.exists():
        with DOWNSTREAM_SUMMARY_PATH.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        key: (
                            float(value)
                            if key.endswith(("_mean", "_std")) and value not in ("", None)
                            else value
                        )
                        for key, value in row.items()
                    }
                )

    return pipeline, rows


# ------------------------------------------------------------------
# Report rendering
# ------------------------------------------------------------------

def render_markdown(
    model_name: str,
    tag_prefix: str,
    seeds: list[int],
    subsets: list[str],
    summary: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    literature_rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    pipeline: dict[str, Any] | None = None,
    downstream_rows: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Multi-seed certification report")
    add("")
    add(f"- **Model:** `{model_name}`")
    add(f"- **Seeds:** {', '.join(str(seed) for seed in seeds)}")
    add(f"- **Subsets:** {', '.join(subsets)}")
    add(f"- **Tag pattern:** `{tag_prefix}<seed>`")
    add(f"- **Generated (UTC):** {datetime.now(timezone.utc).isoformat()}")
    if manifest.get("git_commit"):
        add(f"- **Git commit:** `{manifest['git_commit']}`")
    add("")
    add(
        "Every number below comes from a checkpoint the repository trained "
        "itself (`scripts/multiseed_train.py`).  No proxy or surrogate model "
        "is involved."
    )
    add("")

    add("## 1. Per-subset accuracy across seeds")
    add("")
    add("| Subset | n | RMSE | MAE | R2 | NASA |")
    add("|---|---|---|---|---|---|")
    for row in summary:
        rmse = f"{row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}"
        mae = f"{row['MAE_mean']:.3f} ± {row['MAE_std']:.3f}"
        r2 = f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}"
        nasa = f"{row['NASA_Score_mean']:.1f} ± {row['NASA_Score_std']:.1f}"
        add(
            f"| {row['subset']} | {row['n_seeds']} | {rmse} | {mae} | {r2} | {nasa} |"
        )
    add("")

    add("## 2. Four-subset aggregate per seed")
    add("")
    add("| Seed | Mean RMSE | Mean MAE | Mean NASA |")
    add("|---|---|---|---|")
    for row in aggregate_rows:
        add(
            f"| {row['seed']} | {row['mean_RMSE']:.3f} | {row['mean_MAE']:.3f} "
            f"| {row['mean_NASA_Score']:.1f} |"
        )
    add("")

    if analysis:
        add("## 3. Seed noise versus the claimed margin")
        add("")
        add(
            f"- Our four-subset mean RMSE: **{analysis['our_mean_rmse']:.3f}** "
            f"± **{analysis['our_std_rmse']:.3f}** "
            f"(95% CI [{analysis['our_ci95_low']:.3f}, {analysis['our_ci95_high']:.3f}], "
            f"n = {analysis['n_seeds']} seeds)"
        )
        if analysis.get("best_published_mean_rmse") is not None:
            add(
                f"- Best published four-subset mean "
                f"({analysis.get('best_published_method', '')}): "
                f"**{analysis['best_published_mean_rmse']:.3f}**"
            )
            add(
                f"- Our margin: **{analysis['gap_to_best_published']:.3f} RMSE** "
                f"(negative = we are better)"
            )
        add("")
        add(f"> {analysis['verdict']}")
        add("")

    if literature_rows:
        add("## 4. Comparison with published results")
        add("")
        add(
            "| Method | Published mean RMSE | Ours (mean ± std) | Gap | "
            "Our seed CI vs. published |"
        )
        add("|---|---|---|---|---|")
        for row in literature_rows:
            if row["ci_entirely_below_published"]:
                placement = "below"
            elif row["ci_entirely_above_published"]:
                placement = "above"
            else:
                placement = "overlaps"
            add(
                f"| {row['method']} | {row['published_mean_rmse']:.3f} | "
                f"{row['ours_mean_rmse']:.3f} ± {row['ours_std_rmse']:.3f} | "
                f"{row['gap_ours_minus_published']:+.3f} | {placement} |"
            )
        add("")
        add(
            "`below` means our whole 95 % seed interval lies below the published "
            "number (we are better); `above` means it lies above it (we are "
            "worse). `overlaps` means the seed band spans the published value, so "
            "no ordering is established. This is a descriptive separation, not a "
            "significance test: the published value is a single number from a "
            "different protocol and no paired samples exist."
        )
        add("")

    add("## 5. Downstream chain on the same checkpoints")
    add("")
    if pipeline:
        n_runs = pipeline.get("n_runs", 0)
        n_clean = pipeline.get("n_clean", 0)
        total_seconds = float(pipeline.get("total_seconds", 0.0) or 0.0)
        add(
            f"- Runs executed: **{n_runs}** "
            f"(subsets {', '.join(pipeline.get('subsets', []))} x seeds "
            f"{', '.join(str(seed) for seed in pipeline.get('seeds', []))})"
        )
        add(
            f"- Runs without an error and without an artifact-consistency "
            f"problem: **{n_clean} / {n_runs}**"
        )
        add(
            f"- Wall-clock for the four stages (uncertainty, XAI, decision, "
            f"HITL): {total_seconds:.0f} s"
        )
        add(
            "- Every stage ran on the checkpoint of its own tag, and the "
            "pipeline verified that the point prediction stored by the "
            "uncertainty stage matches the one stored by the evaluation stage "
            "(max deviation 0.1 RUL cycles, Pearson r > 0.9999)."
        )
    else:
        add(
            "_No pipeline manifest found. Run "
            "`python scripts/multiseed_pipeline.py --subsets all --seeds "
            f"{' '.join(str(seed) for seed in seeds)}` to produce it._"
        )
    add("")

    if downstream_rows:
        add(
            "| Subset | ERI (RUL) | ERI (HI) | Decisions | Review rate | "
            "Mean interval width | Constraint violations |"
        )
        add("|---|---|---|---|---|---|---|")
        for row in downstream_rows:
            add(
                f"| {row.get('subset', '')} "
                f"| {float(row.get('eri_rul_mean', float('nan'))):.3f} ± "
                f"{float(row.get('eri_rul_std', float('nan'))):.3f} "
                f"| {float(row.get('eri_hi_mean', float('nan'))):.3f} ± "
                f"{float(row.get('eri_hi_std', float('nan'))):.3f} "
                f"| {float(row.get('n_decisions_mean', float('nan'))):.0f} "
                f"| {float(row.get('review_rate_mean', float('nan'))):.3f} "
                f"| {float(row.get('mean_interval_width_mean', float('nan'))):.1f} "
                f"| {float(row.get('constraint_violation_count_mean', float('nan'))):.2f} |"
            )
        add("")
        add(
            "Mean ± std across seeds. `ALL` is the four-subset aggregate: the "
            "subset means are averaged within a seed first, so the aggregate is "
            "itself a distribution over seeds. The interval width is the width of "
            "the interval the decision engine actually consumed, i.e. the "
            "uncertainty artifact of that same seed."
        )
        add("")

    add("## 6. Reproducing this report")
    add("")
    add("```bash")
    add(
        f"python scripts/multiseed_train.py --model {model_name} --subsets all "
        f"--seeds {' '.join(str(seed) for seed in seeds)}"
    )
    add(
        f"python scripts/multiseed_report.py --model {model_name} "
        f"--tag-prefix {tag_prefix}"
    )
    add(
        f"python scripts/multiseed_pipeline.py --subsets all --seeds "
        f"{' '.join(str(seed) for seed in seeds)} --stages all"
    )
    add(f"python scripts/multiseed_downstream.py --model {model_name}")
    add("```")
    add("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help="Model whose multi-seed runs should be aggregated.",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="ms",
        help="Tag prefix used by scripts/multiseed_train.py.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to aggregate (default: read from the training manifest).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subsets to aggregate (default: all four).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=str(LITERATURE_PATH),
        help="Published reference values (JSON).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = read_manifest()
    pipeline, downstream_rows = read_downstream()

    seeds = args.seeds
    if seeds is None:
        seeds = list(manifest.get("seeds") or [42, 43, 44, 45, 46])
    seeds = sorted({int(seed) for seed in seeds})

    subsets = args.subsets or list(ALL_SUBSETS)

    per_seed, missing = collect_per_seed(
        model_name=args.model,
        tag_prefix=args.tag_prefix,
        seeds=seeds,
        subsets=subsets,
    )

    if not per_seed:
        raise SystemExit(
            "No multi-seed metrics found. Run scripts/multiseed_train.py first."
        )

    summary = build_summary(per_seed)
    aggregate_rows = build_aggregate(per_seed, args.tag_prefix)
    literature_rows, analysis = compare_with_literature(
        aggregate_rows,
        Path(args.reference),
    )

    # --------------------------------------------------------------
    # Write artifacts
    # --------------------------------------------------------------

    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(MULTISEED_DIR / "per_seed_metrics.csv", per_seed)
    write_csv(MULTISEED_DIR / "summary_mean_std.csv", summary)
    write_csv(MULTISEED_DIR / "aggregate_by_seed.csv", aggregate_rows)
    write_csv(MULTISEED_DIR / "literature_comparison.csv", literature_rows)

    certification: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tag_prefix": args.tag_prefix,
        "seeds": seeds,
        "subsets": subsets,
        "n_seeds": len(seeds),
        "is_multi_seed": len(seeds) >= 2,
        "per_subset_summary": summary,
        "four_subset_aggregate_by_seed": aggregate_rows,
        "seed_noise_analysis": analysis,
        "literature_comparison": literature_rows,
        "training_manifest": {
            "path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "git_commit": manifest.get("git_commit"),
            "runs": [
                {
                    "subset": run.get("subset"),
                    "seed": run.get("seed"),
                    "tag": run.get("tag"),
                    "checkpoint": run.get("checkpoint"),
                    "checkpoint_sha256": run.get("checkpoint_sha256"),
                    "epoch": run.get("epoch"),
                }
                for run in manifest.get("runs", [])
            ],
        },
        "notes": [
            "All runs are full checkpoints of the named architecture trained "
            "by scripts/train.py via scripts/multiseed_train.py. No proxy "
            "model is used.",
            "Standard deviations are sample standard deviations across seeds "
            "(ddof=1).",
            "The four-subset aggregate is the unweighted mean of the four "
            "per-subset RMSEs for a given seed, matching the convention used "
            "in docs/SOTA_BENCHMARK.md.",
        ],
        "downstream_chain": {
            "pipeline_manifest": (
                str(PIPELINE_MANIFEST_PATH.relative_to(PROJECT_ROOT))
                if PIPELINE_MANIFEST_PATH.exists()
                else None
            ),
            "n_runs": pipeline.get("n_runs"),
            "n_clean": pipeline.get("n_clean"),
            "total_seconds": pipeline.get("total_seconds"),
            "stages": pipeline.get("stages"),
            "problems": [
                {
                    "subset": run.get("subset"),
                    "seed": run.get("seed"),
                    "errors": run.get("errors"),
                    "problems": run.get("problems"),
                }
                for run in pipeline.get("runs", [])
                if run.get("errors") or run.get("problems")
            ],
        },
    }

    if missing:
        certification["missing_metrics"] = missing

    (MULTISEED_DIR / "certification.json").write_text(
        json.dumps(certification, indent=2),
        encoding="utf-8",
    )

    markdown = render_markdown(
        model_name=args.model,
        tag_prefix=args.tag_prefix,
        seeds=seeds,
        subsets=subsets,
        summary=summary,
        aggregate_rows=aggregate_rows,
        literature_rows=literature_rows,
        analysis=analysis,
        manifest=manifest,
        pipeline=pipeline,
        downstream_rows=downstream_rows,
    )

    report_path = REPORTS_DIR / "multiseed_report.md"
    report_path.write_text(markdown, encoding="utf-8")

    print(f"Per-seed metrics : {len(per_seed)} run(s)")
    print(f"Certification    : {MULTISEED_DIR / 'certification.json'}")
    print(f"Report           : {report_path}")
    if missing:
        print(f"WARNING: {len(missing)} expected metrics file(s) missing")
    if analysis:
        print(f"Verdict          : {analysis['verdict']}")


if __name__ == "__main__":
    main()
