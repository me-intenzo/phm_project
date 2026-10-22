"""Multi-seed benchmark for RUL prediction intervals on real checkpoints.

This benchmark evaluates the **deliverable model** (``gru_att_deg_v2``) that was
actually trained by ``scripts/multiseed_train.py``.  There is **no proxy
model** here — every interval is produced by a real checkpoint the pipeline
trained, using the same conformal machinery as the production
``scripts/uncertainty.py`` pipeline.

Calibration protocol
--------------------
The calibration split is reproduced from the checkpoint's seed (the same split
the model was trained on), so calibration engines are never seen during
training.  Two calibration units are reported:

* ``window`` (primary) — one score per calibration window.  This is the
  classic split-conformal unit and it matches the unit the test set is scored
  on (one window per engine), so empirical coverage tracks the nominal level.
* ``engine`` — one score per calibration engine, namely the maximum over that
  engine's windows.  This is deliberately conservative: it bounds the worst
  window of an engine, so at the same nominal level it produces wider
  intervals and over-covers.

Both units are computed in one pass over the same weights and the same
inference, so the two tables are paired by construction and the comparison
between methods is paired within each unit.

The benchmark reuses ``src.uncertainty.pipeline`` so the computation cannot
drift from the production path.  Outputs are written in the same CSV/JSON
layout as before for compatibility with existing tooling.

Usage
-----
    # Use the multi-seed checkpoints produced by multiseed_train.py
    python scripts/multiseed_conformal_benchmark.py --subsets all

    # Override model/seeds
    python scripts/multiseed_conformal_benchmark.py --model gru_att_deg_v2 --seeds 42 43 44 45 46 --tag-prefix ms

    # Only the primary (window-level) table
    python scripts/multiseed_conformal_benchmark.py --calibration-units window

author: me-intenzo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "conformal_benchmark"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
COVERAGES = (0.80, 0.90, 0.95)
METHODS = ("split_conformal", "cqr", "adaptive", "eara_conformal", "nex_scp", "eara_scp")

# Reuse the production pipeline so benchmark and production cannot drift
from src.uncertainty import pipeline
from src.uncertainty.pipeline import COVERAGE_LEVELS
from src.utils.paths import checkpoint_path

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_arrays(subset: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = "train" if split == "train" else "test"
    x = np.load(DATA / f"{subset}_{suffix}_X.npy").astype(np.float32, copy=False)
    y = np.load(DATA / f"{subset}_{suffix}_y_rul.npy").astype(np.float32).reshape(-1)
    engines = np.load(DATA / f"{subset}_{suffix}_engine_ids.npy").reshape(-1)
    if not (len(x) == len(y) == len(engines)):
        raise ValueError(f"Mismatched {subset} {suffix} arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values found in {subset} {suffix} arrays")
    return x, y, engines


def grouped_split(
    engine_ids: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the engine-wise split the checkpoint was trained with."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=int(seed))
    train_idx, cal_idx = next(
        splitter.split(np.zeros(len(engine_ids)), groups=engine_ids)
    )
    if set(engine_ids[train_idx]) & set(engine_ids[cal_idx]):
        raise RuntimeError("Engine leakage between training and calibration sets")
    return train_idx, cal_idx


# ------------------------------------------------------------------
# Model loading (real checkpoints only)
# ------------------------------------------------------------------


def load_trained_model(
    model_name: str,
    subset: str,
    tag: str,
    input_size: int,
    device: torch.device,
):
    """Load a real checkpoint produced by scripts/train.py/multiseed_train.py."""
    import scripts.uncertainty as uncertainty_script

    # The uncertainty script knows how to rebuild the architecture from the
    # checkpoint's stored config and to strip auxiliary heads safely.
    model, checkpoint = uncertainty_script.load_model(
        model_name=model_name,
        input_size=input_size,
        device=device,
        subset=subset,
        tag=tag,
    )
    seed = uncertainty_script.checkpoint_seed(checkpoint)
    return model, checkpoint, seed


# ------------------------------------------------------------------
# Benchmark execution
# ------------------------------------------------------------------


def run_seed(
    model_name: str,
    subset: str,
    seed: int,
    tag: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    x_all, y_all, engine_all = load_arrays(subset, "train")
    x_test, y_test, test_engines = load_arrays(subset, "test")

    # Load the REAL checkpoint for this seed
    model, checkpoint, ckpt_seed = load_trained_model(
        model_name, subset, tag, x_all.shape[-1], device
    )
    # The calibration split must match the split this checkpoint was trained with
    split_seed = int(ckpt_seed)
    train_idx, cal_idx = grouped_split(engine_all, split_seed)

    y_true_train = y_all[train_idx]
    y_true_cal = y_all[cal_idx]
    engine_ids_train = engine_all[train_idx]
    engine_ids_cal = engine_all[cal_idx]

    # Compute all methods via the shared production pipeline
    results_by_unit = pipeline.compute_all_intervals(
        model=model,
        y_true_train=y_true_train,
        y_true_calibration=y_true_cal,
        y_true_test=y_test,
        engine_ids_train=engine_ids_train,
        engine_ids_calibration=engine_ids_cal,
        X_train=x_all[train_idx],
        X_calibration=x_all[cal_idx],
        X_test=x_test,
        device=device,
        seed=split_seed,
        coverage_levels=COVERAGE_LEVELS,
        n_regimes=args.n_regimes,
        top_k=5,
        batch_size=args.batch_size,
        aci_gamma=args.aci_gamma,
        calibration_units=tuple(args.calibration_units),
    )

    import hashlib

    train_engine_ids = np.sort(np.unique(engine_all[train_idx]))
    calibration_engine_ids = np.sort(np.unique(engine_all[cal_idx]))
    split_hash = hashlib.sha256(
        subset.encode("ascii")
        + str(split_seed).encode("ascii")
        + train_engine_ids.tobytes()
        + calibration_engine_ids.tobytes()
    ).hexdigest()[:16]

    split_manifest = {
        "subset": subset,
        "seed": split_seed,
        "tag": tag,
        "model": model_name,
        "checkpoint": str(checkpoint_path(model_name, subset, tag).relative_to(ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split_hash": split_hash,
        "train_engine_ids": train_engine_ids.tolist(),
        "calibration_engine_ids": calibration_engine_ids.tolist(),
        "calibration_scoring": (
            "Scores are reduced to the requested calibration unit(s): "
            "'window' keeps one score per calibration window, 'engine' keeps "
            "the per-engine maximum (conservative). Calibration split "
            "reproduced from the checkpoint seed."
        ),
        "proxy_model_removed": True,
        "real_model": True,
    }

    records = []
    for unit, results_by_method in results_by_unit.items():
        for method in METHODS:
            for coverage in COVERAGES:
                if method not in results_by_method:
                    continue
                if coverage not in results_by_method[method]:
                    continue
                record = results_by_method[method][coverage]
                lower = record["lower"]
                upper = record["upper"]
                for i in range(len(y_test)):
                    records.append(
                        {
                            "subset": subset,
                            "seed": split_seed,
                            "tag": tag,
                            "model": model_name,
                            "calibration_unit": unit,
                            "calibration_scores": record["calibration_units"],
                            "method": method,
                            "coverage_level": float(coverage),
                            "engine_id": str(test_engines[i]),
                            "y_true": float(y_test[i]),
                            "lower": float(lower[i]),
                            "upper": float(upper[i]),
                            "split_hash": split_hash,
                        }
                    )

    return records, split_manifest


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["subset"],
                row["seed"],
                row["calibration_unit"],
                row["method"],
                row["coverage_level"],
            )
        ].append(row)

    seed_metrics = []
    for (subset, seed, unit, method, coverage), group in grouped.items():
        y = np.asarray([r["y_true"] for r in group])
        lower = np.asarray([r["lower"] for r in group])
        upper = np.asarray([r["upper"] for r in group])
        seed_metrics.append(
            {
                "subset": subset,
                "seed": seed,
                "calibration_unit": unit,
                "method": method,
                **pipeline.interval_metrics(y, lower, upper, coverage),
            }
        )
    return seed_metrics


def aggregate(seed_metrics: list[dict]) -> list[dict]:
    keys = ("empirical_coverage", "coverage_gap", "mean_interval_width", "winkler_score")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_metrics:
        grouped[
            (row["subset"], row["calibration_unit"], row["method"], row["nominal_coverage"])
        ].append(row)

    summary = []
    for (subset, unit, method, coverage), group in sorted(grouped.items()):
        item = {
            "subset": subset,
            "calibration_unit": unit,
            "method": method,
            "nominal_coverage": coverage,
            "n_seeds": len(group),
        }
        for key in keys:
            values = np.asarray([r[key] for r in group], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided paired randomization p-value over seed-level differences."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    signs = np.asarray(
        [
            [1.0 if (mask >> bit) & 1 else -1.0 for bit in range(len(differences))]
            for mask in range(1 << len(differences))
        ]
    )
    null_means = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(null_means >= observed - 1e-12))


def paired_comparisons(seed_metrics: list[dict]) -> list[dict]:
    metrics = ("empirical_coverage", "mean_interval_width", "winkler_score")
    lookup = {
        (
            r["subset"],
            r["calibration_unit"],
            r["seed"],
            r["method"],
            r["nominal_coverage"],
        ): r
        for r in seed_metrics
    }
    comparisons = []
    for subset in sorted({r["subset"] for r in seed_metrics}):
        for unit in sorted({r["calibration_unit"] for r in seed_metrics}):
            for coverage in COVERAGES:
                seeds = sorted(
                    {
                        r["seed"]
                        for r in seed_metrics
                        if r["subset"] == subset
                        and r["calibration_unit"] == unit
                        and r["nominal_coverage"] == coverage
                    }
                )
                if not seeds:
                    continue
                # Report significance for both novel-ish candidates against every
                # baseline, so the new EARA-SCP method is evaluated on the same
                # footing as EARA-Conformal.  Neither EARA nor nex_scp is changed.
                for candidate in ("eara_conformal", "eara_scp"):
                    family = []
                    for metric in metrics:
                        for baseline in METHODS:
                            if baseline == candidate:
                                continue
                            differences = np.asarray(
                                [
                                    lookup[
                                        (subset, unit, seed, candidate, coverage)
                                    ][metric]
                                    - lookup[(subset, unit, seed, baseline, coverage)][metric]
                                    for seed in seeds
                                ]
                            )
                            n = len(differences)
                            mean_diff = float(differences.mean())
                            if n > 1:
                                sem = float(differences.std(ddof=1) / np.sqrt(n))
                                critical = float(student_t.ppf(0.975, df=n - 1))
                                ci_low, ci_high = (
                                    mean_diff - critical * sem,
                                    mean_diff + critical * sem,
                                )
                            else:
                                ci_low = ci_high = float("nan")
                            family.append(
                                {
                                    "subset": subset,
                                    "calibration_unit": unit,
                                    "nominal_coverage": coverage,
                                    "metric": metric,
                                    "comparison": f"{candidate} - {baseline}",
                                    "n_seeds": n,
                                    "mean_paired_difference": mean_diff,
                                    "ci95_low": ci_low,
                                    "ci95_high": ci_high,
                                    "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                                }
                            )
                    for metric in metrics:
                        indices = [
                            i for i, row in enumerate(family) if row["metric"] == metric
                        ]
                        ordered = sorted(indices, key=lambda i: family[i]["exact_sign_flip_p"])
                        running = 0.0
                        for rank, index in enumerate(ordered):
                            adjusted = min(
                                1.0,
                                (len(ordered) - rank) * family[index]["exact_sign_flip_p"],
                            )
                            running = max(running, adjusted)
                            family[index]["holm_adjusted_p"] = running
                    comparisons.extend(family)
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for subset in sorted({r["subset"] for r in summary}):
        units = sorted({r.get("calibration_unit", "engine") for r in summary})
        for unit in units:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for coverage in COVERAGES:
                group = [
                    r
                    for r in summary
                    if r["subset"] == subset
                    and r.get("calibration_unit", "engine") == unit
                    and r["nominal_coverage"] == coverage
                ]
                if not group:
                    continue
                x = np.arange(len(METHODS)) + (coverage - 0.9) * 0.8
                by_method = {r["method"]: r for r in group}
                vals = [by_method[m]["empirical_coverage_mean"] for m in METHODS]
                errs = [by_method[m]["empirical_coverage_std"] for m in METHODS]
                axes[0].errorbar(
                    x, vals, yerr=errs, marker="o", capsize=3, label=f"Target {coverage:.0%}"
                )
                vals = [by_method[m]["mean_interval_width_mean"] for m in METHODS]
                errs = [by_method[m]["mean_interval_width_std"] for m in METHODS]
                axes[1].errorbar(
                    x, vals, yerr=errs, marker="o", capsize=3, label=f"Target {coverage:.0%}"
                )
            for coverage in COVERAGES:
                axes[0].axhline(
                    coverage, color="0.55", linewidth=0.7, linestyle="--", alpha=0.65
                )
            axes[0].set_ylabel("Empirical coverage")
            axes[1].set_ylabel("Mean interval width")
            for axis in axes:
                axis.set_xticks(
                    np.arange(len(METHODS)), [m.replace("_", "\n") for m in METHODS]
                )
                axis.grid(axis="y", alpha=0.25)
                axis.legend(frameon=False)
            fig.suptitle(
                f"{subset}: coverage and interval width by method "
                f"(real model, {unit}-level calibration)"
            )
            fig.tight_layout()
            fig.savefig(OUT / f"{subset}_{unit}_coverage_width.png", dpi=160)
            plt.close(fig)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help="Model architecture to benchmark (must have multi-seed checkpoints).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all"],
        help="FD001..FD004 or all",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="ms",
        help="Tag prefix used by multiseed_train.py (default 'ms' -> tag 'ms42').",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--aci-gamma", type=float, default=0.02)
    parser.add_argument(
        "--calibration-units",
        nargs="+",
        choices=("window", "engine"),
        default=["window", "engine"],
        help=(
            "Calibration units to report. 'window' (one score per calibration "
            "window) matches the unit the test set is scored on; 'engine' (one "
            "max score per engine) is conservative and over-covers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = list(SUBSETS) if "all" in args.subsets else args.subsets
    if set(subsets) - set(SUBSETS):
        raise SystemExit(f"Unknown subset(s): {set(subsets) - set(SUBSETS)}")
    if len(set(args.seeds)) < 2:
        raise SystemExit("At least two distinct seeds are required")
    if not 0.0 < args.aci_gamma <= 1.0:
        raise SystemExit("--aci-gamma must be in (0, 1]")
    if args.n_regimes < 2:
        raise SystemExit("--n-regimes must be at least 2 for EARA-Conformal")

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows: list[dict] = []
    split_manifests = []
    for subset in subsets:
        for seed in args.seeds:
            tag = f"{args.tag_prefix}{seed}"
            print(f"Running {subset}, seed={seed} (tag={tag}) on {device}", flush=True)
            rows, manifest = run_seed(args.model, subset, seed, tag, args, device)
            all_rows.extend(rows)
            split_manifests.append(manifest)

    seed_metrics = summarize(all_rows)
    summary = aggregate(seed_metrics)
    comparisons = paired_comparisons(seed_metrics)

    write_csv(OUT / "per_engine_intervals.csv", all_rows)
    write_csv(OUT / "per_seed_metrics.csv", seed_metrics)
    write_csv(OUT / "summary_mean_std.csv", summary)
    write_csv(OUT / "paired_comparisons.csv", comparisons)
    (OUT / "split_manifest.json").write_text(
        json.dumps(split_manifests, indent=2), encoding="utf-8"
    )
    (OUT / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "tag_prefix": args.tag_prefix,
                "subsets": list(subsets),
                "seeds": args.seeds,
                "batch_size": args.batch_size,
                "n_regimes": args.n_regimes,
                "aci_gamma": args.aci_gamma,
                "coverage_levels": list(COVERAGES),
                "methods": list(METHODS),
                "device": str(device),
                "nex_scp_protocol": "Prequential adaptive miscoverage; each test engine outcome updates alpha for the next engine.",
                "calibration_units": list(args.calibration_units),
                "calibration_unit_note": (
                    "'window' keeps one score per calibration window (the unit "
                    "the test set is scored on); 'engine' keeps the per-engine "
                    "maximum and is deliberately conservative."
                ),
                "proxy_model_removed": True,
                "real_model_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary)
    print(f"Wrote benchmark outputs to {OUT}")


if __name__ == "__main__":
    main()
