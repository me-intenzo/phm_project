"""
sota_benchmark.py

Reproducible SOTA-benchmark evidence for the report in
``docs/SOTA_BENCHMARK.md``.

The script only reads prediction artifacts that already exist in
``outputs/results`` (written by ``scripts/evaluate.py`` and
``scripts/ensemble.py``).  It never re-trains or touches the test labels
in any way, so the numbers are exactly the ones the checkpoints produce.

What it adds on top of ``scripts/summarize.py``
-----------------------------------------------
1. Engine-level paired bootstrap confidence intervals for the
   four-subset mean RUL RMSE (10,000 resamples by default).
2. Paired bootstrap tests of the RMSE difference between a reference
   model and every other model.  The same engine indices are resampled
   for both models, which preserves the correlation between their errors.
3. Five CSVs, so the report tables can be regenerated without editing
   prose: ``sota_benchmark.csv``, ``sota_bootstrap_ci.csv``,
   ``sota_significance.csv``, ``sota_error_profile.csv`` and
   ``sota_error_bands.csv`` under ``outputs/reports/``.

Usage
-----
python scripts/sota_benchmark.py
python scripts/sota_benchmark.py --reference gru_att_deg_v2 --resamples 20000
python scripts/sota_benchmark.py --models gru_att_deg_v2,ensemble,gru_att_deg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import nasa_score

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

DEFAULT_MODELS = [
    "gru_att_deg_v2",
    "ensemble",
    "gru_att_deg",
    "gru",
    "lstm",
    "hybrid",
    "transformer",
]

DEFAULT_REFERENCE = "gru_att_deg_v2"


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for C-MAPSS models."
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model stems to compare.",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help=f"Reference model for paired tests (default: {DEFAULT_REFERENCE}).",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap RNG seed (default: 42).",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_predictions(subset: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` for one subset/model pair."""

    path = RESULTS_DIR / f"{subset}_{model}_predictions.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}\n"
            "Run scripts/evaluate.py (or scripts/ensemble.py) first."
        )

    with np.load(path) as data:
        y_true = np.asarray(data["y_rul_true"], dtype=np.float64)
        y_pred = np.asarray(data["y_rul_pred"], dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch in {path.name}")

    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

def bootstrap_mean_rmse(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the unweighted mean of per-subset RMSE.

    Engines are resampled with replacement inside each subset, so the
    subset's own test size determines its uncertainty.  All subsets stay
    present in every replicate, matching how the aggregate is reported.
    """

    estimates = np.empty(resamples, dtype=np.float64)

    for b in range(resamples):
        subset_rmse = []

        for subset, (y_true, y_pred) in data.items():
            idx = rng.integers(0, len(y_true), size=len(y_true))
            subset_rmse.append(rmse(y_true[idx], y_pred[idx]))

        estimates[b] = float(np.mean(subset_rmse))

    return (
        float(estimates.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def paired_delta(
    reference: dict[str, tuple[np.ndarray, np.ndarray]],
    candidate: dict[str, tuple[np.ndarray, np.ndarray]],
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Paired bootstrap for ``RMSE(candidate) - RMSE(reference)``.

    Negative values mean the candidate is better.  A two-sided p-value is
    approximated as twice the smaller tail mass of the bootstrap
    distribution relative to zero.
    """

    deltas = np.empty(resamples, dtype=np.float64)

    for b in range(resamples):

        per_subset = []

        for subset, (y_true, _) in reference.items():
            y_ref = reference[subset][1]
            y_cand = candidate[subset][1]

            if len(y_cand) != len(y_ref):
                raise ValueError(
                    f"Paired comparison requires equal test sizes on {subset}: "
                    f"{len(y_ref)} vs {len(y_cand)}"
                )

            idx = rng.integers(0, len(y_true), size=len(y_true))

            per_subset.append(
                rmse(y_true[idx], y_cand[idx])
                - rmse(y_true[idx], y_ref[idx])
            )

        deltas[b] = float(np.mean(per_subset))

    observed = float(
        np.mean([
            rmse(reference[s][0], candidate[s][1])
            - rmse(reference[s][0], reference[s][1])
            for s in reference
        ])
    )

    lower = float(np.percentile(deltas, 2.5))
    upper = float(np.percentile(deltas, 97.5))

    tail = float(np.mean(deltas >= 0.0)) if observed < 0 else float(np.mean(deltas <= 0.0))
    p_value = min(1.0, 2.0 * tail)

    return {
        "delta": observed,
        "ci_low": lower,
        "ci_high": upper,
        "p_boot": p_value,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rng = np.random.default_rng(args.seed)

    if args.reference not in models:
        models.append(args.reference)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Per-subset metrics
    # --------------------------------------------------------------

    rows: list[dict] = []
    all_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    print()
    print("Per-subset RUL accuracy (recomputed from saved predictions)")
    print("-" * 78)

    for model in models:

        all_data[model] = {}

        for subset in SUBSETS:
            y_true, y_pred = load_predictions(subset, model)
            all_data[model][subset] = (y_true, y_pred)

            rows.append(
                {
                    "model": model,
                    "subset": subset,
                    "n": len(y_true),
                    "rul_mae": mae(y_true, y_pred),
                    "rul_rmse": rmse(y_true, y_pred),
                    "rul_r2": r2(y_true, y_pred),
                    "nasa": nasa_score(y_true, y_pred),
                }
            )

    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        line = "  ".join(
            f"{r['subset']}={r['rul_rmse']:.3f}" for r in model_rows
        )
        mean_rmse = float(np.mean([r["rul_rmse"] for r in model_rows]))
        print(f"{model:<18} {line}  mean={mean_rmse:.3f}")

    # --------------------------------------------------------------
    # Bootstrap CIs for the four-subset mean
    # --------------------------------------------------------------

    print()
    print(f"Bootstrap 95% CI for mean RUL RMSE ({args.resamples:,} resamples)")
    print("-" * 78)

    ci_rows: list[dict] = []

    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        mean_est, lower, upper = bootstrap_mean_rmse(
            all_data[model],
            resamples=args.resamples,
            rng=rng,
        )
        ci_rows.append(
            {
                "model": model,
                "mean_mae": float(np.mean([r["rul_mae"] for r in model_rows])),
                "mean_rmse": mean_est,
                "ci_low": lower,
                "ci_high": upper,
                "mean_r2": float(np.mean([r["rul_r2"] for r in model_rows])),
                "mean_nasa": float(np.mean([r["nasa"] for r in model_rows])),
                "resamples": args.resamples,
            }
        )
        print(f"{model:<18} {mean_est:.3f}  [{lower:.3f}, {upper:.3f}]")

    # --------------------------------------------------------------
    # Paired tests against the reference
    # --------------------------------------------------------------

    print()
    print(f"Paired bootstrap vs {args.reference} (negative delta is better)")
    print("-" * 78)

    sig_rows: list[dict] = []

    for model in models:

        if model == args.reference:
            continue

        result = paired_delta(
            all_data[args.reference],
            all_data[model],
            resamples=args.resamples,
            rng=rng,
        )
        sig_rows.append({"model": model, **result})
        print(
            f"{model:<18} delta={result['delta']:+.3f}  "
            f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]  "
            f"p_boot={result['p_boot']:.4f}"
        )

    # --------------------------------------------------------------
    # Error anatomy (bands + per-subset regression profile)
    # --------------------------------------------------------------

    bands = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]

    band_rows: list[dict] = []
    profile_rows: list[dict] = []

    print()
    print("Error anatomy (all models; per-subset details for the reference)")
    print("-" * 78)

    for model in models:

        for subset in SUBSETS:
            y_true, y_pred = all_data[model][subset]
            errors = y_pred - y_true

            slope = (
                float(np.polyfit(y_true, y_pred, 1)[0])
                if float(np.std(y_true)) > 0
                else float("nan")
            )

            profile_rows.append(
                {
                    "model": model,
                    "subset": subset,
                    "slope": slope,
                    "mean_error": float(np.mean(errors)),
                    "p90_abs_error": float(np.percentile(np.abs(errors), 90)),
                    "pct_late": float(100.0 * np.mean(errors > 0)),
                    "pct_abs_gt_20": float(100.0 * np.mean(np.abs(errors) > 20)),
                }
            )

        pooled_true = np.concatenate(
            [all_data[model][s][0] for s in SUBSETS]
        )
        pooled_pred = np.concatenate(
            [all_data[model][s][1] for s in SUBSETS]
        )

        for low, high in bands:
            mask = (pooled_true >= low) & (pooled_true < high)
            errors = pooled_pred[mask] - pooled_true[mask]

            band_rows.append(
                {
                    "model": model,
                    "band": f"{low}-{high}",
                    "n": int(mask.sum()),
                    "mae": float(np.mean(np.abs(errors))),
                    "bias": float(np.mean(errors)),
                }
            )

    if args.reference in models:
        print(f"Reference model: {args.reference}")
        for row in profile_rows:
            if row["model"] == args.reference:
                print(
                    f"  {row['subset']} slope={row['slope']:.3f} "
                    f"bias={row['mean_error']:+.2f} "
                    f"p90|e|={row['p90_abs_error']:.2f} "
                    f"late={row['pct_late']:.1f}% "
                    f"|e|>20={row['pct_abs_gt_20']:.1f}%"
                )
        for row in band_rows:
            if row["model"] == args.reference:
                print(
                    f"  band {row['band']:>7}  n={row['n']:3d} "
                    f"mae={row['mae']:.2f} bias={row['bias']:+.2f}"
                )

    # --------------------------------------------------------------
    # CSV outputs
    # --------------------------------------------------------------

    benchmark_path = REPORTS_DIR / "sota_benchmark.csv"

    with open(benchmark_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "subset",
                "n",
                "rul_mae",
                "rul_rmse",
                "rul_r2",
                "nasa",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ci_path = REPORTS_DIR / "sota_bootstrap_ci.csv"

    with open(ci_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "mean_mae",
                "mean_rmse",
                "ci_low",
                "ci_high",
                "mean_r2",
                "mean_nasa",
                "resamples",
            ],
        )
        writer.writeheader()
        writer.writerows(ci_rows)

    significance_path = REPORTS_DIR / "sota_significance.csv"

    with open(significance_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "delta", "ci_low", "ci_high", "p_boot"],
        )
        writer.writeheader()
        writer.writerows(sig_rows)

    print()
    print(f"Wrote {benchmark_path}")
    print(f"Wrote {ci_path}")

    bands_path = REPORTS_DIR / "sota_error_bands.csv"

    with open(bands_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "band", "n", "mae", "bias"],
        )
        writer.writeheader()
        writer.writerows(band_rows)

    profile_path = REPORTS_DIR / "sota_error_profile.csv"

    with open(profile_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "subset",
                "slope",
                "mean_error",
                "p90_abs_error",
                "pct_late",
                "pct_abs_gt_20",
            ],
        )
        writer.writeheader()
        writer.writerows(profile_rows)

    print(f"Wrote {bands_path}")
    print(f"Wrote {profile_path}")
    print(f"Wrote {significance_path}")


if __name__ == "__main__":
    main()
