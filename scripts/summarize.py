"""
summarize.py

Aggregate every saved prediction file into one accuracy table.

This is the script that answers "is the RUL RMSE target met?" without
reading any prose.  It scans ``outputs/results/*_predictions.npz``,
recomputes the metrics from the stored arrays (so the numbers cannot drift
from the logs), writes ``outputs/reports/accuracy_summary.csv`` and prints
per-subset and averaged results.

Usage
-----
python scripts/summarize.py
python scripts/summarize.py --threshold 13
python scripts/summarize.py --models gru_att_deg_v2,ensemble
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import predictive_accuracy

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

PREDICTION_RE = re.compile(
    r"^(?P<subset>FD00[1-4])_(?P<model>.+)_predictions\.npz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate saved C-MAPSS prediction metrics."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=13.0,
        help="Target mean RUL RMSE (default: 13.0).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model stems to include "
            "(default: every prediction file found)."
        ),
    )
    return parser.parse_args()


def collect(models: set[str] | None) -> list[dict]:
    """Read every prediction file and recompute its metrics."""

    rows: list[dict] = []

    for path in sorted(RESULTS_DIR.glob("*_predictions.npz")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue

        subset = match.group("subset")
        model = match.group("model")

        if models is not None and model not in models:
            continue

        data = np.load(path)

        metrics = predictive_accuracy(
            y_rul_true=data["y_rul_true"],
            y_rul_pred=data["y_rul_pred"],
            y_hi_true=data["y_hi_true"],
            y_hi_pred=data["y_hi_pred"],
        )

        rows.append(
            {
                "subset": subset,
                "model": model,
                "n": int(len(data["y_rul_true"])),
                "rul_mae": metrics["rul"]["MAE"],
                "rul_rmse": metrics["rul"]["RMSE"],
                "rul_r2": metrics["rul"]["R2"],
                "nasa": metrics["rul"]["NASA_Score"],
                "hi_mae": metrics["hi"]["MAE"],
                "hi_rmse": metrics["hi"]["RMSE"],
                "hi_r2": metrics["hi"]["R2"],
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    models = None
    if args.models:
        models = {m.strip() for m in args.models.split(",") if m.strip()}

    rows = collect(models)

    if not rows:
        print("No prediction files found in", RESULTS_DIR)
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "accuracy_summary.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------------
    # Per-subset table
    # --------------------------------------------------------------

    header = (
        f"{'MODEL':22s} {'SUBSET':7s} {'N':>4s} "
        f"{'RUL_MAE':>8s} {'RUL_RMSE':>9s} {'RUL_R2':>7s} "
        f"{'NASA':>8s} {'HI_MAE':>7s} {'HI_RMSE':>8s} {'HI_R2':>7s}"
    )

    print("\n" + header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda r: (r["model"], r["subset"])):
        print(
            f"{row['model']:22s} {row['subset']:7s} {row['n']:4d} "
            f"{row['rul_mae']:8.3f} {row['rul_rmse']:9.3f} {row['rul_r2']:7.4f} "
            f"{row['nasa']:8.1f} {row['hi_mae']:7.4f} {row['hi_rmse']:8.4f} "
            f"{row['hi_r2']:7.4f}"
        )

    # --------------------------------------------------------------
    # Per-model average
    # --------------------------------------------------------------

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    print("\n=== AVERAGE ACROSS AVAILABLE SUBSETS ===")
    print(
        f"{'MODEL':22s} {'#DS':>4s} {'RUL_MAE':>8s} {'RUL_RMSE':>9s} "
        f"{'RUL_R2':>7s} {'NASA':>9s} {'HI_MAE':>7s} {'HI_RMSE':>8s} {'HI_R2':>7s}"
    )

    averages = []
    for model, group in sorted(by_model.items()):
        # Only average models that cover every subset — a partial average
        # would look artificially good.
        subsets = {g["subset"] for g in group}
        avg = {
            key: float(np.mean([g[key] for g in group]))
            for key in ("rul_mae", "rul_rmse", "rul_r2", "nasa",
                        "hi_mae", "hi_rmse", "hi_r2")
        }
        avg["model"] = model
        avg["n_subsets"] = len(subsets)
        avg["complete"] = subsets == set(SUBSETS)
        averages.append(avg)

        print(
            f"{model:22s} {len(subsets):4d} "
            f"{avg['rul_mae']:8.3f} {avg['rul_rmse']:9.3f} {avg['rul_r2']:7.4f} "
            f"{avg['nasa']:9.1f} {avg['hi_mae']:7.4f} {avg['hi_rmse']:8.4f} "
            f"{avg['hi_r2']:7.4f}"
        )

    # --------------------------------------------------------------
    # Threshold check
    # --------------------------------------------------------------

    print(f"\n=== RUL RMSE TARGET (< {args.threshold:g}) ===")

    complete = [a for a in averages if a["complete"]]
    for avg in sorted(complete, key=lambda a: a["rul_rmse"]):
        status = "PASS" if avg["rul_rmse"] < args.threshold else "FAIL"
        print(
            f"  {avg['model']:22s} mean RUL_RMSE={avg['rul_rmse']:7.3f}  {status}"
        )

    if not complete:
        print("  No model has predictions for all four subsets yet.")

    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
