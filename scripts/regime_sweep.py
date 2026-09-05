"""
regime_sweep.py — Find optimal N_REGIMES for EARA-Conformal.

Sweeps k in {2, 3, 4, 5, 6, 7, 8, 10} across all four C-MAPSS subsets
using the GRU model (same engine-disjoint split as uncertainty.py).

Composite score = mean_coverage_error + mean_PINAW  (lower is better).

Usage
-----
python scripts/regime_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uncertainty import (
    create_calibration_split,
    get_device,
    load_model,
    load_test_data,
    load_training_data,
    predict,
)
from src.uncertainty.calibration import calibrate_eara_levels
from src.uncertainty.conformal import (
    adaptive_prediction_interval,
    assign_regimes,
    fit_regime_detector,
)
from src.uncertainty.coverage import evaluate_interval

SUBSETS         = ["FD001", "FD002", "FD003", "FD004"]
K_VALUES        = [2, 3, 4, 5, 6, 7, 8, 10]
COVERAGE_LEVELS = [0.80, 0.90, 0.95]
MODEL_NAME      = "gru"
SEED            = 42
OUT_PATH        = PROJECT_ROOT / "outputs" / "results" / "regime_sweep.json"


def sweep_subset(subset: str, device) -> list[dict]:
    X_all, y_rul_all, _, engine_ids = load_training_data(subset)
    train_idx, cal_idx = create_calibration_split(X_all, y_rul_all, engine_ids)
    X_train   = X_all[train_idx]
    X_cal     = X_all[cal_idx]
    y_rul_cal = y_rul_all[cal_idx]
    X_test, y_rul_test, _ = load_test_data(subset)

    model, _ = load_model(MODEL_NAME, X_all.shape[-1], device, subset)
    rul_cal,  scale_cal  = predict(model, X_cal,  device)
    rul_test, scale_test = predict(model, X_test, device)

    rows = []
    for k in K_VALUES:
        km       = fit_regime_detector(X_train, n_regimes=k, random_state=SEED)
        reg_cal  = assign_regimes(km, X_cal)
        reg_test = assign_regimes(km, X_test)
        cal      = calibrate_eara_levels(y_rul_cal, rul_cal, scale_cal, reg_cal, k)

        errs, pinaws, per_cov = [], [], {}
        for cov in COVERAGE_LEVELS:
            lo, hi = adaptive_prediction_interval(
                rul_test, scale_test, cal[cov]["q_hats"], reg_test
            )
            m = evaluate_interval(y_rul_test, lo, hi, cov)
            errs.append(m["coverage_error"])
            pinaws.append(m["pinaw"])
            per_cov[cov] = {
                "empirical_coverage": round(m["empirical_coverage"], 4),
                "coverage_error":     round(m["coverage_error"],     4),
                "mpiw":               round(m["mean_interval_width"], 4),
                "pinaw":              round(m["pinaw"],               4),
            }

        score = round(float(np.mean(errs)) + float(np.mean(pinaws)), 4)
        rows.append({
            "subset":              subset,
            "k":                   k,
            "score":               score,
            "mean_coverage_error": round(float(np.mean(errs)),   4),
            "mean_pinaw":          round(float(np.mean(pinaws)), 4),
            "per_coverage":        per_cov,
        })
        print(
            f"  {subset} k={k:2d}  score={score:.4f}  "
            f"err={np.mean(errs):.4f}  pinaw={np.mean(pinaws):.4f}"
        )
    return rows


def main() -> None:
    device = get_device()
    print(f"Device: {device}\n")

    all_rows: list[dict] = []
    for subset in SUBSETS:
        print(f"=== {subset} ===")
        all_rows.extend(sweep_subset(subset, device))

    # aggregate score per k across all subsets
    agg: dict[int, list[float]] = {k: [] for k in K_VALUES}
    for row in all_rows:
        agg[row["k"]].append(row["score"])

    print("\n=== AGGREGATE SCORE (mean across subsets, lower is better) ===")
    agg_summary = []
    for k in K_VALUES:
        mean_score = round(float(np.mean(agg[k])), 4)
        agg_summary.append({"k": k, "mean_score": mean_score})
        print(f"  k={k:2d}  mean_score={mean_score:.4f}")

    best_k = min(agg_summary, key=lambda x: x["mean_score"])["k"]
    print(f"\nBest k = {best_k}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"rows": all_rows, "aggregate": agg_summary, "best_k": best_k}, f, indent=2)
    print(f"Results saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
