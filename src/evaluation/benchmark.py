"""
src/evaluation/benchmark.py

Comparative performance analysis across all model architectures and
C-MAPSS subsets.

Loads prediction artifacts written by scripts/evaluate.py and uncertainty
artifacts written by scripts/uncertainty.py, then assembles a cross-model
comparison table for each metric dimension.

Usage (from scripts/evaluate.py or demo.py)
-------------------------------------------
    from src.evaluation.benchmark import comparative_benchmark
    table = comparative_benchmark(results_dir, subsets, models)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import predictive_accuracy, uncertainty_reliability


def comparative_benchmark(
    results_dir: str | Path,
    subsets: list[str],
    models: list[str],
    uncertainty_variant: str = "engine_joint_conformal",
) -> dict[str, Any]:
    """
    Load pipeline artifacts and compute predictive + uncertainty metrics
    for every (subset, model) combination.

    Parameters
    ----------
    results_dir         : ``outputs/results/``
    subsets             : e.g. ["FD001", "FD002", "FD003", "FD004"]
    models              : e.g. ["lstm", "gru", "transformer", "hybrid"]
    uncertainty_variant : suffix used in the .npz filename

    Returns
    -------
    dict keyed by ``"{subset}_{model}"`` → metric sub-dicts, plus a
    ``"summary"`` key with the best model per metric.
    """
    results_dir = Path(results_dir)
    report: dict[str, Any] = {}

    for subset in subsets:
        for model in models:
            key = f"{subset}_{model}"

            pred_path = results_dir / f"{subset}_{model}_predictions.npz"
            unc_path  = results_dir / f"{subset}_{model}_{uncertainty_variant}.npz"

            if not pred_path.exists():
                report[key] = {"error": f"missing {pred_path.name}"}
                continue

            pred = np.load(pred_path)
            entry: dict[str, Any] = {}

            entry["predictive_accuracy"] = predictive_accuracy(
                pred["y_rul_true"], pred["y_rul_pred"],
                pred["y_hi_true"],  pred["y_hi_pred"],
            )

            if unc_path.exists():
                unc = np.load(unc_path)
                # engine_ids may be stored as test_engine_ids or engine_ids
                engine_ids = unc.get(
                    "test_engine_ids",
                    unc.get("engine_ids", np.zeros(len(pred["y_rul_true"]), dtype=np.int64))
                )
                entry["uncertainty_reliability"] = uncertainty_reliability(
                    pred["y_rul_true"],
                    unc["lower_90"],
                    unc["upper_90"],
                    engine_ids,
                )
            else:
                entry["uncertainty_reliability"] = {"note": f"missing {unc_path.name}"}

            report[key] = entry

    # Summary: best model per RUL MAE per subset
    summary: dict[str, Any] = {}
    for subset in subsets:
        best_mae   = float("inf")
        best_model = None
        for model in models:
            key = f"{subset}_{model}"
            mae = report.get(key, {}).get("predictive_accuracy", {}).get("rul", {}).get("MAE")
            if mae is not None and mae < best_mae:
                best_mae   = mae
                best_model = model
        summary[subset] = {"best_model_by_rul_mae": best_model, "best_rul_mae": best_mae}

    report["summary"] = summary
    return report
