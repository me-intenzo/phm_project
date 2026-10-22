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
from src.utils.provenance import (
    STATUS_MISMATCH,
    STATUS_UNKNOWN,
    compare_provenance,
    unpack_provenance,
)


def comparative_benchmark(
    results_dir: str | Path,
    subsets: list[str],
    models: list[str],
    uncertainty_variant: str = "engine_joint_conformal",
    tag: str = "",
    strict_provenance: bool = False,
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
    tag                 : optional run tag (e.g. ``"ms42"``) selecting one
                          specific set of weights.  Without a tag the
                          historical untagged artifacts are read.
    strict_provenance   : when ``True``, a pair of artifacts that
                          declares two *different* checkpoints raises
                          ``ValueError`` instead of being reported.

    Returns
    -------
    dict keyed by ``"{subset}_{run_name}"`` → metric sub-dicts, plus a
    ``"summary"`` key with the best model per metric.
    """
    results_dir = Path(results_dir)
    report: dict[str, Any] = {}

    for subset in subsets:
        for model in models:
            run_name = f"{model}_{tag}" if tag else model
            key = f"{subset}_{run_name}"

            pred_path = results_dir / f"{subset}_{run_name}_predictions.npz"
            unc_path  = results_dir / f"{subset}_{run_name}_{uncertainty_variant}.npz"

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

                # ------------------------------------------------------------
                # Provenance
                #
                # The predictions and the intervals are compared against
                # each other here, so they have to come from the same
                # weights.  Neither artifact used to record which
                # checkpoint produced it, which meant regenerating one
                # side and not the other produced a comparison between two
                # different models.
                # ------------------------------------------------------------
                comparison = compare_provenance(
                    unpack_provenance(pred),
                    unpack_provenance(unc),
                )
                entry["provenance"] = comparison

                if comparison["status"] == STATUS_MISMATCH:
                    message = (
                        f"{key}: the predictions and the "
                        f"{uncertainty_variant} intervals come from "
                        f"different checkpoints "
                        f"({comparison['left_sha256'][:12]} vs "
                        f"{comparison['right_sha256'][:12]}). "
                        "Regenerate both from the same run before "
                        "comparing them."
                    )
                    if strict_provenance:
                        raise ValueError(message)
                    entry["provenance"]["warning"] = message
            else:
                entry["uncertainty_reliability"] = {"note": f"missing {unc_path.name}"}
                entry["provenance"] = {
                    "status": STATUS_UNKNOWN,
                    "left_sha256": "",
                    "right_sha256": "",
                }

            report[key] = entry

    # Summary: best model per RUL MAE per subset
    summary: dict[str, Any] = {}
    for subset in subsets:
        best_mae   = float("inf")
        best_model = None
        for model in models:
            # The report is keyed by the *run* name (which carries the tag),
            # so this lookup has to use the same key the loop above wrote.
            # Looking up "{subset}_{model}" silently returned nothing for
            # every tagged run.
            run_name = f"{model}_{tag}" if tag else model
            key = f"{subset}_{run_name}"
            mae = report.get(key, {}).get("predictive_accuracy", {}).get("rul", {}).get("MAE")
            if mae is not None and mae < best_mae:
                best_mae   = mae
                best_model = run_name
        summary[subset] = {"best_model_by_rul_mae": best_model, "best_rul_mae": best_mae}

    report["summary"] = summary
    return report
