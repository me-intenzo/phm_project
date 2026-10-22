"""
ensemble.py

Uniform ensemble of several trained prognostics models.

Averaging the predictions of models that make *different* errors is one of
the most reliable ways to lower RMSE on C-MAPSS, because it cancels part of
the variance that survives regularisation.  The ensemble here uses a plain
uniform average — no weights are fitted on the test set, so the reported
metrics remain honest.

Usage
-----
# ensemble every available v2 checkpoint with the two strongest baselines
python scripts/ensemble.py --subset all \
    --models gru_att_deg_v2,gru_att_deg,gru

# ensemble three seeds of the tuned model
python scripts/ensemble.py --subset all \
    --models gru_att_deg_v2@seed1,gru_att_deg_v2@seed2,gru_att_deg_v2@seed3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import load_model, load_test_data, predict
from src.evaluation.metrics import predictive_accuracy
from src.utils.config import ALL_SUBSETS
from src.utils.logger import get_script_logger

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Uniform ensemble of C-MAPSS prognostics models."
    )
    parser.add_argument("--subset", default="FD001", help="Subset or 'all'.")
    parser.add_argument(
        "--models",
        default="gru_att_deg_v2,gru_att_deg,gru",
        help=(
            "Comma-separated members. Use 'name' or 'name@tag' for a "
            "tagged checkpoint directory."
        ),
    )
    parser.add_argument(
        "--name",
        default="ensemble",
        help="Stem used for the saved prediction/metric files.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force 'cpu' or 'cuda' (default: auto).",
    )
    return parser.parse_args()


def parse_members(raw: str) -> list[tuple[str, str]]:
    """Parse ``name@tag,name`` into ``[(name, tag), ...]``."""

    members: list[tuple[str, str]] = []

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "@" in item:
            name, tag = item.split("@", 1)
        else:
            name, tag = item, ""
        members.append((name.strip(), tag.strip()))

    if len(members) < 2:
        raise ValueError(
            "An ensemble needs at least two members; got "
            f"{len(members)}."
        )

    return members


def get_device(preference: str | None) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(subset: str, members: list[tuple[str, str]], name: str, device) -> dict:
    """Build the ensemble for one subset and persist its results."""

    log = get_script_logger("evaluation", f"ensemble_{subset}_{name}")

    X_test, y_rul_test, y_hi_test = load_test_data(subset)

    rul_members = []
    hi_members = []
    used = []

    for model_name, tag in members:
        try:
            model, _checkpoint, _arch = load_model(
                model_name=model_name,
                input_size=X_test.shape[-1],
                device=device,
                subset=subset,
                tag=tag,
            )
        except FileNotFoundError as exc:
            log.warning("Skipping %s@%s: %s", model_name, tag or "-", exc)
            continue

        pred_rul, pred_hi = predict(model, X_test, device)

        rul_members.append(pred_rul)
        hi_members.append(pred_hi)
        used.append(f"{model_name}@{tag}" if tag else model_name)

        log.info(
            "Member %-28s RUL RMSE=%.4f",
            f"{model_name}@{tag}" if tag else model_name,
            float(np.sqrt(np.mean((pred_rul - y_rul_test) ** 2))),
        )

    if not rul_members:
        raise RuntimeError(f"No ensemble members found for {subset}.")

    # Uniform average — deliberately not fitted on the test split.
    pred_rul = np.mean(np.stack(rul_members, axis=0), axis=0)
    pred_hi = np.mean(np.stack(hi_members, axis=0), axis=0)

    results = predictive_accuracy(
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    log.info("=" * 60)
    log.info("ENSEMBLE %s (%d members)", subset, len(used))
    log.info("Members: %s", ", ".join(used))
    log.info("=" * 60)
    log.info("RUL | MAE=%.4f RMSE=%.4f R2=%.4f NASA=%.2f",
             results["rul"]["MAE"], results["rul"]["RMSE"],
             results["rul"]["R2"], results["rul"]["NASA_Score"])
    log.info("HI  | MAE=%.4f RMSE=%.4f R2=%.4f",
             results["hi"]["MAE"], results["hi"]["RMSE"], results["hi"]["R2"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        RESULTS_DIR / f"{subset}_{name}_predictions.npz",
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    with open(RESULTS_DIR / f"{subset}_{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "subset": subset,
                "model": name,
                "members": used,
                "metrics": results,
            },
            f,
            indent=2,
        )

    return results


def main():
    args = parse_args()

    members = parse_members(args.models)
    device = get_device(args.device)

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]

    for subset in subsets:
        run(subset, members, args.name, device)


if __name__ == "__main__":
    main()
