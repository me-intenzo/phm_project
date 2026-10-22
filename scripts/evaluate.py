"""
evaluate.py

Evaluation pipeline for NASA C-MAPSS prognostics models.

The model architecture is rebuilt from the ``model_config`` stored inside
the checkpoint, so a run can always be evaluated with the exact
hyper-parameters it was trained with (no duplicated constants).

Usage
-----
python scripts/evaluate.py --model gru_att_deg_v2
python scripts/evaluate.py --model gru_att_deg_v2 --subset all
python scripts/evaluate.py --model gru_att_deg_v2 --subset FD001 --tag seed7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import predictive_accuracy
from src.models.checkpoint import load_state_dict_compat
from src.models.registry import ALL_MODELS, build_model
from src.utils.config import ALL_SUBSETS
from src.utils.logger import get_script_logger

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

BATCH_SIZE = 64


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Evaluate a C-MAPSS prognostics model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help=f"Model architecture to evaluate, or 'all'. Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        help="C-MAPSS subset to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix of the checkpoint directory (e.g. 'seed7').",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

def checkpoint_path(model_name: str, subset: str, tag: str = "") -> Path:
    dir_name = model_name + (f"_{tag}" if tag else "")
    return CHECKPOINT_DIR / dir_name / subset / "best_model.pt"


def load_model(
    model_name: str,
    input_size: int,
    device: torch.device,
    subset: str = "FD001",
    tag: str = "",
):
    """Rebuild the model described by the checkpoint and load its weights."""

    path = checkpoint_path(model_name, subset, tag)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    ckpt_input_size = checkpoint.get("input_size", input_size)
    saved_config = dict(checkpoint.get("model_config") or {})

    arch = saved_config.pop("arch", model_name)

    if saved_config:
        # Remove training-only keys; build_model ignores unknown kwargs
        # anyway, but keeping the call explicit documents the contract.
        model = build_model(
            arch,
            input_size=ckpt_input_size,
            **saved_config,
        )
    else:
        # Legacy checkpoint: no architecture metadata was stored.
        model = build_model(arch, input_size=ckpt_input_size)

    load_state_dict_compat(
        model,
        checkpoint["model_state_dict"],
        device=device,
    )

    model.to(device)
    model.eval()

    return model, checkpoint, arch


# ------------------------------------------------------------------
# Load test data
# ------------------------------------------------------------------

def load_test_data(subset: str):

    X = np.load(DATA_DIR / f"{subset}_test_X.npy")
    y_rul = np.load(DATA_DIR / f"{subset}_test_y_rul.npy")
    y_hi = np.load(DATA_DIR / f"{subset}_test_y_hi.npy")

    return X, y_rul, y_hi


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict(model, X: np.ndarray, device: torch.device):

    predictions_rul = []
    predictions_hi = []

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            ).float().to(device)

            pred = model(batch)
            if isinstance(pred, tuple) and len(pred) >= 2:
                pred_rul, pred_hi = pred[0], pred[1]
            else:
                pred_rul, pred_hi = pred

            predictions_rul.append(pred_rul.cpu().numpy())
            predictions_hi.append(pred_hi.cpu().numpy())

    return (
        np.concatenate(predictions_rul),
        np.concatenate(predictions_hi),
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(model_name: str, subset: str, tag: str = "") -> None:
    """Run the full evaluation pipeline for a single model/subset combination."""

    dir_name = model_name + (f"_{tag}" if tag else "")

    log = get_script_logger("evaluation", f"evaluate_{subset}_{dir_name}")

    device = get_device()

    log.info("Using device: %s", device)

    # --------------------------------------------------------------
    # Test data
    # --------------------------------------------------------------

    log.info("Loading processed test data...")

    X_test, y_rul_test, y_hi_test = load_test_data(subset)

    log.info("X_test shape : %s", X_test.shape)

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model, checkpoint, arch = load_model(
        model_name=model_name,
        input_size=X_test.shape[-1],
        device=device,
        subset=subset,
        tag=tag,
    )

    log.info(
        "Loaded %s checkpoint from epoch %d (arch=%s)",
        model_name.upper(),
        checkpoint["epoch"],
        arch,
    )

    if checkpoint.get("ema_decay"):
        log.info(
            "Checkpoint stores EMA weights (decay=%.4f).",
            checkpoint["ema_decay"],
        )

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    log.info("Generating predictions...")

    pred_rul, pred_hi = predict(model, X_test, device)

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    results = predictive_accuracy(
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    log.info("=" * 60)
    log.info("%s FINAL EVALUATION", model_name.upper())
    log.info("=" * 60)

    log.info("RUL metrics:")

    for metric, value in results["rul"].items():
        log.info("  %-12s : %.6f", metric, value)

    log.info("HI metrics:")

    for metric, value in results["hi"].items():
        log.info("  %-12s : %.6f", metric, value)

    # --------------------------------------------------------------
    # Save predictions + metrics
    # --------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_path = RESULTS_DIR / f"{subset}_{dir_name}_predictions.npz"

    np.savez(
        prediction_path,
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    metrics_path = RESULTS_DIR / f"{subset}_{dir_name}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subset": subset,
                "model": model_name,
                "tag": tag,
                "arch": arch,
                "epoch": checkpoint.get("epoch"),
                "metrics": results,
            },
            f,
            indent=2,
        )

    log.info("Predictions saved to: %s", prediction_path)
    log.info("Metrics saved to: %s", metrics_path)


def main():
    args = parse_args()

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    models = ALL_MODELS if args.model == "all" else [args.model]

    for subset in subsets:
        for model_name in models:
            run(model_name, subset, tag=args.tag)


if __name__ == "__main__":
    main()
