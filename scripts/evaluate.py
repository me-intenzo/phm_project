"""
evaluate.py

Generic evaluation pipeline for NASA C-MAPSS prognostics models.

Supported models
----------------
- LSTM
- GRU
- Transformer
- Hybrid

Usage
-----
python scripts/evaluate.py --model lstm
python scripts/evaluate.py --model gru
python scripts/evaluate.py --model transformer
python scripts/evaluate.py --model hybrid
"""

from __future__ import annotations

import argparse
import logging
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

from src.models.evaluator import PrognosticsEvaluator
from src.models.gru import GRUPrognosticsModel
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.models.hybrid import HybridPrognosticsModel


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

SUBSET = "FD001"

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
)

LOG_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
)

BATCH_SIZE = 64

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.3


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
        choices=["lstm", "gru", "transformer", "hybrid"],
        required=True,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
    "--subset",
    type=str,
    choices=[
        "FD001",
        "FD002",
        "FD003",
        "FD004",
    ],
    default="FD001",
    help="C-MAPSS subset to evaluate.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def configure_logging(
    model_name: str,
    subset: str,
) -> logging.Logger:

    log_dir = LOG_DIR / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_dir
                / f"evaluate_{subset}_{model_name}.log",
                mode="w",
            ),
        ],
        force=True,
    )

    return logging.getLogger(__name__)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------

def build_model(
    model_name: str,
    input_size: int,
):

    if model_name == "lstm":

        return LSTMPrognosticsModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )

    if model_name == "gru":

        return GRUPrognosticsModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )

    if model_name == "transformer":

        return TransformerPrognosticsModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=4,
            dropout=DROPOUT,
        )

    if model_name == "hybrid":

        return HybridPrognosticsModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_gru_layers=NUM_LAYERS,
            num_transformer_layers=NUM_LAYERS,
            num_heads=4,
            dropout=DROPOUT,
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

def load_model(
    model_name: str,
    input_size: int,
    device: torch.device,
    subset: str,
):

    checkpoint_path = (
        CHECKPOINT_DIR
        / model_name
        / subset
        / "best_model.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    ckpt_input_size = checkpoint.get("input_size", input_size)

    model = build_model(
        model_name,
        ckpt_input_size,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, checkpoint


# ------------------------------------------------------------------
# Load test data
# ------------------------------------------------------------------

def load_test_data(subset: str):

    X = np.load(
        DATA_DIR
        / f"{subset}_test_X.npy"
    )

    y_rul = np.load(
        DATA_DIR
        / f"{subset}_test_y_rul.npy"
    )

    y_hi = np.load(
        DATA_DIR
        / f"{subset}_test_y_hi.npy"
    )

    return X, y_rul, y_hi


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict(
    model,
    X: np.ndarray,
    device: torch.device,
):

    predictions_rul = []

    predictions_hi = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            batch = torch.from_numpy(
                X[
                    start:
                    start + BATCH_SIZE
                ]
            ).float().to(device)

            pred_rul, pred_hi, _ = model(
                batch
            )

            predictions_rul.append(
                pred_rul.cpu().numpy()
            )

            predictions_hi.append(
                pred_hi.cpu().numpy()
            )

    return (
        np.concatenate(predictions_rul),
        np.concatenate(predictions_hi),
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    args = parse_args()

    model_name = args.model

    subset = args.subset

    log = configure_logging(
        model_name, subset
    )

    device = get_device()

    log.info(
        "Using device: %s",
        device,
    )

    # --------------------------------------------------------------
    # Test data
    # --------------------------------------------------------------

    log.info(
        "Loading processed test data..."
    )

    X_test, y_rul_test, y_hi_test = (
        load_test_data(subset)
    )

    log.info(
        "X_test shape : %s",
        X_test.shape,
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model, checkpoint = load_model(
        model_name=model_name,
        input_size=X_test.shape[-1],
        device=device,
        subset=subset,
    )

    log.info(
        "Loaded %s checkpoint from epoch %d",
        model_name.upper(),
        checkpoint["epoch"],
    )

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    log.info(
        "Generating predictions..."
    )

    pred_rul, pred_hi = predict(
        model,
        X_test,
        device,
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    evaluator = PrognosticsEvaluator()

    results = evaluator.evaluate(
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    log.info("=" * 60)

    log.info(
        "%s FINAL EVALUATION",
        model_name.upper(),
    )

    log.info("=" * 60)

    log.info("RUL metrics:")

    for metric, value in results["RUL"].items():

        log.info(
            "  %-12s : %.6f",
            metric,
            value,
        )

    log.info("HI metrics:")

    for metric, value in results["HI"].items():

        log.info(
            "  %-12s : %.6f",
            metric,
            value,
        )

    # --------------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------------

    prediction_path = (
        RESULTS_DIR
        / f"{subset}_{model_name}_predictions.npz"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        prediction_path,
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    log.info(
        "Predictions saved to: %s",
        prediction_path,
    )


if __name__ == "__main__":
    main()