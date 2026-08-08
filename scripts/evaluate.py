"""
evaluate.py

Evaluate the trained LSTM prognostics model on NASA C-MAPSS FD001.

Evaluation protocol
-------------------
For each test engine:
    1. Select the final available sequence.
    2. Predict RUL and HI.
    3. Compare predicted RUL with the official RUL label.

Metrics
-------
RUL:
    - MAE
    - RMSE
    - R2
    - NASA Score

HI:
    - MAE
    - RMSE
    - R2
"""

from __future__ import annotations

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
from src.models.lstm import LSTMPrognosticsModel


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

SUBSET = "FD001"

DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "best_model.pt"
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


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "evaluate.log",
            mode="w",
        ),
    ],
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info("Using device: %s", device)

    return device


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

def load_model(
    input_size: int,
    device: torch.device,
) -> LSTMPrognosticsModel:

    model = LSTMPrognosticsModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    log.info(
        "Loaded checkpoint from epoch %d",
        checkpoint["epoch"],
    )

    return model


# ------------------------------------------------------------------
# Select final window from each test engine
# ------------------------------------------------------------------

def load_test_data():

    X = np.load(
        DATA_DIR / f"{SUBSET}_test_X.npy"
    )

    y_rul = np.load(
        DATA_DIR / f"{SUBSET}_test_y_rul.npy"
    )

    y_hi = np.load(
        DATA_DIR / f"{SUBSET}_test_y_hi.npy"
    )

    return X, y_rul, y_hi


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def predict(
    model,
    X,
    device,
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

            pred_rul, pred_hi = model(batch)

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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    # --------------------------------------------------------------
    # Load test data
    # --------------------------------------------------------------

    log.info("Loading processed test data...")

    X_test, y_rul_test, y_hi_test = (
        load_test_data()
    )

    log.info(
        "X_test shape : %s",
        X_test.shape,
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    input_size = X_test.shape[-1]

    model = load_model(
        input_size=input_size,
        device=device,
    )

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    log.info("Generating predictions...")

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
    # Display results
    # --------------------------------------------------------------

    log.info("=" * 60)
    log.info("FINAL LSTM EVALUATION")
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

    np.savez(
        RESULTS_DIR / f"{SUBSET}_lstm_predictions.npz",
        y_rul_true=y_rul_test,
        y_rul_pred=pred_rul,
        y_hi_true=y_hi_test,
        y_hi_pred=pred_hi,
    )

    log.info(
        "Predictions saved to %s",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()