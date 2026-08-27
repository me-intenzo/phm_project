"""
evaluate.py

Generic evaluation pipeline for NASA C-MAPSS prognostics models.

Supported models
----------------
- LSTM
- GRU
- Transformer
- Hybrid

The evaluator:
1. Loads processed test data.
2. Loads the model checkpoint for the selected subset.
3. Generates RUL and HI predictions.
4. Calculates RUL and HI metrics.
5. Saves predictions.

Usage
-----
python scripts/evaluate.py --model lstm --subset FD001
python scripts/evaluate.py --model gru --subset FD001
python scripts/evaluate.py --model transformer --subset FD001
python scripts/evaluate.py --model hybrid --subset FD001
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


# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------

from src.models.evaluator import PrognosticsEvaluator
from src.models.gru import GRUPrognosticsModel
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.models.hybrid import HybridPrognosticsModel


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_SUBSET = "FD001"

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
# Argument parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse evaluation configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a NASA C-MAPSS "
            "prognostics model."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "lstm",
            "gru",
            "transformer",
            "hybrid",
        ],
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
        default=DEFAULT_SUBSET,
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
    """
    Configure model-specific evaluation logging.
    """

    log_dir = LOG_DIR / "evaluation"

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / f"evaluate_{subset}_{model_name}.log"
    )

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
                log_path,
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    return logging.getLogger(__name__)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Select CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

def build_model(
    model_name: str,
    input_size: int,
):
    """
    Construct the requested prognostics model.
    """

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
            num_layers=NUM_LAYERS,
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
    subset: str,
    input_size: int,
    device: torch.device,
):
    """
    Load the best checkpoint for the selected
    model and C-MAPSS subset.

    Expected structure:

    outputs/
        checkpoints/
            <model>/
                <subset>/
                    best_model.pt
    """

    checkpoint_path = (
        CHECKPOINT_DIR
        / model_name
        / subset
        / "best_model.pt"
    )

    print(
        "CHECKPOINT PATH:",
        checkpoint_path.resolve(),
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path.resolve()}\n\n"
            "Expected checkpoint structure:\n"
            f"{CHECKPOINT_DIR / model_name / subset}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------------
    # Verify checkpoint input size
    # --------------------------------------------------------------

    ckpt_input_size = checkpoint.get(
        "input_size",
        input_size,
    )

    if ckpt_input_size != input_size:

        raise ValueError(
            "\nInput feature mismatch!\n"
            f"Checkpoint expects : {ckpt_input_size}\n"
            f"Test data provides  : {input_size}\n"
            f"Checkpoint           : {checkpoint_path.resolve()}\n"
        )

    # --------------------------------------------------------------
    # Build model
    # --------------------------------------------------------------

    model = build_model(
        model_name=model_name,
        input_size=ckpt_input_size,
    )

    # --------------------------------------------------------------
    # Load weights
    # --------------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, checkpoint


# ------------------------------------------------------------------
# Load test data
# ------------------------------------------------------------------

def load_test_data(
    subset: str,
):
    """
    Load processed test arrays.
    """

    x_path = (
        DATA_DIR
        / f"{subset}_test_X.npy"
    )

    rul_path = (
        DATA_DIR
        / f"{subset}_test_y_rul.npy"
    )

    hi_path = (
        DATA_DIR
        / f"{subset}_test_y_hi.npy"
    )

    # --------------------------------------------------------------
    # Check files
    # --------------------------------------------------------------

    required_files = [
        x_path,
        rul_path,
        hi_path,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing processed test files:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    X = np.load(x_path)

    y_rul = np.load(
        rul_path
    )

    y_hi = np.load(
        hi_path
    )

    # --------------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            "Expected X_test to have shape "
            "(samples, sequence_length, features), "
            f"but received {X.shape}."
        )

    if len(X) != len(y_rul):

        raise ValueError(
            "X_test and y_rul_test have "
            "different numbers of samples."
        )

    if len(X) != len(y_hi):

        raise ValueError(
            "X_test and y_hi_test have "
            "different numbers of samples."
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
    """
    Generate RUL and HI predictions.
    """

    predictions_rul = []

    predictions_hi = []

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            batch_array = X[
                start:
                start + BATCH_SIZE
            ]

            batch = (
                torch.from_numpy(
                    batch_array
                )
                .float()
                .to(device)
            )

            pred_rul, pred_hi = model(
                batch
            )

            predictions_rul.append(
                pred_rul
                .detach()
                .cpu()
                .numpy()
            )

            predictions_hi.append(
                pred_hi
                .detach()
                .cpu()
                .numpy()
            )

    return (
        np.concatenate(
            predictions_rul
        ),
        np.concatenate(
            predictions_hi
        ),
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    model_name = args.model

    subset = args.subset

    log = configure_logging(
        model_name,
        subset,
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

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
        load_test_data(
            subset
        )
    )

    log.info(
        "X_test shape : %s",
        X_test.shape,
    )

    log.info(
        "RUL shape    : %s",
        y_rul_test.shape,
    )

    log.info(
        "HI shape     : %s",
        y_hi_test.shape,
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model, checkpoint = load_model(
        model_name=model_name,
        subset=subset,
        input_size=X_test.shape[-1],
        device=device,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    log.info(
        "Loaded %s checkpoint from epoch %s",
        model_name.upper(),
        checkpoint_epoch,
    )

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    log.info(
        "Generating predictions..."
    )

    pred_rul, pred_hi = predict(
        model=model,
        X=X_test,
        device=device,
    )

    # --------------------------------------------------------------
    # Prediction validation
    # --------------------------------------------------------------

    if not np.all(
        np.isfinite(pred_rul)
    ):

        raise ValueError(
            "RUL predictions contain "
            "NaN or infinite values."
        )

    if not np.all(
        np.isfinite(pred_hi)
    ):

        raise ValueError(
            "HI predictions contain "
            "NaN or infinite values."
        )

    log.info(
        "RUL prediction range: %.4f to %.4f",
        float(pred_rul.min()),
        float(pred_rul.max()),
    )

    log.info(
        "HI prediction range : %.4f to %.4f",
        float(pred_hi.min()),
        float(pred_hi.max()),
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    log.info(
        "Starting joint prognostics evaluation."
    )

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

    log.info(
        "RUL metrics:"
    )

    for metric, value in results[
        "RUL"
    ].items():

        log.info(
            "  %-12s : %.6f",
            metric,
            value,
        )

    log.info(
        "HI metrics:"
    )

    for metric, value in results[
        "HI"
    ].items():

        log.info(
            "  %-12s : %.6f",
            metric,
            value,
        )

    # --------------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        RESULTS_DIR
        / f"{subset}_{model_name}_predictions.npz"
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
        prediction_path.resolve(),
    )

    log.info(
        "Evaluation completed successfully."
    )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()