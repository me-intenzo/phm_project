"""
uncertainty.py

Conformal uncertainty estimation for the GRU prognostics model.

Workflow
--------
Processed training data
        ↓
Reproduce engine-wise train/validation split
        ↓
Load trained GRU checkpoint
        ↓
Predict validation RUL
        ↓
Calibrate conformal quantiles
        ↓
Load final test data
        ↓
Predict test RUL
        ↓
Generate prediction intervals
        ↓
Evaluate coverage / interval width

Important
---------
The final test targets are NEVER used during calibration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------

from src.models.gru import GRUPrognosticsModel

from src.uncertainty.calibration import (
    calibrate_multiple_levels,
)

from src.uncertainty.conformal import (
    prediction_interval,
)

from src.uncertainty.coverage import (
    evaluate_interval,
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

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

SEED = 42

BATCH_SIZE = 64

VALIDATION_SIZE = 0.20

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.3

COVERAGE_LEVELS = (
    0.80,
    0.90,
    0.95,
)


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Conformal uncertainty estimation "
            "for the GRU prognostics model."
        )
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
        help="C-MAPSS subset.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def configure_logging(
    subset: str,
) -> logging.Logger:

    log_dir = LOG_DIR / "uncertainty"

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
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
                log_dir
                / f"uncertainty_{subset}.log",
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
# Data Loading
# ------------------------------------------------------------------

def load_training_data(
    subset: str,
):
    """
    Load processed training data.

    Engine IDs are required to reproduce the
    engine-wise validation split.
    """

    X = np.load(
        DATA_DIR
        / f"{subset}_train_X.npy"
    )

    y_rul = np.load(
        DATA_DIR
        / f"{subset}_train_y_rul.npy"
    )

    y_hi = np.load(
        DATA_DIR
        / f"{subset}_train_y_hi.npy"
    )

    engine_ids = np.load(
        DATA_DIR
        / f"{subset}_train_engine_ids.npy"
    )

    if not (
        len(X)
        == len(y_rul)
        == len(y_hi)
        == len(engine_ids)
    ):
        raise ValueError(
            "Training arrays must contain "
            "the same number of samples."
        )

    return (
        X,
        y_rul,
        y_hi,
        engine_ids,
    )


def load_test_data(
    subset: str,
):
    """
    Load untouched final test data.
    """

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

    return (
        X,
        y_rul,
        y_hi,
    )


# ------------------------------------------------------------------
# Reproduce Validation Split
# ------------------------------------------------------------------

def create_calibration_split(
    X: np.ndarray,
    y_rul: np.ndarray,
    engine_ids: np.ndarray,
):
    """
    Reproduce the engine-wise validation split used
    by scripts/train.py.

    random_state=42 must remain identical to training.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )

    train_indices, calibration_indices = next(
        splitter.split(
            X,
            y_rul,
            groups=engine_ids,
        )
    )

    train_engines = set(
        engine_ids[train_indices].tolist()
    )

    calibration_engines = set(
        engine_ids[
            calibration_indices
        ].tolist()
    )

    overlap = (
        train_engines
        & calibration_engines
    )

    if overlap:
        raise RuntimeError(
            "Engine leakage detected between "
            "training and calibration sets: "
            f"{sorted(overlap)}"
        )

    return (
        train_indices,
        calibration_indices,
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_gru(
    input_size: int,
):
    """
    Reconstruct the exact GRU architecture
    used during O1 training.
    """

    return GRUPrognosticsModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )


def load_model(
    input_size: int,
    device: torch.device,
    subset: str = "",
):
    """
    Load the trained GRU checkpoint.
    """

    checkpoint_path = (
        CHECKPOINT_DIR
        / "gru"
        / subset
        / "best_model.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"GRU checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Train the GRU first using:\n"
            "python scripts/train.py --model gru"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    checkpoint_input_size = checkpoint.get("input_size", input_size)

    if checkpoint_input_size != input_size:
        raise RuntimeError(
            f"Checkpoint input_size={checkpoint_input_size} does not match "
            f"data input_size={input_size}. "
            f"Re-train the GRU for {subset} using:\n"
            f"python scripts/train.py --model gru --subset {subset}"
        )

    model = build_gru(
        input_size=input_size,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, checkpoint


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_rul(
    model,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Generate GRU RUL predictions.
    """

    predictions = []

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

            pred_rul, _ = model(
                batch
            )

            predictions.append(
                pred_rul
                .cpu()
                .numpy()
            )

    return np.concatenate(
        predictions
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    args = parse_args()

    subset = args.subset

    log = configure_logging(
        subset
    )

    device = get_device()

    log.info(
        "=" * 60
    )

    log.info(
        "CONFORMAL UNCERTAINTY ESTIMATION"
    )

    log.info(
        "Subset: %s",
        subset,
    )

    log.info(
        "Device: %s",
        device,
    )

    log.info(
        "=" * 60
    )

    # --------------------------------------------------------------
    # Load training data
    # --------------------------------------------------------------

    log.info(
        "Loading training data..."
    )

    (
        X_train_all,
        y_rul_all,
        y_hi_all,
        engine_ids,
    ) = load_training_data(
        subset
    )

    log.info(
        "Training samples: %d",
        len(X_train_all),
    )

    log.info(
        "Input shape: %s",
        X_train_all.shape,
    )

    # --------------------------------------------------------------
    # Reproduce calibration split
    # --------------------------------------------------------------

    log.info(
        "Creating engine-wise calibration split..."
    )

    (
        train_indices,
        calibration_indices,
    ) = create_calibration_split(
        X=X_train_all,
        y_rul=y_rul_all,
        engine_ids=engine_ids,
    )

    X_calibration = X_train_all[
        calibration_indices
    ]

    y_rul_calibration = y_rul_all[
        calibration_indices
    ]

    calibration_engines = np.unique(
        engine_ids[
            calibration_indices
        ]
    )

    training_engines = np.unique(
        engine_ids[
            train_indices
        ]
    )

    log.info(
        "Training engines    : %d",
        len(training_engines),
    )

    log.info(
        "Calibration engines : %d",
        len(calibration_engines),
    )

    log.info(
        "Calibration samples : %d",
        len(X_calibration),
    )

    # --------------------------------------------------------------
    # Load trained GRU
    # --------------------------------------------------------------

    log.info(
        "Loading trained GRU..."
    )

    model, checkpoint = load_model(
        input_size=X_train_all.shape[-1],
        device=device,
        subset=subset,
    )

    log.info(
        "Loaded checkpoint from epoch %d",
        checkpoint["epoch"],
    )

    # --------------------------------------------------------------
    # Calibration predictions
    # --------------------------------------------------------------

    log.info(
        "Generating calibration predictions..."
    )

    calibration_predictions = predict_rul(
        model=model,
        X=X_calibration,
        device=device,
    )

    # --------------------------------------------------------------
    # Conformal calibration
    # --------------------------------------------------------------

    log.info(
        "Calibrating conformal intervals..."
    )

    calibration_results = (
        calibrate_multiple_levels(
            y_true=y_rul_calibration,
            y_pred=calibration_predictions,
            coverage_levels=COVERAGE_LEVELS,
        )
    )

    for coverage, result in (
        calibration_results.items()
    ):

        log.info(
            "Nominal %.0f%% | alpha %.2f | q_hat %.6f",
            coverage * 100.0,
            result["alpha"],
            result["q_hat"],
        )

    # --------------------------------------------------------------
    # Load final test set
    # --------------------------------------------------------------

    log.info(
        "Loading final test data..."
    )

    (
        X_test,
        y_rul_test,
        y_hi_test,
    ) = load_test_data(
        subset
    )

    log.info(
        "Test samples: %d",
        len(X_test),
    )

    # --------------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------------

    log.info(
        "Generating test predictions..."
    )

    test_predictions = predict_rul(
        model=model,
        X=X_test,
        device=device,
    )

    # --------------------------------------------------------------
    # Evaluate intervals
    # --------------------------------------------------------------

    results = {}

    for coverage, calibration in (
        calibration_results.items()
    ):

        q_hat = calibration["q_hat"]

        lower, upper = prediction_interval(
            y_pred=test_predictions,
            q_hat=q_hat,
        )

        metrics = evaluate_interval(
            y_true=y_rul_test,
            lower=lower,
            upper=upper,
            nominal_coverage=coverage,
        )

        results[coverage] = metrics

        log.info(
            (
                "Coverage %.0f%% | "
                "Empirical %.4f | "
                "Error %.4f | "
                "MPIW %.4f"
            ),
            coverage * 100.0,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{subset}_gru_conformal.npz"
    )

    save_data = {
        "y_rul_test": y_rul_test,
        "y_rul_pred": test_predictions,
    }

    for coverage, calibration in (
        calibration_results.items()
    ):

        q_hat = calibration["q_hat"]

        lower, upper = prediction_interval(
            test_predictions,
            q_hat,
        )

        suffix = int(
            coverage * 100
        )

        save_data[
            f"lower_{suffix}"
        ] = lower

        save_data[
            f"upper_{suffix}"
        ] = upper

        save_data[
            f"q_hat_{suffix}"
        ] = np.asarray(
            [q_hat]
        )

    np.savez(
        output_path,
        **save_data,
    )

    log.info(
        "Results saved to: %s",
        output_path,
    )

    log.info(
        "=" * 60
    )

    log.info(
        "UNCERTAINTY ESTIMATION COMPLETE"
    )

    log.info(
        "=" * 60
    )


if __name__ == "__main__":
    main()