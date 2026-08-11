"""
train.py

Generic training pipeline for prognostics models.

Supported models
----------------
- LSTM
- GRU
- Transformer
- Hybrid

Workflow
--------
Load processed data
        ↓
Train / validation split
        ↓
Select model
        ↓
Multi-task loss
        ↓
ModelTrainer
        ↓
Best checkpoint
        ↓
Training history

Usage
-----
python scripts/train.py --model lstm
python scripts/train.py --model gru
python scripts/train.py --model transformer
python scripts/train.py --model hybrid

"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.gru import GRUPrognosticsModel
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.models.hybrid import HybridPrognosticsModel
from src.models.losses import MultiTaskLoss
from src.models.trainer import ModelTrainer


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

SEED = 42

BATCH_SIZE = 64

EPOCHS = 50

PATIENCE = 10

VALIDATION_SIZE = 0.20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.3

RUL_LOSS_WEIGHT = 1.0

HI_LOSS_WEIGHT = 0.5


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def configure_logging(model_name: str) -> logging.Logger:
    """
    Configure model-specific logging.
    """

    log_path = (
        LOG_DIR
        / f"train_{model_name}.log"
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
            ),
        ],
        force=True,
    )

    return logging.getLogger(__name__)


# ------------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train a prognostics model "
            "on NASA C-MAPSS."
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
        default="lstm",
        help="Model architecture to train.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Select CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    return device


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

def load_training_data():
    """
    Load processed training arrays.
    """

    x_path = (
        DATA_DIR
        / f"{SUBSET}_train_X.npy"
    )

    rul_path = (
        DATA_DIR
        / f"{SUBSET}_train_y_rul.npy"
    )

    hi_path = (
        DATA_DIR
        / f"{SUBSET}_train_y_hi.npy"
    )

    X = np.load(x_path)

    y_rul = np.load(rul_path)

    y_hi = np.load(hi_path)

    return X, y_rul, y_hi


# ------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------

def build_model(
    model_name: str,
    input_size: int,
):
    """
    Construct the requested prognostics model.

    All models use the same:
        - input size
        - hidden representation
        - dropout
        - multi-task output structure

    The Transformer additionally uses:
        - 4 attention heads
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
# Main
# ------------------------------------------------------------------

def main():

    args = parse_args()

    model_name = args.model

    log = configure_logging(
        model_name
    )

    set_seed(SEED)

    device = get_device()

    log.info(
        "=" * 60
    )

    log.info(
        "TRAINING MODEL: %s",
        model_name.upper(),
    )

    log.info(
        "=" * 60
    )

    log.info(
        "Using device: %s",
        device,
    )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    log.info(
        "Loading processed training data..."
    )

    X, y_rul, y_hi = (
        load_training_data()
    )

    log.info(
        "X shape      : %s",
        X.shape,
    )

    log.info(
        "RUL shape    : %s",
        y_rul.shape,
    )

    log.info(
        "HI shape     : %s",
        y_hi.shape,
    )

    # --------------------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------------------

    (
        X_train,
        X_val,
        y_rul_train,
        y_rul_val,
        y_hi_train,
        y_hi_val,
    ) = train_test_split(
        X,
        y_rul,
        y_hi,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    log.info(
        "Training samples   : %d",
        len(X_train),
    )

    log.info(
        "Validation samples : %d",
        len(X_val),
    )

    # --------------------------------------------------------------
    # Build Model
    # --------------------------------------------------------------

    input_size = X.shape[-1]

    model = build_model(
        model_name=model_name,
        input_size=input_size,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    log.info(
        "Model input size: %d",
        input_size,
    )

    log.info(
        "Trainable parameters: %d",
        parameter_count,
    )

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------

    criterion = MultiTaskLoss(
        alpha=RUL_LOSS_WEIGHT,
        beta=HI_LOSS_WEIGHT,
    )

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------------
    # Model-specific checkpoint directory
    # --------------------------------------------------------------

    model_checkpoint_dir = (
        CHECKPOINT_DIR
        / model_name
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=BATCH_SIZE,
        checkpoint_dir=model_checkpoint_dir,
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    train_loader, val_loader = (
        trainer.create_dataloaders(
            X_train,
            y_rul_train,
            y_hi_train,
            X_val,
            y_rul_val,
            y_hi_val,
        )
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    # --------------------------------------------------------------
    # Save history
    # --------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        RESULTS_DIR
        / f"{SUBSET}_{model_name}_history.npz"
    )

    np.savez(
        str(history_path),
        **{
            key: np.asarray(value)
            for key, value in history.items()
        },
    )

    # --------------------------------------------------------------
    # Final logging
    # --------------------------------------------------------------

    log.info(
        "Best validation loss: %.6f",
        trainer.best_loss,
    )

    log.info(
        "Training history saved to: %s",
        history_path,
    )

    log.info(
        "Checkpoint saved to: %s",
        model_checkpoint_dir,
    )

    log.info(
        "Training completed successfully."
    )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()