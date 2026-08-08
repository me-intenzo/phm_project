"""
train.py

Training pipeline for the baseline LSTM prognostics model.

Workflow
--------
Load processed training data
        ↓
Train / validation split
        ↓
Build LSTM model
        ↓
Multi-task loss
        ↓
ModelTrainer
        ↓
Best checkpoint
        ↓
Training history
"""

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


from src.models.lstm import LSTMPrognosticsModel
from src.models.losses import MultiTaskLoss
from src.models.trainer import ModelTrainer


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

SUBSET = "FD001"

DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = (
    PROJECT_ROOT / "outputs" / "checkpoints"
)

LOG_DIR = (
    PROJECT_ROOT / "outputs" / "logs"
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "train.log",
            mode="w",
        ),
    ],
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.
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

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    log.info("Using device: %s", device)

    return device


# ------------------------------------------------------------------
# Load processed dataset
# ------------------------------------------------------------------

def load_training_data():
    """
    Load processed training arrays.
    """

    x_path = DATA_DIR / f"{SUBSET}_train_X.npy"

    rul_path = DATA_DIR / f"{SUBSET}_train_y_rul.npy"

    hi_path = DATA_DIR / f"{SUBSET}_train_y_hi.npy"

    log.info("Loading processed training data...")

    X = np.load(x_path)

    y_rul = np.load(rul_path)

    y_hi = np.load(hi_path)

    log.info("X shape      : %s", X.shape)

    log.info("RUL shape    : %s", y_rul.shape)

    log.info("HI shape     : %s", y_hi.shape)

    return X, y_rul, y_hi


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    set_seed(SEED)

    device = get_device()

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    X, y_rul, y_hi = load_training_data()

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
    # Model
    # --------------------------------------------------------------

    input_size = X.shape[-1]

    model = LSTMPrognosticsModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    log.info(
        "Model input size: %d",
        input_size,
    )

    log.info(
        "Trainable parameters: %d",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
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
    # Learning Rate Scheduler
    # --------------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        batch_size=BATCH_SIZE,
        checkpoint_dir=str(CHECKPOINT_DIR),
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
    # Save Training History
    # --------------------------------------------------------------

    history_path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / f"{SUBSET}_lstm_history.npz"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays: dict[str, np.ndarray] = {
        key: np.asarray(value)
        for key, value in history.items()
    }

    np.savez(str(history_path), **arrays)

    log.info(
        "Training history saved to: %s",
        history_path,
    )

    log.info(
        "Best validation loss: %.6f",
        trainer.best_loss,
    )

    log.info(
        "Training completed successfully."
    )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()