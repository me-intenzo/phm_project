"""
plot_degradation.py

Generate RUL and Health Index degradation curves for
the GRU baseline and CNN + GRU + Transformer hybrid.

Uses the validation engines from the same GroupShuffleSplit
configuration used during training.

Usage:
    python scripts/plot_degradation.py --subset FD001 --engine 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ----------------------------------------------------------
# Project root
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------
# Third-party imports
# ----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

# ----------------------------------------------------------
# Project imports
# ----------------------------------------------------------

from src.models.gru import GRUPrognosticsModel
from src.models.hybrid import HybridPrognosticsModel

# ==============================================================
# CONFIGURATION
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "degradation"

HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.3

BATCH_SIZE = 64

VALIDATION_SIZE = 0.20
SEED = 42


# ==============================================================
# ARGUMENTS
# ==============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate RUL and HI degradation curves."
    )

    parser.add_argument(
        "--subset",
        type=str,
        choices=["FD001", "FD002", "FD003", "FD004"],
        default="FD001",
    )

    parser.add_argument(
        "--engine",
        type=int,
        default=24,
        help="Validation engine ID to plot.",
    )

    return parser.parse_args()


# ==============================================================
# DEVICE
# ==============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ==============================================================
# LOAD PROCESSED TRAINING DATA
# ==============================================================

def load_data(subset):

    X = np.load(
        DATA_DIR / f"{subset}_train_X.npy"
    )

    y_rul = np.load(
        DATA_DIR / f"{subset}_train_y_rul.npy"
    )

    y_hi = np.load(
        DATA_DIR / f"{subset}_train_y_hi.npy"
    )

    engine_ids = np.load(
        DATA_DIR / f"{subset}_train_engine_ids.npy"
    )

    return X, y_rul, y_hi, engine_ids


# ==============================================================
# RECREATE THE TRAIN / VALIDATION SPLIT
# ==============================================================

def get_validation_engines(engine_ids):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )

    indices = np.arange(
        len(engine_ids)
    )

    _, val_indices = next(
        splitter.split(
            indices,
            groups=engine_ids,
        )
    )

    validation_engines = sorted(
        np.unique(
            engine_ids[val_indices]
        )
    )

    return validation_engines


# ==============================================================
# BUILD MODEL
# ==============================================================

def build_model(
    model_name,
    input_size,
):

    if model_name == "gru":

        return GRUPrognosticsModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )

    if model_name == "hybrid":

        return HybridPrognosticsModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ==============================================================
# LOAD CHECKPOINT
# ==============================================================

def load_model(
    model_name,
    subset,
    input_size,
    device,
):

    checkpoint_path = (
        CHECKPOINT_DIR
        / model_name
        / subset
        / "best_model.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    checkpoint_input_size = checkpoint.get(
        "input_size",
        input_size,
    )

    if checkpoint_input_size != input_size:

        raise ValueError(
            f"Feature mismatch for {model_name}:\n"
            f"Checkpoint: {checkpoint_input_size}\n"
            f"Data: {input_size}"
        )

    model = build_model(
        model_name,
        checkpoint_input_size,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model


# ==============================================================
# PREDICT
# ==============================================================

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

            rul, hi = model(batch)

            predictions_rul.append(
                rul.cpu().numpy()
            )

            predictions_hi.append(
                hi.cpu().numpy()
            )

    return (
        np.concatenate(
            predictions_rul
        ),
        np.concatenate(
            predictions_hi
        ),
    )


# ==============================================================
# BUILD CYCLE NUMBERS
# ==============================================================

def build_cycle_numbers(
    engine_ids,
    selected_engine,
):

    positions = np.where(
        engine_ids == selected_engine
    )[0]

    # Every sliding window has window_size = 30.
    #
    # First window:
    # cycles 1–30  -> endpoint 30
    #
    # Second window:
    # cycles 2–31 -> endpoint 31
    #
    # Therefore:

    cycles = np.arange(
        30,
        30 + len(positions),
    )

    return cycles


# ==============================================================
# PLOT RUL
# ==============================================================

def plot_rul_curve(
    engine_id,
    cycles,
    actual_rul,
    gru_rul,
    hybrid_rul,
    output_path,
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        cycles,
        actual_rul,
        label="Actual RUL",
        linewidth=2,
    )

    plt.plot(
        cycles,
        gru_rul,
        label="GRU Predicted RUL",
        linewidth=2,
    )

    plt.plot(
        cycles,
        hybrid_rul,
        label="Hybrid Predicted RUL",
        linewidth=2,
    )

    plt.xlabel(
        "Operating Cycle"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        f"Engine {engine_id} - RUL Degradation"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==============================================================
# PLOT HI
# ==============================================================

def plot_hi_curve(
    engine_id,
    cycles,
    actual_hi,
    gru_hi,
    hybrid_hi,
    output_path,
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        cycles,
        actual_hi,
        label="Actual HI",
        linewidth=2,
    )

    plt.plot(
        cycles,
        gru_hi,
        label="GRU Predicted HI",
        linewidth=2,
    )

    plt.plot(
        cycles,
        hybrid_hi,
        label="Hybrid Predicted HI",
        linewidth=2,
    )

    plt.xlabel(
        "Operating Cycle"
    )

    plt.ylabel(
        "Health Index"
    )

    plt.title(
        f"Engine {engine_id} - Health Index Degradation"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==============================================================
# MAIN
# ==============================================================

def main():

    args = parse_args()

    subset = args.subset

    engine_id = args.engine

    device = get_device()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("DEGRADATION CURVE GENERATION")
    print("=" * 60)

    print(
        f"Subset : {subset}"
    )

    print(
        f"Engine : {engine_id}"
    )

    print(
        f"Device : {device}"
    )

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    X, y_rul, y_hi, engine_ids = load_data(
        subset
    )

    print(
        f"Training windows : {X.shape}"
    )

    # ----------------------------------------------------------
    # Recreate validation split
    # ----------------------------------------------------------

    validation_engines = get_validation_engines(
        engine_ids
    )

    print(
        f"Validation engines: {validation_engines}"
    )

    if engine_id not in validation_engines:

        raise ValueError(
            f"Engine {engine_id} is not in the "
            f"validation set.\n\n"
            f"Choose one of:\n"
            f"{validation_engines}"
        )

    # ----------------------------------------------------------
    # Select engine
    # ----------------------------------------------------------

    mask = (
        engine_ids == engine_id
    )

    X_engine = X[mask]

    y_rul_engine = y_rul[mask]

    y_hi_engine = y_hi[mask]

    engine_ids_selected = engine_ids[mask]

    print(
        f"Engine windows: {len(X_engine)}"
    )

    # ----------------------------------------------------------
    # Build cycle numbers
    # ----------------------------------------------------------

    cycles = build_cycle_numbers(
        engine_ids_selected,
        engine_id,
    )

    # ----------------------------------------------------------
    # Load models
    # ----------------------------------------------------------

    print(
        "Loading GRU model..."
    )

    gru_model = load_model(
        "gru",
        subset,
        X.shape[-1],
        device,
    )

    print(
        "Loading hybrid model..."
    )

    hybrid_model = load_model(
        "hybrid",
        subset,
        X.shape[-1],
        device,
    )

    # ----------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------

    print(
        "Generating GRU predictions..."
    )

    gru_rul, gru_hi = predict(
        gru_model,
        X_engine,
        device,
    )

    print(
        "Generating hybrid predictions..."
    )

    hybrid_rul, hybrid_hi = predict(
        hybrid_model,
        X_engine,
        device,
    )

    # ----------------------------------------------------------
    # Save RUL curve
    # ----------------------------------------------------------

    rul_path = (
        OUTPUT_DIR
        / f"{subset}_engine_{engine_id}_rul_degradation.png"
    )

    plot_rul_curve(
        engine_id=engine_id,
        cycles=cycles,
        actual_rul=y_rul_engine,
        gru_rul=gru_rul,
        hybrid_rul=hybrid_rul,
        output_path=rul_path,
    )

    # ----------------------------------------------------------
    # Save HI curve
    # ----------------------------------------------------------

    hi_path = (
        OUTPUT_DIR
        / f"{subset}_engine_{engine_id}_hi_degradation.png"
    )

    plot_hi_curve(
        engine_id=engine_id,
        cycles=cycles,
        actual_hi=y_hi_engine,
        gru_hi=gru_hi,
        hybrid_hi=hybrid_hi,
        output_path=hi_path,
    )

    print()
    print("Generated files:")
    print(
        rul_path.resolve()
    )

    print(
        hi_path.resolve()
    )

    print()
    print(
        "Degradation curves generated successfully."
    )


if __name__ == "__main__":
    main()