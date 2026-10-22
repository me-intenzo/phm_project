"""
train.py

Config-driven training pipeline for prognostics models.

Workflow
--------
Load YAML config (``configs/fd00X.yaml`` merged over defaults)
        ↓
Load processed data (and verify the RUL cap matches the config)
        ↓
Engine-wise train / validation split
        ↓
Window augmentation
        ↓
Multi-task loss (RUL + HI + heteroscedastic scale)
        ↓
ModelTrainer (warm-up, cosine/plateau LR, grad clipping, weight EMA)
        ↓
Best checkpoint (EMA weights) + full training history

The checkpoint records the exact architecture and training config, so
``scripts/evaluate.py`` can rebuild the model without hard-coded
hyper-parameters.

Usage
-----
python scripts/train.py --model gru_att_deg_v2 --subset FD001
python scripts/train.py --model gru_att_deg_v2 --subset all
python scripts/train.py --model gru_att_deg_v2 --subset FD001 --seed 7 --tag seed7
python scripts/train.py --model gru_att_deg_v2 --subset FD001 --no-augment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.augmentation import AugmentationConfig, WindowAugmenter
from src.models.losses import MultiTaskLoss
from src.models.registry import ALL_MODELS, build_model
from src.models.trainer import ModelTrainer
from src.utils.config import ALL_SUBSETS, load_config
from src.utils.config import get_device as get_device_from_config
from src.utils.logger import get_script_logger
from src.utils.seed import set_seed

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


# ------------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train a prognostics model on NASA C-MAPSS."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help=f"Model architecture to train, or 'all'. Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        help="C-MAPSS subset to train, or 'all'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit YAML config path (default: configs/<subset>.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the config.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for the checkpoint directory (e.g. 'seed7').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the maximum number of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable window augmentation.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable the exponential moving average of the weights.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help=(
            "Disable the deterministic torch/CUDA kernels. The run stays "
            "seeded, but bit-for-bit reproducibility is no longer "
            "guaranteed."
        ),
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help=(
            "Additional config overrides as comma-separated key=value "
            "pairs, e.g. 'hidden_size=160,dropout=0.4'."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

# NOTE: ``set_seed`` lives in ``src/utils/seed.py`` so that every entry
# point (training, calibration, bootstrap) shares one implementation.  It
# seeds ``random``/``numpy``/``torch``/``torch.cuda`` and enables the
# deterministic torch/CUDA kernels, so a run is identified by its seed.


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device(preference: str = "cuda") -> torch.device:
    """Select CUDA when available (and requested), otherwise CPU.

    See ``src.utils.config.get_device`` — the single device selection of
    the project.
    """

    return get_device_from_config(preference)


# ------------------------------------------------------------------
# Override parsing
# ------------------------------------------------------------------

def _coerce(value: str):
    """Best-effort conversion of a CLI override string to a scalar."""

    lowered = value.strip().lower()

    if lowered in ("true", "false"):
        return lowered == "true"

    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass

    return value


def parse_overrides(raw: str | None) -> dict:
    """Parse ``key=value,key=value`` into a typed dict."""

    if not raw:
        return {}

    overrides: dict = {}

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected key=value."
            )
        key, value = item.split("=", 1)
        overrides[key.strip()] = _coerce(value)

    return overrides


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

def load_training_data(subset: str, expected_max_rul: int | None = None):
    """Load processed training arrays, engine IDs and metadata.

    Verifies that the RUL cap stored by ``preprocess.py`` matches the one
    requested by the experiment config.  A mismatch means the labels were
    built with a different cap, which silently invalidates the run.
    """

    x_path      = DATA_DIR / f"{subset}_train_X.npy"
    rul_path    = DATA_DIR / f"{subset}_train_y_rul.npy"
    hi_path     = DATA_DIR / f"{subset}_train_y_hi.npy"
    engine_path = DATA_DIR / f"{subset}_train_engine_ids.npy"
    meta_path   = DATA_DIR / f"{subset}_train_metadata.json"

    for p in (x_path, rul_path, hi_path, engine_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Processed file not found: {p}\n"
                f"Run: python scripts/preprocess.py --subset {subset}"
            )

    import json

    with open(meta_path) as f:
        metadata = json.load(f)

    if expected_max_rul is not None:
        stored = metadata.get("max_rul")
        if stored is not None and int(stored) != int(expected_max_rul):
            raise ValueError(
                f"RUL cap mismatch for {subset}: processed data uses "
                f"max_rul={stored} but the config requests "
                f"max_rul={expected_max_rul}. Re-run preprocessing with "
                f"--max-rul {expected_max_rul}."
            )

    X          = np.load(x_path)
    y_rul      = np.load(rul_path)
    y_hi       = np.load(hi_path)
    engine_ids = np.load(engine_path)

    if not (len(X) == len(y_rul) == len(y_hi) == len(engine_ids)):
        raise ValueError(
            "X, RUL, HI and engine IDs must contain the same number of samples."
        )

    if X.shape[-1] != metadata["features"]:
        raise ValueError(
            f"Feature count mismatch: array has {X.shape[-1]} features "
            f"but metadata says {metadata['features']}. "
            f"Re-run preprocess.py --subset {subset}."
        )

    return X, y_rul, y_hi, engine_ids, metadata


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

def build_scheduler(optimizer, config: dict):
    """Build the LR scheduler described by the config."""

    kind = str(config.get("scheduler", "cosine")).lower()

    if kind == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
        )

    if kind == "cosine":
        total = max(1, int(config["epochs"]) - int(config.get("warmup_epochs", 0)))
        return CosineAnnealingLR(
            optimizer,
            T_max=total,
            eta_min=float(config.get("min_lr", 1e-6)),
        )

    if kind in ("none", "null"):
        return None

    raise ValueError(
        f"Unknown scheduler '{kind}'. Expected 'cosine', 'plateau' or 'none'."
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(model_name: str, subset: str, config: dict, tag: str = "") -> None:
    """Run the full training pipeline for a single model/subset combination."""

    dir_name = model_name + (f"_{tag}" if tag else "")

    log = get_script_logger("training", f"train_{subset}_{dir_name}")

    seed = int(config["random_seed"])

    # Seed is set per (model, subset) run so different combinations
    # get different weight initialisations while remaining reproducible.
    # The seed drives weight init, the engine-wise split, dataloader
    # shuffling and the window augmenter, so it identifies the whole run.
    #
    # ``config["deterministic"]`` (default True) additionally enables the
    # deterministic torch/CUDA kernels.  It is recorded in the checkpoint so
    # a reviewer can tell exactly how a published number was produced.
    deterministic = bool(config.get("deterministic", True))
    determinism = set_seed(seed, deterministic=deterministic)

    device = get_device(str(config.get("device", "cuda")))

    log.info("=" * 60)
    log.info("TRAINING MODEL: %s  |  SUBSET: %s", model_name.upper(), subset)
    log.info("=" * 60)
    log.info("Using device: %s", device)
    log.info("Seed        : %d", seed)
    log.info("Determinism : %s", determinism)
    log.info(
        "Config      : %s",
        {k: config[k] for k in sorted(config)},
    )

    # --------------------------------------------------------------
    # Load data + metadata
    # --------------------------------------------------------------

    log.info("Loading processed training data...")

    X, y_rul, y_hi, engine_ids, metadata = load_training_data(
        subset=subset,
        expected_max_rul=config.get("max_rul"),
    )

    window_size = metadata["window_size"]
    log.info("X shape        : %s", X.shape)
    log.info("RUL shape      : %s", y_rul.shape)
    log.info("HI shape       : %s", y_hi.shape)
    log.info("window_size    : %d  (from preprocessing metadata)", window_size)
    log.info("features       : %d", metadata["features"])
    log.info("RUL cap        : %s (from preprocessing metadata)", metadata.get("max_rul"))

    if metadata.get("max_rul") is None:
        log.warning(
            "Processed metadata has no 'max_rul' entry (legacy data). "
            "Cannot verify the label cap; assuming the default of 125. "
            "Re-run scripts/preprocess.py to record it."
        )

    # The health-index construction is part of the label contract: a model
    # trained against a different HI label is not the model that was
    # certified, so the stored mode is checked against the config.
    stored_hi_mode = metadata.get("hi_mode")
    configured_hi_mode = str(config.get("hi_mode", "sensor")).lower()
    if stored_hi_mode is not None and stored_hi_mode != configured_hi_mode:
        raise ValueError(
            f"The processed data was built with hi_mode={stored_hi_mode!r} "
            f"but the config requests {configured_hi_mode!r}. "
            f"Re-run scripts/preprocess.py --subset {subset} "
            f"--hi-mode {configured_hi_mode}."
        )
    log.info("HI label mode   : %s", stored_hi_mode or "unknown (legacy data)")

    # --------------------------------------------------------------
    # Engine-wise Train / Validation Split
    # --------------------------------------------------------------

    log.info("Creating engine-wise train/validation split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_indices, val_indices = next(
        splitter.split(X, y_rul, groups=engine_ids)
    )

    X_train, X_val = X[train_indices], X[val_indices]
    y_rul_train, y_rul_val = y_rul[train_indices], y_rul[val_indices]
    y_hi_train, y_hi_val = y_hi[train_indices], y_hi[val_indices]

    train_engine_ids = engine_ids[train_indices]
    val_engine_ids = engine_ids[val_indices]

    # --------------------------------------------------------------
    # Leakage Verification
    # --------------------------------------------------------------

    train_engines = set(train_engine_ids.tolist())
    val_engines = set(val_engine_ids.tolist())

    overlap = train_engines & val_engines

    if overlap:
        raise RuntimeError(
            "Engine leakage detected! "
            f"Overlapping engines: {sorted(overlap)}"
        )

    log.info("Training engines   : %d", len(train_engines))
    log.info("Validation engines : %d", len(val_engines))
    log.info("Engine overlap     : %d", len(overlap))
    log.info("Training samples   : %d", len(X_train))
    log.info("Validation samples : %d", len(X_val))

    # --------------------------------------------------------------
    # Build Model
    # --------------------------------------------------------------

    input_size = X.shape[-1]

    model = build_model(model_name, input_size=input_size, **config)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    log.info("Model input size: %d", input_size)
    log.info("Trainable parameters: %d", parameter_count)

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------

    # The HI term is a bounded MSE while the RUL term is an MSE in cycles
    # squared, so the raw HI loss would move the shared trunk by ~1e-5 in
    # relative terms.  "rul_max" expresses it in cycle-equivalent units,
    # which makes ``hi_loss_weight`` comparable with ``rul_loss_weight``.
    hi_scale_mode = str(config.get("hi_loss_scale", "rul_max")).lower()
    if hi_scale_mode == "rul_max":
        hi_scale = float(config.get("max_rul", 125)) ** 2
    elif hi_scale_mode == "none":
        hi_scale = 1.0
    else:
        raise ValueError(
            f"Unknown hi_loss_scale {hi_scale_mode!r}; expected 'rul_max' or 'none'."
        )

    criterion = MultiTaskLoss(
        alpha=float(config.get("rul_loss_weight", 1.0)),
        beta=float(config.get("hi_loss_weight", 0.4)),
        gamma=float(config.get("scale_loss_weight", 0.1)),
        hi_scale=hi_scale,
        rul_loss_type=str(config.get("rul_loss_type", "mse")),
        huber_delta=float(config.get("huber_delta", 10.0)),
    )
    log.info(
        "Loss weights: rul=%.3f | hi=%.3f (scale=%.1f) | scale=%.3f",
        criterion.alpha,
        criterion.beta,
        criterion.hi_scale,
        criterion.gamma,
    )

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    optimizer_name = str(config.get("optimizer", "AdamW")).lower()
    optimizer_cls = AdamW if optimizer_name == "adamw" else Adam

    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    scheduler = build_scheduler(optimizer, config)

    # --------------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------------

    augmenter = None
    if config.get("augment", True):
        aug_config = AugmentationConfig.from_config(config)
        if aug_config.enabled:
            augmenter = WindowAugmenter(aug_config, seed=seed)
            log.info("Augmentation enabled: %s", aug_config)
        else:
            log.info("Augmentation configured but all magnitudes are zero.")
    else:
        log.info("Augmentation disabled.")

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    model_checkpoint_dir = CHECKPOINT_DIR / dir_name / subset

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=int(config["batch_size"]),
        checkpoint_dir=model_checkpoint_dir,
        augmenter=augmenter,
        grad_clip=float(config.get("grad_clip", 0.0)),
        ema_decay=0.0 if config.get("no_ema") else float(config.get("ema_decay", 0.0)),
        seed=seed,
    )

    trainer.input_size = input_size
    trainer.window_size = window_size
    trainer.subset = subset

    # Record everything needed to rebuild this exact model later.
    trainer.model_config = {**config, "arch": model_name}

    # Provenance: the seed identifies the run, the determinism block records
    # how it was made repeatable.  Both are stored in the checkpoint so a
    # reviewer can verify that a published number came from this run.
    trainer.run_metadata = {
        "seed": seed,
        "tag": tag,
        "determinism": determinism,
        "arch": model_name,
        "subset": subset,
        "hi_mode": stored_hi_mode,
        "hi_loss_scale": hi_scale_mode,
    }

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_rul_train, y_hi_train,
        X_val, y_rul_val, y_hi_val,
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(config["epochs"]),
        patience=int(config.get("patience", 40)),
        warmup_epochs=int(config.get("warmup_epochs", 0)),
        base_lr=float(config["learning_rate"]),
    )

    # --------------------------------------------------------------
    # Save history
    # --------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history_path = RESULTS_DIR / f"{subset}_{dir_name}_history.npz"

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

    log.info("Best validation loss: %.6f", trainer.best_loss)
    log.info("Training history saved to: %s", history_path)
    log.info("Checkpoint saved to: %s", model_checkpoint_dir)
    log.info("Training completed successfully.")


def main():
    args = parse_args()

    overrides = parse_overrides(args.overrides)

    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.no_augment:
        overrides["augment"] = False

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    models = ALL_MODELS if args.model == "all" else [args.model]

    for subset in subsets:
        config = load_config(
            path=args.config,
            overrides=overrides,
            subset=subset,
        )

        if args.no_ema:
            config["no_ema"] = True

        if args.no_deterministic:
            config["deterministic"] = False

        for model_name in models:
            run(model_name, subset, config, tag=args.tag)


if __name__ == "__main__":
    main()
