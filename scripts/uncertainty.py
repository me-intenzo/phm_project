"""
uncertainty.py — Engine-Aware Regime-Adaptive Conformal Prediction
                 (EARA-Conformal)

Workflow
--------
Processed training data
        ↓
Engine-disjoint train / calibration split  (same seed as train.py)
        ↓
Load trained Model checkpoint
        ↓
Fit k-means regime detector on training windows
        ↓
Assign regime labels to calibration and test windows
        ↓
Predict calibration RUL + scale
        ↓
Calibrate per-regime normalised quantiles  q̂_k
        ↓
Predict test RUL + scale
        ↓
Generate adaptive intervals  C(x) = [ŷ ± q̂_k · s(x)]  clipped [0,125]
        ↓
Evaluate coverage / MPIW / PINAW / per-regime coverage

Important
---------
Final test targets are NEVER used during calibration.
Engine-disjoint split is reproduced with the same seed as training.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru import GRUPrognosticsModel
from src.models.gru_att_deg import GruAttDeg
from src.models.hybrid import HybridPrognosticsModel
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.uncertainty.calibration import calibrate_eara_levels
from src.uncertainty.conformal import (
    adaptive_prediction_interval,
    assign_regimes,
    fit_regime_detector,
)
from src.uncertainty.coverage import evaluate_interval

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_DIR       = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "outputs" / "results"
LOG_DIR        = PROJECT_ROOT / "outputs" / "logs"

SEED            = 42
BATCH_SIZE      = 64
VALIDATION_SIZE = 0.20
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.3
N_REGIMES       = 6
COVERAGE_LEVELS = (0.80, 0.90, 0.95)
ALL_SUBSETS     = ["FD001", "FD002", "FD003", "FD004"]

MODEL_REGISTRY: dict[str, tuple] = {
    "gru": (
        GRUPrognosticsModel,
        {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT},
    ),
    "gru_att_deg": (
        GruAttDeg,
        {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT},
    ),
    "lstm": (
        LSTMPrognosticsModel,
        {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT},
    ),
    "transformer": (
        TransformerPrognosticsModel,
        {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "num_heads": 4, "dropout": DROPOUT},
    ),
    "hybrid": (
        HybridPrognosticsModel,
        {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT},
    ),
}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EARA-Conformal uncertainty estimation."
    )
    parser.add_argument(
        "--subset",
        default="FD001",
        help="C-MAPSS subset or 'all'. Choices: FD001 FD002 FD003 FD004 all",
    )
    parser.add_argument(
        "--model",
        default="hybrid",
        help=f"Model name or 'all'. Choices: {list(MODEL_REGISTRY)} all",
    )
    parser.add_argument(
        "--n_regimes",
        type=int,
        default=N_REGIMES,
        help="Number of operating-condition regimes (k-means clusters).",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def configure_logging(subset: str, model_name: str) -> logging.Logger:
    log_dir = LOG_DIR / "uncertainty"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"uncertainty.{subset}.{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / f"uncertainty_{subset}_{model_name}.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_training_data(subset: str):
    X          = np.load(DATA_DIR / f"{subset}_train_X.npy")
    y_rul      = np.load(DATA_DIR / f"{subset}_train_y_rul.npy")
    y_hi       = np.load(DATA_DIR / f"{subset}_train_y_hi.npy")
    engine_ids = np.load(DATA_DIR / f"{subset}_train_engine_ids.npy")
    return X, y_rul, y_hi, engine_ids


def load_test_data(subset: str):
    X     = np.load(DATA_DIR / f"{subset}_test_X.npy")
    y_rul = np.load(DATA_DIR / f"{subset}_test_y_rul.npy")
    y_hi  = np.load(DATA_DIR / f"{subset}_test_y_hi.npy")
    return X, y_rul, y_hi


# ------------------------------------------------------------------
# Engine-disjoint calibration split
# ------------------------------------------------------------------

def create_calibration_split(X, y_rul, engine_ids):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=SEED
    )
    train_idx, cal_idx = next(splitter.split(X, y_rul, groups=engine_ids))

    overlap = set(engine_ids[train_idx]) & set(engine_ids[cal_idx])
    if overlap:
        raise RuntimeError(f"Engine leakage detected: {sorted(overlap)}")

    return train_idx, cal_idx


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def load_model(model_name: str, input_size: int, device: torch.device, subset: str):
    ckpt_path = CHECKPOINT_DIR / model_name / subset / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first: python scripts/train.py --model {model_name}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_input = ckpt.get("input_size", input_size)
    if ckpt_input != input_size:
        raise RuntimeError(
            f"Checkpoint input_size={ckpt_input} != data input_size={input_size}."
        )
    cls, kwargs = MODEL_REGISTRY[model_name]
    model = cls(input_size=input_size, **kwargs)
    missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # scale_head absent in older checkpoints — init to neutral scale ≈ 1.0
    if missing:
        import math
        nn.init.zeros_(model.scale_head.weight)
        nn.init.constant_(model.scale_head.bias, math.log(math.exp(1.0) - 1.0))
    model.to(device).eval()
    return model, ckpt


# ------------------------------------------------------------------
# Inference — returns (rul_pred, scale_pred)
# ------------------------------------------------------------------

def predict(model, X: np.ndarray, device: torch.device):
    rul_preds, scale_preds = [], []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = (
                torch.from_numpy(X[start : start + BATCH_SIZE])
                .float()
                .to(device)
            )
            rul, _hi, scale = model(batch)
            rul_preds.append(rul.cpu().numpy())
            scale_preds.append(scale.cpu().numpy())
    return np.concatenate(rul_preds), np.concatenate(scale_preds)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(subset: str, model_name: str, n_reg: int) -> None:
    """Run EARA-Conformal estimation for a single subset/model combination."""
    log    = configure_logging(subset, model_name)
    device = get_device()

    log.info("=" * 60)
    log.info("EARA-CONFORMAL UNCERTAINTY ESTIMATION")
    log.info("Subset   : %s", subset)
    log.info("Model    : %s", model_name)
    log.info("Device   : %s", device)
    log.info("Regimes  : %d", n_reg)
    log.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────
    X_all, y_rul_all, _, engine_ids = load_training_data(subset)
    log.info("Training windows : %d  |  features : %d", len(X_all), X_all.shape[-1])

    train_idx, cal_idx = create_calibration_split(X_all, y_rul_all, engine_ids)

    X_cal     = X_all[cal_idx]
    y_rul_cal = y_rul_all[cal_idx]
    X_train   = X_all[train_idx]

    log.info(
        "Train engines : %d  |  Cal engines : %d  |  Cal windows : %d",
        len(np.unique(engine_ids[train_idx])),
        len(np.unique(engine_ids[cal_idx])),
        len(X_cal),
    )

    X_test, y_rul_test, _ = load_test_data(subset)
    log.info("Test windows : %d", len(X_test))

    # ── Model ─────────────────────────────────────────────────────
    model, ckpt = load_model(model_name, X_all.shape[-1], device, subset)
    log.info("Loaded checkpoint from epoch %d", ckpt["epoch"])

    # ── Regime detection (fit on training windows only) ───────────
    log.info("Fitting regime detector (k=%d) on training windows...", n_reg)
    km = fit_regime_detector(X_train, n_regimes=n_reg, random_state=SEED)

    regimes_cal  = assign_regimes(km, X_cal)
    regimes_test = assign_regimes(km, X_test)

    unique, counts = np.unique(regimes_cal, return_counts=True)
    for r, c in zip(unique, counts):
        flag = "  [< 30, global fallback]" if c < 30 else ""
        log.info("  Calibration regime %d : %d windows%s", r, c, flag)

    # ── Calibration predictions ───────────────────────────────────
    log.info("Generating calibration predictions...")
    rul_cal, scale_cal = predict(model, X_cal, device)

    # ── EARA-Conformal calibration ────────────────────────────────
    log.info("Calibrating per-regime quantiles...")
    cal_results = calibrate_eara_levels(
        y_true=y_rul_cal,
        y_pred=rul_cal,
        scale=scale_cal,
        regimes=regimes_cal,
        n_regimes=n_reg,
        coverage_levels=COVERAGE_LEVELS,
    )

    for cov, res in cal_results.items():
        log.info(
            "Nominal %3.0f%%  |  q_hat per regime: %s",
            cov * 100,
            np.round(res["q_hats"], 4),
        )

    # ── Test predictions ──────────────────────────────────────────
    log.info("Generating test predictions...")
    rul_test, scale_test = predict(model, X_test, device)

    # ── Evaluate intervals ────────────────────────────────────────
    eval_results = {}
    for cov, cal in cal_results.items():
        lower, upper = adaptive_prediction_interval(
            y_pred=rul_test,
            scale=scale_test,
            q_hats=cal["q_hats"],
            regimes=regimes_test,
        )
        metrics = evaluate_interval(
            y_true=y_rul_test,
            lower=lower,
            upper=upper,
            nominal_coverage=cov,
            regimes=regimes_test,
            n_regimes=n_reg,
        )
        eval_results[cov] = {**metrics, "lower": lower, "upper": upper}

        log.info(
            "Coverage %3.0f%%  |  Empirical %.4f  |  Error %.4f  |  "
            "MPIW %.4f  |  PINAW %.4f",
            cov * 100,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
            metrics["pinaw"],
        )
        for k, rc in metrics.get("regime_coverage", {}).items():
            log.info("    Regime %d coverage : %.4f", k, rc)

    # ── Save ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{subset}_{model_name}_eara_conformal.npz"

    save_data: dict[str, np.ndarray] = {
        "y_rul_test":    y_rul_test,
        "y_rul_pred":    rul_test,
        "scale_pred":    scale_test,
        "regimes_test":  regimes_test,
        "regimes_cal":   regimes_cal,
        "cal_scores":    np.abs(y_rul_cal - rul_cal) / (scale_cal + 1e-6),
    }

    for cov, res in eval_results.items():
        suffix = int(cov * 100)
        save_data[f"lower_{suffix}"]  = res["lower"]
        save_data[f"upper_{suffix}"]  = res["upper"]
        save_data[f"q_hats_{suffix}"] = cal_results[cov]["q_hats"]

    np.savez(out_path, **save_data)
    log.info("Results saved to: %s", out_path)
    log.info("=" * 60)
    log.info("EARA-CONFORMAL ESTIMATION COMPLETE")
    log.info("=" * 60)


def main():
    args = parse_args()
    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    models  = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for subset in subsets:
        for model_name in models:
            try:
                run(subset, model_name, args.n_regimes)
            except FileNotFoundError as e:
                print(f"[SKIP] {model_name}/{subset}: {e}")
            except Exception as e:
                print(f"[ERROR] {model_name}/{subset}: {e}")


if __name__ == "__main__":
    main()
