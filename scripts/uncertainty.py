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
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_script_logger
from src.models.gru import GRUPrognosticsModel
from src.models.gru_att_deg import GruAttDeg
from src.models.hybrid import HybridPrognosticsModel
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.uncertainty.calibration import calibrate_eara_levels
from src.uncertainty.conformal import (
    adaptive_conformal_interval,
    adaptive_prediction_interval,
    assign_regimes,
    conformal_interval,
    engine_joint_interval,
    fit_regime_detector,
    scale_conformalized_interval,
)
from src.uncertainty.coverage import evaluate_interval

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_DIR       = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "outputs" / "results"
UNCERTAINTY_RESULTS_DIR = RESULTS_DIR / "uncertainty"

SEED            = 42
BATCH_SIZE      = 64
VALIDATION_SIZE = 0.20
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.3
N_REGIMES       = 7  # optimal: regime sweep (k=2..10) → best composite score at k=7
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
    parser.add_argument(
        "--variant",
        default="eara_conformal",
        choices=("eara_conformal", "conformal", "adaptive_conformal", "cqr", "engine_joint_conformal"),
        help="Uncertainty method to run.",
    )
    return parser.parse_args()


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
    engine_ids = np.load(DATA_DIR / f"{subset}_test_engine_ids.npy")
    return X, y_rul, y_hi, engine_ids


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

def _split_hash(train_ids: np.ndarray, cal_ids: np.ndarray) -> str:
    payload = np.concatenate((np.sort(np.unique(train_ids)), np.sort(np.unique(cal_ids))))
    return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


def _calibrate_variant(
    variant: str,
    y_rul_cal: np.ndarray,
    rul_cal: np.ndarray,
    scale_cal: np.ndarray,
    regimes_cal: np.ndarray,
    engine_ids_cal: np.ndarray,
    n_reg: int,
    rul_test: np.ndarray,
    scale_test: np.ndarray,
    regimes_test: np.ndarray,
) -> dict[float, dict[str, object]]:
    results: dict[float, dict[str, object]] = {}
    for coverage in COVERAGE_LEVELS:
        alpha = 1.0 - coverage
        if variant == "eara_conformal":
            calibrated = calibrate_eara_levels(
                y_true=y_rul_cal, y_pred=rul_cal, scale=scale_cal,
                regimes=regimes_cal, n_regimes=n_reg,
                coverage_levels=(coverage,),
            )[coverage]
            lower, upper = adaptive_prediction_interval(
                y_pred=rul_test, scale=scale_test,
                q_hats=calibrated["q_hats"], regimes=regimes_test,
            )
            results[coverage] = {**calibrated, "lower": lower, "upper": upper}
        elif variant == "conformal":
            results[coverage] = conformal_interval(y_rul_cal, rul_cal, rul_test, alpha)
        elif variant == "adaptive_conformal":
            results[coverage] = adaptive_conformal_interval(
                y_rul_cal, rul_cal, scale_cal, rul_test, scale_test, alpha,
            )
        elif variant == "cqr":
            results[coverage] = scale_conformalized_interval(
                y_rul_cal, rul_cal, scale_cal, rul_test, scale_test, alpha,
            )
        elif variant == "engine_joint_conformal":
            results[coverage] = engine_joint_interval(
                y_rul_cal, rul_cal, engine_ids_cal, rul_test, alpha,
            )
        else:
            raise ValueError(f"Unknown uncertainty variant: {variant}")
    return results


def run(subset: str, model_name: str, n_reg: int, variant: str) -> None:
    """Run one uncertainty variant for a single subset/model combination."""
    log    = get_script_logger("uncertainty", f"uncertainty_{subset}_{model_name}")
    device = get_device()

    log.info("=" * 60)
    log.info("%s UNCERTAINTY ESTIMATION", variant.upper())
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

    X_test, y_rul_test, _, test_engine_ids = load_test_data(subset)
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

    # ── Test predictions ──────────────────────────────────────────
    log.info("Generating test predictions...")
    rul_test, scale_test = predict(model, X_test, device)

    log.info("Calibrating %s...", variant)
    cal_results = _calibrate_variant(
        variant, y_rul_cal, rul_cal, scale_cal, regimes_cal,
        engine_ids[cal_idx], n_reg, rul_test, scale_test, regimes_test,
    )

    # ── Evaluate intervals ────────────────────────────────────────
    # Each variant has already produced its own calibrated interval in
    # `cal_results`, so evaluate those bounds directly. Re-deriving them
    # here would assume EARA's per-regime `q_hats` and raise KeyError for
    # the global variants, which return a scalar `q_hat`.
    eval_results = {}
    for cov, cal in cal_results.items():
        lower = cal["lower"]
        upper = cal["upper"]
        metrics = evaluate_interval(
            y_true=y_rul_test,
            lower=lower,
            upper=upper,
            nominal_coverage=cov,
            regimes=regimes_test,
            n_regimes=n_reg,
            engine_ids=test_engine_ids,
        )
        eval_results[cov] = {**cal, **metrics}

        log.info(
            "Coverage %3.0f%%  |  Empirical %.4f  |  Error %.4f  |  "
            "MPIW %.4f  |  PINAW %.4f",
            cov * 100,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
            metrics["pinaw"],
        )
        log.info(
            "Coverage units | windows: %.4f | engines: %.4f",
            metrics["empirical_coverage"],
            metrics["engine_level_coverage"],
        )
        for k, rc in metrics.get("regime_coverage", {}).items():
            log.info("    Regime %d coverage : %.4f", k, rc)

    # ── Save ──────────────────────────────────────────────────────
    output_dir = UNCERTAINTY_RESULTS_DIR / variant / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}.npz"
    metadata_path = output_dir / f"{model_name}.json"

    save_data: dict[str, np.ndarray] = {
        "y_rul_test":    y_rul_test,
        "y_rul_pred":    rul_test,
        "scale_pred":    scale_test,
        "regimes_test":  regimes_test,
        "regimes_cal":   regimes_cal,
        "cal_scores":    np.abs(y_rul_cal - rul_cal) / (scale_cal + 1e-6),
        "cal_engine_ids": engine_ids[cal_idx],
        "test_engine_ids": test_engine_ids,
    }

    for cov, res in eval_results.items():
        suffix = int(cov * 100)
        save_data[f"lower_{suffix}"]  = res["lower"]
        save_data[f"upper_{suffix}"]  = res["upper"]
        save_data[f"window_coverage_{suffix}"] = np.asarray(res["empirical_coverage"])
        save_data[f"engine_coverage_{suffix}"] = np.asarray(res["engine_level_coverage"])
        save_data[f"q_hat_{suffix}"] = np.asarray(res.get("q_hat", res.get("q_hats")))
        if "q_hats" in res:
            save_data[f"q_hats_{suffix}"] = res["q_hats"]
        if "engine_scores" in res:
            save_data[f"engine_scores_{suffix}"] = res["engine_scores"]

    np.savez(out_path, **save_data)
    metadata = {
        "schema_version": 2,
        "variant": variant,
        "subset": subset,
        "model": model_name,
        "seed": SEED,
        "n_regimes": n_reg,
        "coverage_levels": list(COVERAGE_LEVELS),
        "calibration_engine_count": int(len(np.unique(engine_ids[cal_idx]))),
        "test_engine_count": int(len(np.unique(test_engine_ids))),
        "split_hash": _split_hash(engine_ids[train_idx], engine_ids[cal_idx]),
        "cqr_note": (
            "Scale-based conformalized interval (CQR-style): checkpoint point "
            "prediction +/- scale is used as the conditional bound; the current "
            "model has no separately trained quantile heads."
            if variant == "cqr" else None
        ),
        "coverage_units": {
            "window_level": "fraction of windows covered",
            "engine_level": "fraction of engines with every window covered",
            "primary_for_engine_joint_conformal": "engine_level",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Results saved to: %s", out_path)
    log.info("=" * 60)
    log.info("%s ESTIMATION COMPLETE", variant.upper())
    log.info("=" * 60)


def main():
    args = parse_args()
    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    models  = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for subset in subsets:
        for model_name in models:
            try:
                run(subset, model_name, args.n_regimes, args.variant)
            except FileNotFoundError as e:
                print(f"[SKIP] {model_name}/{subset}: {e}")
            except Exception as e:
                print(f"[ERROR] {model_name}/{subset}: {e}")


if __name__ == "__main__":
    main()
