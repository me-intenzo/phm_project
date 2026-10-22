"""
uncertainty.py

Conformal uncertainty estimation for trained prognostics models.

Workflow
--------
Processed training data
        ↓
Reproduce engine-wise train/validation split
        ↓
Load trained model checkpoint
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

python scripts/uncertainty.py --model lstm --subset FD001 --method split
python scripts/uncertainty.py --model lstm --subset FD001 --method cqr
python scripts/uncertainty.py --model lstm --subset FD001 --method adaptive
python scripts/uncertainty.py --model lstm --subset FD001 --method engine_joint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

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

from src.models.outputs import unpack_model_output
from src.models.registry import ALL_MODELS, load_model_from_checkpoint
from src.models.uncertainty_heads import QuantileHead, ScaleHead
from src.uncertainty import pipeline
from src.uncertainty.conformal import (
    SCALE_FLOOR,
    calibrate_multiple_levels,
    conformal_quantile,
    floor_scale,
    normalised_nonconformity,
    prediction_interval,
    scale_adaptive_interval,
    signed_conformal_quantile,
)
from src.uncertainty.coverage import (
    evaluate_interval,
)
from src.utils.config import get_device
from src.utils.paths import (
    checkpoint_path,
    results_path,
)
from src.utils.provenance import build_provenance, pack_provenance

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
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

# Fallback seed.  The real seed is read from the checkpoint that is being
# evaluated, because the calibration split MUST reproduce the training
# split of that exact checkpoint.  Using a hard-coded seed here would
# calibrate a seed-43 checkpoint against the seed-42 hold-out, whose
# engines the seed-43 model was trained on — a silent leakage bug.
DEFAULT_SEED = SEED

BATCH_SIZE = 64

VALIDATION_SIZE = 0.20

# NOTE: the architecture (hidden size, layer count, dropout) is no longer
# hard-coded here.  It is read from the ``model_config`` stored inside each
# checkpoint, so a model trained with a per-subset YAML override (e.g.
# FD002/FD004 use ``hidden_size: 160``) is rebuilt correctly.

COVERAGE_LEVELS = (
    0.80,
    0.90,
    0.95,
)

# NOTE: the scale floor is imported from ``src.uncertainty.conformal``
# (``SCALE_FLOOR``).  It used to be declared a third time in this script
# with a different value and a different bounding operator, so the
# calibration score and the interval radius of the EARA path did not
# use the same scale.


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Conformal uncertainty estimation "
            "for a trained prognostics model."
        )
    )

    parser.add_argument(
        "--model",
        choices=list(ALL_MODELS),
        default="gru",
        help="Model architecture whose checkpoint will be evaluated.",
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

    parser.add_argument(
        "--method",
        choices=["split", "cqr", "adaptive", "engine_joint", "eara_scp"],
        default="split",
        help=(
            "Uncertainty method: global split conformal ('split'), "
            "conformalized quantile regression ('cqr'), normalized "
            "adaptive conformal ('adaptive'), engine-block joint "
            "conformal ('engine_joint'), or EARA-SCP — engine/regime-aware "
            "scale with prequential online adaptation ('eara_scp')."
        ),
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Checkpoint tag (e.g. 'ms42'). Selects a specific run of the "
            "model and makes the output paths tag-aware."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def configure_logging(
    model_name: str,
    subset: str,
    method: str,
    tag: str = "",
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
                / f"uncertainty_{model_name}_{subset}_{method}"
                f"{'_' + tag if tag else ''}.log",
                mode="w",
            ),
        ],
        force=True,
    )

    return logging.getLogger(__name__)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

# ``get_device`` lives in ``src.utils.config`` — the single device
# selection of the project.


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
    seed: int = DEFAULT_SEED,
):
    """
    Reproduce the engine-wise validation split used
    by scripts/train.py.

    ``seed`` must be the seed of the checkpoint being evaluated.  The
    trainer splits the engines with ``GroupShuffleSplit(random_state=seed)``,
    so using a different seed here would select a *different* hold-out —
    one that can contain engines the model was trained on.  That is a
    leakage bug, not a cosmetic difference, so the seed is taken from the
    checkpoint rather than hard-coded.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=int(seed),
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

def build_model(
    model_name: str,
    input_size: int,
    **hyperparameters,
) -> torch.nn.Module:
    """Reconstruct the architecture used by ``scripts/train.py``.

    Delegates to the shared registry so every script builds a model the
    same way.  Unknown keyword arguments are ignored by the registry, which
    lets a flat training config be splatted straight in.
    """
    from src.models.registry import build_model as build_model_from_registry

    return build_model_from_registry(
        model_name,
        input_size=input_size,
        **hyperparameters,
    )


def load_model(
    model_name: str,
    input_size: int,
    device: torch.device,
    subset: str = "",
    tag: str = "",
):
    """
    Load a trained architecture-specific checkpoint.

    The checkpoint stores the seed it was trained with.  That seed is
    returned alongside the model because the conformal calibration must
    reproduce *this* checkpoint's training split.

    The loader itself lives in ``src/models/registry.py``; three scripts
    used to carry their own copy of it.
    """

    model, checkpoint, _arch = load_model_from_checkpoint(
        model_name=model_name,
        input_size=input_size,
        device=device,
        subset=subset,
        tag=tag,
        strict_input_size=True,
    )

    return model, checkpoint


def build_artifact_provenance(
    model_name: str,
    subset: str,
    tag: str = "",
    variant: str = "",
    seed: int | None = None,
    checkpoint: dict | None = None,
) -> dict[str, Any]:
    """Provenance stamp for one uncertainty artifact.

    The stamp records the SHA-256 digest of the checkpoint the intervals
    were produced from.  It is embedded in the ``.npz`` so that a reader
    can tell whether two artifacts came from the same weights — which is
    what makes a comparison between predictions and intervals
    trustworthy.
    """

    extra: dict[str, Any] = {"variant": variant}
    if seed is not None:
        extra["seed"] = int(seed)

    return build_provenance(
        checkpoint_path=checkpoint_path(model_name, subset, tag),
        checkpoint=checkpoint,
        subset=subset,
        model=model_name,
        tag=tag,
        extra=extra,
    )


def checkpoint_seed(
    checkpoint: dict,
    fallback: int = DEFAULT_SEED,
) -> int:
    """Read the training seed out of a checkpoint.

    Priority order:

    1. ``run_metadata["seed"]`` — written by the current trainer;
    2. ``checkpoint["seed"]`` — the flat copy of the same value;
    3. ``model_config["random_seed"]`` — the config value used by older runs;
    4. ``fallback``.

    The returned seed identifies the engine-wise split the model was
    trained with, so the conformal calibration can reproduce it exactly.
    """

    metadata = checkpoint.get("run_metadata") or {}

    for candidate in (
        metadata.get("seed"),
        checkpoint.get("seed"),
        (checkpoint.get("model_config") or {}).get("random_seed"),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue

    return int(fallback)


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

            pred_rul, _, _ = unpack_model_output(model(batch))

            predictions.append(
                pred_rul
                .cpu()
                .numpy()
                .reshape(-1)
            )

    return np.concatenate(
        predictions
    )


def predict_rul_hi(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate both model outputs for joint RUL/HI conformal scoring."""
    rul_predictions, hi_predictions = [], []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[start:start + BATCH_SIZE]).float().to(device)
            pred_rul, pred_hi, _ = unpack_model_output(model(batch))
            if pred_hi is None:
                pred_hi = torch.zeros_like(pred_rul)
            rul_predictions.append(pred_rul.cpu().numpy().reshape(-1))
            hi_predictions.append(pred_hi.cpu().numpy().reshape(-1))
    return np.concatenate(rul_predictions), np.concatenate(hi_predictions)


def predict(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict RUL together with the model's own scale estimate.

    Compatibility entry point used by ``scripts/regime_sweep.py``. The
    returned scale is floored with the shared ``SCALE_FLOOR`` so callers
    that build EARA intervals from it stay consistent with the
    calibration path.
    """
    rul_predictions, scale_predictions = [], []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[start:start + BATCH_SIZE]).float().to(device)
            pred_rul, _, pred_scale = unpack_model_output(model(batch))
            if pred_scale is None:
                pred_scale = torch.ones_like(pred_rul)
            rul_predictions.append(pred_rul.cpu().numpy().reshape(-1))
            scale_predictions.append(pred_scale.cpu().numpy().reshape(-1))
    rul = np.concatenate(rul_predictions)
    scale = floor_scale(np.concatenate(scale_predictions))
    return rul, scale


def _robust_scale(values: np.ndarray, floor: float = SCALE_FLOOR) -> float:
    """Robust residual scale estimate."""

    return pipeline._robust_scale(values, floor)


def _build_training_engine_profiles(
    features: torch.Tensor,
    residuals: np.ndarray,
    engine_ids: np.ndarray,
    predicted_rul: np.ndarray,
) -> dict:
    """
    Build engine/regime profiles using TRAINING ENGINES ONLY.

    Delegates to ``src.uncertainty.pipeline`` so the production CLI and the
    multi-seed benchmark build identical profiles.  The scale function is
    deliberately constructed before conformal calibration, which preserves
    the split-conformal separation between calibration and test data.
    """

    return pipeline.build_training_engine_profiles(
        features,
        residuals,
        engine_ids,
        predicted_rul,
    )


def _nearest_engine_regime_scale(
    feature: np.ndarray,
    predicted_rul: float,
    profile_data: dict,
    top_k: int = 5,
) -> tuple[float, list[int], np.ndarray, int]:
    """Retrieve nearest training engines and produce a regime-aware scale."""

    return pipeline.nearest_engine_regime_scale(
        feature,
        predicted_rul,
        profile_data,
        top_k=top_k,
    )


def run_engine_joint_conformal(
    model: torch.nn.Module,
    model_name: str,
    subset: str,
    X_train: np.ndarray,
    y_rul_train: np.ndarray,
    y_hi_train: np.ndarray,
    X_calibration: np.ndarray,
    y_rul_calibration: np.ndarray,
    y_hi_calibration: np.ndarray,
    calibration_engine_ids: np.ndarray,
    device: torch.device,
    log: logging.Logger,
    training_engine_ids: np.ndarray | None = None,
    tag: str = "",
    seed: int = DEFAULT_SEED,
    checkpoint: dict | None = None,
) -> None:
    """
    Engine-Aware Regime-Adaptive Conformal Prediction (EARA-Conformal).

    ``engine_joint`` is retained as the CLI compatibility endpoint.

    Novel mechanism:
      1. Learn engine profiles from training engines only.
      2. Extract a frozen prognostic representation from the backbone.
      3. Retrieve the nearest training-engine profiles at each query.
      4. Condition the residual scale on a learned RUL degradation regime.
      5. Normalize calibration residuals by that query-specific scale.
      6. Apply split-conformal calibration to the normalized scores.
      7. Produce query-specific RUL intervals using the calibrated q-hat.

    No calibration or test target is used to learn the scale function.
    """
    if training_engine_ids is None:
        raise ValueError(
            "EARA-Conformal requires training_engine_ids for engine-aware profiles."
        )

    log.info(
        "============================================================"
    )
    log.info(
        "ENGINE-AWARE REGIME-ADAPTIVE CONFORMAL PREDICTION (EARA)"
    )
    log.info(
        "Building engine/regime profiles from training engines only..."
    )

    train_features = extract_model_features(model, X_train, device)
    calibration_features = extract_model_features(model, X_calibration, device)

    train_rul = predict_rul(model, X_train, device)
    calibration_predictions = predict_rul(model, X_calibration, device)

    train_residuals = np.abs(y_rul_train - train_rul)
    calibration_residuals = np.abs(
        y_rul_calibration - calibration_predictions
    )

    profile_data = _build_training_engine_profiles(
        train_features,
        train_residuals,
        training_engine_ids,
        train_rul,
    )

    # --------------------------------------------------------------
    # Engine/regime-adaptive calibration scores
    # --------------------------------------------------------------
    calibration_scales = np.empty(len(X_calibration), dtype=np.float64)
    calibration_regimes = np.empty(len(X_calibration), dtype=np.int64)

    for i, (feature, pred) in enumerate(
        zip(
            calibration_features.detach().cpu().numpy(),
            calibration_predictions,
        )
    ):
        scale, _, _, regime = _nearest_engine_regime_scale(
            feature,
            float(pred),
            profile_data,
            top_k=5,
        )
        calibration_scales[i] = scale
        calibration_regimes[i] = regime

    normalized_scores = calibration_residuals / floor_scale(
        calibration_scales,
    )

    log.info(
        "EARA calibration: %d training engines -> %d calibration engines",
        len(np.unique(training_engine_ids)),
        len(np.unique(calibration_engine_ids)),
    )
    log.info(
        "EARA retrieval: top-k=5; degradation regimes=3; "
        "normalized-score calibration enabled"
    )
    log.info(
        "Normalized calibration score median=%.6f, 90th percentile=%.6f",
        float(np.median(normalized_scores)),
        float(np.quantile(normalized_scores, 0.90)),
    )

    # --------------------------------------------------------------
    # Final test predictions
    # --------------------------------------------------------------
    X_test, y_rul_test, y_hi_test = load_test_data(subset)
    test_features = extract_model_features(model, X_test, device)
    test_predictions = predict_rul(model, X_test, device)

    test_scales = np.empty(len(X_test), dtype=np.float64)
    test_regimes = np.empty(len(X_test), dtype=np.int64)
    test_neighbors = []

    for feature, pred in zip(
        test_features.detach().cpu().numpy(),
        test_predictions,
    ):
        scale, neighbors, _, regime = _nearest_engine_regime_scale(
            feature,
            float(pred),
            profile_data,
            top_k=5,
        )
        test_scales[len(test_neighbors)] = scale
        test_regimes[len(test_neighbors)] = regime
        test_neighbors.append(neighbors)

    save_data: dict[str, np.ndarray] = {
        "y_rul_test": y_rul_test,
        "y_hi_test": y_hi_test,
        "y_rul_pred": test_predictions,
        "test_scales": test_scales,
        "test_regimes": test_regimes,
        "calibration_scales": calibration_scales,
        "calibration_regimes": calibration_regimes,
        "normalized_calibration_scores": normalized_scores,
    }

    # --------------------------------------------------------------
    # Conformal prediction
    # --------------------------------------------------------------
    for coverage in COVERAGE_LEVELS:
        alpha = 1.0 - coverage

        # This is the only learned conformal quantile. It is calibrated on
        # held-out engines after the EARA scale function has been fixed using
        # training engines only.
        q_hat = conformal_quantile(
            normalized_scores,
            alpha,
        )

        radius = q_hat * floor_scale(test_scales)
        lower = test_predictions - radius
        upper = test_predictions + radius

        metrics = evaluate_interval(
            y_true=y_rul_test,
            lower=lower,
            upper=upper,
            nominal_coverage=coverage,
        )

        suffix = int(coverage * 100)
        save_data[f"lower_{suffix}"] = lower
        save_data[f"upper_{suffix}"] = upper
        save_data[f"q_hat_{suffix}"] = np.asarray([q_hat])

        log.info(
            "EARA-Conformal %.0f%% | q_hat %.6f | "
            "Empirical %.4f | Error %.4f | MPIW %.4f | "
            "MeanScale %.4f",
            coverage * 100.0,
            q_hat,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
            float(np.mean(test_scales)),
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(
        subset,
        model_name,
        tag,
        suffix="engine_joint_conformal",
    )
    save_data["meta_json"] = pack_provenance(
        build_artifact_provenance(
            model_name,
            subset,
            tag,
            variant="engine_joint_conformal",
            seed=seed,
            checkpoint=checkpoint,
        )
    )
    np.savez(output_path, **save_data)
    log.info("EARA-Conformal results saved to: %s", output_path)
    log.info(
        "EARA complete: intervals are regime- and engine-neighbor adaptive."
    )


# ------------------------------------------------------------------
# Frozen-feature helpers
#
# The implementations live in ``src.uncertainty.pipeline`` so that the
# production pipeline (``scripts/uncertainty.py``) and the multi-seed
# benchmark (``scripts/multiseed_conformal_benchmark.py``) share ONE
# implementation of every primitive.  The wrappers below keep the
# historical local names so the rest of this file is unchanged.
# ------------------------------------------------------------------


def extract_model_features(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Capture the input to the architecture's RUL head via a safe hook."""

    return pipeline.extract_model_features(model, X, device, BATCH_SIZE)


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Pinball (quantile) loss."""

    return pipeline.pinball_loss(prediction, target, quantile)


def train_quantile_head(
    features: torch.Tensor,
    y_true: np.ndarray,
    device: torch.device,
    lower_quantile: float,
    upper_quantile: float,
    epochs: int = 50,
    seed: int = DEFAULT_SEED,
    base_pred: np.ndarray | None = None,
) -> QuantileHead:
    """Fit a quantile head using only the training-engine features."""

    return pipeline.fit_quantile_head(
        features=features,
        y_true=y_true,
        device=device,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        epochs=epochs,
        seed=seed,
        base_pred=base_pred,
    )


def predict_quantiles(
    head: QuantileHead,
    features: torch.Tensor,
    device: torch.device,
    base_pred: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict conditional quantiles from a fitted quantile head."""

    return pipeline.predict_quantiles(head, features, device, base_pred=base_pred)


def cqr_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile for **signed** CQR scores.

    Thin wrapper around :func:`src.uncertainty.conformal.signed_conformal_quantile`
    — the canonical implementation.  This script used to carry its own
    copy of the finite-sample quantile, which accepted non-finite scores
    and clamped the rank.
    """
    return signed_conformal_quantile(scores, alpha)


def train_scale_head(
    features: torch.Tensor,
    residuals: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    seed: int = DEFAULT_SEED,
) -> ScaleHead:
    """Fit a positive residual-scale model on training engines only."""

    return pipeline.fit_scale_head(
        features=features,
        residuals=residuals,
        device=device,
        epochs=epochs,
        seed=seed,
    )


def predict_scales(
    head: ScaleHead,
    features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Predict positive residual scales from a fitted scale head."""

    return pipeline.predict_scales(head, features, device)



def run_eara_scp(
    model: torch.nn.Module,
    model_name: str,
    subset: str,
    X_train: np.ndarray,
    y_rul_train: np.ndarray,
    X_calibration: np.ndarray,
    y_rul_calibration: np.ndarray,
    calibration_engine_ids: np.ndarray,
    training_engine_ids: np.ndarray,
    device: torch.device,
    log: logging.Logger,
    tag: str = "",
    seed: int = DEFAULT_SEED,
    checkpoint: dict | None = None,
    aci_gamma: float = 0.02,
) -> None:
    """EARA-SCP: engine/regime-aware scale + prequential online adaptation.

    Combines the EARA-Conformal scale (kNN retrieval over training-engine
    profiles, conditioned on degradation regime) with the NEX-SCP prequential
    miscoverage update.  The scale is sharper than the generic scale head
    because it is conditioned on the engine's degradation regime; the level
    adapts from each realised test outcome so coverage tracks the nominal
    target even when the scale estimates are imperfect.
    """
    log.info("=" * 60)
    log.info("EARA-SCP: ENGINE/REGIME-AWARE SCALE + PREQUENTIAL ADAPTATION")

    train_features = extract_model_features(model, X_train, device)
    calibration_features = extract_model_features(model, X_calibration, device)

    train_rul = predict_rul(model, X_train, device)
    calibration_predictions = predict_rul(model, X_calibration, device)

    train_residuals = np.abs(y_rul_train - train_rul)
    calibration_residuals = np.abs(y_rul_calibration - calibration_predictions)

    profile_data = _build_training_engine_profiles(
        train_features, train_residuals, training_engine_ids, train_rul
    )

    calibration_scales = np.empty(len(X_calibration), dtype=np.float64)
    for i, (feature, pred) in enumerate(
        zip(calibration_features.detach().cpu().numpy(), calibration_predictions)
    ):
        scale, _, _, _ = _nearest_engine_regime_scale(
            feature, float(pred), profile_data, top_k=5
        )
        calibration_scales[i] = scale

    normalized_scores = calibration_residuals / floor_scale(calibration_scales)

    X_test, y_rul_test, y_hi_test = load_test_data(subset)
    test_features = extract_model_features(model, X_test, device)
    test_predictions = predict_rul(model, X_test, device)

    test_scales = np.empty(len(X_test), dtype=np.float64)
    for i, (feature, pred) in enumerate(
        zip(test_features.detach().cpu().numpy(), test_predictions)
    ):
        scale, _, _, _ = _nearest_engine_regime_scale(
            feature, float(pred), profile_data, top_k=5
        )
        test_scales[i] = scale

    save_data: dict[str, np.ndarray] = {
        "y_rul_test": y_rul_test,
        "y_hi_test": y_hi_test,
        "y_rul_pred": test_predictions,
        "test_scales": test_scales,
        "calibration_scales": calibration_scales,
        "normalized_calibration_scores": normalized_scores,
    }

    for coverage in COVERAGE_LEVELS:
        alpha = 1.0 - coverage
        lower, upper, alpha_stream = pipeline.prequential_aci_intervals(
            scores=normalized_scores,
            y_pred=test_predictions,
            scales=test_scales,
            y_true=y_rul_test,
            alpha=alpha,
            gamma=aci_gamma,
            scale_floor=SCALE_FLOOR,
        )
        metrics = evaluate_interval(y_rul_test, lower, upper, coverage)
        suffix = int(coverage * 100)
        save_data[f"lower_{suffix}"] = lower
        save_data[f"upper_{suffix}"] = upper
        # q_hat stores the median adapted alpha level (not a radius quantile).
        save_data[f"q_hat_{suffix}"] = np.asarray(
            [float(np.median(alpha_stream))] if alpha_stream else [alpha]
        )
        save_data[f"final_alpha_{suffix}"] = np.asarray(
            [alpha_stream[-1]] if alpha_stream else [alpha]
        )
        log.info(
            "EARA-SCP %.0f%% | median_alpha %.4f | final_alpha %.4f | "
            "Empirical %.4f | Error %.4f | MPIW %.4f | MeanScale %.4f",
            coverage * 100.0,
            float(np.median(alpha_stream)) if alpha_stream else alpha,
            alpha_stream[-1] if alpha_stream else alpha,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
            float(np.mean(test_scales)),
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(subset, model_name, tag, suffix="eara_scp")
    save_data["meta_json"] = pack_provenance(
        build_artifact_provenance(
            model_name, subset, tag, variant="eara_scp", seed=seed,
            checkpoint=checkpoint,
        )
    )
    np.savez(output_path, **save_data)
    log.info("EARA-SCP results saved to: %s", output_path)


def run_adaptive_conformal(
    model: torch.nn.Module,
    model_name: str,
    subset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    device: torch.device,
    log: logging.Logger,
    tag: str = "",
    seed: int = DEFAULT_SEED,
    checkpoint: dict | None = None,
) -> None:
    """Run normalized conformal prediction with a frozen-GRU scale head."""
    log.info("Extracting frozen GRU features for adaptive conformal...")
    train_features = extract_model_features(model, X_train, device)
    calibration_features = extract_model_features(model, X_calibration, device)
    train_predictions = predict_rul(model, X_train, device)
    log.info("Training residual-scale head on training engines only...")
    head = train_scale_head(
        train_features,
        np.abs(y_train - train_predictions),
        device,
        seed=seed,
    )

    calibration_predictions = predict_rul(model, X_calibration, device)
    calibration_scales = predict_scales(head, calibration_features, device)

    X_test, y_test, _ = load_test_data(subset)
    test_features = extract_model_features(model, X_test, device)
    test_predictions = predict_rul(model, X_test, device)
    test_scales = predict_scales(head, test_features, device)
    save_data: dict[str, np.ndarray] = {
        "y_rul_test": y_test,
        "y_rul_pred": test_predictions,
        "scales": test_scales,
    }
    for coverage in COVERAGE_LEVELS:
        q_hat = conformal_quantile(
            normalised_nonconformity(
                y_calibration,
                calibration_predictions,
                calibration_scales,
            ),
            1.0 - coverage,
        )
        lower, upper = scale_adaptive_interval(
            test_predictions,
            q_hat,
            test_scales,
        )
        metrics = evaluate_interval(y_test, lower, upper, coverage)
        suffix = int(coverage * 100)
        save_data[f"lower_{suffix}"] = lower
        save_data[f"upper_{suffix}"] = upper
        save_data[f"q_hat_{suffix}"] = np.asarray([q_hat])
        log.info(
            "Adaptive %.0f%% | Empirical %.4f | Error %.4f | MPIW %.4f",
            coverage * 100.0,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(
        subset,
        model_name,
        tag,
        suffix="adaptive_conformal",
    )
    save_data["meta_json"] = pack_provenance(
        build_artifact_provenance(
            model_name,
            subset,
            tag,
            variant="adaptive_conformal",
            seed=seed,
            checkpoint=checkpoint,
        )
    )
    np.savez(output_path, **save_data)
    log.info("Adaptive results saved to: %s", output_path)


def run_cqr(
    model: torch.nn.Module,
    model_name: str,
    subset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    device: torch.device,
    log: logging.Logger,
    tag: str = "",
    seed: int = DEFAULT_SEED,
    checkpoint: dict | None = None,
) -> None:
    """Run CQR with quantile-head fitting and conformal calibration."""
    log.info("Extracting frozen model features for CQR...")
    train_features = extract_model_features(model, X_train, device)
    calibration_features = extract_model_features(model, X_calibration, device)
    X_test, y_test, _ = load_test_data(subset)
    test_features = extract_model_features(model, X_test, device)
    # The point predictions of the *same* checkpoint anchor the quantile
    # head (centre = point prediction + learned correction) and are stored
    # so a downstream consumer can pair the interval with its prediction.
    train_pred = predict_rul(model, X_train, device)
    calibration_pred = predict_rul(model, X_calibration, device)
    test_pred = predict_rul(model, X_test, device)
    save_data: dict[str, np.ndarray] = {"y_rul_test": y_test}

    for coverage in COVERAGE_LEVELS:
        alpha = 1.0 - coverage
        lower_quantile = alpha / 2.0
        upper_quantile = 1.0 - lower_quantile
        log.info(
            "Training CQR quantile head for %.0f%% coverage...",
            coverage * 100.0,
        )
        head = train_quantile_head(
            train_features,
            y_train,
            device,
            lower_quantile,
            upper_quantile,
            seed=seed,
            base_pred=train_pred,
        )
        cal_lower, cal_upper = predict_quantiles(
            head, calibration_features, device, base_pred=calibration_pred
        )
        calibration_scores = np.maximum(
            cal_lower - y_calibration,
            y_calibration - cal_upper,
        )
        test_lower, test_upper = predict_quantiles(
            head, test_features, device, base_pred=test_pred
        )
        q_hat = cqr_quantile(calibration_scores, alpha)
        lower, upper = test_lower - q_hat, test_upper + q_hat
        metrics = evaluate_interval(y_test, lower, upper, coverage)
        suffix = int(coverage * 100)
        save_data[f"lower_{suffix}"] = lower
        save_data[f"upper_{suffix}"] = upper
        save_data[f"q_hat_{suffix}"] = np.asarray([q_hat])
        save_data[f"quantile_lower_{suffix}"] = test_lower
        save_data[f"quantile_upper_{suffix}"] = test_upper
        # The point prediction of the *same* checkpoint, so a downstream
        # consumer can pair the interval with the prediction it belongs to.
        save_data["y_rul_pred"] = test_pred
        log.info(
            "CQR %.0f%% | Empirical %.4f | Error %.4f | MPIW %.4f",
            coverage * 100.0,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(
        subset,
        model_name,
        tag,
        suffix="cqr",
    )
    save_data["meta_json"] = pack_provenance(
        build_artifact_provenance(
            model_name,
            subset,
            tag,
            variant="cqr",
            seed=seed,
            checkpoint=checkpoint,
        )
    )
    np.savez(output_path, **save_data)
    log.info("CQR results saved to: %s", output_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    args = parse_args()

    subset = args.subset
    model_name = args.model
    tag = args.tag

    log = configure_logging(
        model_name,
        subset,
        args.method,
        tag,
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
        "Method: %s",
        args.method,
    )

    log.info(
        "Model: %s",
        model_name,
    )

    if tag:
        log.info(
            "Tag: %s",
            tag,
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
    # Load trained model
    #
    # The model is loaded BEFORE the calibration split is created,
    # because the split has to reproduce the split the checkpoint was
    # trained with.  That seed lives in the checkpoint.
    # --------------------------------------------------------------

    log.info(
        "Loading trained %s...",
        model_name.upper(),
    )

    model, checkpoint = load_model(
        model_name=model_name,
        input_size=X_train_all.shape[-1],
        device=device,
        subset=subset,
        tag=tag,
    )

    seed = checkpoint_seed(checkpoint)

    log.info(
        "Loaded checkpoint from epoch %d",
        checkpoint["epoch"],
    )

    log.info(
        "Checkpoint seed: %d (calibration split reproduces this split)",
        seed,
    )

    # --------------------------------------------------------------
    # Reproduce calibration split
    # --------------------------------------------------------------

    log.info(
        "Creating engine-wise calibration split (seed=%d)...",
        seed,
    )

    (
        train_indices,
        calibration_indices,
    ) = create_calibration_split(
        X=X_train_all,
        y_rul=y_rul_all,
        engine_ids=engine_ids,
        seed=seed,
    )

    X_calibration = X_train_all[
        calibration_indices
    ]

    y_rul_calibration = y_rul_all[
        calibration_indices
    ]

    y_hi_calibration = y_hi_all[
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
    # Method dispatch
    # --------------------------------------------------------------
    # IMPORTANT: method-specific implementations must return here.
    # Otherwise every method falls through to the original global split
    # conformal path and produces identical results.
    X_train = X_train_all[train_indices]
    y_rul_train = y_rul_all[train_indices]
    y_hi_train = y_hi_all[train_indices]
    training_engine_ids = engine_ids[train_indices]
    calibration_engine_ids = engine_ids[calibration_indices]

    if args.method == "adaptive":
        run_adaptive_conformal(
            model=model,
            model_name=model_name,
            subset=subset,
            X_train=X_train,
            y_train=y_rul_train,
            X_calibration=X_calibration,
            y_calibration=y_rul_calibration,
            device=device,
            log=log,
            tag=tag,
            seed=seed,
            checkpoint=checkpoint,
        )
        return

    if args.method == "cqr":
        run_cqr(
            model=model,
            model_name=model_name,
            subset=subset,
            X_train=X_train,
            y_train=y_rul_train,
            X_calibration=X_calibration,
            y_calibration=y_rul_calibration,
            device=device,
            log=log,
            tag=tag,
            seed=seed,
            checkpoint=checkpoint,
        )
        return

    if args.method == "engine_joint":
        run_engine_joint_conformal(
            model=model,
            model_name=model_name,
            subset=subset,
            X_train=X_train,
            y_rul_train=y_rul_train,
            y_hi_train=y_hi_train,
            X_calibration=X_calibration,
            y_rul_calibration=y_rul_calibration,
            y_hi_calibration=y_hi_calibration,
            calibration_engine_ids=calibration_engine_ids,
            training_engine_ids=training_engine_ids,
            device=device,
            log=log,
            tag=tag,
            seed=seed,
            checkpoint=checkpoint,
        )
        return

    if args.method == "eara_scp":
        run_eara_scp(
            model=model,
            model_name=model_name,
            subset=subset,
            X_train=X_train,
            y_rul_train=y_rul_train,
            X_calibration=X_calibration,
            y_rul_calibration=y_rul_calibration,
            calibration_engine_ids=calibration_engine_ids,
            training_engine_ids=training_engine_ids,
            device=device,
            log=log,
            tag=tag,
            seed=seed,
            checkpoint=checkpoint,
        )
        return

    # --------------------------------------------------------------
    # Global split conformal
    # --------------------------------------------------------------
    log.info("Generating calibration predictions...")
    calibration_predictions = predict_rul(
        model=model,
        X=X_calibration,
        device=device,
    )

    log.info("Calibrating global split-conformal intervals...")
    calibration_results = calibrate_multiple_levels(
        y_true=y_rul_calibration,
        y_pred=calibration_predictions,
        coverage_levels=COVERAGE_LEVELS,
    )

    for coverage, result in calibration_results.items():
        log.info(
            "Nominal %.0f%% | alpha %.2f | q_hat %.6f",
            coverage * 100.0,
            result["alpha"],
            result["q_hat"],
        )

    # --------------------------------------------------------------
    # Load final test set
    # --------------------------------------------------------------
    log.info("Loading final test data...")
    (
        X_test,
        y_rul_test,
        y_hi_test,
    ) = load_test_data(subset)

    log.info("Test samples: %d", len(X_test))

    # --------------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------------
    log.info("Generating test predictions...")
    test_predictions = predict_rul(
        model=model,
        X=X_test,
        device=device,
    )

    # --------------------------------------------------------------
    # Evaluate intervals
    # --------------------------------------------------------------
    results = {}
    save_data = {
        "y_rul_test": y_rul_test,
        "y_rul_pred": test_predictions,
    }

    for coverage, calibration in calibration_results.items():
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

        suffix = int(coverage * 100)
        save_data[f"lower_{suffix}"] = lower
        save_data[f"upper_{suffix}"] = upper
        save_data[f"q_hat_{suffix}"] = np.asarray([q_hat])

        log.info(
            "Coverage %.0f%% | Empirical %.4f | Error %.4f | MPIW %.4f",
            coverage * 100.0,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = results_path(
        subset,
        model_name,
        tag,
        suffix="conformal",
    )
    save_data["meta_json"] = pack_provenance(
        build_artifact_provenance(
            model_name,
            subset,
            tag,
            variant="conformal",
            seed=seed,
            checkpoint=checkpoint,
        )
    )
    np.savez(output_path, **save_data)
    log.info("Results saved to: %s", output_path)

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
