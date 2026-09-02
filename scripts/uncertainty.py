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

import numpy as np
import torch
import torch.nn as nn
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
from src.models.hybrid import CNNGRUTransformerPrognostics
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.models.uncertainty_heads import QuantileHead, ScaleHead

from src.uncertainty.calibration import (
    calibrate_multiple_levels,
)

from src.uncertainty.adaptive import (
    adaptive_quantile,
    prediction_interval as adaptive_prediction_interval,
)
from src.uncertainty.conformal import conformal_quantile, prediction_interval

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
            "for a trained prognostics model."
        )
    )

    parser.add_argument(
        "--model",
        choices=["lstm", "gru", "transformer", "hybrid"],
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
        choices=["split", "cqr", "adaptive", "engine_joint"],
        default="split",
        help=(
            "Uncertainty method: global split conformal ('split'), "
            "conformalized quantile regression ('cqr'), or normalized "
            "adaptive conformal ('adaptive'), or engine-block joint "
            "conformal ('engine_joint')."
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
                / f"uncertainty_{model_name}_{subset}_{method}.log",
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

def build_model(
    model_name: str,
    input_size: int,
 ) -> torch.nn.Module:
    """Reconstruct the architecture used by ``scripts/train.py``."""
    if model_name == "lstm":
        return LSTMPrognosticsModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    if model_name == "gru":
        return GRUPrognosticsModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    if model_name == "transformer":
        return TransformerPrognosticsModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 4, DROPOUT)
    if model_name == "hybrid":
        return CNNGRUTransformerPrognostics(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_gru_layers=NUM_LAYERS,
            num_transformer_layers=NUM_LAYERS,
            num_heads=4,
            dropout=DROPOUT,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_model(
    model_name: str,
    input_size: int,
    device: torch.device,
    subset: str = "",
):
    """
    Load a trained architecture-specific checkpoint.
    """

    checkpoint_path = (
        CHECKPOINT_DIR
        / model_name
        / subset
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        legacy_checkpoint_path = (
            CHECKPOINT_DIR
            / model_name
            / "best_model.pt"
        )
        if legacy_checkpoint_path.exists():
            checkpoint_path = legacy_checkpoint_path

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"{model_name.upper()} checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            f"Train the model first using:\n"
            f"python scripts/train.py --model {model_name} --subset {subset}"
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
            f"Re-train {model_name} for {subset} using:\n"
            f"python scripts/train.py --model {model_name} --subset {subset}"
        )

    model = build_model(
        model_name=model_name,
        input_size=input_size,
    )

    state_dict = checkpoint["model_state_dict"]

    # Adaptive conformal trains its own ScaleHead separately.
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("scale_head.")
    }

    model.load_state_dict(state_dict)

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

            pred = model(batch)

            if isinstance(pred, tuple) and len(pred) == 3:
                pred_rul, pred_hi, _ = pred
            else:
                pred_rul, _ = pred

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
            # Handle variable return counts from hybrid model
            outputs = model(batch)
            if isinstance(outputs, (tuple, list)):
                pred_rul, pred_hi = outputs[0], outputs[1]
            else:
                pred_rul = outputs
                pred_hi = torch.zeros_like(pred_rul)
            rul_predictions.append(pred_rul.cpu().numpy().reshape(-1))
            hi_predictions.append(pred_hi.cpu().numpy().reshape(-1))
    return np.concatenate(rul_predictions), np.concatenate(hi_predictions)


def _robust_scale(values: np.ndarray, floor: float = 1e-6) -> float:
    """Robust residual scale estimate."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return floor
    return max(float(np.median(np.abs(values))), floor)


def _build_training_engine_profiles(
    features: torch.Tensor,
    residuals: np.ndarray,
    engine_ids: np.ndarray,
    predicted_rul: np.ndarray,
) -> dict:
    """
    Build engine/regime profiles using TRAINING ENGINES ONLY.

    This is deliberately constructed before conformal calibration. Therefore
    the calibration scores and final test intervals are evaluated against a
    fixed, learned scale function, preserving the split-conformal separation.
    """
    feature_np = features.detach().cpu().numpy().astype(np.float64)
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    predicted_rul = np.asarray(predicted_rul, dtype=np.float64).reshape(-1)
    engine_ids = np.asarray(engine_ids)

    center = np.median(feature_np, axis=0)
    mad = np.median(np.abs(feature_np - center), axis=0)
    mad[mad < 1e-8] = 1.0
    z = (feature_np - center) / mad

    # Regimes are learned ONLY from training predictions.
    q1, q2 = np.quantile(predicted_rul, [1.0 / 3.0, 2.0 / 3.0])
    regimes = np.digitize(predicted_rul, [q1, q2], right=False)

    profiles = []
    for engine in np.unique(engine_ids):
        mask = engine_ids == engine
        if not np.any(mask):
            continue

        engine_residuals = np.abs(residuals[mask])
        engine_z = z[mask]
        regime_scales = {}

        for regime in range(3):
            values = engine_residuals[regimes[mask] == regime]
            if len(values) >= 8:
                regime_scales[regime] = _robust_scale(values)

        profiles.append(
            {
                "engine_id": int(engine),
                "centroid": np.mean(engine_z, axis=0),
                "global_scale": _robust_scale(engine_residuals),
                "regime_scales": regime_scales,
            }
        )

    if not profiles:
        raise RuntimeError("EARA-Conformal could not build training-engine profiles.")

    return {
        "profiles": profiles,
        "center": center,
        "mad": mad,
        "regime_edges": np.asarray([q1, q2], dtype=np.float64),
    }


def _nearest_engine_regime_scale(
    feature: np.ndarray,
    predicted_rul: float,
    profile_data: dict,
    top_k: int = 5,
) -> tuple[float, list[int], np.ndarray, int]:
    """Retrieve nearest training engines and produce a regime-aware scale."""
    center = profile_data["center"]
    mad = profile_data["mad"]
    q = (np.asarray(feature, dtype=np.float64) - center) / mad

    profiles = profile_data["profiles"]
    centroids = np.stack([p["centroid"] for p in profiles], axis=0)
    distances = np.linalg.norm(centroids - q[None, :], axis=1)

    k = min(top_k, len(profiles))
    indices = np.argsort(distances)[:k]
    d = distances[indices]

    # Soft nearest-engine weighting.
    weights = 1.0 / (d + 1e-6)
    weights /= weights.sum()

    regime = int(
        np.digitize(
            predicted_rul,
            profile_data["regime_edges"],
            right=False,
        )
    )

    scales = []
    for idx in indices:
        profile = profiles[idx]
        scales.append(
            profile["regime_scales"].get(
                regime,
                profile["global_scale"],
            )
        )

    scale = float(np.sum(weights * np.asarray(scales)))
    return max(scale, 1e-6), [profiles[i]["engine_id"] for i in indices], weights, regime


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

    normalized_scores = calibration_residuals / np.maximum(
        calibration_scales,
        1e-6,
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

        radius = q_hat * test_scales
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
    output_path = (
        RESULTS_DIR
        / f"{subset}_{model_name}_engine_joint_conformal.npz"
    )
    np.savez(output_path, **save_data)
    log.info("EARA-Conformal results saved to: %s", output_path)
    log.info(
        "EARA complete: intervals are regime- and engine-neighbor adaptive."
    )


def extract_model_features(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Extract the input to each architecture's RUL head via a safe hook."""
    features = []
    captured: list[torch.Tensor] = []

    def capture_head_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(inputs[0].detach().cpu())

    handle = model.rul_head.register_forward_pre_hook(capture_head_input)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(X), BATCH_SIZE):
                captured.clear()
                batch = torch.from_numpy(X[start:start + BATCH_SIZE]).float().to(device)
                model(batch)
                if len(captured) != 1:
                    raise RuntimeError("Could not capture one RUL-head feature batch.")
                features.append(captured[0])
    finally:
        handle.remove()
    return torch.cat(features, dim=0)


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(
        quantile * residual,
        (quantile - 1.0) * residual,
    ).mean()


def train_quantile_head(
    features: torch.Tensor,
    y_true: np.ndarray,
    device: torch.device,
    lower_quantile: float,
    upper_quantile: float,
    epochs: int = 50,
) -> QuantileHead:
    """Fit a quantile head using only the training-engine features."""
    torch.manual_seed(SEED)
    head = QuantileHead(HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    targets = torch.from_numpy(y_true).float()

    head.train()
    optimization_batch_size = 2048
    for _ in range(epochs):
        for start in torch.randperm(len(features)).split(optimization_batch_size):
            x_batch = features[start].to(device)
            y_batch = targets[start].to(device)
            lower, upper = head(x_batch)
            loss = (
                pinball_loss(lower, y_batch, lower_quantile)
                + pinball_loss(upper, y_batch, upper_quantile)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


def predict_quantiles(
    head: QuantileHead,
    features: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    lowers, uppers = [], []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            lower, upper = head(features[start:start + BATCH_SIZE].to(device))
            lowers.append(lower.cpu().numpy())
            uppers.append(upper.cpu().numpy())
    return np.concatenate(lowers), np.concatenate(uppers)


def cqr_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile for signed CQR scores."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("CQR calibration scores must be finite and non-empty.")
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    return float(np.sort(scores)[min(max(rank, 1), len(scores)) - 1])


def train_scale_head(
    features: torch.Tensor,
    residuals: np.ndarray,
    device: torch.device,
    epochs: int = 50,
) -> ScaleHead:
    """Fit a positive residual-scale model on training engines only."""
    torch.manual_seed(SEED)
    head = ScaleHead(HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    # Ensure targets are a tensor and correctly shaped
    targets = torch.from_numpy(np.asarray(residuals, dtype=np.float32)).to(device)
    optimization_batch_size = 2048

    # Ensure targets are a tensor and flattened for easy indexing
    targets = torch.from_numpy(np.asarray(residuals, dtype=np.float32)).flatten().to(device)
    optimization_batch_size = 2048

    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(features))
        # Use range based indexing to correctly slice both features and targets
        for start_idx in range(0, len(features), optimization_batch_size):
            end_idx = min(start_idx + optimization_batch_size, len(features))
            indices = perm[start_idx : end_idx]
            
            batch_features = features[indices].to(device)
            batch_targets = targets[indices]
            
            # Ensure batch_targets is at least 1D (if batch_size=1)
            if batch_targets.dim() == 0:
                batch_targets = batch_targets.unsqueeze(0)
            
            prediction = head(batch_features).squeeze()
            
            # Ensure prediction has at least one dimension for safety (e.g. if batch_size is 1)
            if prediction.dim() == 0:
                prediction = prediction.unsqueeze(0)
            
            # Ensure shapes are consistent (N,)
            loss = nn.functional.l1_loss(prediction.view(-1), batch_targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


def predict_scales(
    head: ScaleHead,
    features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    scales = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            batch = features[start:start + BATCH_SIZE].to(device)
            scales.append(head(batch).cpu().numpy().flatten())
    return np.concatenate(scales)


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
) -> None:
    """Run normalized conformal prediction with a frozen-GRU scale head."""
    log.info("Extracting frozen GRU features for adaptive conformal...")
    train_features = extract_model_features(model, X_train, device)
    calibration_features = extract_model_features(model, X_calibration, device)
    train_predictions = predict_rul(model, X_train, device)
    log.info("Training residual-scale head on training engines only...")
    head = train_scale_head(train_features, np.abs(y_train - train_predictions), device)

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
        q_hat = adaptive_quantile(
            y_calibration,
            calibration_predictions,
            calibration_scales,
            1.0 - coverage,
        )
        lower, upper = adaptive_prediction_interval(test_predictions, q_hat, test_scales)
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
    output_path = RESULTS_DIR / f"{subset}_{model_name}_adaptive_conformal.npz"
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
) -> None:
    """Run CQR with quantile-head fitting and conformal calibration."""
    log.info("Extracting frozen model features for CQR...")
    train_features = extract_model_features(model, X_train, device)
    calibration_features = extract_model_features(model, X_calibration, device)
    X_test, y_test, _ = load_test_data(subset)
    test_features = extract_model_features(model, X_test, device)
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
        )
        cal_lower, cal_upper = predict_quantiles(head, calibration_features, device)
        calibration_scores = np.maximum(
            cal_lower - y_calibration,
            y_calibration - cal_upper,
        )
        test_lower, test_upper = predict_quantiles(head, test_features, device)
        q_hat = cqr_quantile(calibration_scores, alpha)
        lower, upper = test_lower - q_hat, test_upper + q_hat
        metrics = evaluate_interval(y_test, lower, upper, coverage)
        suffix = int(coverage * 100)
        save_data[f"lower_{suffix}"] = lower
        save_data[f"upper_{suffix}"] = upper
        save_data[f"q_hat_{suffix}"] = np.asarray([q_hat])
        save_data[f"quantile_lower_{suffix}"] = test_lower
        save_data[f"quantile_upper_{suffix}"] = test_upper
        log.info(
            "CQR %.0f%% | Empirical %.4f | Error %.4f | MPIW %.4f",
            coverage * 100.0,
            metrics["empirical_coverage"],
            metrics["coverage_error"],
            metrics["mean_interval_width"],
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{subset}_{model_name}_cqr.npz"
    np.savez(output_path, **save_data)
    log.info("CQR results saved to: %s", output_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    args = parse_args()

    subset = args.subset
    model_name = args.model

    log = configure_logging(
        model_name,
        subset
        , args.method
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
    # Load trained model
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
    )

    log.info(
        "Loaded checkpoint from epoch %d",
        checkpoint["epoch"],
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
    output_path = RESULTS_DIR / f"{subset}_{model_name}_conformal.npz"
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
