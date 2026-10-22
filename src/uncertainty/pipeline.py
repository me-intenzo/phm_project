"""Model-agnostic conformal interval computations.

This module holds the interval math shared by the production pipeline
(``scripts/uncertainty.py``) and the multi-seed benchmark
(``scripts/multiseed_conformal_benchmark.py``).  Keeping it in one place is
what makes the benchmark meaningful: the benchmark measures **the same code
path**, on **the same trained checkpoints**, that the single-run pipeline
uses.  There is no proxy/surrogate model anywhere in this module — every
interval is produced by the checkpoint the repository actually trained.

Calibration protocol
--------------------
The calibration unit is the **engine**.

For every held-out calibration engine the window residuals are reduced to a
single nonconformity score (the maximum over that engine's windows).  The
conformal sample size is therefore the number of held-out engines, not the
number of windows.  This matters for three reasons:

1. it matches the test unit (one final window per test engine);
2. it keeps the finite-sample rank ``k = ceil((n + 1)(1 - alpha))`` attainable
   at the 95 % level with the 20 % engine hold-out used by the trainer; and
3. it removes the window-count dependence that lets a window-level
   max-residual quantile collapse to an artificially tight interval.

The split is the *training* split of the checkpoint being evaluated, so the
calibration engines are exactly the engines the model never saw.  The seed
is read from the checkpoint, so a seed-43 checkpoint is calibrated against
its own hold-out and never against another seed's engines.

author: me-intenzo
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from src.models.outputs import unpack_model_output
from src.models.uncertainty_heads import QuantileHead, ScaleHead

from .conformal import (
    SCALE_FLOOR,
    clip_interval,
    conformal_quantile,
    conformal_rank,
    floor_scale,
    signed_conformal_quantile,
)
from .coverage import (
    empirical_coverage,
    mean_prediction_interval_width,
    winkler_score,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: Shared scale floor.  It is imported from :mod:`src.uncertainty.conformal`
#: so that this module cannot drift away from the calibration path: the
#: *same* constant and the *same* bounding operator (``max``) are applied
#: when a predicted scale is normalised into a calibration score and when
#: it is multiplied into an interval radius.

DEFAULT_BATCH_SIZE: int = 64

COVERAGE_LEVELS: tuple[float, ...] = (0.80, 0.90, 0.95)

METHODS: tuple[str, ...] = (
    "split_conformal",
    "cqr",
    "adaptive",
    "eara_conformal",
    "nex_scp",
    "eara_scp",
)


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

@torch.no_grad()
def predict_rul(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """RUL predictions for a batch of windows."""

    predictions = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        outputs = model(batch)
        pred_rul, _, _ = unpack_model_output(outputs)
        predictions.append(pred_rul.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions) if predictions else np.empty(0)


@torch.no_grad()
def predict_rul_hi(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """RUL and HI predictions for a batch of windows."""

    rul_predictions, hi_predictions = [], []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        outputs = model(batch)
        pred_rul, pred_hi, _ = unpack_model_output(outputs)
        if pred_hi is None:
            pred_hi = torch.zeros_like(pred_rul)
        rul_predictions.append(pred_rul.detach().cpu().numpy().reshape(-1))
        hi_predictions.append(pred_hi.detach().cpu().numpy().reshape(-1))
    return np.concatenate(rul_predictions), np.concatenate(hi_predictions)


@torch.no_grad()
def predict_rul_scale(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    scale_floor: float = SCALE_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """RUL predictions together with the model's own scale head output."""

    rul_predictions, scale_predictions = [], []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        outputs = model(batch)
        pred_rul, _, pred_scale = unpack_model_output(outputs)
        if pred_scale is None:
            pred_scale = torch.ones_like(pred_rul)
        rul_predictions.append(pred_rul.detach().cpu().numpy().reshape(-1))
        scale_predictions.append(pred_scale.detach().cpu().numpy().reshape(-1))

    rul = np.concatenate(rul_predictions) if rul_predictions else np.empty(0)
    scale = np.concatenate(scale_predictions) if scale_predictions else np.empty(0)
    return rul, floor_scale(scale, scale_floor)


@torch.no_grad()
def extract_model_features(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    """Capture the input to the architecture's RUL head via a safe hook."""

    features: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture_head_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(inputs[0].detach().cpu())

    if not hasattr(model, "rul_head"):
        raise AttributeError(
            "Model does not expose a 'rul_head'; frozen-feature calibration "
            "(CQR / adaptive) is not available for this architecture."
        )

    handle = model.rul_head.register_forward_pre_hook(capture_head_input)
    model.eval()
    try:
        for start in range(0, len(X), batch_size):
            captured.clear()
            batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
            model(batch)
            if len(captured) != 1:
                raise RuntimeError("Could not capture one RUL-head feature batch.")
            features.append(captured[0])
    finally:
        handle.remove()

    if not features:
        return torch.empty((0, 0))
    return torch.cat(features, dim=0)


# ------------------------------------------------------------------
# Auxiliary heads (fitted on training engines only)
# ------------------------------------------------------------------

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


def fit_quantile_head(
    features: torch.Tensor,
    y_true: np.ndarray,
    device: torch.device,
    lower_quantile: float,
    upper_quantile: float,
    epochs: int = 50,
    seed: int = 42,
    batch_size: int = 2048,
    base_pred: np.ndarray | None = None,
) -> QuantileHead:
    """Fit a quantile head using only training-engine features.

    ``base_pred`` is the model's own point prediction for the same windows.
    It anchors the head's centre so the linear correction starts at the
    already-accurate point forecast instead of at zero.
    """

    torch.manual_seed(int(seed))

    head = QuantileHead(int(features.shape[-1])).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    targets = torch.from_numpy(
        np.asarray(y_true, dtype=np.float32)
    ).reshape(-1).to(device)
    offsets = None
    if base_pred is not None:
        offsets = torch.from_numpy(
            np.asarray(base_pred, dtype=np.float32)
        ).reshape(-1).to(device)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    head.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            indices = permutation[start:start + batch_size]
            x_batch = features[indices].to(device)
            y_batch = targets[indices]
            offset_batch = offsets[indices] if offsets is not None else None
            lower, upper = head(x_batch, offset_batch)
            loss = (
                pinball_loss(lower, y_batch, lower_quantile)
                + pinball_loss(upper, y_batch, upper_quantile)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


@torch.no_grad()
def predict_quantiles(
    head: QuantileHead,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
    base_pred: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lowers, uppers = [], []
    for start in range(0, len(features), batch_size):
        stop = start + batch_size
        offset = None
        if base_pred is not None:
            offset = torch.from_numpy(
                np.asarray(base_pred[start:stop], dtype=np.float32)
            ).to(device)
        lower, upper = head(features[start:stop].to(device), offset)
        lowers.append(lower.detach().cpu().numpy().reshape(-1))
        uppers.append(upper.detach().cpu().numpy().reshape(-1))
    return np.concatenate(lowers), np.concatenate(uppers)


def fit_scale_head(
    features: torch.Tensor,
    residuals: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    seed: int = 42,
    batch_size: int = 2048,
) -> ScaleHead:
    """Fit a positive residual-scale model on training engines only."""

    torch.manual_seed(int(seed))

    head = ScaleHead(int(features.shape[-1])).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    targets = torch.from_numpy(
        np.asarray(residuals, dtype=np.float32)
    ).reshape(-1).to(device)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    head.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            indices = permutation[start:start + batch_size]
            batch_features = features[indices].to(device)
            batch_targets = targets[indices]
            prediction = head(batch_features).squeeze()
            if prediction.dim() == 0:
                prediction = prediction.unsqueeze(0)
            loss = torch.nn.functional.l1_loss(
                prediction.view(-1), batch_targets.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


@torch.no_grad()
def predict_scales(
    head: ScaleHead,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    scales = []
    head.eval()
    for start in range(0, len(features), batch_size):
        scales.append(
            head(features[start:start + batch_size].to(device))
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return np.concatenate(scales)


# ------------------------------------------------------------------
# Engine-level calibration helpers
# ------------------------------------------------------------------

def engine_max(
    scores: np.ndarray,
    engine_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce window scores to one maximum score per engine."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    engine_ids = np.asarray(engine_ids).reshape(-1)
    if scores.shape != engine_ids.shape:
        raise ValueError("scores and engine_ids must have the same length.")

    ids = np.unique(engine_ids)
    maxima = np.asarray(
        [float(np.max(scores[engine_ids == engine])) for engine in ids],
        dtype=np.float64,
    )
    return ids, maxima


CALIBRATION_UNITS = ("window", "engine")


def reduce_calibration_scores(
    scores: np.ndarray,
    engine_ids: np.ndarray,
    unit: str,
) -> np.ndarray:
    """Reduce window-level scores to the calibration unit that will be used.

    ``"window"`` keeps every score (classic split conformal: the calibration
    unit is the window, which is also the unit the test set is scored on).
    ``"engine"`` keeps one score per engine — its maximum.  The max is
    deliberately conservative: an engine with hundreds of calibration windows
    contributes its worst one, so the resulting interval is wider than a
    window-level interval at the same nominal coverage.
    """

    if unit not in CALIBRATION_UNITS:
        raise ValueError(
            f"Unknown calibration unit {unit!r}; expected one of {CALIBRATION_UNITS}."
        )

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if unit == "window":
        return scores

    _, maxima = engine_max(scores, engine_ids)
    return maxima


def conformal_rank_attainable(n: int, alpha: float) -> tuple[int, bool]:
    """Return the finite-sample conformal rank and whether it is attainable.

    ``k = ceil((n + 1)(1 - alpha))``.  When ``k > n`` the mathematically
    correct quantile is ``+inf`` — the calibration set is too small to
    certify the requested coverage.
    """

    k = conformal_rank(n, alpha)
    return k, k <= n


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile; ``inf`` when the rank is unattainable."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Cannot calibrate with no engine scores.")
    return conformal_quantile(scores, alpha)


def signed_finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile for **signed** scores (CQR).

    CQR nonconformity scores are negative whenever the true value lies
    inside the predicted quantile interval, so the non-negativity check of
    :func:`conformal_quantile` must not be applied to them.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Cannot calibrate with no engine scores.")
    return signed_conformal_quantile(scores, alpha)


# ------------------------------------------------------------------
# Interval metrics
# ------------------------------------------------------------------

def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    """Coverage, width and Winkler score for one interval set.

    The metrics themselves are computed by
    :mod:`src.uncertainty.coverage` — the single implementation of the
    interval metrics.  This function only maps them onto the key names
    the benchmark tables use.
    """

    alpha = 1.0 - coverage

    empirical = empirical_coverage(y_true, lower, upper)

    return {
        "nominal_coverage": float(coverage),
        "empirical_coverage": empirical,
        "coverage_gap": empirical - float(coverage),
        "mean_interval_width": mean_prediction_interval_width(lower, upper),
        "winkler_score": winkler_score(y_true, lower, upper, alpha),
    }


def clip_to_rul_domain(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip intervals to the physical RUL range, keeping them ordered.

    Delegates to the single interval builder of
    :mod:`src.uncertainty.conformal` so that every module in the project
    clips and orders an interval the same way.
    """

    return clip_interval(lower, upper)


# ------------------------------------------------------------------
# EARA-Conformal: engine-aware, regime-adaptive scale
# ------------------------------------------------------------------

def _robust_scale(values: np.ndarray, floor: float = SCALE_FLOOR) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return floor
    return max(float(np.median(np.abs(values))), floor)


def build_training_engine_profiles(
    features: torch.Tensor,
    residuals: np.ndarray,
    engine_ids: np.ndarray,
    predicted_rul: np.ndarray,
    n_regimes: int = 3,
) -> dict[str, Any]:
    """Build engine/regime residual profiles from **training engines only**.

    The scale function is frozen before any calibration score is computed,
    which preserves the split-conformal separation between calibration and
    test data.
    """

    feature_np = features.detach().cpu().numpy().astype(np.float64)
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    predicted_rul = np.asarray(predicted_rul, dtype=np.float64).reshape(-1)
    engine_ids = np.asarray(engine_ids).reshape(-1)

    center = np.median(feature_np, axis=0)
    mad = np.median(np.abs(feature_np - center), axis=0)
    mad[mad < 1e-8] = 1.0
    z = (feature_np - center) / mad

    # Degradation regimes are learned from the *training* predictions only.
    # np.digitize with these edges returns values in [0, n_regimes].  Index 0
    # (below the first edge) and n_regimes (above the last edge) are both valid
    # regime labels; regime_scales is keyed 0..n_regimes so all are covered.
    edges = np.quantile(predicted_rul, np.linspace(0.0, 1.0, n_regimes + 1)[1:-1])
    regimes = np.digitize(predicted_rul, edges, right=False)  # values in [0, n_regimes]

    profiles = []
    for engine in np.unique(engine_ids):
        mask = engine_ids == engine
        if not np.any(mask):
            continue

        engine_residuals = np.abs(residuals[mask])
        engine_z = z[mask]
        regime_scales: dict[int, float] = {}

        # Iterate over all possible regime labels: 0..n_regimes (inclusive).
        # np.digitize returns n_regimes for values above the last edge, so
        # that label must be stored or it silently falls back to global_scale.
        for regime in range(n_regimes + 1):
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
        raise RuntimeError(
            "EARA-Conformal could not build training-engine profiles."
        )

    return {
        "profiles": profiles,
        "center": center,
        "mad": mad,
        "regime_edges": np.asarray(edges, dtype=np.float64),
        "n_regimes": n_regimes,
    }


def nearest_engine_regime_scale(
    feature: np.ndarray,
    predicted_rul: float,
    profile_data: dict[str, Any],
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

    weights = 1.0 / (d + 1e-6)
    weights = weights / weights.sum()

    # np.digitize returns values in [0, n_regimes]; all are valid keys in
    # regime_scales (built with the same range in build_training_engine_profiles).
    regime = int(
        np.digitize(predicted_rul, profile_data["regime_edges"], right=False)
    )

    scales = []
    for idx in indices:
        profile = profiles[idx]
        scales.append(
            profile["regime_scales"].get(regime, profile["global_scale"])
        )

    scale = float(np.sum(weights * np.asarray(scales)))
    return (
        max(scale, SCALE_FLOOR),
        [profiles[i]["engine_id"] for i in indices],
        weights,
        regime,
    )


# ------------------------------------------------------------------
# Prequential adaptive conformal inference (nex-SCP)
# ------------------------------------------------------------------

def prequential_aci_intervals(
    scores: np.ndarray,
    y_pred: np.ndarray,
    scales: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    gamma: float,
    scale_floor: float = SCALE_FLOOR,
    alpha_floor: float = 0.01,
    engine_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Prequential adaptive conformal inference (Gibbs & Candes, 2021).

    The miscoverage level adapts online from the realised coverage::

        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), floor, 1 - floor)

    The test units are consumed in stored order, so each engine's outcome
    updates the level used for the next engine.  No test label is used
    before its own interval has been produced.

    ``alpha`` is kept strictly inside ``(0, 1)``: the unclipped update can
    reach exactly 0 after a run of misses, and a 0 miscoverage level is not a
    valid conformal level.  The 0.01 floor means the adaptor never targets
    more than 99 % coverage, which keeps the quantile finite for a window-level
    calibration set.  For the engine-level unit the floor is sometimes below
    the attainable rank, in which case the interval correctly degenerates to
    the full RUL domain instead of crashing.

    ``engine_ids`` is optional.  When supplied the alpha update fires once per
    **engine** (after the last window of that engine) rather than once per
    window.  On C-MAPSS the test set has one window per engine so the two
    behaviours are identical; the parameter exists so the function is correct
    for multi-window test sets without changing the C-MAPSS results.
    """

    if not 0.0 < alpha_floor < 0.5:
        raise ValueError("alpha_floor must be in (0, 0.5).")

    scores = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    if scores.size == 0:
        raise ValueError("Cannot calibrate with no calibration scores.")

    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if not (y_pred.shape == scales.shape == y_true.shape):
        raise ValueError("Predictions, scales, and targets must align.")

    lower = np.empty_like(y_pred)
    upper = np.empty_like(y_pred)
    alpha_stream: list[float] = []
    alpha_t = float(np.clip(alpha, alpha_floor, 1.0 - alpha_floor))

    # Determine the last window index for each engine so the alpha update
    # fires once per engine rather than once per window when engine_ids are
    # supplied.  On C-MAPSS the test set has one window per engine, so the
    # two paths produce identical results; the engine-aware path is correct
    # for multi-window test sets.
    if engine_ids is not None:
        engine_ids_arr = np.asarray(engine_ids).reshape(-1)
        if engine_ids_arr.shape != y_pred.shape:
            raise ValueError("engine_ids must have the same length as y_pred.")
        # last index of each engine in stored order
        last_window: set[int] = set()
        seen: dict = {}
        for i, eid in enumerate(engine_ids_arr):
            seen[int(eid)] = i
        last_window = set(seen.values())
    else:
        # update after every window (original behaviour; correct when the
        # test set has exactly one window per engine)
        last_window = set(range(len(y_pred)))

    for index in range(len(y_pred)):
        quantile = finite_sample_quantile(scores, alpha_t)
        radius = quantile * max(float(scales[index]), scale_floor)
        lo, hi = clip_interval(
            np.asarray([y_pred[index] - radius]),
            np.asarray([y_pred[index] + radius]),
        )
        lower[index] = lo[0]
        upper[index] = hi[0]
        alpha_stream.append(alpha_t)

        # Update alpha only at the last window of each engine.
        if index in last_window:
            lo_val = float(lo[0])
            hi_val = float(hi[0])
            error = 0.0 if lo_val <= float(y_true[index]) <= hi_val else 1.0
            alpha_t = float(
                np.clip(
                    alpha_t + gamma * (alpha - error),
                    alpha_floor,
                    1.0 - alpha_floor,
                )
            )

    return lower, upper, alpha_stream


# ------------------------------------------------------------------
# Method implementations
# ------------------------------------------------------------------

def _build_result(
    method: str,
    coverage: float,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_hat: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lower, upper = clip_to_rul_domain(lower, upper)
    result: dict[str, Any] = {
        "method": method,
        "coverage_level": float(coverage),
        "q_hat": float(q_hat),
        "lower": lower,
        "upper": upper,
    }
    result.update(interval_metrics(y_true, lower, upper, coverage))
    if extra:
        result.update(extra)
    return result


def compute_all_intervals(
    model: nn.Module,
    y_true_train: np.ndarray,
    y_true_calibration: np.ndarray,
    y_true_test: np.ndarray,
    engine_ids_train: np.ndarray,
    engine_ids_calibration: np.ndarray,
    X_train: np.ndarray,
    X_calibration: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    seed: int = 42,
    coverage_levels: Iterable[float] = COVERAGE_LEVELS,
    n_regimes: int = 3,
    top_k: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    aci_gamma: float = 0.02,
    scale_floor: float = SCALE_FLOOR,
    quantile_head_epochs: int = 50,
    calibration_units: Iterable[str] = ("engine",),
) -> dict[str, dict[str, dict[float, dict[str, Any]]]]:
    """Compute every conformal interval method for one trained checkpoint.

    All methods see the same model, the same calibration engines and the
    same test engines, so the comparison between them is paired.

    ``calibration_units`` selects how the calibration window scores are
    reduced.  ``"engine"`` keeps one max-score per engine (conservative);
    ``"window"`` keeps one score per calibration window (the classic split
    conformal unit, and the unit the test set is scored on).  Both units can
    be requested in one call: the expensive work (inference, the quantile
    head, the EARA profiles) is shared, only the score reduction differs.

    Returns
    -------
    dict
        ``results[unit][method][coverage]`` → interval record with ``lower``,
        ``upper``, ``q_hat`` and the interval metrics.
    """

    coverage_levels = tuple(float(level) for level in coverage_levels)

    if isinstance(calibration_units, str):
        calibration_units = (calibration_units,)
    unit_list = tuple(dict.fromkeys(str(unit) for unit in calibration_units))
    for unit in unit_list:
        if unit not in CALIBRATION_UNITS:
            raise ValueError(
                f"Unknown calibration unit {unit!r}; expected one of "
                f"{CALIBRATION_UNITS}."
            )
    if not unit_list:
        raise ValueError("At least one calibration unit is required.")

    y_true_train = np.asarray(y_true_train, dtype=np.float64).reshape(-1)
    y_true_calibration = np.asarray(y_true_calibration, dtype=np.float64).reshape(-1)
    y_true_test = np.asarray(y_true_test, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Point predictions and model-native scale
    # ------------------------------------------------------------------

    train_pred = predict_rul(model, X_train, device, batch_size)
    cal_pred = predict_rul(model, X_calibration, device, batch_size)
    test_pred = predict_rul(model, X_test, device, batch_size)

    # ------------------------------------------------------------------
    # Frozen features for the auxiliary heads
    # ------------------------------------------------------------------

    train_features = extract_model_features(model, X_train, device, batch_size)
    cal_features = extract_model_features(model, X_calibration, device, batch_size)
    test_features = extract_model_features(model, X_test, device, batch_size)

    train_residuals = np.abs(y_true_train - train_pred)
    cal_residuals = np.abs(y_true_calibration - cal_pred)

    # ------------------------------------------------------------------
    # Adaptive residual-scale head (training engines only)
    # ------------------------------------------------------------------
    # This mirrors `run_adaptive_conformal` in scripts/uncertainty.py: the
    # scale is learned post hoc from training-engine residuals, not read from
    # the checkpoint's own scale head.  Using the checkpoint head here made
    # the "adaptive" baseline identical to split conformal whenever the
    # loader neutralised that head, which is exactly what used to happen.
    scale_head = fit_scale_head(
        train_features,
        train_residuals,
        device,
        epochs=quantile_head_epochs,
        seed=seed,
    )
    cal_scale = predict_scales(scale_head, cal_features, device, batch_size)
    test_scale = predict_scales(scale_head, test_features, device, batch_size)

    # ------------------------------------------------------------------
    # Window-level calibration scores (reduced to the unit below)
    # ------------------------------------------------------------------

    adaptive_raw = cal_residuals / floor_scale(cal_scale, scale_floor)

    # ------------------------------------------------------------------
    # EARA profiles (training engines only)
    # ------------------------------------------------------------------

    profile_data = build_training_engine_profiles(
        train_features,
        train_residuals,
        engine_ids_train,
        train_pred,
        n_regimes=n_regimes,
    )

    cal_scales = np.empty(len(X_calibration), dtype=np.float64)
    cal_regimes = np.empty(len(X_calibration), dtype=np.int64)
    for index, (feature, pred) in enumerate(
        zip(cal_features.detach().cpu().numpy(), cal_pred)
    ):
        scale, _, _, regime = nearest_engine_regime_scale(
            feature, float(pred), profile_data, top_k=top_k
        )
        cal_scales[index] = scale
        cal_regimes[index] = regime

    test_scales = np.empty(len(X_test), dtype=np.float64)
    test_regimes = np.empty(len(X_test), dtype=np.int64)
    for index, (feature, pred) in enumerate(
        zip(test_features.detach().cpu().numpy(), test_pred)
    ):
        scale, _, _, regime = nearest_engine_regime_scale(
            feature, float(pred), profile_data, top_k=top_k
        )
        test_scales[index] = scale
        test_regimes[index] = regime

    eara_raw = (
        np.abs(y_true_calibration - cal_pred)
        / floor_scale(cal_scales, scale_floor)
    )

    window_scores = {
        "split_conformal": cal_residuals,
        "adaptive": adaptive_raw,
        "eara_conformal": eara_raw,
    }
    unit_scores = {
        unit: {
            method: reduce_calibration_scores(values, engine_ids_calibration, unit)
            for method, values in window_scores.items()
        }
        for unit in unit_list
    }

    results: dict[str, dict[str, dict[float, dict[str, Any]]]] = {
        unit: {method: {} for method in METHODS} for unit in unit_list
    }

    for coverage in coverage_levels:
        alpha = 1.0 - coverage

        # ---------------- CQR quantile head ----------------
        # Fitted on training engines only, once per coverage level, because
        # the lower/upper quantile levels depend on alpha.  The head does not
        # depend on the calibration unit.
        lower_quantile = alpha / 2.0
        upper_quantile = 1.0 - lower_quantile

        head = fit_quantile_head(
            train_features,
            y_true_train,
            device,
            lower_quantile,
            upper_quantile,
            epochs=quantile_head_epochs,
            seed=seed,
            base_pred=train_pred,
        )

        cal_lower, cal_upper = predict_quantiles(
            head, cal_features, device, base_pred=cal_pred
        )
        # Standard CQR nonconformity score (Romano et al., 2019):
        #     E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
        # The score is SIGNED: it is negative when the true value falls
        # inside the predicted quantile interval, which is exactly what
        # allows the conformal correction to tighten the interval rather
        # than only widen it.
        cqr_nonconformity = np.maximum(
            cal_lower - y_true_calibration,
            y_true_calibration - cal_upper,
        )
        test_lower, test_upper = predict_quantiles(
            head, test_features, device, base_pred=test_pred
        )

        for unit in unit_list:
            scores = unit_scores[unit]
            split_scores = scores["split_conformal"]
            adaptive_scores = scores["adaptive"]
            eara_scores = scores["eara_conformal"]
            cqr_scores = reduce_calibration_scores(
                cqr_nonconformity, engine_ids_calibration, unit
            )
            unit_info = {
                "calibration_unit": unit,
                "calibration_units": int(len(split_scores)),
            }

            # ---------------- split conformal ----------------
            q_split = finite_sample_quantile(split_scores, alpha)
            results[unit]["split_conformal"][coverage] = _build_result(
                "split_conformal",
                coverage,
                y_true_test,
                test_pred - q_split,
                test_pred + q_split,
                q_split,
                dict(unit_info),
            )

            # ---------------- adaptive (scale-normalised) ----------------
            q_adaptive = finite_sample_quantile(adaptive_scores, alpha)
            radius = q_adaptive * floor_scale(test_scale, scale_floor)
            results[unit]["adaptive"][coverage] = _build_result(
                "adaptive",
                coverage,
                y_true_test,
                test_pred - radius,
                test_pred + radius,
                q_adaptive,
                dict(unit_info),
            )

            # ---------------- EARA-Conformal ----------------
            q_eara = finite_sample_quantile(eara_scores, alpha)
            radius = q_eara * floor_scale(test_scales, scale_floor)
            results[unit]["eara_conformal"][coverage] = _build_result(
                "eara_conformal",
                coverage,
                y_true_test,
                test_pred - radius,
                test_pred + radius,
                q_eara,
                dict(unit_info),
            )

            # ---------------- nex-SCP (prequential, generic scale head) ----------------
            # Uses the generic scale-head scale (test_scale, singular).
            nex_lower, nex_upper, alpha_stream = prequential_aci_intervals(
                adaptive_scores,
                test_pred,
                test_scale,          # generic scale head — distinct from test_scales (EARA)
                y_true_test,
                alpha,
                aci_gamma,
                scale_floor,
            )
            results[unit]["nex_scp"][coverage] = _build_result(
                "nex_scp",
                coverage,
                y_true_test,
                nex_lower,
                nex_upper,
                # q_hat for prequential methods is the median adapted alpha level,
                # not a radius quantile.  Stored under the same key for schema
                # consistency; the "final_alpha" key carries the terminal value.
                float(np.median(alpha_stream)) if alpha_stream else float(alpha),
                {
                    **unit_info,
                    "final_alpha": alpha_stream[-1] if alpha_stream else float(alpha),
                    "q_hat_is_median_alpha": True,
                },
            )

            # ---------------- EARA-SCP (prequential, EARA scale) ----------------
            # Novel combination: EARA's engine/regime-aware scale (test_scales,
            # plural — kNN over training-engine profiles) with NEX-SCP's online
            # miscoverage adaptation.  The scale is sharper than the generic head
            # because it is conditioned on the engine's degradation regime; the
            # level adapts from each realised test outcome so coverage tracks the
            # nominal target even when the scale estimates are imperfect.
            # Neither EARA-Conformal nor nex_scp is modified; this is a clean
            # composition reported alongside them.
            eara_scp_lower, eara_scp_upper, eara_scp_alpha_stream = (
                prequential_aci_intervals(
                    eara_scores,
                    test_pred,
                    test_scales,     # EARA kNN scale — distinct from test_scale (generic)
                    y_true_test,
                    alpha,
                    aci_gamma,
                    scale_floor,
                )
            )
            results[unit]["eara_scp"][coverage] = _build_result(
                "eara_scp",
                coverage,
                y_true_test,
                eara_scp_lower,
                eara_scp_upper,
                float(np.median(eara_scp_alpha_stream))
                if eara_scp_alpha_stream
                else float(alpha),
                {
                    **unit_info,
                    "final_alpha": (
                        eara_scp_alpha_stream[-1]
                        if eara_scp_alpha_stream
                        else float(alpha)
                    ),
                    "q_hat_is_median_alpha": True,
                },
            )

            # ---------------- CQR ----------------
            q_cqr = signed_finite_sample_quantile(cqr_scores, alpha)
            results[unit]["cqr"][coverage] = _build_result(
                "cqr",
                coverage,
                y_true_test,
                test_lower - q_cqr,
                test_upper + q_cqr,
                q_cqr,
                dict(unit_info),
            )

    # ------------------------------------------------------------------
    # Attach the point predictions of this exact checkpoint
    #
    # Downstream consumers (the decision engine, the HITL workflow) need
    # the point prediction that goes with the interval.  Attaching it
    # here guarantees that the interval and the point prediction come
    # from the same weights — there is no second model to disagree with.
    # ------------------------------------------------------------------

    for unit_results in results.values():
        for method_results in unit_results.values():
            for record in method_results.values():
                record["y_rul_pred"] = test_pred
                record["y_rul_true"] = y_true_test

    return results
