"""Configuration helpers.

Every pipeline script reads its knobs from a YAML file under ``configs/``
(``configs/fd001.yaml`` … ``configs/fd004.yaml``).  :func:`load_config`
merges that file over :data:`DEFAULT_CONFIG` so a script can always rely on
every key being present, and :func:`config_path_for` resolves the file that
belongs to a dataset subset.

The defaults encode the *tuned* recipe (see ``docs/RMSE_TARGET_PLAN.md``):
LayerNorm + multi-head attention, cosine schedule with warm-up, gradient
clipping, weight EMA and window augmentation.  A subset may override any of
them in its own YAML file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from src.utils.paths import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

#: The maintenance actions the decision engine may emit.  The list lives
#: here, in the lowest layer of the package, so that the decision engine
#: (``src/decision_engine/constraints.py``) and the HITL feedback layer
#: (``src/hitl/feedback.py``) accept exactly the same vocabulary.  The two
#: modules used to declare their own copy, and a new action had to be
#: added twice.
VALID_ACTIONS: frozenset[str] = frozenset({
    "CONTINUE_OPERATION",
    "MONITOR",
    "INSPECT",
    "SCHEDULE_MAINTENANCE",
    "URGENT_MAINTENANCE",
})


#: Explainability-Reliability-Index band thresholds.  They live here, in the
#: lowest layer of the package, so that the explainability report
#: (``src/explainability/eri.py``) and the decision policy
#: (``src/decision_engine/rules.py``) classify the same ERI value with the
#: same comparison.  The two modules used to disagree on the boundary values
#: 0.60 and 0.80.
ERI_LOW_THRESHOLD = 0.60
ERI_HIGH_THRESHOLD = 0.80

#: Continuous bands used to group the expert-feedback evidence by the
#: continuous risk / urgency values.  They live here so that the HITL policy
#: refinement and the decision-sensitivity analysis share one definition.
RISK_BANDS: tuple[tuple[float, str], ...] = ((0.33, "LOW"), (0.66, "MODERATE"))
URGENCY_BANDS: tuple[tuple[float, str], ...] = ((0.33, "LOW"), (0.66, "MODERATE"))


DEFAULT_CONFIG: dict[str, Any] = {
    # ---------------------------------------------------------- data
    "window_size": 40,
    "random_seed": 42,
    "device": "cuda",

    # ---------------------------------------------------------- labelling
    # RUL cap used when generating the piecewise-linear RUL label.  Also
    # applied to the test labels so train/test stay consistent.
    "max_rul": 125,
    # Health-index construction.  "sensor" builds the index from the sensor
    # signal (the default); "rul" reproduces the legacy HI = RUL / max_rul
    # label, which is a deterministic function of RUL.
    "hi_mode": "sensor",
    # Leading cycles per engine used as that engine's healthy reference when
    # the sensor-derived index is built.
    "hi_baseline_cycles": 30,

    # ---------------------------------------------------------- model
    # The architecture is *not* a config key: it is selected on the command
    # line (``scripts/train.py --model``) and recorded in the checkpoint's
    # ``model_config``, so evaluation rebuilds the architecture the weights
    # were actually trained with.  A ``model`` key here would be inert and
    # could only ever disagree with the checkpoint.
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "num_heads": 4,
    "use_layer_norm": True,

    # ---------------------------------------------------------- optimisation
    "batch_size": 256,
    "epochs": 300,
    "learning_rate": 3e-4,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": "cosine",        # "cosine" | "plateau" | "none"
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "patience": 40,
    "grad_clip": 1.0,
    "ema_decay": 0.995,           # 0 disables the weight EMA

    # ---------------------------------------------------------- loss
    "rul_loss_type": "mse",       # "mse" | "huber" | "smooth_l1"
    "huber_delta": 10.0,
    "rul_loss_weight": 1.0,
    "hi_loss_weight": 0.4,
    "scale_loss_weight": 0.1,
    # How the HI term is put on the same scale as the RUL term.  The RUL
    # loss is an unscaled MSE in cycles squared (order 1e2) while the HI
    # loss is a bounded MSE (order 1e-2), so without a scale factor the HI
    # term moves the shared trunk by ~1e-5 in relative terms.  "rul_max"
    # multiplies the HI loss by max_rul^2, which expresses it in
    # cycle-equivalent units; "none" keeps the raw bounded HI loss.
    "hi_loss_scale": "rul_max",   # "rul_max" | "none"

    # ---------------------------------------------------------- augmentation
    # All magnitudes are expressed on the *standardised* feature scale.
    "augment": True,
    "aug_noise_std": 0.02,        # per-timestep Gaussian sensor noise
    "aug_bias_std": 0.05,         # per-sample per-sensor constant offset (drift)
    "aug_gain_std": 0.02,         # per-sample per-sensor gain jitter
    "aug_sensor_dropout": 0.05,   # probability of masking a whole sensor channel
    "aug_time_mask_prob": 0.15,   # probability of masking a contiguous time span
    "aug_time_mask_len": 8,       # max length (in steps) of a masked span
    "aug_history_crop": 8,        # max number of oldest steps dropped per sample
}


def config_path_for(subset: str) -> Path:
    """Return the YAML config path that belongs to *subset*."""

    return CONFIG_DIR / f"{str(subset).lower()}.yaml"


def get_device(
    preference: str | None = None,
    *,
    strict: bool = False,
) -> torch.device:
    """Return the device the pipeline should run on.

    This is the single device selection in the project.  Four scripts
    used to carry their own ``get_device`` helper, and they disagreed
    about what happens when CUDA is requested but not available: one
    fell back to the CPU, another raised, and a third ignored the
    preference completely.

    Parameters
    ----------
    preference:
        ``"cuda"``, ``"cpu"`` or ``None`` (auto-detect).
    strict:
        When ``True``, an unavailable CUDA device raises instead of
        falling back to the CPU.
    """

    if preference in (None, ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preference not in ("cuda", "cpu"):
        raise ValueError(f"Unknown device preference: {preference}")

    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if strict:
        raise RuntimeError("CUDA requested but not available.")

    return torch.device("cpu")


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    subset: str | None = None,
) -> dict[str, Any]:
    """Load a YAML config merged over :data:`DEFAULT_CONFIG`.

    Parameters
    ----------
    path:
        Explicit YAML file.  When omitted, the file for *subset* is used.
    overrides:
        Optional flat mapping applied last (highest precedence).
    subset:
        Dataset subset used to resolve the default YAML path.

    Returns
    -------
    dict
        A new dict; :data:`DEFAULT_CONFIG` is never mutated.
    """

    if path is None:
        if subset is None:
            raise ValueError("Either 'path' or 'subset' must be provided.")
        path = config_path_for(subset)

    config = copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Config file {config_path} must contain a YAML mapping."
            )
        config.update(loaded)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})

    return config


__all__ = [
    "ALL_SUBSETS",
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "VALID_ACTIONS",
    "config_path_for",
    "get_device",
    "load_config",
]
