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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


DEFAULT_CONFIG: dict[str, Any] = {
    # ---------------------------------------------------------- dataset
    "dataset": "FD001",
    "window_size": 40,
    "random_seed": 42,
    "device": "cuda",

    # ---------------------------------------------------------- labelling
    # RUL cap used when generating the piecewise-linear RUL label.  Also
    # applied to the test labels so train/test stay consistent.
    "max_rul": 125,

    # ---------------------------------------------------------- model
    "model": "gru_att_deg_v2",
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

    if subset is not None:
        config["dataset"] = subset

    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})

    return config


__all__ = [
    "ALL_SUBSETS",
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "config_path_for",
    "load_config",
]
