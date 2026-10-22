"""
checkpoint.py

Robust model state-dictionary loading.

Pipeline scripts and the model trainer used to load checkpoints strictly,
which fails whenever the checkpoint and the model disagree on the scale
head. The scale head is an optional auxiliary module: the adaptive /
EARA pipelines sometimes strip `scale_head.*` keys from a checkpoint
before loading, and some model revisions add or remove the head entirely.
Strict loading turns those legitimate differences into RuntimeErrors.

This module centralises a tolerant loader. It uses ``strict=False`` and,
when a model expects a scale head that the checkpoint does not provide,
re-initialises that head to a neutral scale so downstream intervals stay
sensible instead of aborting.

author: me-intenzo
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

SCALE_HEAD_PREFIX = "scale_head."

# Inverse softplus of 1.0, i.e. softplus(log(e - 1)) == 1.0.
_NEUTRAL_SOFTPLUS_LOGIT = math.log(math.exp(1.0) - 1.0)


def init_neutral_scale_head(model: nn.Module) -> None:
    """
    Reset a model's ``scale_head`` to predict a neutral scale of ~1.0.

    Missing scale-head weights are initialised to zero weights and a bias
    of ``softplus^{-1}(1.0)`` so that ``softplus(z) ≈ 1.0`` for every
    window. Adaptive / EARA intervals then behave like a plain split
    conformal radius until the head is re-trained or re-loaded.
    """
    scale_head = getattr(model, "scale_head", None)
    if scale_head is None:
        return
    if isinstance(scale_head, nn.Linear):
        nn.init.zeros_(scale_head.weight)
        nn.init.constant_(scale_head.bias, _NEUTRAL_SOFTPLUS_LOGIT)
    else:
        # Fallback: neutralise every parameter we can identify.
        for param in scale_head.parameters():
            with torch.no_grad():
                param.zero_()


def load_state_dict_compat(
    model: nn.Module,
    state_dict: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
    init_missing_scale_head: bool = True,
    device: torch.device | str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load ``state_dict`` into ``model`` tolerantly.

    Unlike ``nn.Module.load_state_dict(..., strict=True)``, a checkpoint
    with (or without) a ``scale_head`` can be loaded into a model with
    (or without) one. Missing ``scale_head.*`` parameters are re-initialised
    to a neutral scale rather than failing the load.

    Parameters
    ----------
    model:
        Target module to load weights into.
    state_dict:
        Checkpoint weights.
    logger:
        Optional logger for missing/unexpected key diagnostics.
    init_missing_scale_head:
        Re-initialise a neutral scale head when its keys are missing.
    device:
        Optional device to move the loaded state to (bypasses strict
        checks on parameter device placement).

    Returns
    -------
    tuple[list[str], list[str]]
        (missing_keys, unexpected_keys) reported by PyTorch.
    """
    state = dict(state_dict)
    if device is not None:
        state = {key: value.to(device) for key, value in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    missing_scale = [
        key for key in missing
        if key.startswith(SCALE_HEAD_PREFIX)
    ]
    if missing_scale and init_missing_scale_head:
        init_neutral_scale_head(model)

    if logger is not None:
        if missing:
            logger.warning(
                "Missing state_dict keys (loaded non-strictly): %s",
                missing,
            )
        if unexpected:
            logger.warning(
                "Unexpected state_dict keys (dropped): %s",
                unexpected,
            )
        if missing_scale:
            logger.info(
                "Scale-head weights missing from checkpoint; "
                "initialised to neutral scale ~= 1.0"
            )

    return list(missing), list(unexpected)


__all__ = [
    "SCALE_HEAD_PREFIX",
    "init_neutral_scale_head",
    "load_state_dict_compat",
]
