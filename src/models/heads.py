"""
src/models/heads.py

Shared multi-task heads for the prognostics models.

Every architecture in :mod:`src.models.registry` ends in the same three
task heads:

``rul_head``
    Remaining Useful Life regression.
``hi_head``
    Health Index regression.  Some architectures squash the output into
    ``(0, 1)`` with a sigmoid; the others emit a raw scalar.
``scale_head``
    The heteroscedastic error scale used by the EARA-Conformal pipeline.
    Its output is passed through a softplus so that it is strictly
    positive.

The heads used to be copy-pasted into all six architectures.  They are
built here instead, but the *modules* are still assigned to the same
attribute names (``rul_head`` / ``hi_head`` / ``scale_head``) on the
owning model, so a checkpoint written before this refactor still loads
with an unchanged ``state_dict`` key layout.

author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The supported head layouts.
#:
#: ``"linear"``
#:     Plain linear heads (GRU, LSTM, Transformer, Hybrid).
#: ``"att_deg"``
#:     Linear RUL head, ReLU + sigmoid degradation head.
#: ``"att_deg_v2"``
#:     GELU MLP RUL head, GELU MLP + sigmoid degradation head.
HEAD_STYLES: tuple[str, ...] = ("linear", "att_deg", "att_deg_v2")


def linear_head(hidden_size: int) -> nn.Linear:
    """A plain single-layer regression head."""

    return nn.Linear(hidden_size, 1)


def scale_head(hidden_size: int) -> nn.Linear:
    """The heteroscedastic scale head (softplus is applied by the caller)."""

    return nn.Linear(hidden_size, 1)


def degradation_head(
    hidden_size: int,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """A degradation (health-index) head whose output is squashed to (0, 1).

    The hidden width is half of the trunk width, which is what the
    original copy-pasted heads used.
    """

    inner = max(1, hidden_size // 2)

    layers: list[nn.Module] = [nn.Linear(hidden_size, inner)]

    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "gelu":
        layers.append(nn.GELU())
    else:
        raise ValueError(
            f"Unknown activation {activation!r}; expected 'relu' or 'gelu'."
        )

    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(inner, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def regression_head(
    hidden_size: int,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """A small MLP regression head (used for the RUL head of the v2 model)."""

    inner = max(1, hidden_size // 2)

    layers: list[nn.Module] = [nn.Linear(hidden_size, inner)]

    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "gelu":
        layers.append(nn.GELU())
    else:
        raise ValueError(
            f"Unknown activation {activation!r}; expected 'relu' or 'gelu'."
        )

    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(inner, 1))

    return nn.Sequential(*layers)


def build_task_heads(
    hidden_size: int,
    style: str = "linear",
    dropout: float = 0.0,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build the ``(rul_head, hi_head, scale_head)`` triple for *style*.

    The returned modules are meant to be assigned to attributes with
    exactly those names so that the ``state_dict`` layout of the owning
    model is unchanged.
    """

    if style == "linear":
        return (
            linear_head(hidden_size),
            linear_head(hidden_size),
            scale_head(hidden_size),
        )

    if style == "att_deg":
        return (
            linear_head(hidden_size),
            degradation_head(hidden_size, activation="relu"),
            scale_head(hidden_size),
        )

    if style == "att_deg_v2":
        return (
            regression_head(hidden_size, activation="gelu", dropout=dropout),
            degradation_head(hidden_size, activation="gelu", dropout=dropout),
            scale_head(hidden_size),
        )

    raise ValueError(
        f"Unknown head style {style!r}; expected one of {HEAD_STYLES}."
    )


def forward_task_heads(
    rul_head: nn.Module,
    hi_head: nn.Module,
    scale_head: nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the three heads to *features*.

    Returns
    -------
    tuple
        ``(pred_rul, pred_hi, pred_scale)``, each of shape ``(batch,)``.
        The scale is softplus-activated, so it is strictly positive.
    """

    pred_rul = rul_head(features).squeeze(-1)
    pred_hi = hi_head(features).squeeze(-1)
    pred_scale = F.softplus(scale_head(features)).squeeze(-1)

    return pred_rul, pred_hi, pred_scale


__all__ = [
    "HEAD_STYLES",
    "build_task_heads",
    "degradation_head",
    "forward_task_heads",
    "linear_head",
    "regression_head",
    "scale_head",
]
