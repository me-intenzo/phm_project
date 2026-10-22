"""
uncertainty_heads.py

Auxiliary heads for uncertainty estimation and
adaptive scaling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleHead(nn.Module):
    """
    Adaptive scale head for regime-conditional
    conformal prediction.

    Learns a positive scaling factor s(x) from
    latent features.
    """

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # Ensures s(x) > 0
        )

    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            Latent features from the GRU (batch, hidden_size)

        Returns
        -------
        torch.Tensor
            Positive scale factor s(x) (batch,)
        """
        return self.net(features).squeeze(-1)


class QuantileHead(nn.Module):
    """Predict non-crossing lower and upper response quantiles.

    The head is trained on frozen features of a model that already predicts
    the response well, so it can be anchored to that point prediction: the
    centre becomes ``base_pred + delta(features)`` and the linear ``delta``
    only has to learn a correction.  Without the anchor the head has to
    re-learn the whole response from the frozen features with a single linear
    layer, which underfits and collapses to (very wide) marginal quantiles.

    ``base_pred`` is optional so the head can still be used standalone.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.center = nn.Linear(hidden_size, 1)
        self.half_width = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        features: torch.Tensor,
        base_pred: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = self.center(features).squeeze(-1)
        if base_pred is not None:
            center = center + base_pred.reshape(-1)
        half_width = self.half_width(features).squeeze(-1)
        return center - half_width, center + half_width
