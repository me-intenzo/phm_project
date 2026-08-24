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
    """Predict non-crossing lower and upper response quantiles."""

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = self.center(features).squeeze(-1)
        half_width = self.half_width(features).squeeze(-1)
        return center - half_width, center + half_width
