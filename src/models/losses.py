"""
losses.py

Loss functions for multi-task prognostics models.

Author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint RUL and HI prediction.

    Total Loss =
        alpha * RUL Loss +
        beta  * HI Loss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
    ) -> None:

        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.rul_loss = nn.MSELoss()
        self.hi_loss = nn.MSELoss()

    def forward(
        self,
        pred_rul,
        pred_hi,
        target_rul,
        target_hi,
    ):

        loss_rul = self.rul_loss(
            pred_rul,
            target_rul,
        )

        loss_hi = self.hi_loss(
            pred_hi,
            target_hi,
        )

        total_loss = (
            self.alpha * loss_rul
            +
            self.beta * loss_hi
        )

        return {
            "total_loss": total_loss,
            "rul_loss": loss_rul,
            "hi_loss": loss_hi,
        }