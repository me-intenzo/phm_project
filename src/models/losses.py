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
        beta  * HI Loss +
        gamma * Scale Loss

    The scale term is the Gaussian negative log-likelihood of the RUL
    residual under the model's predicted per-sample scale s(x):

        scale_loss = mean( 0.5 * ((y - y_hat) / s)^2 + log(s) )

    Minimising this drives s(x) toward |y - y_hat|, which is exactly what
    the EARA-Conformal adaptive intervals assume s(x) represents. Without
    this term the scale head receives no gradient and stays at its
    initialisation, so adaptive interval widths are not error-calibrated.

    Parameters
    ----------
    alpha : float
        Weight of the RUL MSE term.
    beta : float
        Weight of the HI MSE term.
    gamma : float
        Weight of the scale (NLL) term. Deliberately small: Adam largely
        normalises per-parameter gradient scale, so this controls how much
        the scale objective competes with RUL accuracy rather than how
        fast the scale head learns.
    eps : float
        Numerical floor keeping s(x) strictly positive in the log term.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1e-6,
    ) -> None:

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

        self.rul_loss = nn.MSELoss()
        self.hi_loss = nn.MSELoss()

    def forward(
        self,
        pred_rul,
        pred_hi,
        target_rul,
        target_hi,
        pred_scale=None,
    ):

        loss_rul = self.rul_loss(
            pred_rul,
            target_rul,
        )

        loss_hi = self.hi_loss(
            pred_hi,
            target_hi,
        )

        if pred_scale is None:
            # Models without a scale head, or callers opting out.
            scale_loss = torch.zeros((), device=loss_rul.device)
        else:
            scale = pred_scale + self.eps
            normalised_residual = (target_rul - pred_rul) / scale
            scale_loss = (
                0.5 * normalised_residual.pow(2)
                + torch.log(scale)
            ).mean()

        total_loss = (
            self.alpha * loss_rul
            +
            self.beta * loss_hi
            +
            self.gamma * scale_loss
        )

        return {
            "total_loss": total_loss,
            "rul_loss": loss_rul,
            "hi_loss": loss_hi,
            "scale_loss": scale_loss,
        }