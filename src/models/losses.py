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
    hi_scale : float
        Factor applied to the HI term so it lives on the same scale as the
        RUL term.  The RUL loss is an unscaled MSE in cycles squared (order
        1e2 for a 12-cycle RMSE) while the HI loss is a bounded MSE (order
        1e-2), so without a factor the HI term moves the shared trunk by
        ~1e-5 in relative terms and ``beta`` is effectively inert.  The
        pipeline passes ``max_rul ** 2`` (cycle-equivalent units); the
        default of ``1.0`` reproduces the raw bounded HI loss.
    eps : float
        Numerical floor keeping s(x) strictly positive in the log term.
    rul_loss_type : str
        ``"mse"`` (default, matched to RMSE), ``"huber"`` or
        ``"smooth_l1"``.  Huber/SmoothL1 clip the influence of the few
        badly mis-estimated windows, which regularises training but
        slightly biases the optimum away from the least-squares solution.
    huber_delta : float
        Transition point of the Huber loss, expressed in RUL cycles.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        hi_scale: float = 1.0,
        eps: float = 1e-6,
        rul_loss_type: str = "mse",
        huber_delta: float = 10.0,
    ) -> None:

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hi_scale = float(hi_scale)
        self.eps = eps
        self.rul_loss_type = rul_loss_type
        self.huber_delta = huber_delta

        if rul_loss_type == "mse":
            self.rul_loss = nn.MSELoss()
        elif rul_loss_type == "huber":
            self.rul_loss = nn.HuberLoss(delta=huber_delta)
        elif rul_loss_type == "smooth_l1":
            self.rul_loss = nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError(
                f"Unknown rul_loss_type: {rul_loss_type}. "
                "Expected 'mse', 'huber' or 'smooth_l1'."
            )

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

        # The HI term is put on the RUL scale before it enters the total, so
        # ``beta`` is comparable with ``alpha`` and the HI head actually
        # contributes gradient to the shared trunk.
        loss_hi_scaled = self.hi_scale * loss_hi

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
            self.beta * loss_hi_scaled
            +
            self.gamma * scale_loss
        )

        return {
            "total_loss": total_loss,
            "rul_loss": loss_rul,
            "hi_loss": loss_hi,
            "hi_loss_scaled": loss_hi_scaled,
            "scale_loss": scale_loss,
        }
