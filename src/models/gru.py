"""
gru.py

Baseline GRU model for joint Remaining Useful Life (RUL)
and Health Index (HI) prediction.

The architecture intentionally mirrors the LSTM baseline so that
the recurrent encoder is the primary experimental variable.

Author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  


class GRUPrognosticsModel(nn.Module):
    """
    Multi-task GRU prognostics model.

    Architecture
    ------------
    Input sequence
        ↓
    GRU encoder
        ↓
    Last hidden representation
        ↓
    Shared feature layer
        ↓
    ┌──────────────┬──────────────┐
    ↓              ↓
    RUL head       HI head

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.

    hidden_size : int
        GRU hidden representation size.

    num_layers : int
        Number of stacked GRU layers.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --------------------------------------------------
        # Shared GRU Encoder
        # --------------------------------------------------

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --------------------------------------------------
        # Shared Feature Representation
        # --------------------------------------------------

        self.shared = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.BatchNorm1d(
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # Task Heads
        # --------------------------------------------------

        self.rul_head = nn.Linear(
            hidden_size,
            1,
        )

        self.hi_head = nn.Linear(
            hidden_size,
            1,
        )

        # Heteroscedastic scale head for EARA-Conformal.
        # Outputs log-scale to ensure positivity via softplus.
        self.scale_head = nn.Linear(
            hidden_size,
            1,
        )

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (batch, sequence_length, input_size)

        Returns
        -------
        tuple
            pred_rul:
                Shape (batch,)

            pred_hi:
                Shape (batch,)
        """

        _, hidden = self.encoder(x)

        # Last GRU layer representation
        features = hidden[-1]

        # Shared representation
        features = self.shared(features)

        # Task predictions
        pred_rul = self.rul_head(features)

        pred_hi = self.hi_head(features)

        # Softplus ensures scale > 0
        pred_scale = torch.nn.functional.softplus(
            self.scale_head(features)
        )

        return (
            pred_rul.squeeze(-1),
            pred_hi.squeeze(-1),
            pred_scale.squeeze(-1),
        )


# ----------------------------------------------------------
# Standalone Model Test
# ----------------------------------------------------------

if __name__ == "__main__":

    model = GRUPrognosticsModel(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )

    x = torch.randn(
        64,
        30,
        14,
    )

    rul, hi = model(x)

    print(
        f"RUL Output Shape : {rul.shape}"
    )

    print(
        f"HI Output Shape  : {hi.shape}"
    )

    print(
        f"Trainable Parameters : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )