"""
lstm.py

Baseline LSTM model for joint Remaining Useful Life (RUL)
and Health Index (HI) prediction.

Author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPrognosticsModel(nn.Module):
    """
    Multi-task LSTM Prognostics Model.

    Outputs
    -------
    - Remaining Useful Life (RUL)
    - Health Index (HI)
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
        # Shared LSTM Encoder
        # --------------------------------------------------

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # --------------------------------------------------
        # Shared Feature Representation
        # --------------------------------------------------

        self.shared = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),

            nn.BatchNorm1d(hidden_size),

            nn.ReLU(),

            nn.Dropout(dropout)

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

    # ------------------------------------------------------

    def forward(self, x):

        """
        Parameters
        ----------
        x : Tensor

        Shape

        (batch, window, features)

        Returns
        -------

        rul : Tensor

        hi : Tensor
        """

        _, (hidden, _) = self.encoder(x)

        features = hidden[-1]

        features = self.shared(features)

        rul = self.rul_head(features)

        hi = self.hi_head(features)

        return rul.squeeze(-1), hi.squeeze(-1)


# --------------------------------------------------------
# Model Test
# --------------------------------------------------------

if __name__ == "__main__":

    model = LSTMPrognosticsModel(
        input_size=18
    )

    x = torch.randn(
        64,
        30,
        18,
    )

    rul, hi = model(x)

    print(f"RUL Output Shape : {rul.shape}")
    print(f"HI Output Shape  : {hi.shape}")