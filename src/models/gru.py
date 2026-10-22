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

from .heads import build_task_heads, forward_task_heads


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
        #
        # The head layout is shared with the other architectures (see
        # ``src/models/heads.py``); only the modules are created here so
        # that the ``state_dict`` keys stay ``rul_head.*`` / ``hi_head.*``
        # / ``scale_head.*``.
        # --------------------------------------------------

        (
            self.rul_head,
            self.hi_head,
            self.scale_head,
        ) = build_task_heads(
            hidden_size,
            style="linear",
        )

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            pred_scale:
                Shape (batch,); softplus-activated per-sample scale.
        """

        _, hidden = self.encoder(x)

        # Last GRU layer representation
        features = hidden[-1]

        # Shared representation
        features = self.shared(features)

        # Task predictions
        pred_rul, pred_hi, pred_scale = forward_task_heads(
            self.rul_head,
            self.hi_head,
            self.scale_head,
            features,
        )

        return (
            pred_rul,
            pred_hi,
            pred_scale,
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

    rul, hi, _ = model(x)

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
