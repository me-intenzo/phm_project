"""
hybrid.py

Hybrid LSTM + Transformer prognostics model for joint
Remaining Useful Life (RUL) and Health Index (HI) prediction.

Architecture
------------
Input sequence
      │
      ├───────────────┐
      │               │
      ▼               ▼
   LSTM Branch    Transformer Branch
      │               │
      ▼               ▼
 Temporal State   Temporal Representation
      │               │
      └───────┬───────┘
              ▼
       Feature Fusion
              │
       Shared Representation
              │
        ┌─────┴─────┐
        ▼           ▼
       RUL          HI

Author: me-intenzo
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x + self.pe[ # type: ignore[index]
            :, :x.size(1), :
        ]

        return self.dropout(x)


class HybridPrognosticsModel(nn.Module):
    """
    Parallel LSTM + Transformer multi-task prognostics model.

    Parameters
    ----------
    input_size : int
        Number of sensor features.

    hidden_size : int
        Hidden representation size.

    num_layers : int
        Number of LSTM/Transformer layers.

    num_heads : int
        Number of Transformer attention heads.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        # --------------------------------------------------
        # LSTM Branch
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # --------------------------------------------------
        # Transformer Branch
        # --------------------------------------------------

        self.transformer_projection = (
            nn.Linear(
                input_size,
                hidden_size,
            )
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=hidden_size,
                max_len=500,
                dropout=dropout,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        # --------------------------------------------------
        # Fusion Layer
        # --------------------------------------------------

        fusion_size = hidden_size * 2

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_size,
                hidden_size,
            ),
            nn.BatchNorm1d(
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # Multi-task Heads
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
    # Forward
    # ------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # ==================================================
        # LSTM branch
        # ==================================================

        _, (lstm_hidden, _) = (
            self.lstm(x)
        )

        lstm_features = (
            lstm_hidden[-1]
        )

        # ==================================================
        # Transformer branch
        # ==================================================

        transformer_x = (
            self.transformer_projection(x)
        )

        transformer_x = (
            self.positional_encoding(
                transformer_x
            )
        )

        transformer_x = (
            self.transformer(
                transformer_x
            )
        )

        transformer_features = (
            transformer_x.mean(
                dim=1
            )
        )

        # ==================================================
        # Feature Fusion
        # ==================================================

        fused_features = torch.cat(
            [
                lstm_features,
                transformer_features,
            ],
            dim=1,
        )

        features = self.fusion(
            fused_features
        )

        # ==================================================
        # Multi-task prediction
        # ==================================================

        pred_rul = self.rul_head(
            features
        )

        pred_hi = self.hi_head(
            features
        )

        return (
            pred_rul.squeeze(-1),
            pred_hi.squeeze(-1),
        )


# ----------------------------------------------------------
# Standalone Test
# ----------------------------------------------------------

if __name__ == "__main__":

    model = HybridPrognosticsModel(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
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
        "Trainable Parameters : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )