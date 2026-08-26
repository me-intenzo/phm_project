"""
transformer.py

Transformer encoder model for joint Remaining Useful Life (RUL)
and Health Index (HI) prediction.

Architecture
------------
Input sequence
    ↓
Feature Projection
    ↓
Positional Encoding
    ↓
Transformer Encoder
    ↓
Temporal Aggregation
    ↓
Shared Feature Representation
    ↓
┌──────────────┬
↓              ↓
RUL Head       HI Head

Author: me-intenzo
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Parameters
    ----------
    d_model : int
        Transformer embedding dimension.

    max_len : int
        Maximum supported sequence length.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(
            p=dropout
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

        # Shape:
        # (1, max_len, d_model)

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (batch, sequence_length, d_model)

        Returns
        -------
        torch.Tensor
            Position-aware sequence representation.
        """

        x = x + self.pe[  # type: ignore[index]
            :, :x.size(1), :
        ]

        return self.dropout(x)


class TransformerPrognosticsModel(nn.Module):
    """
    Transformer-based multi-task prognostics model.

    Predicts:
        1. Remaining Useful Life (RUL)
        2. Health Index (HI)
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

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.num_heads = num_heads

        # --------------------------------------------------
        # Feature Projection
        # --------------------------------------------------

        self.input_projection = nn.Linear(
            input_size,
            hidden_size,
        )

        # --------------------------------------------------
        # Positional Encoding
        # --------------------------------------------------

        self.positional_encoding = (
            PositionalEncoding(
                d_model=hidden_size,
                max_len=500,
                dropout=dropout,
            )
        )

        # --------------------------------------------------
        # Transformer Encoder
        # --------------------------------------------------

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

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        x : torch.Tensor

        Shape
        -----
        (batch, sequence_length, input_size)

        Returns
        -------
        pred_rul : torch.Tensor
            Shape: (batch,)

        pred_hi : torch.Tensor
            Shape: (batch,)
        """

        # --------------------------------------------------
        # Project sensor features into Transformer space
        # --------------------------------------------------

        x = self.input_projection(x)

        # --------------------------------------------------
        # Add temporal positional information
        # --------------------------------------------------

        x = self.positional_encoding(x)

        # --------------------------------------------------
        # Transformer encoding
        # --------------------------------------------------

        x = self.encoder(x)

        # --------------------------------------------------
        # Temporal aggregation
        #
        # Mean pooling allows the model to use information
        # from the complete degradation window.
        # --------------------------------------------------

        features = x.mean(
            dim=1
        )

        # --------------------------------------------------
        # Shared representation
        # --------------------------------------------------

        features = self.shared(
            features
        )

        # --------------------------------------------------
        # Multi-task prediction
        # --------------------------------------------------

        pred_rul = self.rul_head(
            features
        )

        pred_hi = self.hi_head(
            features
        )

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

    model = TransformerPrognosticsModel(
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