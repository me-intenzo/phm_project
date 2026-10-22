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

import torch
import torch.nn as nn

from .heads import build_task_heads, forward_task_heads
from .layers import PositionalEncoding


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
        # Task Heads (shared layout, see ``src/models/heads.py``)
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
    ) -> tuple[
        torch.Tensor,
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

        pred_scale : torch.Tensor
            Shape: (batch,); softplus-activated per-sample scale.
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

    rul, hi, scale = model(x)

    print(
        f"RUL Output Shape : {rul.shape}"
    )

    print(
        f"HI Output Shape  : {hi.shape}"
    )

    print(
        f"Scale Output Shape : {scale.shape}"
    )

    print(
        "Trainable Parameters : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
