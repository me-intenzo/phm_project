<<<<<<< HEAD
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ==============================================================
# POSITIONAL ENCODING
# ==============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Position indices
        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Frequency terms
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

        # Positional encoding matrix
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
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x + self.pe[
            :, :x.size(1), :
        ]

        return self.dropout(x)


# ==============================================================
# HYBRID CNN + GRU + TRANSFORMER MODEL
# ==============================================================

class CNNGRUTransformerPrognostics(nn.Module):
    """
    Parallel 1D CNN + GRU + Transformer multi-task
    prognostics model.

    Input:
        x shape = (batch, sequence_length, input_size)

    Outputs:
        RUL prediction
        HI prediction
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_gru_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        cnn_channels: tuple = (64, 128),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        # ======================================================
        # 1D CNN BRANCH
        # ======================================================

        cnn_layers = []

        in_channels = input_size

        for out_channels in cnn_channels:

            cnn_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),

                    nn.BatchNorm1d(
                        out_channels
                    ),

                    nn.ReLU(),
                ]
            )

            in_channels = out_channels

        self.cnn = nn.Sequential(
            *cnn_layers
        )

        # Spatial features output dimension
        self.spatial_dim = cnn_channels[-1]

        # ======================================================
        # TEMPORAL BRANCH (BiGRU + Transformer)
        # ======================================================

        self.bigru = nn.GRU(
            input_size=self.spatial_dim,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        self.transformer_projection = nn.Linear(
            self.spatial_dim,
            hidden_size,
        )

        self.bigru_projection = nn.Linear(hidden_size * 2, hidden_size)

        self.positional_encoding = PositionalEncoding(
            d_model=hidden_size,
            max_len=500,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Attention weight for fusion: 1 scalar per branch
        self.attn_weights = nn.Parameter(torch.ones(2))

        # CNN + BiGRU + Transformer
        # Temporal features are already fused via attention (size=hidden_size)
        fusion_input_size = hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_size,
                hidden_size,
            ),

            nn.BatchNorm1d(
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),
        )

        # ======================================================
        # MULTI-TASK HEADS
        # ======================================================

        # Remaining Useful Life
        self.rul_head = nn.Linear(
            hidden_size,
            1,
        )

        # Health Index
        self.hi_head = nn.Linear(
            hidden_size,
            1,
        )

    # ==========================================================
    # FORWARD PASS
    # ==========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        """
        Forward pass.

        Input:
            x = (batch, sequence_length, input_size)

        Returns:
            pred_rul = (batch,)
            pred_hi  = (batch,)
        """

        # ======================================================
        # SPATIAL FEATURE EXTRACTION (CNN)
        # ======================================================

        # Conv1D expects: (batch, channels, sequence)
        cnn_x = x.permute(0, 2, 1)
        cnn_features = self.cnn(cnn_x)

        # Transpose back to (batch, sequence, channels) for temporal branches
        spatial_features = cnn_features.permute(0, 2, 1)

        # ======================================================
        # TEMPORAL FEATURE EXTRACTION (BiGRU + Transformer)
        # ======================================================

        # BiGRU
        gru_output, _ = self.bigru(spatial_features)
        # Take last time step for BiGRU (batch, hidden * 2)
        bigru_features = gru_output[:, -1, :]
        bigru_features = self.bigru_projection(bigru_features)

        # Transformer
        transformer_x = self.transformer_projection(spatial_features)
        transformer_x = self.positional_encoding(transformer_x)
        transformer_x = self.transformer(transformer_x)
        # Global pooling over sequence dimension
        transformer_features = transformer_x.mean(dim=1)

        # Attention-weighted sum
        weights = torch.softmax(self.attn_weights, dim=0)
        temporal_features = (weights[0] * bigru_features) + (weights[1] * transformer_features)

        # ======================================================
        # FEATURE FUSION
        # ======================================================

        # Shape:
        # (batch, features)
        features = self.fusion(temporal_features)

        # ======================================================
        # MULTI-TASK PREDICTION
        # ======================================================

        # RUL prediction
        pred_rul = self.rul_head(
            features
        )

        # HI prediction
        pred_hi = self.hi_head(
            features
        )

        # Return (batch, 1) to match loss function target dimensions
        return (
            pred_rul,
            pred_hi,
            torch.ones_like(pred_rul),
        )


# Alias for compatibility with existing scripts
HybridPrognosticsModel = CNNGRUTransformerPrognostics


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Model configuration
    # ----------------------------------------------------------

    model = CNNGRUTransformerPrognostics(

        # Number of input sensor features
        input_size=14,

        # Common hidden representation
        hidden_size=128,

        # GRU layers
        num_gru_layers=2,

        # Transformer layers
        num_transformer_layers=2,

        # Attention heads
        num_heads=4,

        # CNN channels
        cnn_channels=(64, 128),

        # CNN kernel
        kernel_size=3,

        # Dropout
        dropout=0.3,
    )

    # ----------------------------------------------------------
    # Example input
    # ----------------------------------------------------------

    # Batch size = 64
    # Sequence length = 30
    # Number of sensors = 14

    x = torch.randn(
        64,
        30,
        14,
    )

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    rul, hi = model(x)

    # ----------------------------------------------------------
    # Output shapes
    # ----------------------------------------------------------

    print(
        f"Input Shape       : {x.shape}"
    )

    print(
        f"RUL Output Shape  : {rul.shape}"
    )

    print(
        f"HI Output Shape   : {hi.shape}"
    )

    # ----------------------------------------------------------
    # Number of trainable parameters
    # ----------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable Parameters : {total_params:,}"
    )
=======
"""
hybrid.py

Hybrid TCN + GRU prognostics model for joint
Remaining Useful Life (RUL) and Health Index (HI) prediction.

Architecture
------------
Input sequence
      │
      ├───────────────┐
      │               │
      ▼               ▼
  TCN Branch      GRU Branch
      │               │
      ▼               ▼
 Local Patterns   Sequential State
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

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    Single TCN block: two dilated causal Conv1d layers with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:

        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def _causal_trim(self, x: torch.Tensor, size: int) -> torch.Tensor:
        return x[:, :, :size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)

        out = self.relu(self._causal_trim(self.conv1(x), T))
        out = self.dropout(out)
        out = self.relu(self._causal_trim(self.conv2(out), T))
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with exponentially increasing dilations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(
                TemporalBlock(in_ch, hidden_size, kernel_size, dilation=2**i, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for Conv1d
        return self.network(x.permute(0, 2, 1))


class HybridPrognosticsModel(nn.Module):
    """
    Parallel TCN + GRU multi-task prognostics model.

    Parameters
    ----------
    input_size : int
        Number of sensor features.

    hidden_size : int
        Hidden representation size.

    num_layers : int
        Number of TCN blocks / GRU layers.

    kernel_size : int
        Kernel size for TCN convolutions.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        # --------------------------------------------------
        # TCN Branch
        # --------------------------------------------------

        self.tcn = TCN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # --------------------------------------------------
        # GRU Branch
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --------------------------------------------------
        # Fusion Layer
        # --------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # Multi-task Heads
        # --------------------------------------------------

        self.rul_head = nn.Linear(hidden_size, 1)
        self.hi_head = nn.Linear(hidden_size, 1)
        self.scale_head = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ==================================================
        # TCN branch  →  last time-step of (B, C, T)
        # ==================================================

        tcn_features = self.tcn(x)[:, :, -1]

        # ==================================================
        # GRU branch  →  last hidden state
        # ==================================================

        _, gru_hidden = self.gru(x)
        gru_features = gru_hidden[-1]

        # ==================================================
        # Feature Fusion
        # ==================================================

        features = self.fusion(
            torch.cat([tcn_features, gru_features], dim=1)
        )

        # ==================================================
        # Multi-task prediction
        # ==================================================

        pred_scale = torch.nn.functional.softplus(self.scale_head(features))

        return (
            self.rul_head(features).squeeze(-1),
            self.hi_head(features).squeeze(-1),
            pred_scale.squeeze(-1),
        )


# ----------------------------------------------------------
# Standalone Test
# ----------------------------------------------------------

if __name__ == "__main__":

    model = HybridPrognosticsModel(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        kernel_size=3,
        dropout=0.3,
    )

    x = torch.randn(64, 30, 14)

    rul, hi, scale = model(x)

    print(f"RUL Output Shape   : {rul.shape}")
    print(f"HI Output Shape    : {hi.shape}")
    print(f"Scale Output Shape : {scale.shape}")
    print(
        "Trainable Parameters : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
>>>>>>> 232c8a9acfc44f8cf76034ecb550fec335210471
