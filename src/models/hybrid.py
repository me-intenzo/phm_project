"""
hybrid.py

Hybrid TCN + GRU prognostics model for joint
Hybrid CNN + GRU + Transformer prognostics model for joint
Remaining Useful Life (RUL) and Health Index (HI) prediction.

Architecture
------------
Input Sequence
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
      ├──────────────────────┬──────────────────────┐
      │                      │                      │
      ▼                      ▼                      ▼
   1D CNN                  GRU               Transformer
      │                      │                      │
      ▼                      ▼                      ▼
Local Temporal         Sequential          Long-Range
  Features             Features            Dependencies
      │                      │                      │
      └──────────────────────┼──────────────────────┘
                             ▼
                     Feature Concatenation
                             │
                             ▼
                       Fusion Layer
                             │
                             ▼
                   Shared Representation
                             │
                       ┌─────┴─────┐
                       ▼           ▼
                      RUL         HI

Author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ==============================================================
# POSITIONAL ENCODING
# ==============================================================

class TemporalBlock(nn.Module):
    """
    Single TCN block: two dilated causal Conv1d layers with residual connection.
    Sinusoidal positional encoding for Transformer input.
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
        self.dropout = nn.Dropout(dropout)

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
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
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
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x + self.pe[
            :, :x.size(1), :
        ]

        return self.dropout(x)


# ==============================================================
# HYBRID CNN + GRU + TRANSFORMER MODEL
# ==============================================================

class HybridPrognosticsModel(nn.Module):
    """
    Parallel TCN + GRU multi-task prognostics model.
    Parallel CNN + GRU + Transformer multi-task prognostics model.

    Parameters
    ----------
    input_size : int
        Number of input sensor features.

    hidden_size : int
        Shared hidden representation size.

    num_layers : int
        Number of TCN blocks / GRU layers.
        Number of GRU and Transformer layers.

    kernel_size : int
        Kernel size for TCN convolutions.

    dropout : float
        Dropout probability.

    Outputs
    -------
    pred_rul : torch.Tensor
        Predicted Remaining Useful Life.

    pred_hi : torch.Tensor
        Predicted Health Index.
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
        # ----------------------------------------------------------
        # 1. CNN BRANCH
        # ----------------------------------------------------------
        #
        # Input:
        #   (batch, sequence, features)
        #
        # Conv1D requires:
        #   (batch, channels, sequence)
        #
        # Therefore the input is permuted in forward().
        #

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(dropout),
        )

        # Reduce temporal dimension
        self.cnn_pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
        )

        # Convert temporal features to fixed vector
        self.cnn_global_pool = nn.AdaptiveAvgPool1d(
            1
        )

        cnn_output_size = 128

        # ----------------------------------------------------------
        # 2. GRU BRANCH
        # ----------------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # --------------------------------------------------
        # GRU Branch
        # --------------------------------------------------
        gru_output_size = hidden_size

        # ----------------------------------------------------------
        # 3. TRANSFORMER BRANCH
        # ----------------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.transformer_projection = nn.Linear(
            input_size,
            hidden_size,
        )

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
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        transformer_output_size = hidden_size

        # --------------------------------------------------
        # Fusion Layer
        # --------------------------------------------------
        # ----------------------------------------------------------
        # 4. FEATURE FUSION
        # ----------------------------------------------------------

        fusion_input_size = (
            cnn_output_size
            + gru_output_size
            + transformer_output_size
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),

            nn.Linear(
                fusion_input_size,
                hidden_size,
            ),

            nn.BatchNorm1d(
                hidden_size,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),
        )

        # ----------------------------------------------------------
        # 5. MULTI-TASK OUTPUT HEADS
        # ----------------------------------------------------------

        self.rul_head = nn.Linear(hidden_size, 1)
        self.hi_head = nn.Linear(hidden_size, 1)
        self.scale_head = nn.Linear(hidden_size, 1)

    # ==============================================================
    # FORWARD PASS
    # ==============================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ==================================================
        # TCN branch  →  last time-step of (B, C, T)
        # ==================================================

        tcn_features = self.tcn(x)[:, :, -1]
        # ==========================================================
        # CNN BRANCH
        # ==========================================================

        # Input:
        # (batch, sequence, features)

        cnn_x = x.permute(
            0,
            2,
            1,
        )

        # CNN feature extraction
        cnn_x = self.cnn(
            cnn_x
        )

        # Temporal downsampling
        cnn_x = self.cnn_pool(
            cnn_x
        )

        # Global average pooling
        # (batch, 128, sequence)
        #            ↓
        # (batch, 128, 1)

        cnn_x = self.cnn_global_pool(
            cnn_x
        )

        # (batch, 128, 1)
        #       ↓
        # (batch, 128)

        cnn_features = cnn_x.squeeze(
            -1
        )

        # ==================================================
        # GRU branch  →  last hidden state
        # ==================================================
        # ==========================================================
        # GRU BRANCH
        # ==========================================================

        _, gru_hidden = self.gru(x)
        gru_features = gru_hidden[-1]
        _, gru_hidden = self.gru(x)

        # Last GRU layer hidden state
        #
        # Shape:
        # (num_layers, batch, hidden_size)
        #
        # Take the final layer:

        gru_features = gru_hidden[-1]

        # Shape:
        # (batch, hidden_size)

        # ==========================================================
        # TRANSFORMER BRANCH
        # ==========================================================

        transformer_x = self.transformer_projection(
            x
        )

        transformer_x = self.positional_encoding(
            transformer_x
        )

        transformer_x = self.transformer(
            transformer_x
        )

        # Global average pooling across time
        #
        # (batch, sequence, hidden)
        #          ↓
        # (batch, hidden)

        transformer_features = transformer_x.mean(
            dim=1
        )

        # ==================================================
        # Feature Fusion
        # ==================================================
        # ==========================================================
        # FEATURE FUSION
        # ==========================================================

        fused_features = torch.cat(
            [
                cnn_features,
                gru_features,
                transformer_features,
            ],
            dim=1,
        )

        # Shape:
        #
        # CNN        = 128
        # GRU        = hidden_size
        # Transformer= hidden_size
        #
        # For hidden_size=128:
        # 128 + 128 + 128 = 384

        features = self.fusion(
            torch.cat([tcn_features, gru_features], dim=1)
        )

        # ==========================================================
        # RUL PREDICTION
        # ==========================================================

        pred_scale = torch.nn.functional.softplus(self.scale_head(features))
        pred_rul = self.rul_head(
            features
        )

        # ==========================================================
        # HI PREDICTION
        # ==========================================================

        pred_hi = self.hi_head(
            features
        )

        # Remove final dimension:
        #
        # (batch, 1)
        #    ↓
        # (batch,)

        return (
            self.rul_head(features).squeeze(-1),
            self.hi_head(features).squeeze(-1),
            pred_scale.squeeze(-1),
        )


# ==============================================================
# STANDALONE MODEL TEST
# ==============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Hybrid CNN + GRU + Transformer Model")
    print("=" * 60)

    # ----------------------------------------------------------
    # Model configuration
    # ----------------------------------------------------------

    model = HybridPrognosticsModel(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        kernel_size=3,
        dropout=0.3,
    )

    # ----------------------------------------------------------
    # Example input
    # ----------------------------------------------------------

    # Batch size = 64
    # Sequence length = 30
    # Sensor features = 14

    x = torch.randn(64, 30, 14)
    x = torch.randn(
        64,
        30,
        14,
    )

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    with torch.no_grad():

    rul, hi, scale = model(x)
        rul, hi = model(x)

    print(f"RUL Output Shape   : {rul.shape}")
    print(f"HI Output Shape    : {hi.shape}")
    print(f"Scale Output Shape : {scale.shape}")
    # ----------------------------------------------------------
    # Output information
    # ----------------------------------------------------------

    print(
        f"Input Shape        : {x.shape}"
    )

    print(
        f"RUL Output Shape   : {rul.shape}"
    )

    print(
        f"HI Output Shape    : {hi.shape}"
    )

    # ----------------------------------------------------------
    # Parameter count
    # ----------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable Parameters : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

        f"Trainable Parameters: {total_params:,}"
    )

    # ----------------------------------------------------------
    # Expected checks
    # ----------------------------------------------------------

    assert x.shape == (
        64,
        30,
        14,
    )

    assert rul.shape == (
        64,
    )

    assert hi.shape == (
        64,
    )

    print()
    print("Model forward-pass test: PASSED")
    print("=" * 60)