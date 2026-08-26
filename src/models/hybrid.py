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
