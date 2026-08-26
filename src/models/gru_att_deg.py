"""
gru_att_deg.py

Hybrid GRU + Temporal Attention + Degradation prognostics model
for joint RUL and Health Index (HI) prediction.

Architecture
------------
Input sequence
      │
      ▼
   GRU Encoder
      │
      ▼
 Temporal Attention  ←── learns which time-steps matter most
      │
      ▼
 Context Vector
      │
   ┌──┴──────────────┐
   ▼                 ▼
  RUL Head     Degradation Head
                (monotonic HI)

Author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over GRU hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, gru_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # gru_out: (B, T, H)
        scores = self.attn(gru_out).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=1)               # (B, T)
        context = (weights.unsqueeze(-1) * gru_out).sum(dim=1)  # (B, H)
        return context, weights


class GruAttDeg(nn.Module):
    """
    GRU + Temporal Attention + Degradation multi-task prognostics model.

    Parameters
    ----------
    input_size : int
        Number of sensor features.
    hidden_size : int
        GRU hidden size.
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

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_size)

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # RUL head
        self.rul_head = nn.Linear(hidden_size, 1)
        self.scale_head = nn.Linear(hidden_size, 1)

        # Degradation head — sigmoid keeps HI in (0, 1)
        self.hi_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        gru_out, _ = self.gru(x)                         # (B, T, H)
        context, _attn_weights = self.attention(gru_out) # (B, H)
        features = self.shared(context)                  # (B, H)

        pred_rul = self.rul_head(features).squeeze(-1)
        pred_hi = self.hi_head(features).squeeze(-1)
        pred_scale = F.softplus(self.scale_head(features)).squeeze(-1)

        return pred_rul, pred_hi, pred_scale


# ----------------------------------------------------------
# Standalone Test
# ----------------------------------------------------------

if __name__ == "__main__":

    model = GruAttDeg(input_size=14, hidden_size=128, num_layers=2, dropout=0.3)

    x = torch.randn(64, 30, 14)
    rul, hi, scale = model(x)

    print(f"RUL Output Shape   : {rul.shape}")
    print(f"HI Output Shape    : {hi.shape}")
    print(f"Scale Output Shape : {scale.shape}")
    print(
        "Trainable Parameters : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
