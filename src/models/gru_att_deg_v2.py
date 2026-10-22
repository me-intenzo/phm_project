"""
gru_att_deg_v2.py

GRU + Multi-Head Temporal Attention + Degradation multi-task model (v2).

Why v2 exists
-------------
The v1 model (``gru_att_deg``) reached a good average RUL RMSE but showed
two structural weaknesses on the NASA C-MAPSS subsets:

1. **BatchNorm in the shared layer.**  With 80–260 training engines the
   batch statistics are dominated by whichever engines happen to land in a
   batch, which made validation loss swing by 40 % between consecutive
   epochs and pushed the model towards a regression-to-the-mean solution.
2. **Single-head additive attention.**  One attention vector per window is
   not enough to separate "this engine is degrading fast" from "this
   engine is degrading slowly" on the multi-condition subsets (FD002,
   FD004).

v2 therefore:

* replaces BatchNorm with LayerNorm (batch-size independent, stable);
* uses multi-head scaled dot-product temporal attention with a residual
  connection;
* fuses the attention context with the last GRU hidden state, so the model
  sees both "what is salient in the window" and "the most recent state";
* widens the task heads and keeps the heteroscedastic ``scale`` head used
  by the EARA-Conformal pipeline.

The module returns exactly the same triple as v1 —
``(pred_rul, pred_hi, pred_scale)`` — so the trainer, the evaluator, the
uncertainty pipeline and the XAI scripts all work unchanged.

Author: me-intenzo
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head scaled dot-product attention over GRU timesteps.

    Parameters
    ----------
    hidden_size:
        Width of the GRU representation.  Must be divisible by *num_heads*.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout applied to the attention weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, H).

        Returns
        -------
        tuple
            ``(context, attention_weights)`` where *context* has shape
            (B, T, H) and *attention_weights* has shape (B, heads, T, T).
        """

        B, T, H = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = (weights @ v).transpose(1, 2).reshape(B, T, H)
        context = self.out_proj(context)

        return context, weights


class GruAttDegV2(nn.Module):
    """
    GRU + multi-head temporal attention + degradation multi-task model.

    Parameters
    ----------
    input_size : int
        Number of sensor features per timestep.
    hidden_size : int
        GRU hidden size.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability.
    num_heads : int
        Attention heads.  Ignored when *use_attention* is ``False``.
    use_layer_norm : bool
        Use LayerNorm instead of BatchNorm in the shared trunk.
    use_attention : bool
        Disable to fall back to a last-hidden-state-only encoder.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        use_layer_norm: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.encoder_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        # --------------------------------------------------
        # Temporal attention
        # --------------------------------------------------

        if use_attention:
            self.attention = MultiHeadTemporalAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.attn_norm = (
                nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
            )
            fused_size = hidden_size * 2
        else:
            self.attention = None
            self.attn_norm = None
            fused_size = hidden_size

        # --------------------------------------------------
        # Shared trunk
        # --------------------------------------------------

        self.shared = nn.Sequential(
            nn.Linear(fused_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # Task heads
        # --------------------------------------------------

        self.rul_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Degradation head — sigmoid keeps HI in (0, 1)
        self.hi_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Heteroscedastic scale head for EARA-Conformal.
        self.scale_head = nn.Linear(hidden_size, 1)

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
            Shape (batch, sequence_length, input_size).

        Returns
        -------
        tuple
            ``(pred_rul, pred_hi, pred_scale)`` each of shape (batch,).
        """

        gru_out, hidden = self.gru(x)                 # (B, T, H)
        gru_out = self.encoder_norm(gru_out)          # (B, T, H)

        last_hidden = hidden[-1]                      # (B, H)

        if self.attention is not None:
            context, _weights = self.attention(gru_out)
            if getattr(self, "attn_norm", None) is not None:
                context = self.attn_norm(gru_out + context)   # residual
            else:
                context = gru_out + context
            features = torch.cat([context[:, -1, :], last_hidden], dim=-1)
        else:
            features = last_hidden

        features = self.shared(features)               # (B, H)

        pred_rul = self.rul_head(features).squeeze(-1)
        pred_hi = self.hi_head(features).squeeze(-1)
        pred_scale = F.softplus(self.scale_head(features)).squeeze(-1)

        return pred_rul, pred_hi, pred_scale


# ----------------------------------------------------------
# Standalone Test
# ----------------------------------------------------------

if __name__ == "__main__":

    model = GruAttDegV2(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        num_heads=4,
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
