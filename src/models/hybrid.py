from __future__ import annotations

import torch
import torch.nn as nn

from .heads import build_task_heads, forward_task_heads
from .layers import PositionalEncoding

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
        scale prediction (learned per-sample error scale; softplus-
        activated to stay positive, matching the other model families)
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
        # MULTI-TASK HEADS (shared layout, see src/models/heads.py)
        # ======================================================

        (
            self.rul_head,
            self.hi_head,
            self.scale_head,
        ) = build_task_heads(
            hidden_size,
            style="linear",
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
        torch.Tensor,
    ]:

        """
        Forward pass.

        Input:
            x = (batch, sequence_length, input_size)

        Returns:
            pred_rul = (batch,)
            pred_hi  = (batch,)
            pred_scale = (batch,); softplus-activated error scale
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

        # RUL / HI / softplus-activated error scale.
        # The return signature (batch,) is shared by every prognostics
        # model in the project.
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

    rul, hi, scale = model(x)

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

    print(
        f"Scale Output Shape: {scale.shape}"
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
