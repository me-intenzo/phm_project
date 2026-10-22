"""
src/models/layers.py

Reusable network layers shared by the prognostics architectures.

The sinusoidal :class:`PositionalEncoding` used to be duplicated verbatim
in ``src/models/transformer.py`` and ``src/models/hybrid.py`` (the two
copies had even drifted apart in their default dropout).  There is one
implementation now, and both architectures import it.

The layer keeps the same module structure and the same buffer name
(``pe``), so a checkpoint written by either architecture still loads with
an unchanged ``state_dict`` key layout.

author: me-intenzo
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

        self.dropout = nn.Dropout(p=dropout)

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

        x = x + self.pe[
            :, :x.size(1), :
        ]

        return self.dropout(x)


__all__ = ["PositionalEncoding"]
