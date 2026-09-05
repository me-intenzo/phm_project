"""
Temporal attention-style explanation for the GRU.

Important
---------
This module does NOT modify the trained GRU architecture.
Temporal relevance is derived from input gradients.

author: me-intenzo
"""

from __future__ import annotations

import torch


def _select_output(
    model,
    x: torch.Tensor,
    target: str,
) -> torch.Tensor:
    pred_rul, pred_hi, *_ = model(x)
    target = target.lower()
    if target == "rul":
        return pred_rul
    if target == "hi":
        return pred_hi
    raise ValueError("target must be either 'rul' or 'hi'.")


def temporal_gradient_relevance(
    model,
    x: torch.Tensor,
    target: str = "rul",
) -> torch.Tensor:
    """
    Calculate timestep relevance using input gradients.

    Returns
    -------
    torch.Tensor  shape (batch, timesteps)
    """
    x = x.detach().clone()
    x.requires_grad_(True)

    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        output = _select_output(model=model, x=x, target=target)
        scalar_output = output.sum()
        gradients = torch.autograd.grad(
            outputs=scalar_output,
            inputs=x,
            retain_graph=False,
            create_graph=False,
        )[0]

    relevance = gradients.abs().mean(dim=2)
    denominator = relevance.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (relevance / denominator).detach()


def explain_attention(
    model,
    data,
    target: str = "rul",
):
    """Generate temporal attention-style explanations."""
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)

    device = next(model.parameters()).device
    data = data.to(device)

    relevance = temporal_gradient_relevance(model=model, x=data, target=target)

    return {
        "method": "temporal_attention",
        "target": target,
        "temporal_relevance": relevance,
        "global_temporal_relevance": relevance.mean(dim=0),
    }
