"""
Integrated Gradients for GRU-based RUL/HI explanations.

Input:
    (batch, timesteps, features)

Output:
    attribution tensor with the same shape as the input.

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


def integrated_gradients(
    model,
    x: torch.Tensor,
    target: str = "rul",
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """
    Calculate Integrated Gradients.

    IG_i(x) = (x_i - x'_i) * integral_0^1 dF(x' + a(x-x'))/dx_i da
    """
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    model.train()
    x = x.detach()

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach()
        if baseline.shape != x.shape:
            raise ValueError("baseline and x must have the same shape.")

    device = x.device
    baseline = baseline.to(device)
    delta = x - baseline
    total_gradients = torch.zeros_like(x, device=device)

    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        interpolated = (baseline + alpha * delta)
        interpolated.requires_grad_()

        output = _select_output(model=model, x=interpolated, target=target)
        scalar_output = output.sum()

        gradients = torch.autograd.grad(
            outputs=scalar_output,
            inputs=interpolated,
            retain_graph=False,
            create_graph=False,
        )[0]

        total_gradients += gradients.detach()

    model.eval()
    attributions = delta * (total_gradients / float(steps))
    return attributions


def explain_with_integrated_gradients(
    model,
    data,
    target: str = "rul",
    baseline=None,
    steps: int = 50,
):
    """Public explanation interface."""
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)

    attributions = integrated_gradients(
        model=model,
        x=data,
        target=target,
        baseline=baseline,
        steps=steps,
    )

    return {
        "method": "integrated_gradients",
        "target": target,
        "attributions": attributions,
        "sensor_importance": attributions.abs().mean(dim=(0, 1)),
        "temporal_importance": attributions.abs().mean(dim=(0, 2)),
    }
