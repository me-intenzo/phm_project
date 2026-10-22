"""
Integrated Gradients for GRU-based RUL/HI explanations.

Input:
    (batch, timesteps, features)

Output:
    attribution tensor with the same shape as the input.

The module also owns the *temporal gradient relevance* attribution: both
attributions need the same guarded gradient path, so they share one
implementation instead of living in two modules with different (and
inconsistent) model-mode handling.

author: me-intenzo
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def _eval_guard(model) -> Iterator[None]:
    """
    Run the enclosed block with the model in eval mode.

    ``train()`` would (a) enable Dropout, making the attribution
    non-deterministic across calls, and (b) update the BatchNorm running
    statistics, permanently corrupting the model that is being explained
    (and every downstream stage that reuses it).  The caller's mode is
    restored on exit, so explaining a model never leaves it in a different
    state than it was found in.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


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

    # `torch.enable_grad()` keeps the function usable from a `no_grad` block.
    #
    # cuDNN has to be disabled for the backward pass: its fused RNN kernels
    # only implement the backward pass in *training* mode, so with the model
    # in eval mode the gradient computation raises
    #   "cudnn RNN backward can only be called in training mode".
    # PyTorch's native RNN kernels support eval-mode backward and give the
    # same gradients.
    with _eval_guard(model):
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
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


def temporal_gradient_relevance(
    model,
    x: torch.Tensor,
    target: str = "rul",
) -> torch.Tensor:
    """
    Timestep relevance derived from the input gradients.

    Shares the eval-guarded gradient path with ``integrated_gradients``, so
    the result is deterministic and cannot corrupt the BatchNorm running
    statistics of the explained model.

    Returns
    -------
    torch.Tensor  shape (batch, timesteps)
    """
    with _eval_guard(model):
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            x = x.detach().clone().requires_grad_(True)

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
