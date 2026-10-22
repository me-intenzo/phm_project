"""
src/models/outputs.py

One place that knows how to read a prognostics model's output.

Every architecture in this repository returns the same triple

    ``(pred_rul, pred_hi, pred_scale)``

but the *callers* each had their own unpacking pattern — five different
variants across the XAI modules, the uncertainty pipeline, the trainer
and the evaluation layer.  They disagreed about what to do with a model
that returns a bare tensor or a two-element tuple, and one of them
silently mis-indexed a three-element tuple.

:func:`unpack_model_output` is the single implementation.  It always
returns ``(pred_rul, pred_hi, pred_scale)``; the two optional entries are
``None`` when the model does not produce them.

author: me-intenzo
"""

from __future__ import annotations

from typing import Any

import torch


def unpack_model_output(
    outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Split a model output into ``(pred_rul, pred_hi, pred_scale)``.

    Parameters
    ----------
    outputs:
        Whatever the model returned:

        * a bare tensor — treated as the RUL prediction;
        * a tuple/list of one, two or three tensors — read positionally.

    Returns
    -------
    tuple
        ``(pred_rul, pred_hi, pred_scale)``.  ``pred_hi`` and
        ``pred_scale`` are ``None`` when the model does not provide them.

    Raises
    ------
    TypeError
        If *outputs* is neither a tensor nor a sequence of tensors.
    """

    if isinstance(outputs, torch.Tensor):
        return outputs, None, None

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise TypeError("The model returned an empty output sequence.")
        if len(outputs) == 1:
            return outputs[0], None, None
        if len(outputs) == 2:
            return outputs[0], outputs[1], None
        return outputs[0], outputs[1], outputs[2]

    raise TypeError(
        "Unsupported model output of type "
        f"{type(outputs).__name__}; expected a tensor or a sequence of tensors."
    )


def select_task_output(
    outputs: Any,
    target: str = "rul",
) -> torch.Tensor:
    """Return the RUL or the HI prediction from a model output.

    Parameters
    ----------
    outputs:
        The raw output of a prognostics model.
    target:
        ``"rul"`` or ``"hi"`` (case-insensitive).

    Raises
    ------
    ValueError
        If *target* is not a known task, or the model does not provide the
        requested head.
    """

    pred_rul, pred_hi, _ = unpack_model_output(outputs)

    task = str(target).lower()

    if task == "rul":
        return pred_rul
    if task == "hi":
        if pred_hi is None:
            raise ValueError(
                "The model does not expose a health-index head, so the "
                "'hi' target cannot be explained."
            )
        return pred_hi

    raise ValueError("target must be either 'rul' or 'hi'.")


__all__ = ["select_task_output", "unpack_model_output"]
