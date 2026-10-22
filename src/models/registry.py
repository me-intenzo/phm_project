"""
registry.py

Single source of truth for building prognostics models.

``scripts/train.py`` and ``scripts/evaluate.py`` previously each carried
their own copy of a ``build_model`` function with hard-coded
hyper-parameters.  That was a latent bug: training with a different
hidden size than evaluation rebuilt would load a mismatched
state-dict.  The registry below removes the duplication and lets a
checkpoint record the exact architecture it was trained with.

Author: me-intenzo
"""

from __future__ import annotations

from typing import Any

from .gru import GRUPrognosticsModel
from .gru_att_deg import GruAttDeg
from .gru_att_deg_v2 import GruAttDegV2
from .hybrid import HybridPrognosticsModel
from .lstm import LSTMPrognosticsModel
from .transformer import TransformerPrognosticsModel

#: name -> (class, extra constructor kwargs)
ARCHS: dict[str, tuple[type, dict[str, Any]]] = {
    "lstm": (LSTMPrognosticsModel, {}),
    "gru": (GRUPrognosticsModel, {}),
    "transformer": (TransformerPrognosticsModel, {}),
    "hybrid": (HybridPrognosticsModel, {}),
    "gru_att_deg": (GruAttDeg, {}),
    "gru_att_deg_v2": (GruAttDegV2, {}),
}

#: Every architecture name, in a stable order.
ALL_MODELS: list[str] = list(ARCHS)


def build_model(
    model_name: str,
    input_size: int,
    **hyperparameters: Any,
):
    """Instantiate *model_name*.

    Parameters
    ----------
    model_name:
        Key of :data:`ARCHS` (e.g. ``"gru_att_deg_v2"``).
    input_size:
        Number of input features per timestep.
    **hyperparameters:
        Architecture hyper-parameters.  Unknown keys are ignored so a
        single flat config mapping can be splatted in.

    Raises
    ------
    ValueError
        If *model_name* is not a known architecture.
    """

    if model_name not in ARCHS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Available: {sorted(ARCHS)}"
        )

    model_cls, extra = ARCHS[model_name]

    kwargs: dict[str, Any] = {
        "input_size": input_size,
        "hidden_size": hyperparameters.get("hidden_size", 128),
        "num_layers": hyperparameters.get("num_layers", 2),
        "dropout": hyperparameters.get("dropout", 0.3),
    }

    if model_cls is HybridPrognosticsModel:
        kwargs = {
            "input_size": input_size,
            "hidden_size": hyperparameters.get("hidden_size", 128),
            "num_gru_layers": hyperparameters.get("num_layers", 2),
            "num_transformer_layers": hyperparameters.get("num_layers", 2),
            "num_heads": hyperparameters.get("num_heads", 4),
            "dropout": hyperparameters.get("dropout", 0.3),
        }
    elif model_cls is TransformerPrognosticsModel:
        kwargs["num_heads"] = hyperparameters.get("num_heads", 4)
    elif model_cls is GruAttDegV2:
        kwargs["num_heads"] = hyperparameters.get("num_heads", 4)
        kwargs["use_layer_norm"] = hyperparameters.get("use_layer_norm", True)
        kwargs["use_attention"] = hyperparameters.get("use_attention", True)

    kwargs.update(extra)

    return model_cls(**kwargs)


def model_config(model_name: str, **hyperparameters: Any) -> dict[str, Any]:
    """Return the architecture description stored inside a checkpoint.

    The mapping is what :func:`build_model` needs to rebuild the exact
    same module later, so a checkpoint can always be re-evaluated with
    the configuration it was trained with.
    """

    if model_name not in ARCHS:
        raise ValueError(f"Unsupported model: {model_name}")

    config: dict[str, Any] = {
        "arch": model_name,
        "hidden_size": int(hyperparameters.get("hidden_size", 128)),
        "num_layers": int(hyperparameters.get("num_layers", 2)),
        "dropout": float(hyperparameters.get("dropout", 0.3)),
        "num_heads": int(hyperparameters.get("num_heads", 4)),
    }

    if model_name == "gru_att_deg_v2":
        config["use_layer_norm"] = bool(
            hyperparameters.get("use_layer_norm", True)
        )
        config["use_attention"] = bool(
            hyperparameters.get("use_attention", True)
        )

    return config


__all__ = ["ALL_MODELS", "ARCHS", "build_model", "model_config"]
