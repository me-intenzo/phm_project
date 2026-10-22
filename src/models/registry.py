"""
registry.py

Single source of truth for building **and loading** prognostics models.

``scripts/train.py`` and ``scripts/evaluate.py`` previously each carried
their own copy of a ``build_model`` function with hard-coded
hyper-parameters.  That was a latent bug: training with a different
hidden size than evaluation rebuilt would load a mismatched
state-dict.  The registry below removes the duplication and lets a
checkpoint record the exact architecture it was trained with.

The same module also owns :func:`load_model_from_checkpoint`.  Three
scripts used to carry their own ``load_model`` implementation
(``scripts/evaluate.py``, ``scripts/explain.py`` and
``scripts/uncertainty.py``); they had already drifted apart in how they
resolved the checkpoint path and how they handled a checkpoint without
architecture metadata.  There is one loader now.

Author: me-intenzo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.paths import CHECKPOINT_DIR, checkpoint_path

from .checkpoint import load_state_dict_compat
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


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def resolve_checkpoint_path(
    model_name: str,
    subset: str = "FD001",
    tag: str = "",
    root: Path | None = None,
) -> Path:
    """Return the checkpoint path of a (model, subset, tag) run.

    When *tag* is empty, the historical untagged layout
    (``<root>/<model>/<subset>/best_model.pt``) is tried first and the
    older flat layout (``<root>/<model>/best_model.pt``) is used as a
    fallback.  The fallback exists because early runs wrote their
    checkpoint without the subset directory.
    """

    base = root if root is not None else CHECKPOINT_DIR

    path = checkpoint_path(model_name, subset, tag, root=base)

    if path.exists() or tag:
        return path

    legacy = base / model_name / "best_model.pt"
    if legacy.exists():
        return legacy

    return path


def load_model_from_checkpoint(
    model_name: str,
    input_size: int,
    device: torch.device | str = "cpu",
    subset: str = "FD001",
    tag: str = "",
    *,
    root: Path | None = None,
    strict_input_size: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Rebuild the model described by a checkpoint and load its weights.

    The architecture is taken from the ``model_config`` stored inside the
    checkpoint, so a model trained with a per-subset YAML override (for
    example FD002/FD004, which train with ``hidden_size: 160``) is
    rebuilt correctly.  A checkpoint without architecture metadata falls
    back to the default hyper-parameters.

    Parameters
    ----------
    model_name:
        Architecture to load.  Only used as a fallback when the
        checkpoint does not record its own architecture.
    input_size:
        Number of input features per timestep.  Used as a fallback when
        the checkpoint does not record its own input size.
    device:
        Device the model is moved to.
    subset:
        Dataset subset the checkpoint belongs to.
    tag:
        Run tag of a multi-seed checkpoint (e.g. ``"ms42"``).
    root:
        Optional override of the checkpoint root directory.
    strict_input_size:
        Raise when the checkpoint's input size does not match
        *input_size*.  The conformal pipeline needs this: a mismatch
        would silently calibrate a model against the wrong features.

    Returns
    -------
    tuple
        ``(model, checkpoint, arch)``.
    """

    path = resolve_checkpoint_path(model_name, subset, tag, root=root)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train the model first using:\n"
            f"python scripts/train.py --model {model_name} --subset {subset}"
            + (f" --tag {tag}" if tag else "")
        )

    device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    checkpoint_input_size = int(checkpoint.get("input_size", input_size))

    if strict_input_size and checkpoint_input_size != int(input_size):
        raise RuntimeError(
            f"Checkpoint input_size={checkpoint_input_size} does not match "
            f"data input_size={input_size}. "
            f"Re-train {model_name} for {subset} using:\n"
            f"python scripts/train.py --model {model_name} --subset {subset}"
        )

    saved_config = dict(checkpoint.get("model_config") or {})
    arch = str(saved_config.pop("arch", model_name))

    if saved_config:
        # ``build_model`` ignores unknown keys, so a flat training
        # config can be splatted straight in.
        model = build_model(
            arch,
            input_size=checkpoint_input_size,
            **saved_config,
        )
    else:
        # Legacy checkpoint: no architecture metadata was stored.
        model = build_model(arch, input_size=checkpoint_input_size)

    load_state_dict_compat(
        model,
        checkpoint["model_state_dict"],
        device=device,
    )

    model.to(device)
    model.eval()

    return model, checkpoint, arch


__all__ = [
    "ALL_MODELS",
    "ARCHS",
    "build_model",
    "load_model_from_checkpoint",
    "model_config",
    "resolve_checkpoint_path",
]
