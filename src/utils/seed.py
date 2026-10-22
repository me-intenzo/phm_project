"""Seed and determinism helpers.

This module is the **single source of truth** for reproducibility in the
project.  Every entry point that trains a model, fits a calibration head or
samples a bootstrap must call :func:`set_seed` before it touches an RNG, so
that a run is identified by its seed alone.

Two things are configured:

* the RNG streams — ``random``, ``numpy``, ``torch`` and ``torch.cuda``; and
* the determinism flags that make a CUDA run repeatable bit-for-bit.

The determinism flags are enabled with ``warn_only=True`` so an operator
without a deterministic kernel degrades to a warning instead of aborting the
run.  ``set_seed`` returns a dictionary describing exactly what was applied;
callers persist that dictionary in the run manifest so a reviewer can verify
that the reported numbers came from a deterministic configuration.

Note that this module used to seed only ``random`` and ``numpy`` and was
never imported by the training scripts.  That is fixed: ``scripts/train.py``
and the multi-seed harnesses now import :func:`set_seed` from here.

author: me-intenzo
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

DEFAULT_SEED: int = 42


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------

def enable_determinism() -> dict[str, Any]:
    """Enable deterministic kernels and report what was applied.

    Returns
    -------
    dict
        A machine-readable description of the determinism configuration.
    """

    # cuBLAS needs this set *before* the first CUDA context is created for
    # deterministic GEMMs; ``setdefault`` keeps a caller's choice intact.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32 changes results run-to-run across GPU architectures, which is
    # exactly the kind of drift a multi-seed certification must avoid.
    for backend in (torch.backends.cuda, torch.backends.cudnn):
        matmul = getattr(backend, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False
        if hasattr(backend, "allow_tf32"):
            try:
                backend.allow_tf32 = False
            except (AttributeError, RuntimeError):  # pragma: no cover
                pass

    return {
        "torch_use_deterministic_algorithms": True,
        "torch_deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32_allowed": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


# ------------------------------------------------------------------
# Seeding
# ------------------------------------------------------------------

def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG the project uses and configure determinism.

    Parameters
    ----------
    seed:
        The seed.  It drives weight initialisation, the engine-wise
        train/validation split, dataloader shuffling and window
        augmentation, so a seed identifies a complete training run.
    deterministic:
        When ``True`` (the default) the torch/CUDA determinism flags are
        enabled as well.

    Returns
    -------
    dict
        The applied configuration, suitable for persisting in a manifest.
    """

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    settings: dict[str, Any] = {"seed": seed, "deterministic": bool(deterministic)}

    if deterministic:
        settings.update(enable_determinism())

    return settings


def seed_worker(worker_id: int) -> None:  # pragma: no cover - torch hook
    """Seed a ``DataLoader`` worker from the base seed torch hands it."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for dataloader shuffling."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


__all__ = [
    "DEFAULT_SEED",
    "enable_determinism",
    "seed_worker",
    "set_seed",
    "torch_generator",
]
