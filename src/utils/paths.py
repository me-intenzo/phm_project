"""Artifact path helpers.

Every stage of the pipeline (training, evaluation, uncertainty, XAI,
decision, HITL) writes its artifacts under ``outputs/``.  Once the
multi-seed family exists, the same (model, subset) pair can have several
artifacts — one per seed — so every path has to be able to carry a **tag**.

This module is the single place that knows how a tag is spelled into a
path.  Keeping it here is what lets the whole chain (uncertainty → XAI →
decision → HITL) consume a specific multi-seed checkpoint without any
stage having to know how another stage names its files.

Naming rules
------------
* Checkpoints: ``outputs/checkpoints/<model>[_<tag>]/<subset>/best_model.pt``
* Flat results: ``outputs/results/<subset>_<model>[_<tag>]_<suffix>.<ext>``
* Structured results:
  ``outputs/results/uncertainty/<variant>/<subset>/<model>[_<tag>].<ext>``

The ``[<tag>]`` part is optional: without a tag the historical, untagged
paths are produced unchanged, so every existing artifact and every existing
consumer keeps working.

author: me-intenzo
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

UNCERTAINTY_DIR = RESULTS_DIR / "uncertainty"


# ------------------------------------------------------------------
# Directory / file naming
# ------------------------------------------------------------------

def run_dir_name(model_name: str, tag: str = "") -> str:
    """Directory name of a run: ``<model>`` or ``<model>_<tag>``."""

    return f"{model_name}_{tag}" if tag else model_name


def flat_stem(subset: str, model_name: str, tag: str = "", suffix: str = "") -> str:
    """Flat artifact stem: ``<subset>_<model>[_<tag>]_<suffix>``."""

    parts = [subset, run_dir_name(model_name, tag)]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def checkpoint_path(
    model_name: str,
    subset: str,
    tag: str = "",
    root: Path | None = None,
) -> Path:
    """Path of the best-model checkpoint for a (model, subset, tag)."""

    base = root if root is not None else CHECKPOINT_DIR
    return base / run_dir_name(model_name, tag) / subset / "best_model.pt"


def results_path(
    subset: str,
    model_name: str,
    tag: str = "",
    suffix: str = "",
    extension: str = "npz",
    root: Path | None = None,
) -> Path:
    """Flat path under ``outputs/results/``."""

    base = root if root is not None else RESULTS_DIR
    return base / f"{flat_stem(subset, model_name, tag, suffix)}.{extension}"


def structured_results_path(
    variant: str,
    subset: str,
    model_name: str,
    tag: str = "",
    extension: str = "npz",
    root: Path | None = None,
) -> Path:
    """Structured path under ``outputs/results/<group>/<variant>/...``."""

    base = root if root is not None else RESULTS_DIR
    return (
        base
        / "uncertainty"
        / variant
        / subset
        / f"{run_dir_name(model_name, tag)}.{extension}"
    )


def resolve_tagged_path(
    candidates: list[Path],
) -> Path | None:
    """Return the first candidate that exists, else ``None``."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


__all__ = [
    "CHECKPOINT_DIR",
    "LOGS_DIR",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "UNCERTAINTY_DIR",
    "checkpoint_path",
    "flat_stem",
    "resolve_tagged_path",
    "results_path",
    "run_dir_name",
    "structured_results_path",
]
