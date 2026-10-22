"""
src/utils/provenance.py

Run identity for generated artifacts.

Every artifact this repository writes is derived from a **checkpoint**.
Two artifacts that are compared against each other — the predictions from
``scripts/evaluate.py`` and the intervals from ``scripts/uncertainty.py`` —
are only comparable if they came from the *same* weights.  Before this
module existed nothing recorded which checkpoint produced which file, so
regenerating one side and not the other silently produced a comparison
between two different models.

The stamp is embedded **inside** the ``.npz`` rather than written as a
sidecar file, so it can never drift away from the data it describes: an
artifact and its provenance are written and moved together.

Format
------
The stamp is stored under the ``meta_json`` key as a 0-dimensional numpy
unicode array holding a JSON document.  A unicode array (not an object
array) means ``np.load`` can read it **without** ``allow_pickle=True``.

Statuses
--------
:func:`compare_provenance` returns one of:

``"ok"``
    Both artifacts declare the same checkpoint.
``"mismatch"``
    Both artifacts declare a checkpoint, but different ones.  The
    comparison is not trustworthy.
``"unknown"``
    At least one artifact predates this module and carries no stamp.
    Nothing can be asserted, but nothing is wrong either.

author: me-intenzo
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.paths import PROJECT_ROOT

#: Key under which the provenance stamp is stored inside an ``.npz``.
META_KEY = "meta_json"

#: Provenance statuses returned by :func:`compare_provenance`.
STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_UNKNOWN = "unknown"


# ------------------------------------------------------------------
# Digests and repository state
# ------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, or an empty string if absent."""

    path = Path(path)
    if not path.exists():
        return ""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path | None = None) -> str:
    """Return the current git commit hash, or an empty string."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root is not None else str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if result.returncode != 0:
        return ""

    return result.stdout.strip()


# ------------------------------------------------------------------
# Building the stamp
# ------------------------------------------------------------------

def build_provenance(
    checkpoint_path: str | Path | None = None,
    checkpoint: dict[str, Any] | None = None,
    subset: str = "",
    model: str = "",
    tag: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance document for one artifact.

    Parameters
    ----------
    checkpoint_path:
        Path of the checkpoint the artifact was produced from.  Its
        SHA-256 digest is the artifact's identity.
    checkpoint:
        The loaded checkpoint, when available.  The seed and epoch are
        read from it so the stamp carries the run's identity even when
        the checkpoint file has since been overwritten.
    subset, model, tag:
        The run coordinates.
    extra:
        Additional key/value pairs to merge into the document.

    Returns
    -------
    dict
        A JSON-serialisable document.
    """

    meta: dict[str, Any] = {
        "subset": subset,
        "model": model,
        "tag": tag,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else "",
        "git_commit": git_commit(),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }

    if checkpoint is not None:
        metadata = checkpoint.get("run_metadata") or {}
        seed = checkpoint.get("seed")
        if seed is None:
            seed = metadata.get("seed")
        if seed is None:
            seed = (checkpoint.get("model_config") or {}).get("random_seed")
        meta["seed"] = int(seed) if seed is not None else None
        meta["epoch"] = checkpoint.get("epoch")
        meta["arch"] = (checkpoint.get("model_config") or {}).get("arch")

    if extra:
        meta.update(extra)

    return meta


def pack_provenance(meta: dict[str, Any]) -> np.ndarray:
    """Serialise a provenance document into a 0-d numpy unicode array."""

    return np.array(json.dumps(meta, sort_keys=True, default=str))


def unpack_provenance(npz: Any) -> dict[str, Any] | None:
    """Read the provenance stamp out of an open ``.npz``.

    Returns ``None`` when the artifact carries no stamp (it predates
    this module) or when the stamp cannot be parsed.
    """

    try:
        files = set(npz.files)
    except AttributeError:
        return None

    if META_KEY not in files:
        return None

    try:
        raw = npz[META_KEY]
        text = raw.item() if hasattr(raw, "item") else str(raw)
        document = json.loads(text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None

    if not isinstance(document, dict):
        return None

    return document


# ------------------------------------------------------------------
# Comparing two stamps
# ------------------------------------------------------------------

def compare_provenance(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare the checkpoint identities of two artifacts.

    Parameters
    ----------
    left, right:
        Provenance documents as returned by :func:`unpack_provenance`,
        or ``None`` for an artifact with no stamp.

    Returns
    -------
    dict
        ``{"status": ..., "left_sha256": ..., "right_sha256": ...}``
        where ``status`` is ``"ok"``, ``"mismatch"`` or ``"unknown"``.
    """

    left_sha = (left or {}).get("checkpoint_sha256") or ""
    right_sha = (right or {}).get("checkpoint_sha256") or ""

    if not left_sha or not right_sha:
        return {
            "status": STATUS_UNKNOWN,
            "left_sha256": left_sha,
            "right_sha256": right_sha,
        }

    if left_sha == right_sha:
        return {
            "status": STATUS_OK,
            "left_sha256": left_sha,
            "right_sha256": right_sha,
        }

    return {
        "status": STATUS_MISMATCH,
        "left_sha256": left_sha,
        "right_sha256": right_sha,
    }


__all__ = [
    "META_KEY",
    "STATUS_MISMATCH",
    "STATUS_OK",
    "STATUS_UNKNOWN",
    "build_provenance",
    "compare_provenance",
    "git_commit",
    "pack_provenance",
    "sha256_file",
    "unpack_provenance",
]
