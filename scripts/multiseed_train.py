"""Multi-seed training harness for certification-grade results.

Why this script exists
----------------------
A single training run cannot certify a claim.  The headline point-prediction
numbers of this project used to come from *one* checkpoint per subset
(``gru_att_deg_v2`` at seed 42).  A reviewer who asks "how much of the
reported margin is seed noise?" needs an answer, and the honest answer
requires the deliverable model itself to be trained several times.

This harness trains the **deliverable architecture** (``gru_att_deg_v2`` by
default) across a grid of seeds and subsets, and records a manifest that
makes every run auditable:

* the seed, the resolved config, and the git commit;
* the SHA-256 of every checkpoint, so a reported number can be traced to the
  exact weights that produced it;
* the determinism configuration that was active during the run.

There is **no proxy model** anywhere in this harness.  Every checkpoint it
produces is a full ``gru_att_deg_v2`` checkpoint that the rest of the
pipeline (uncertainty, XAI, decision, HITL) can consume directly.

Workflow
--------
    for subset in subsets:
        for seed in seeds:
            train   -> outputs/checkpoints/<model>_ms<seed>/<subset>/best_model.pt
            evaluate-> outputs/results/<subset>_<model>_ms<seed>_metrics.json
    write manifest -> results/multiseed/training_manifest.json

Tag convention
--------------
Runs are tagged ``ms<seed>`` (e.g. ``ms42``), so the checkpoint directory is
``gru_att_deg_v2_ms42``.  The ``ms`` namespace is deliberate: it keeps the
multi-seed family separate from the historical single-run tags (``seed7``,
``noaug``, ``mild``) and makes it obvious which checkpoints form the
certified set.

Usage
-----
    python scripts/multiseed_train.py --model gru_att_deg_v2 --subsets all
    python scripts/multiseed_train.py --seeds 42 43 44 45 46 --subsets all
    python scripts/multiseed_train.py --subsets all --force

author: me-intenzo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.evaluate as evaluate_module
import scripts.train as train_module
from src.utils.config import ALL_SUBSETS, load_config
from src.utils.logger import get_script_logger

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
MANIFEST_DIR = PROJECT_ROOT / "results" / "multiseed"

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_TAG_PREFIX = "ms"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, or an empty string if absent."""

    if not path.exists():
        return ""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the current git commit hash, or an empty string."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
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


def checkpoint_path(model_name: str, subset: str, tag: str) -> Path:
    """Checkpoint path for a (model, subset, tag) triple."""

    dir_name = model_name + (f"_{tag}" if tag else "")
    return CHECKPOINT_DIR / dir_name / subset / "best_model.pt"


def metrics_path(model_name: str, subset: str, tag: str) -> Path:
    """Metrics JSON path for a (model, subset, tag) triple."""

    dir_name = model_name + (f"_{tag}" if tag else "")
    return RESULTS_DIR / f"{subset}_{dir_name}_metrics.json"


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_one(
    model_name: str,
    subset: str,
    seed: int,
    tag: str,
    overrides: dict[str, Any],
    force: bool,
    log,
) -> dict[str, Any]:
    """Train and evaluate one (subset, seed) combination.

    Returns a manifest entry describing the run.
    """

    ckpt_path = checkpoint_path(model_name, subset, tag)
    metrics_file = metrics_path(model_name, subset, tag)

    if not force and ckpt_path.exists() and metrics_file.exists():
        log.info("[skip] %s / seed=%d (checkpoint + metrics already exist)", subset, seed)
        return {
            "subset": subset,
            "seed": seed,
            "tag": tag,
            "status": "skipped",
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "metrics": str(metrics_file.relative_to(PROJECT_ROOT)),
        }

    # --------------------------------------------------------------
    # Resolve the config for this subset
    # --------------------------------------------------------------

    config = load_config(
        path=None,
        overrides=overrides,
        subset=subset,
    )
    config["random_seed"] = int(seed)

    log.info("[run ] %s / seed=%d", subset, seed)

    started = time.perf_counter()

    # ``scripts.train.run`` is the single training entry point, so the
    # multi-seed family is trained by exactly the same code path as a
    # normal run.  No separate implementation exists.
    train_module.run(
        model_name=model_name,
        subset=subset,
        config=config,
        tag=tag,
    )

    # Evaluate the checkpoint that was just trained.
    evaluate_module.run(
        model_name=model_name,
        subset=subset,
        tag=tag,
    )

    elapsed = time.perf_counter() - started

    entry: dict[str, Any] = {
        "subset": subset,
        "seed": int(seed),
        "tag": tag,
        "status": "trained",
        "seconds": round(elapsed, 2),
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "metrics": str(metrics_file.relative_to(PROJECT_ROOT)),
    }

    # Record the run-level metadata stored inside the checkpoint so the
    # manifest mirrors what the checkpoint itself claims.
    try:
        import torch

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        entry["epoch"] = checkpoint.get("epoch")
        entry["run_metadata"] = checkpoint.get("run_metadata")
        entry["input_size"] = checkpoint.get("input_size")
        entry["window_size"] = checkpoint.get("window_size")
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not read checkpoint metadata for %s: %s", ckpt_path, exc)

    return entry


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=str,
        default="gru_att_deg_v2",
        help="Architecture to train across seeds.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all"],
        help="C-MAPSS subsets to cover (FD001..FD004) or 'all'.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to train.",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default=DEFAULT_TAG_PREFIX,
        help="Prefix of the per-run tag (default 'ms' -> tag 'ms42').",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help=(
            "Config overrides applied to every run, as comma-separated "
            "key=value pairs (e.g. 'epochs=150')."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when the checkpoint and metrics already exist.",
    )
    parser.add_argument(
        "--allow-single-seed",
        action="store_true",
        help=(
            "Allow a one-seed run (smoke tests only). A single-seed run is "
            "NOT a multi-seed certification and is recorded as such in the "
            "manifest."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subsets = list(ALL_SUBSETS) if "all" in args.subsets else list(args.subsets)
    unknown = set(subsets) - set(ALL_SUBSETS)
    if unknown:
        raise SystemExit(f"Unknown subset(s): {sorted(unknown)}")

    seeds = sorted({int(seed) for seed in args.seeds})
    if len(seeds) < 2 and not args.allow_single_seed:
        raise SystemExit(
            "At least two distinct seeds are required for a multi-seed "
            "certification; a single seed is not multi-seed."
        )

    overrides: dict[str, Any] = {}
    if args.overrides:
        overrides = train_module.parse_overrides(args.overrides)

    log = get_script_logger("multiseed", "multiseed_train")

    log.info("=" * 68)
    log.info("MULTI-SEED TRAINING HARNESS")
    log.info("=" * 68)
    log.info("Model   : %s", args.model)
    log.info("Subsets : %s", ", ".join(subsets))
    log.info("Seeds   : %s", ", ".join(str(seed) for seed in seeds))
    log.info("Commit  : %s", git_commit() or "(unknown)")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()

    for subset in subsets:
        for seed in seeds:
            tag = f"{args.tag_prefix}{seed}"
            runs.append(
                train_one(
                    model_name=args.model,
                    subset=subset,
                    seed=seed,
                    tag=tag,
                    overrides=overrides,
                    force=args.force,
                    log=log,
                )
            )

    elapsed = time.perf_counter() - started

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": args.model,
        "subsets": subsets,
        "seeds": seeds,
        "tag_prefix": args.tag_prefix,
        "tag_pattern": f"{args.tag_prefix}<seed>",
        "overrides": overrides,
        "total_seconds": round(elapsed, 2),
        "device": str(train_module.get_device()),
        "note": (
            "Every run in this manifest is a full checkpoint of the named "
            "architecture trained by scripts/train.py. No proxy or surrogate "
            "model is involved."
        ),
        "is_multi_seed_certification": len(seeds) >= 2,
        "runs": runs,
    }

    manifest_path = MANIFEST_DIR / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    log.info("=" * 68)
    log.info("Trained %d run(s) in %.1f s", len(runs), elapsed)
    log.info("Manifest written to: %s", manifest_path)
    log.info("=" * 68)


if __name__ == "__main__":
    main()
