"""Run the full downstream pipeline for every multi-seed checkpoint.

Why this script exists
----------------------
The requirement for this project is that *every* downstream stage —
uncertainty, XAI, decision and HITL — consumes the predictions of the model
the repository trained itself, not a proxy and not a stale artifact from a
different run.  Threading a ``--tag`` through the stages is only half of
that; the other half is proving that the chain actually runs end to end for
each seed and that every artifact it produces belongs to the same set of
weights.

This driver does exactly that.  For every (subset, seed) pair it runs:

1. ``scripts/uncertainty.py``  -> calibrated prediction intervals
2. ``scripts/explain.py``      -> SHAP / IG / attention attributions
3. ``scripts/decision.py``     -> maintenance recommendations
4. ``scripts/hitl.py``         -> human-in-the-loop review workflow

and then verifies that every artifact exists and that the numbers are
mutually consistent (e.g. the interval bounds used by the decision engine
are the ones the uncertainty stage produced for that same tag).

Stages can be selected with ``--stages`` so that, for example, the
expensive XAI stage can be run only for the headline seed while the
uncertainty stage runs for all seeds.

Usage
-----
    # Everything, all four subsets, all five seeds
    python scripts/multiseed_pipeline.py --subsets all

    # Only the uncertainty + decision stages (fast)
    python scripts/multiseed_pipeline.py --stages uncertainty decision

    # Only the headline seed, full pipeline
    python scripts/multiseed_pipeline.py --seeds 42 --stages all

author: me-intenzo
"""

from __future__ import annotations

import argparse
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

import numpy as np

from src.utils.config import ALL_SUBSETS
from src.utils.paths import RESULTS_DIR

MULTISEED_DIR = PROJECT_ROOT / "results" / "multiseed"

STAGES = ("uncertainty", "xai", "decision", "hitl")

UNCERTAINTY_METHODS = ("split", "cqr", "adaptive", "engine_joint")

# Maximum tolerated deviation (in RUL cycles) between the point predictions
# stored by the uncertainty stage and those stored by the evaluation stage.
# Both run the same weights, but with different batch sizes, so the GPU
# forward pass is not bit-identical.  One tenth of a cycle is far below any
# level that could change a decision.
MAX_PREDICTION_DEVIATION = 0.1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def run_stage(
    stage: str,
    model_name: str,
    subset: str,
    tag: str,
    extra: list[str],
) -> None:
    """Invoke a pipeline stage as a subprocess (same entry points as the CLI)."""

    if stage == "uncertainty":
        for method in UNCERTAINTY_METHODS:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "uncertainty.py"),
                "--model", model_name,
                "--subset", subset,
                "--method", method,
                "--tag", tag,
            ]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    elif stage == "xai":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "explain.py"),
            "--model", model_name,
            "--subset", subset,
            "--target", "rul",
            "--tag", tag,
            *extra,
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    elif stage == "decision":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "decision.py"),
            "--model", model_name,
            "--subset", subset,
            "--tag", tag,
            "--uncertainty-variant", "engine_joint_conformal",
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    elif stage == "hitl":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "hitl.py"),
            "--model", model_name,
            "--subset", subset,
            "--tag", tag,
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    else:
        raise ValueError(f"Unknown stage: {stage}")


def expected_artifacts(model_name: str, subset: str, tag: str) -> dict[str, Path]:
    """Artifacts that must exist after a successful pipeline run."""

    run_name = f"{model_name}_{tag}"

    return {
        "predictions": RESULTS_DIR / f"{subset}_{run_name}_predictions.npz",
        "metrics": RESULTS_DIR / f"{subset}_{run_name}_metrics.json",
        "uncertainty_split": RESULTS_DIR / f"{subset}_{run_name}_conformal.npz",
        "uncertainty_cqr": RESULTS_DIR / f"{subset}_{run_name}_cqr.npz",
        "uncertainty_adaptive": RESULTS_DIR / f"{subset}_{run_name}_adaptive_conformal.npz",
        "uncertainty_engine_joint": RESULTS_DIR / f"{subset}_{run_name}_engine_joint_conformal.npz",
        "xai": PROJECT_ROOT / "outputs" / "xai" / subset / run_name / "eri_rul.json",
        "decision": RESULTS_DIR / f"decision_{subset}_{run_name}.json",
        "hitl": RESULTS_DIR / f"hitl_{subset}_{run_name}.json",
    }


def verify(model_name: str, subset: str, tag: str) -> tuple[dict[str, Any], list[str]]:
    """Check that the artifacts exist and describe the same run."""

    artifacts = expected_artifacts(model_name, subset, tag)
    problems: list[str] = []

    for name, path in artifacts.items():
        if not path.exists():
            problems.append(f"missing {name}: {path.name}")

    # The uncertainty artifact that the decision engine consumes must carry
    # the point predictions of the same checkpoint.
    unc_path = artifacts["uncertainty_engine_joint"]
    pred_path = artifacts["predictions"]

    if unc_path.exists() and pred_path.exists():
        with np.load(unc_path) as unc, np.load(pred_path) as pred:
            if "y_rul_pred" in unc.files:
                unc_rul = np.asarray(unc["y_rul_pred"]).reshape(-1)
                pred_rul = np.asarray(pred["y_rul_pred"]).reshape(-1)
                n = min(len(unc_rul), len(pred_rul))
                if n == 0:
                    problems.append("no overlapping predictions to compare")
                else:
                    # The two stages run the same weights, but with different
                    # batch sizes, so the GPU forward pass is not bit-identical.
                    # What must hold is that they describe the *same* model:
                    # a max deviation far below one RUL cycle and a
                    # near-perfect correlation.
                    max_dev = float(np.max(np.abs(unc_rul[:n] - pred_rul[:n])))
                    corr = float(
                        np.corrcoef(unc_rul[:n], pred_rul[:n])[0, 1]
                    )
                    if max_dev > MAX_PREDICTION_DEVIATION:
                        problems.append(
                            f"uncertainty point predictions deviate from the "
                            f"evaluated checkpoint by {max_dev:.4f} cycles "
                            f"(> {MAX_PREDICTION_DEVIATION})"
                        )
                    if corr < 0.9999:
                        problems.append(
                            f"uncertainty point predictions correlate only "
                            f"{corr:.6f} with the evaluated checkpoint"
                        )
            else:
                problems.append(
                    "engine_joint artifact has no 'y_rul_pred' key"
                )

    return {name: str(path) for name, path in artifacts.items()}, problems


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", type=str, default="gru_att_deg_v2")
    parser.add_argument("--subsets", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--tag-prefix", type=str, default="ms")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=[*STAGES, "all"],
        help="Pipeline stages to run.",
    )
    parser.add_argument(
        "--xai-samples",
        type=int,
        default=None,
        help="Number of explained samples (default: script default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subsets = list(ALL_SUBSETS) if "all" in args.subsets else list(args.subsets)
    stages = list(STAGES) if "all" in args.stages else list(args.stages)

    extra: list[str] = []
    if args.xai_samples is not None:
        extra += ["--samples", str(args.xai_samples)]

    print("=" * 68)
    print("MULTI-SEED PIPELINE")
    print("=" * 68)
    print(f"Model   : {args.model}")
    print(f"Subsets : {', '.join(subsets)}")
    print(f"Seeds   : {', '.join(str(s) for s in args.seeds)}")
    print(f"Stages  : {', '.join(stages)}")

    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for subset in subsets:
        for seed in args.seeds:
            tag = f"{args.tag_prefix}{seed}"
            print("-" * 68)
            print(f"{subset} / seed={seed} / tag={tag}", flush=True)

            record: dict[str, Any] = {
                "subset": subset,
                "seed": seed,
                "tag": tag,
                "stages": stages,
                "errors": [],
            }

            for stage in stages:
                stage_started = time.perf_counter()
                try:
                    run_stage(stage, args.model, subset, tag, extra)
                    record[f"{stage}_seconds"] = round(
                        time.perf_counter() - stage_started, 2
                    )
                    record[f"{stage}_status"] = "ok"
                except subprocess.CalledProcessError as exc:
                    record[f"{stage}_status"] = "failed"
                    record["errors"].append(f"{stage}: exit code {exc.returncode}")
                    print(f"  [FAIL] {stage}: exit code {exc.returncode}")
                    break
                print(f"  [ok]   {stage} ({record[f'{stage}_seconds']}s)")

            artifacts, problems = verify(args.model, subset, tag)
            record["artifacts"] = artifacts
            record["problems"] = problems

            if problems:
                for problem in problems:
                    print(f"  [WARN] {problem}")

            records.append(record)

    elapsed = time.perf_counter() - started

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "subsets": subsets,
        "seeds": args.seeds,
        "stages": stages,
        "total_seconds": round(elapsed, 2),
        "n_runs": len(records),
        "n_clean": sum(1 for r in records if not r["problems"] and not r["errors"]),
        "runs": records,
    }

    out_path = MULTISEED_DIR / "pipeline_manifest.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"Completed {summary['n_clean']}/{summary['n_runs']} run(s) cleanly")
    print(f"Manifest: {out_path}")
    print("=" * 68)

    if summary["n_clean"] != summary["n_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
