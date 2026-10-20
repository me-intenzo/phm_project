"""
demo.py -- one-command, complete PHM-XAI pipeline.

Runs every stage of the prognostics + uncertainty + XAI + decision chain:

    1. EDA          (optional, FD001 only)   python main.py
    2. Preprocess                            scripts/preprocess.py   --subset <subset>
    3. Uncertainty + seeded training         scripts/uncertainty.py  --model <model> --subset <s> --method eara --seeds ...
    4. Evaluate each seed                   scripts/evaluate.py     --model <model> --subset <subset> --seeds ...
    5. Explain each seed                    scripts/explain.py      --model <model> --subset <subset> --target all --seeds ...
    6. Decision engine (O4)                 scripts/decision.py     --subset <subset> --model <model> --uncertainty-variant eara_conformal --seeds ...
    7. HITL review + policy refinement (O5) scripts/hitl.py         --subset <subset> --model <model>

Each stage shells out to the same CLI entry points documented in the
README, so the demo behaves exactly like running the stages by hand, with
each stage writing its own structured logs under ``outputs/logs/``.

Usage
-----
    python scripts/demo.py                              # full pipeline, FD001 / gru
    python scripts/demo.py --subset all                 # all four C-MAPSS subsets
    python scripts/demo.py --subset FD002 --model lstm
    python scripts/demo.py --eda                        # also run the EDA entry point (FD001)
    python scripts/demo.py --seeds 42 43 44 45 46
    python scripts/demo.py --minimum-safe-rul 10 --maintenance-lead-time 10

The demo aborts with a non-zero exit code on the first failed stage and
finishes by verifying that every expected artifact was produced.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

ALL_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
ALL_MODELS = ("lstm", "gru", "transformer", "hybrid")
DEFAULT_SUBSET = "FD001"
DEFAULT_MODEL = "gru"

UNCERTAINTY_VARIANT = "eara_conformal"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

SEPARATOR = "=" * 72


def py(*names: str) -> list[str]:
    """Command line for running one of the project scripts with this Python."""
    return [sys.executable, str(SCRIPTS.joinpath(*names))]


def run_stage(
    title: str,
    argv: list[str],
    outputs: list[Path],
) -> bool:
    """Run one stage and report success based on its exit code + artifacts.

    Some stages (e.g. the explain/XAI script) can crash on interpreter
    teardown on Windows even after producing all outputs; in that case the
    exit code is negative but the artifacts exist. Success therefore requires
    the artifacts to exist; a non-zero exit code is only fatal when the
    outputs are also missing.
    """
    print(SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)
    print("  $ " + " ".join(str(a) for a in argv))
    print()

    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    result = subprocess.run(argv, cwd=str(ROOT), env=env)

    missing = [str(p) for p in outputs if not p.exists()]
    if missing:
        print()
        print(f"  [FAIL] {title}: exit code {result.returncode}")
        print("         Missing artifact(s):")
        for path in missing:
            print(f"           {path}")
        return False

    if result.returncode != 0:
        print()
        print(f"  [WARN] {title}: exit code {result.returncode} but all "
              f"expected artifacts were produced.")

    print(f"  [OK]   {title}")
    return True


def subset_list(spec: str) -> list[str]:
    return list(ALL_SUBSETS) if spec.lower() == "all" else [spec.upper()]


def _seeded_evaluation_summary(
    results_dir: Path,
    subsets: list[str],
    model: str,
    seeds: list[int],
) -> dict:
    from src.evaluation.metrics import predictive_accuracy

    report = {}
    for subset in subsets:
        per_seed = []
        for seed in seeds:
            path = (
                results_dir / "uncertainty" / model / subset / "eara"
                / f"seed_{seed}" / "results" / f"{subset}_{model}_predictions.npz"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing seed prediction artifact: {path}")
            with np.load(path) as pred:
                per_seed.append(predictive_accuracy(
                    pred["y_rul_true"], pred["y_rul_pred"],
                    pred["y_hi_true"], pred["y_hi_pred"],
                ))

        metrics = {}
        for target in ("rul", "hi"):
            metrics[target] = {}
            for metric in per_seed[0][target]:
                values = [float(row[target][metric]) for row in per_seed]
                metrics[target][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }

        interval_path = (
            results_dir / "uncertainty" / model / subset / "eara" / "summary_mean_std.csv"
        )
        if not interval_path.exists():
            raise FileNotFoundError(f"Missing EARA summary: {interval_path}")
        with interval_path.open(newline="", encoding="utf-8") as handle:
            interval_metrics = list(csv.DictReader(handle))
        report[subset] = {
            "n_seeds": len(seeds),
            "prediction_metrics_mean_std": metrics,
            "eara_interval_metrics_mean_std": interval_metrics,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end PHM-XAI pipeline: EDA (optional) -> preprocess -> "
            "multi-seed EARA-Conformal training/calibration -> evaluation -> explain -> "
            "decision -> HITL review."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subset", type=str, default=DEFAULT_SUBSET,
        help="C-MAPSS subset, or 'all' for FD001-FD004.",
    )
    parser.add_argument(
        "--model", type=str, choices=ALL_MODELS, default=DEFAULT_MODEL,
        help="Model architecture used for every trainable stage. Only the "
             "architectures that run through the full chain are allowed "
             "(gru_att_deg is not wired into the decision engine).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="At least five distinct seeds (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--eda", action="store_true",
        help="Run the exploratory data analysis entry point (main.py) first. "
             "FD001 only; note that main.py also appends a demo expert-"
             "feedback record to outputs/hitl/feedback.jsonl.",
    )

    # Per-stage skip switches (handy when re-running only later stages).
    for stage in ("preprocess", "evaluate", "uncertainty",
                  "explain", "decision", "hitl"):
        parser.add_argument(
            f"--skip-{stage}", action="store_true",
            help=f"Skip the {stage} stage.",
        )

    # Optional operational constraints, forwarded to scripts/decision.py.
    parser.add_argument("--maintenance-window", action="store_true",
                        help="Scheduled maintenance can be performed now.")
    parser.add_argument("--no-maintenance-window", action="store_true",
                        help="No maintenance window is currently available.")
    parser.add_argument("--minimum-safe-rul", type=float, default=None,
                        help="Minimum RUL (cycles) required to defer maintenance.")
    parser.add_argument("--maintenance-lead-time", type=float, default=None,
                        help="Cycles required before maintenance can begin.")
    parser.add_argument("--max-allowable-delay", type=float, default=None,
                        help="Longest wait the operator will accept (cycles).")
    parser.add_argument("--resource-availability", type=str, default=None,
                        choices=["true", "false"],
                        help="Whether crew and parts are available.")

    args = parser.parse_args()
    if len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds requires at least five distinct integers")

    model = args.model.lower()
    subsets = subset_list(args.subset)

    print(SEPARATOR)
    print("  PHM-XAI COMPLETE PIPELINE DEMO")
    print(SEPARATOR)
    print(f"  Subset(s): {', '.join(subsets)}")
    print(f"  Model    : {model}")
    print()

    # ------------------------------------------------------------------
    # Stage 1 - EDA (optional, FD001 only)
    # ------------------------------------------------------------------
    if args.eda and "FD001" in subsets:
        ok = run_stage(
            "EDA",
            [sys.executable, str(ROOT / "main.py")],
            [ROOT / "outputs" / "reports" / "EDA" / "training_summary.csv"],
        )
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2 - Preprocess
    # ------------------------------------------------------------------
    for subset in subsets:
        if not args.skip_preprocess:
            ok = run_stage(
                f"Preprocess {subset}",
                py("preprocess.py") + ["--subset", subset],
                [ROOT / "data" / "processed" / f"{subset}_train_X.npy",
                 ROOT / "data" / "processed" / f"{subset}_test_X.npy",
                 ROOT / "data" / "processed" / f"{subset}_train_engine_ids.npy"],
            )
            if not ok:
                sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 3 - EARA-Conformal (trains each seed and calibrates its intervals)
    # ------------------------------------------------------------------
    for subset in subsets:
        if not args.skip_uncertainty:
            ok = run_stage(
                f"Uncertainty (EARA-Conformal) {model} on {subset}",
                py("uncertainty.py")
                + ["--model", model, "--subset", subset, "--method", "eara",
                   "--seeds", *[str(seed) for seed in args.seeds]],
                [ROOT / "outputs" / "results" / "uncertainty" / model / subset
                 / "eara" / "run_config.json",
                 ROOT / "outputs" / "results" / "uncertainty" / model / subset
                 / "eara" / "summary_mean_std.csv"],
            )
            if not ok:
                sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 4 - Evaluate each seed's checkpoint
    # ------------------------------------------------------------------
    for subset in subsets:
        if not args.skip_evaluate:
            prediction_artifacts = [
                ROOT / "outputs" / "results" / "uncertainty" / model / subset
                / "eara" / f"seed_{seed}" / "results"
                / f"{subset}_{model}_predictions.npz"
                for seed in args.seeds
            ]
            ok = run_stage(
                f"Evaluate {model} on {subset} across seeds",
                py("evaluate.py") + ["--model", model, "--subset", subset,
                                      "--seeds", *[str(seed) for seed in args.seeds]],
                prediction_artifacts,
            )
            if not ok:
                sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 5 - Explainability (IG, SHAP, temporal relevance, ERI)
    # ------------------------------------------------------------------
    for subset in subsets:
        if not args.skip_explain:
            eri_artifacts = [
                ROOT / "outputs" / "results" / "uncertainty" / model / subset
                / "eara" / f"seed_{seed}" / "xai" / subset / model / f"eri_{target}.json"
                for seed in args.seeds for target in ("rul", "hi")
            ]
            ok = run_stage(
                f"Explain {model} on {subset} across seeds (RUL + HI)",
                py("explain.py") + ["--model", model, "--subset", subset,
                                    "--target", "all", "--seeds",
                                    *[str(seed) for seed in args.seeds]],
                eri_artifacts,
            )
            if not ok:
                sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 6 - Decision engine (O4)
    # ------------------------------------------------------------------
    decision_extra: list[str] = []
    if args.maintenance_window:
        decision_extra.append("--maintenance-window")
    if args.no_maintenance_window:
        decision_extra.append("--no-maintenance-window")
    for flag, value in (
        ("--minimum-safe-rul", args.minimum_safe_rul),
        ("--maintenance-lead-time", args.maintenance_lead_time),
        ("--max-allowable-delay", args.max_allowable_delay),
    ):
        if value is not None:
            decision_extra += [flag, str(value)]
    if args.resource_availability is not None:
        decision_extra += ["--resource-availability", args.resource_availability]

    for subset in subsets:
        if not args.skip_decision:
            ok = run_stage(
                f"Decision engine {subset} / {model}",
                py("decision.py")
                + ["--subset", subset, "--model", model,
                   "--uncertainty-variant", UNCERTAINTY_VARIANT, "--seeds",
                   *[str(seed) for seed in args.seeds]]
                + decision_extra,
                [ROOT / "outputs" / "results" / f"decision_{subset}_{model}.json",
                 ROOT / "outputs" / "reports" / f"decision_{subset}_{model}.html"],
            )
            if not ok:
                sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 7 - HITL review + policy refinement (O5)
    # ------------------------------------------------------------------
    for subset in subsets:
        if not args.skip_hitl:
            ok = run_stage(
                f"HITL review {subset} / {model}",
                py("hitl.py") + ["--subset", subset, "--model", model],
                [ROOT / "outputs" / "results" / f"hitl_{subset}_{model}.json",
                 ROOT / "outputs" / "reports" / f"hitl_{subset}_{model}.html"],
            )
            if not ok:
                sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 9 - Multi-dimensional evaluation (src/evaluation)
    # ------------------------------------------------------------------
    print(SEPARATOR)
    print("  MULTI-DIMENSIONAL EVALUATION")
    print(SEPARATOR)

    import json as _json
    sys.path.insert(0, str(ROOT))
    from src.evaluation import run_ablation

    results_dir = ROOT / "outputs" / "results"

    # 9a — aggregate prediction and interval metrics across the requested seeds
    print("  [benchmark] multi-seed prediction and EARA interval summaries...")
    bench = _seeded_evaluation_summary(results_dir, subsets, model, args.seeds)
    bench_path = results_dir / f"benchmark_{model}.json"
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w") as _f:
        _json.dump(bench, _f, indent=2, default=str)
    print(f"  [OK]   benchmark saved -> {bench_path.relative_to(ROOT)}")

    # 9b — ablation study per subset
    for subset in subsets:
        print(f"  [ablation] {subset} / {model}...")
        ablation = run_ablation(results_dir, subset, model)
        ablation["point_only_seed_aggregate"] = bench[subset][
            "prediction_metrics_mean_std"
        ]
        abl_path = results_dir / f"ablation_{subset}_{model}.json"
        with open(abl_path, "w") as _f:
            _json.dump(ablation, _f, indent=2, default=str)
        print(f"  [OK]   ablation saved -> {abl_path.relative_to(ROOT)}")

    # 9c — cross-subset consistency from multi-seed RUL MAE summaries
    print(f"  [robustness] cross-subset consistency for {model}...")
    subset_maes = {
        subset: bench[subset]["prediction_metrics_mean_std"]["rul"]["MAE"]["mean"]
        for subset in subsets
    }
    subset_values = list(subset_maes.values())
    mean_mae = float(np.mean(subset_values)) if subset_values else float("nan")
    std_mae = float(np.std(subset_values)) if subset_values else float("nan")
    cv = std_mae / mean_mae if len(subset_values) > 1 and mean_mae > 0 else float("nan")
    consistency = {
        "per_subset_rul_mae": subset_maes,
        "mean_rul_mae": mean_mae,
        "std_rul_mae": std_mae,
        "cv_rul_mae": cv,
        "consistency_band": (
            "NOT_APPLICABLE" if len(subset_values) < 2 else
            "HIGH" if cv < 0.20 else "MODERATE" if cv < 0.40 else "LOW"
        ),
        "aggregation": "mean of per-seed RUL MAE for each subset",
    }
    con_path = results_dir / f"consistency_{model}.json"
    with open(con_path, "w") as _f:
        _json.dump(consistency, _f, indent=2, default=str)
    cv = consistency.get("cv_rul_mae", float("nan"))
    band = consistency.get("consistency_band", "?")
    print(f"  [OK]   consistency CV={cv:.3f} ({band}) -> {con_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(SEPARATOR)
    print("  PIPELINE SUMMARY")
    print(SEPARATOR)

    all_ok = True
    for subset in subsets:
        artifacts = [
            ROOT / "data" / "processed" / f"{subset}_train_X.npy",
            ROOT / "outputs" / "results" / "uncertainty" / model / subset
            / "eara" / "run_config.json",
            ROOT / "outputs" / "results" / f"decision_{subset}_{model}.json",
            ROOT / "outputs" / "reports" / f"decision_{subset}_{model}.html",
            ROOT / "outputs" / "results" / f"hitl_{subset}_{model}.json",
            ROOT / "outputs" / "reports" / f"hitl_{subset}_{model}.html",
            ROOT / "outputs" / "results" / f"ablation_{subset}_{model}.json",
        ]
        artifacts.extend(
            ROOT / "outputs" / "results" / "uncertainty" / model / subset
            / "eara" / f"seed_{seed}" / "results" / f"{subset}_{model}_eara_conformal.npz"
            for seed in args.seeds
        )
        artifacts.extend(
            ROOT / "outputs" / "results" / "uncertainty" / model / subset
            / "eara" / f"seed_{seed}" / "checkpoints" / model / subset / "best_model.pt"
            for seed in args.seeds
        )
        artifacts.extend(
            ROOT / "outputs" / "results" / "uncertainty" / model / subset
            / "eara" / f"seed_{seed}" / "results" / f"{subset}_{model}_predictions.npz"
            for seed in args.seeds
        )
        artifacts.extend(
            ROOT / "outputs" / "results" / "uncertainty" / model / subset
            / "eara" / f"seed_{seed}" / "xai" / subset / model / "eri_rul.json"
            for seed in args.seeds
        )
        artifacts.extend(
            ROOT / "outputs" / "results" / "uncertainty" / model / subset
            / "eara" / f"seed_{seed}" / "xai" / subset / model / "eri_hi.json"
            for seed in args.seeds
        )
        print(f"  {subset} / {model}")
        for path in artifacts:
            status = "OK" if path.exists() else "MISSING"
            if status != "OK":
                all_ok = False
            print(f"    [{status:<7}] {path.relative_to(ROOT)}")

    print(SEPARATOR)
    if all_ok:
        print("  COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    else:
        print("  PIPELINE FINISHED WITH MISSING ARTIFACTS")
        sys.exit(1)


if __name__ == "__main__":
    main()
