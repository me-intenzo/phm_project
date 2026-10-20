"""Compare original EARA with research-inspired EARA adaptations.

The source benchmark provides one final window per test engine. This runner
replays those engines in a fixed order and reveals each target only after its
interval is emitted (prequential, retrospective oracle feedback).

Examples (activate .venv first):
    .\\.venv\\Scripts\\Activate.ps1
    python scripts/eara_novel_benchmarks.py --subset all
    python scripts/eara_novel_benchmarks.py --subset FD001 FD003 --seeds 42 43 44 45 46

Outputs for original EARA and both novel EARA variants are written to one
consolidated result folder. No non-EARA baselines are evaluated here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "results" / "multiseed_conformal_benchmark" / "per_engine_intervals.csv"
OUT_ROOT = ROOT / "outputs" / "results"
OUT = OUT_ROOT / "eara_novel_benchmark"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
VARIANTS = {
    "eara_kowcpi": "Kernel-weighted asymmetric residual calibration (KOWCPI-inspired)",
    "eara_pid_expert": "PID-expert-inspired online coverage controller",
    "eara_up_ocp": "UP-OCP parameter-free online coverage controller, warm-started from EARA width",
    "eara_moco_eg": "Monotone online conformal exponentiated-gradient threshold controller",
}


def load_eara_rows() -> list[dict]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["method"] == "eara_conformal"]
    if not rows:
        raise RuntimeError(f"No EARA-Conformal rows found in {SOURCE}")
    return rows


def metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: float) -> dict:
    alpha = 1.0 - target
    covered = (lo <= y) & (y <= hi)
    score = hi - lo + (2.0 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return {
        "nominal_coverage": target,
        "empirical_coverage": float(np.mean(covered)),
        "coverage_gap": float(np.mean(covered) - target),
        "absolute_coverage_error": float(abs(np.mean(covered) - target)),
        "mean_interval_width": float(np.mean(hi - lo)),
        "winkler_score": float(np.mean(score)),
    }


def kernel_weighted_intervals(group: list[dict], target: float, history_size: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Prequential asymmetric residual quantiles, with base EARA warm-up."""
    n = len(group)
    y = np.asarray([float(r["y_true"]) for r in group])
    base_lo = np.asarray([float(r["lower"]) for r in group])
    base_hi = np.asarray([float(r["upper"]) for r in group])
    center = (base_lo + base_hi) / 2.0
    lo, hi = base_lo.copy(), base_hi.copy()
    revealed: list[float] = []
    alpha = 1.0 - target
    beta_grid = np.linspace(0.0, alpha, 21)
    for t in range(n):
        hist = np.asarray(revealed[-history_size:], dtype=float)
        if len(hist) >= 10:
            ages = np.arange(len(hist) - 1, -1, -1, dtype=float)
            bandwidth = max(3.0, np.sqrt(len(hist)))
            weights = np.exp(-0.5 * (ages / bandwidth) ** 2)
            order = np.argsort(hist)
            sorted_residuals, sorted_weights = hist[order], weights[order]
            cdf = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            candidates = []
            for beta in beta_grid:
                qlo = float(sorted_residuals[min(np.searchsorted(cdf, beta, side="left"), len(hist) - 1)]) if beta > 0 else float(sorted_residuals[0])
                qhi = float(sorted_residuals[min(np.searchsorted(cdf, 1.0 - alpha + beta, side="left"), len(hist) - 1)])
                candidates.append((qhi - qlo, qlo, qhi))
            _, left, right = min(candidates, key=lambda item: item[0])
            lo[t], hi[t] = np.clip(center[t] + left, 0.0, 125.0), np.clip(center[t] + right, 0.0, 125.0)
        revealed.append(float(y[t] - center[t]))
    return lo, hi


def pid_expert_intervals(group: list[dict], target: float) -> tuple[np.ndarray, np.ndarray]:
    """A small expert-mixture coverage controller with delayed feedback."""
    base_lo = np.asarray([float(r["lower"]) for r in group])
    base_hi = np.asarray([float(r["upper"]) for r in group])
    center = (base_lo + base_hi) / 2.0
    half_width = (base_hi - base_lo) / 2.0
    scales = np.asarray([0.75, 1.0, 1.25, 1.5, 2.0])
    log_weights = np.zeros(len(scales), dtype=float)
    integral_error = 0.0
    previous_error = 0.0
    chosen = np.ones(len(group), dtype=float)
    for t, row in enumerate(group):
        probs = np.exp(log_weights - np.max(log_weights))
        probs /= probs.sum()
        chosen[t] = float(np.dot(probs, scales))
        lo_t = max(0.0, center[t] - chosen[t] * half_width[t])
        hi_t = min(125.0, center[t] + chosen[t] * half_width[t])
        y = float(row["y_true"])
        error = (0.0 if lo_t <= y <= hi_t else 1.0) - (1.0 - target)
        integral_error = float(np.clip(integral_error + error, -10.0, 10.0))
        derivative = error - previous_error
        previous_error = error
        # PID-like proportional/integral/derivative correction selects a
        # neighboring width expert; multiplicative weights penalize misses.
        control = float(np.clip(0.9 * error + 0.08 * integral_error + 0.1 * derivative, -0.3, 0.6))
        desired = chosen[t] * np.exp(control)
        log_weights -= 0.5 * ((scales - desired) / 0.5) ** 2
        miss_loss = np.where((y < center[t] - scales * half_width[t]) | (y > center[t] + scales * half_width[t]), 1.0, 0.0)
        log_weights -= 0.7 * miss_loss
        log_weights -= np.max(log_weights)
    lo = np.clip(center - chosen * half_width, 0.0, 125.0)
    hi = np.clip(center + chosen * half_width, 0.0, 125.0)
    return lo, hi


def up_ocp_intervals(group: list[dict], target: float) -> tuple[np.ndarray, np.ndarray]:
    """UP-OCP wealth update, warm-started at the first EARA interval radius."""
    centers = np.asarray([(float(r["lower"]) + float(r["upper"])) / 2.0 for r in group])
    y = np.asarray([float(r["y_true"]) for r in group])
    base_radius = np.asarray([(float(r["upper"]) - float(r["lower"])) / 2.0 for r in group])
    alpha = 1.0 - target
    wealth = max(float(base_radius[0]), 1e-6) * alpha * (1.0 - alpha) / (0.5 - alpha)
    misses = 0
    lower = np.empty(len(group), dtype=float)
    upper = np.empty(len(group), dtype=float)
    for index in range(len(group)):
        t = index + 1
        lam = (misses + 0.5) / t
        radius = max(0.0, wealth * (lam - alpha) / (alpha * (1.0 - alpha)))
        lower[index] = max(0.0, centers[index] - radius)
        upper[index] = min(125.0, centers[index] + radius)
        miss = abs(y[index] - centers[index]) > radius
        if miss:
            misses += 1
            wealth *= lam / alpha
        else:
            wealth *= (1.0 - lam) / (1.0 - alpha)
    return lower, upper


def _truncated_simplex_projection(values: np.ndarray, floor: float) -> np.ndarray:
    """KL projection onto the simplex with a positive coordinate floor."""
    projected = np.asarray(values, dtype=float).copy()
    free = np.ones(len(projected), dtype=bool)
    while True:
        fixed_mass = float(np.count_nonzero(~free)) * floor
        free_values = projected[free]
        scale = (1.0 - fixed_mass) / max(float(free_values.sum()), 1e-15)
        below = free & (projected * scale < floor)
        if not below.any():
            projected[free] *= scale
            projected[~free] = floor
            return projected / projected.sum()
        free[below] = False


def monotone_eg_intervals(
    grouped_by_coverage: dict[float, list[dict]],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Joint EG updates of bounded residual quantiles across all targets."""
    targets = sorted(grouped_by_coverage)
    # Increasing alpha corresponds to decreasing radius; thresholds are kept
    # nested by representing their adjacent gaps on a simplex.
    alphas = np.asarray([1.0 - target for target in targets[::-1]], dtype=float)
    ordered_groups = [sorted(grouped_by_coverage[target], key=lambda r: int(r["engine_id"]))
                      for target in targets[::-1]]
    if not ordered_groups or any(len(group) != len(ordered_groups[0]) for group in ordered_groups):
        raise ValueError("Monotone EG needs matching engine sequences at all coverage levels")
    n_levels = len(alphas)
    bound = 125.0
    floor = 1e-5
    eta = 1.0 / bound
    base_q = np.asarray([
        np.median([(float(r["upper"]) - float(r["lower"])) / 2.0 for r in group])
        for group in ordered_groups
    ])
    # Project warm-start radii into q(alpha_1) >= ... >= q(alpha_K).
    base_q = np.minimum.accumulate(np.clip(base_q, 0.0, bound))
    gaps = np.concatenate(([bound - base_q[0]], base_q[:-1] - base_q[1:], [base_q[-1]])) / bound
    weights = _truncated_simplex_projection(np.maximum(gaps, floor), floor)
    centers = [
        np.asarray([(float(r["lower"]) + float(r["upper"])) / 2.0 for r in group])
        for group in ordered_groups
    ]
    truths = [np.asarray([float(r["y_true"]) for r in group]) for group in ordered_groups]
    lows = [np.empty(len(group), dtype=float) for group in ordered_groups]
    highs = [np.empty(len(group), dtype=float) for group in ordered_groups]

    for t in range(len(ordered_groups[0])):
        q = bound * np.cumsum(weights[:0:-1])[::-1]
        center = np.mean([c[t] for c in centers])
        truth = truths[0][t]
        for level in range(n_levels):
            radius = float(q[level])
            lows[level][t] = max(0.0, center - radius)
            highs[level][t] = min(125.0, center + radius)
        errors = np.asarray([
            float(abs(truth - center) > q[level]) for level in range(n_levels)
        ])
        gradient = np.zeros(n_levels + 1, dtype=float)
        for coordinate in range(1, n_levels + 1):
            gradient[coordinate] = bound * float(np.sum(alphas[coordinate - 1:] - errors[coordinate - 1:]))
        tilted = weights * np.exp(np.clip(-eta * gradient, -50.0, 50.0))
        weights = _truncated_simplex_projection(tilted, floor)

    result = {}
    for index, target in enumerate(targets[::-1]):
        result[target] = (lows[index], highs[index])
    return result


def grouped(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["subset"], int(row["seed"]), float(row["coverage_level"]))
        buckets[key].append(row)
    result = []
    for (subset, seed, target), group in sorted(buckets.items()):
        group.sort(key=lambda r: (int(r["engine_id"]), r["engine_id"]))
        y = np.asarray([float(r["y_true"]) for r in group])
        base_lo = np.asarray([float(r["lower"]) for r in group])
        base_hi = np.asarray([float(r["upper"]) for r in group])
        variants = {
            "eara_conformal": (base_lo, base_hi),
            "eara_kowcpi": kernel_weighted_intervals(group, target),
            "eara_pid_expert": pid_expert_intervals(group, target),
            "eara_up_ocp": up_ocp_intervals(group, target),
        }
        for method, (lo, hi) in variants.items():
            for index, row in enumerate(group):
                result.append({
                    "subset": subset, "seed": seed, "method": method,
                    "nominal_coverage": target, "engine_id": row["engine_id"],
                    "sequence_index": index, "y_true": float(y[index]),
                    "lower": float(lo[index]), "upper": float(hi[index]),
                })
    # The 2026 monotone EG method updates all target levels jointly so the
    # resulting intervals remain nested after every sequential feedback step.
    joint: dict[tuple[str, int], dict[float, list[dict]]] = defaultdict(dict)
    for row in result:
        if row["method"] == "eara_conformal":
            joint[(row["subset"], row["seed"])].setdefault(row["nominal_coverage"], []).append(row)
    for (subset, seed), by_coverage in joint.items():
        intervals = monotone_eg_intervals(by_coverage)
        for target, (lower, upper) in intervals.items():
            source = sorted(by_coverage[target], key=lambda r: int(r["engine_id"]))
            for index, row in enumerate(source):
                result.append({
                    "subset": subset, "seed": seed, "method": "eara_moco_eg",
                    "nominal_coverage": target, "engine_id": row["engine_id"],
                    "sequence_index": index, "y_true": row["y_true"],
                    "lower": float(lower[index]), "upper": float(upper[index]),
                })
    return result


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["subset"], row["seed"], row["method"], row["nominal_coverage"])].append(row)
    per_seed = []
    for (subset, seed, method, target), group in sorted(buckets.items()):
        y = np.asarray([r["y_true"] for r in group])
        lo = np.asarray([r["lower"] for r in group])
        hi = np.asarray([r["upper"] for r in group])
        per_seed.append({"subset": subset, "seed": seed, "method": method, **metrics(y, lo, hi, target)})
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_seed:
        agg[(row["subset"], row["method"], row["nominal_coverage"])].append(row)
    summary = []
    metric_keys = ("empirical_coverage", "coverage_gap", "absolute_coverage_error", "mean_interval_width", "winkler_score")
    for (subset, method, target), group in sorted(agg.items()):
        entry = {"subset": subset, "method": method, "nominal_coverage": target, "n_seeds": len(group)}
        for key in metric_keys:
            vals = np.asarray([r[key] for r in group])
            entry[f"{key}_mean"] = float(vals.mean())
            entry[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(entry)
    return per_seed, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_summary(summary: list[dict], out: Path) -> None:
    methods = (
        "eara_conformal", "eara_kowcpi", "eara_pid_expert",
        "eara_up_ocp", "eara_moco_eg",
    )
    coverages = sorted({float(row["nominal_coverage"]) for row in summary})
    for subset in sorted({row["subset"] for row in summary}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for coverage in coverages:
            group = [
                row for row in summary
                if row["subset"] == subset and row["nominal_coverage"] == coverage
            ]
            by_method = {row["method"]: row for row in group}
            x = np.arange(len(methods)) + (coverage - 0.9) * 0.8
            for axis, metric, std_metric in (
                (axes[0], "empirical_coverage", "empirical_coverage_std"),
                (axes[1], "mean_interval_width", "mean_interval_width_std"),
            ):
                axis.errorbar(
                    x,
                    [by_method[method][f"{metric}_mean"] for method in methods],
                    yerr=[by_method[method][std_metric] for method in methods],
                    marker="o", capsize=3, label=f"Target {coverage:.0%}",
                )
        for coverage in coverages:
            axes[0].axhline(coverage, color="0.55", linewidth=0.7, linestyle="--", alpha=0.65)
        axes[0].set_ylabel("Empirical coverage")
        axes[1].set_ylabel("Mean interval width")
        for axis in axes:
            labels = [
                {
                    "eara_conformal": "Original EARA",
                    "eara_kowcpi": "EARA-KOWCPI",
                    "eara_pid_expert": "EARA-PID\nExpert",
                    "eara_up_ocp": "EARA-UP-OCP",
                    "eara_moco_eg": "EARA-MOCO-EG",
                }[method]
                for method in methods
            ]
            axis.set_xticks(np.arange(len(methods)), labels)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(frameon=False)
        fig.suptitle(f"{subset}: Original EARA and research-based EARA variants")
        fig.tight_layout()
        fig.savefig(out / f"{subset}_coverage_width.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--subset", "--subsets", dest="subsets", nargs="+", default=["all"],
        help="FD001..FD004, or 'all' to execute every subset (default: all).",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) < 5:
        raise SystemExit("At least five distinct seeds are required")
    subsets = list(SUBSETS) if "all" in args.subsets else args.subsets
    if set(subsets) - set(SUBSETS):
        raise SystemExit("Subsets must be selected from FD001..FD004")
    source_rows = load_eara_rows()
    selected = [r for r in source_rows if r["subset"] in subsets and int(r["seed"]) in args.seeds]
    found = {(r["subset"], int(r["seed"])) for r in selected}
    missing = [(s, seed) for s in subsets for seed in args.seeds if (s, seed) not in found]
    if missing:
        raise SystemExit(f"Missing base EARA runs: {missing}")
    all_rows = grouped(selected)
    per_seed, summary = summarize(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "per_engine_intervals.csv", all_rows)
    write_csv(OUT / "per_seed_metrics.csv", per_seed)
    write_csv(OUT / "summary_mean_std.csv", summary)
    plot_summary(summary, OUT)
    config = {
        "methods": ["eara_conformal", *VARIANTS],
        "variant_descriptions": VARIANTS,
        "reference": "eara_conformal",
        "reference_description": "Original EARA-Conformal with its original regime-conditioned calibration and empirical coverage evaluation.",
        "reference_calibration": "One seeded random window per calibration engine; regime-specific normalized residual quantiles with global fallback for small regime pools.",
        "reference_coverage_metric": "Fraction of held-out engine endpoints covered by the original EARA intervals, reported against each nominal target.",
        "research_references": {
            "eara_up_ocp": {
                "title": "Online Conformal Prediction via Universal Portfolio Algorithms",
                "url": "https://arxiv.org/abs/2602.03168",
                "note": "UP-OCP update is warm-started using the first original EARA interval radius; this warm start is an adaptation, not the paper's default initialization.",
            },
            "eara_moco_eg": {
                "title": "Online Conformal Prediction: Enforcing Monotonicity via Online Optimization",
                "url": "https://arxiv.org/abs/2605.12668",
                "note": "Joint exponentiated-gradient update over the configured coverage levels, initialized from original EARA widths and bounded by the RUL range.",
            },
        },
        "subsets": subsets, "seeds": args.seeds,
        "coverage_levels": sorted({float(r["coverage_level"]) for r in selected}),
        "feedback_protocol": "Retrospective prequential replay over sorted held-out engine IDs; each y_true is used only after its interval is emitted.",
        "limitation": "C-MAPSS processed test data has one endpoint per engine. This is not cycle-level online deployment evidence; realized RUL feedback is unavailable during real monitoring.",
        "source_eara_results": str(SOURCE),
    }
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
