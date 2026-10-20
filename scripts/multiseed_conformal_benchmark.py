"""Multi-seed engine-level benchmark for RUL prediction intervals.

Example:
    # Run all four subsets with the default five seeds:
    python scripts/multiseed_conformal_benchmark.py --subsets all

    # Explicitly set at least five distinct seeds:
    python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46

The comparison includes split conformal, CQR, adaptive conformal, and
EARA-Conformal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import t as student_t
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.uncertainty.coverage import evaluate_interval

DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "results" / "multiseed_conformal_benchmark"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
COVERAGES = (0.80, 0.90, 0.95)
METHODS = ("split_conformal", "cqr", "adaptive", "eara_conformal")
EARA_METHODS = ("eara_conformal",)
COMPARISON_BASELINES = ("split_conformal", "cqr", "adaptive")
QUANTILE_LEVELS = (0.025, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.975)
# Shared scale floor. The same constant is applied whenever a predicted scale
# is normalised into a calibration score and whenever it multiplies a quantile
# into an interval radius, so calibration and prediction stay aligned.
SCALE_FLOOR = 1e-3


class IntervalRegressor(nn.Module):
    """Shared GRU backbone with point, quantile, and scale outputs."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.point = nn.Linear(hidden_size, 1)
        self.quantiles = nn.Linear(hidden_size, len(QUANTILE_LEVELS))
        self.scale = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, hidden = self.encoder(x)
        state = hidden[-1]
        point = self.point(state).squeeze(-1)
        quantiles = torch.sort(self.quantiles(state), dim=-1).values
        scale = torch.nn.functional.softplus(self.scale(state).squeeze(-1)) + SCALE_FLOOR
        return point, quantiles, scale


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(subset: str, split: str) -> tuple[np.ndarray, ...]:
    suffix = "train" if split == "train" else "test"
    x = np.load(DATA / f"{subset}_{suffix}_X.npy").astype(np.float32, copy=False)
    y = np.load(DATA / f"{subset}_{suffix}_y_rul.npy").astype(np.float32).reshape(-1)
    engines = np.load(DATA / f"{subset}_{suffix}_engine_ids.npy").reshape(-1)
    if not (len(x) == len(y) == len(engines)):
        raise ValueError(f"Mismatched {subset} {suffix} arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values found in {subset} {suffix} arrays")
    return x, y, engines


def grouped_split(engine_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, cal_idx = next(splitter.split(np.zeros(len(engine_ids)), groups=engine_ids))
    if set(engine_ids[train_idx]) & set(engine_ids[cal_idx]):
        raise RuntimeError("Engine leakage between training and calibration sets")
    return train_idx, cal_idx


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> IntervalRegressor:
    seed_everything(seed)
    model = IntervalRegressor(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            point, predicted_quantiles, scale = model(xb)
            residual = yb - point
            qloss = torch.zeros((), device=device)
            for index, level in enumerate(QUANTILE_LEVELS):
                prediction = predicted_quantiles[:, index]
                error = yb - prediction
                qloss += torch.maximum(level * error, (level - 1.0) * error).mean()
            loss = torch.nn.functional.smooth_l1_loss(point, yb)
            loss += 0.5 * qloss
            loss += 0.1 * torch.nn.functional.smooth_l1_loss(scale, residual.abs())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: IntervalRegressor, x: np.ndarray, device: torch.device, batch_size: int):
    outputs: list[list[np.ndarray]] = [[], [], []]
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        for bucket, values in zip(outputs, model(batch)):
            bucket.append(values.detach().cpu().numpy())
    return tuple(np.concatenate(values) for values in outputs)


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if scores.size == 0:
        raise ValueError("Cannot calibrate with no engine scores")
    rank = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    return float(scores[min(max(rank, 1), scores.size) - 1])


def engine_max(scores: np.ndarray, engines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(engines)
    return ids, np.asarray([np.max(scores[engines == engine]) for engine in ids])


def sample_calibration_indices(engine_ids: np.ndarray, seed: int) -> np.ndarray:
    """Sample one reproducible window per engine to mimic censored test endpoints."""
    rng = np.random.default_rng(seed)
    return np.asarray([
        rng.choice(np.flatnonzero(engine_ids == engine))
        for engine in np.unique(engine_ids)
    ], dtype=np.int64)


def interval_metrics(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float,
    engine_ids: np.ndarray, regimes: np.ndarray, n_regimes: int,
) -> dict[str, float]:
    alpha = 1.0 - coverage
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    winkler = (upper - lower) + (2.0 / alpha) * (below + above)
    coverage_result = evaluate_interval(
        y_true=y, lower=lower, upper=upper, nominal_coverage=coverage,
        regimes=regimes, n_regimes=n_regimes, engine_ids=engine_ids,
    )
    empirical = float(coverage_result["empirical_coverage"])
    return {
        "nominal_coverage": coverage,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - coverage,
        "absolute_coverage_error": float(coverage_result["coverage_error"]),
        "mean_interval_width": float(coverage_result["mean_interval_width"]),
        "normalized_interval_width": float(coverage_result["pinaw"]),
        "engine_level_coverage": float(coverage_result["engine_level_coverage"]),
        "winkler_score": float(np.mean(winkler)),
    }


def eara_scores(
    y: np.ndarray, point: np.ndarray, scale: np.ndarray, regimes: np.ndarray,
    engines: np.ndarray, n_regimes: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    raw = np.abs(y - point) / np.maximum(scale, SCALE_FLOOR)
    _, global_scores = engine_max(raw, engines)
    by_regime: dict[int, np.ndarray] = {}
    for regime in range(n_regimes):
        per_engine = []
        for engine in np.unique(engines):
            mask = (engines == engine) & (regimes == regime)
            if mask.any():
                per_engine.append(float(np.max(raw[mask])))
        by_regime[regime] = np.asarray(per_engine, dtype=np.float64)
    return by_regime, global_scores


def run_seed(
    subset: str, seed: int, args: argparse.Namespace, device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    x_all, y_all, engine_all = load_arrays(subset, "train")
    x_test, y_test, test_engines = load_arrays(subset, "test")
    train_idx, cal_idx = grouped_split(engine_all, seed)
    model = train_model(x_all[train_idx], y_all[train_idx], seed, args.epochs, args.batch_size, device)
    cal_sample_idx = sample_calibration_indices(engine_all[cal_idx], seed)
    x_cal_sample = x_all[cal_idx][cal_sample_idx]
    y_cal_sample = y_all[cal_idx][cal_sample_idx]
    cal_engines = engine_all[cal_idx][cal_sample_idx]
    point_cal, quantiles_cal, scale_cal = predict(model, x_cal_sample, device, args.batch_size)
    point_test, quantiles_test, scale_test = predict(model, x_test, device, args.batch_size)
    _, split_scores = engine_max(np.abs(y_cal_sample - point_cal), cal_engines)
    # Scale-normalised scores shared by adaptive and learned EARA methods.
    adaptive_raw = np.abs(y_cal_sample - point_cal) / np.maximum(scale_cal, SCALE_FLOOR)
    _, adaptive_scores = engine_max(adaptive_raw, cal_engines)
    regimes = None
    eara_cal_scores = None
    if args.n_regimes > 0:
        clusterer = KMeans(n_clusters=args.n_regimes, random_state=seed, n_init=10)
        clusterer.fit(x_all[train_idx, -1, :])
        regimes_cal = clusterer.predict(x_cal_sample[:, -1, :])
        regimes = clusterer.predict(x_test[:, -1, :])
        eara_cal_scores, _ = eara_scores(
            y_cal_sample, point_cal, scale_cal, regimes_cal, cal_engines, args.n_regimes
        )

    records = []
    calibration_records = []
    train_engine_ids = np.sort(np.unique(engine_all[train_idx]))
    calibration_engine_ids = np.sort(np.unique(engine_all[cal_idx]))
    split_hash = hashlib.sha256(
        subset.encode("ascii") + str(seed).encode("ascii")
        + train_engine_ids.tobytes() + calibration_engine_ids.tobytes()
    ).hexdigest()[:16]
    split_manifest = {
        "subset": subset,
        "seed": seed,
        "split_hash": split_hash,
        "train_engine_ids": train_engine_ids.tolist(),
        "calibration_engine_ids": calibration_engine_ids.tolist(),
        "calibration_scoring": "One seeded random window per calibration engine",
    }
    for coverage in COVERAGES:
        alpha = 1.0 - coverage
        radius = finite_sample_quantile(split_scores, alpha)
        lower_index = min(range(len(QUANTILE_LEVELS)), key=lambda i: abs(QUANTILE_LEVELS[i] - alpha / 2.0))
        upper_index = min(range(len(QUANTILE_LEVELS)), key=lambda i: abs(QUANTILE_LEVELS[i] - (1.0 - alpha / 2.0)))
        qlo_cal, qhi_cal = quantiles_cal[:, lower_index], quantiles_cal[:, upper_index]
        qlo_test, qhi_test = quantiles_test[:, lower_index], quantiles_test[:, upper_index]
        cqr_nonconformity = np.maximum.reduce((
            qlo_cal - y_cal_sample, y_cal_sample - qhi_cal,
            np.zeros(len(y_cal_sample)),
        ))
        _, cqr_scores = engine_max(cqr_nonconformity, cal_engines)
        cqr_delta = finite_sample_quantile(cqr_scores, alpha)
        eara_global = finite_sample_quantile(adaptive_raw, alpha)
        # Global scale-adaptive split conformal.
        adaptive_q = finite_sample_quantile(adaptive_scores, alpha)
        for method, scores, q_hat in (
            ("split_conformal", split_scores, radius),
            ("cqr", cqr_scores, cqr_delta),
            ("adaptive", adaptive_scores, adaptive_q),
        ):
            calibration_records.append({
                "subset": subset, "seed": seed, "method": method,
                "nominal_coverage": coverage, "regime_id": -1,
                "calibration_score_count": int(len(scores)),
                "calibration_score_mean": float(np.mean(scores)),
                "calibration_score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "calibrated_quantile": float(q_hat),
                "used_global_fallback": False,
            })
        for regime_id in range(args.n_regimes):
            scores = eara_cal_scores.get(regime_id, np.empty(0))
            fallback = scores.size < 5
            q_hat = eara_global if fallback else finite_sample_quantile(scores, alpha)
            calibration_records.append({
                "subset": subset, "seed": seed, "method": "eara_conformal",
                "nominal_coverage": coverage, "regime_id": regime_id,
                "calibration_score_count": int(len(scores)),
                "calibration_score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "calibration_score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "calibrated_quantile": float(q_hat),
                "used_global_fallback": fallback,
            })
        for i in range(len(y_test)):
            adaptive_radius = adaptive_q * max(float(scale_test[i]), SCALE_FLOOR)
            candidates: dict[str, tuple[float, float]] = {
                "split_conformal": (point_test[i] - radius, point_test[i] + radius),
                "cqr": (qlo_test[i] - cqr_delta, qhi_test[i] + cqr_delta),
                "adaptive": (point_test[i] - adaptive_radius, point_test[i] + adaptive_radius),
            }
            regime = int(regimes[i]) if regimes is not None else -1
            regime_scores = eara_cal_scores.get(regime, np.empty(0)) if eara_cal_scores is not None else np.empty(0)
            # Small or absent regime calibration sets use the engine-level global calibration pool.
            q_regime = finite_sample_quantile(regime_scores, alpha) if regime_scores.size >= 5 else eara_global
            scaled = max(float(scale_test[i]), SCALE_FLOOR)
            candidates["eara_conformal"] = (
                point_test[i] - q_regime * scaled,
                point_test[i] + q_regime * scaled,
            )
            for method, (lo, hi) in candidates.items():
                lo, hi = float(np.clip(lo, 0.0, 125.0)), float(np.clip(hi, 0.0, 125.0))
                row = {
                    "subset": subset, "seed": seed, "method": method,
                    "coverage_level": coverage, "engine_id": str(test_engines[i]),
                    "y_true": float(y_test[i]), "lower": float(lo), "upper": float(hi),
                    "regime_id": regime, "n_regimes": args.n_regimes,
                    "split_hash": split_hash,
                }
                records.append(row)
    return records, calibration_records, split_manifest


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subset"], row["seed"], row["method"], row["coverage_level"])].append(row)
    seed_metrics = []
    for (subset, seed, method, coverage), group in grouped.items():
        y = np.asarray([r["y_true"] for r in group])
        lower = np.asarray([r["lower"] for r in group])
        upper = np.asarray([r["upper"] for r in group])
        engines = np.asarray([r["engine_id"] for r in group])
        regimes = np.asarray([r["regime_id"] for r in group], dtype=np.int32)
        n_regimes = int(group[0]["n_regimes"])
        seed_metrics.append({"subset": subset, "seed": seed, "method": method,
                             **interval_metrics(y, lower, upper, coverage, engines, regimes, n_regimes)})
    return seed_metrics


def aggregate(seed_metrics: list[dict]) -> list[dict]:
    keys = (
        "empirical_coverage", "coverage_gap", "absolute_coverage_error",
        "mean_interval_width", "normalized_interval_width",
        "engine_level_coverage", "winkler_score",
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_metrics:
        grouped[(row["subset"], row["method"], row["nominal_coverage"])].append(row)
    summary = []
    for (subset, method, coverage), group in sorted(grouped.items()):
        item = {"subset": subset, "method": method, "nominal_coverage": coverage,
                "n_seeds": len(group)}
        for key in keys:
            values = np.asarray([r[key] for r in group], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def regime_coverage(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subset"], row["seed"], row["method"], row["coverage_level"], row["regime_id"])].append(row)
    result = []
    for (subset, seed, method, coverage, regime), group in sorted(grouped.items()):
        if regime < 0:
            continue
        result.append({
            "subset": subset, "seed": seed, "method": method,
            "nominal_coverage": coverage, "regime_id": regime,
            "n_test_engines": len(group),
            "empirical_coverage": float(np.mean([
                r["lower"] <= r["y_true"] <= r["upper"] for r in group
            ])),
        })
    return result


def aggregate_calibration(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subset"], row["method"], row["nominal_coverage"], row["regime_id"])].append(row)
    result = []
    for (subset, method, coverage, regime), group in sorted(grouped.items()):
        item = {
            "subset": subset, "method": method, "nominal_coverage": coverage,
            "regime_id": regime, "n_seeds": len(group),
            "fallback_seed_count": sum(bool(r["used_global_fallback"]) for r in group),
        }
        for key in ("calibration_score_count", "calibration_score_mean", "calibration_score_std", "calibrated_quantile"):
            values = np.asarray([r[key] for r in group], dtype=float)
            values = values[np.isfinite(values)]
            item[f"{key}_mean"] = float(values.mean()) if len(values) else float("nan")
            item[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(item)
    return result


SIGN_FLIP_RESAMPLES = 100_000
_SIGN_FLIP_CACHE: dict[int, np.ndarray] = {}


def sign_flip_pvalue(differences: np.ndarray) -> tuple[float, str]:
    """Two-sided paired sign-flip p-value, exact for small n and sampled otherwise."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    n = len(differences)
    if n <= 16:
        signs = np.asarray([
            [1.0 if (mask >> bit) & 1 else -1.0 for bit in range(n)]
            for mask in range(1 << n)
        ])
        null_means = np.abs((signs * differences).mean(axis=1))
        return float(np.mean(null_means >= observed - 1e-12)), "exact"

    if n not in _SIGN_FLIP_CACHE:
        rng = np.random.default_rng(20261019 + n)
        _SIGN_FLIP_CACHE[n] = (
            2 * rng.integers(0, 2, size=(SIGN_FLIP_RESAMPLES, n), dtype=np.int8) - 1
        )
    null_means = np.abs(_SIGN_FLIP_CACHE[n] @ differences / n)
    exceedances = np.count_nonzero(null_means >= observed - 1e-12)
    return float((1 + exceedances) / (SIGN_FLIP_RESAMPLES + 1)), "monte_carlo"


def paired_comparisons(seed_metrics: list[dict]) -> list[dict]:
    metrics = ("empirical_coverage", "mean_interval_width", "winkler_score")
    comparison_pairs = [
        (method, baseline)
        for method in EARA_METHODS
        for baseline in COMPARISON_BASELINES
    ]
    lookup = {(r["subset"], r["seed"], r["method"], r["nominal_coverage"]): r
              for r in seed_metrics}
    comparisons = []
    for subset in sorted({r["subset"] for r in seed_metrics}):
        for coverage in COVERAGES:
            seeds = sorted({r["seed"] for r in seed_metrics
                            if r["subset"] == subset and r["nominal_coverage"] == coverage})
            family = []
            for metric in metrics:
                for method, baseline in comparison_pairs:
                    differences = np.asarray([
                        lookup[(subset, seed, method, coverage)][metric]
                        - lookup[(subset, seed, baseline, coverage)][metric]
                        for seed in seeds
                    ])
                    n = len(differences)
                    mean_diff = float(differences.mean())
                    if n > 1:
                        sem = float(differences.std(ddof=1) / np.sqrt(n))
                        critical = float(student_t.ppf(0.975, df=n - 1))
                        ci_low, ci_high = mean_diff - critical * sem, mean_diff + critical * sem
                    else:
                        ci_low = ci_high = float("nan")
                    p_value, p_value_method = sign_flip_pvalue(differences)
                    family.append({
                        "subset": subset, "nominal_coverage": coverage,
                        "metric": metric, "comparison": f"{method} - {baseline}",
                        "n_seeds": n, "mean_paired_difference": mean_diff,
                        "ci95_low": ci_low, "ci95_high": ci_high,
                        "sign_flip_p": p_value,
                        "sign_flip_p_method": p_value_method,
                    })
            for metric in metrics:
                indices = [i for i, row in enumerate(family) if row["metric"] == metric]
                ordered = sorted(indices, key=lambda i: family[i]["sign_flip_p"])
                running = 0.0
                for rank, index in enumerate(ordered):
                    adjusted = min(1.0, (len(ordered) - rank) * family[index]["sign_flip_p"])
                    running = max(running, adjusted)
                    family[index]["holm_adjusted_p"] = running
            comparisons.extend(family)
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict]) -> None:
    for subset in sorted({r["subset"] for r in summary}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for coverage in COVERAGES:
            group = [r for r in summary if r["subset"] == subset and r["nominal_coverage"] == coverage]
            x = np.arange(len(METHODS)) + (coverage - 0.9) * 0.8
            by_method = {r["method"]: r for r in group}
            vals = [by_method[m]["empirical_coverage_mean"] for m in METHODS]
            errs = [by_method[m]["empirical_coverage_std"] for m in METHODS]
            axes[0].errorbar(x, vals, yerr=errs, marker="o", capsize=3, label=f"Target {coverage:.0%}")
            vals = [by_method[m]["mean_interval_width_mean"] for m in METHODS]
            errs = [by_method[m]["mean_interval_width_std"] for m in METHODS]
            axes[1].errorbar(x, vals, yerr=errs, marker="o", capsize=3, label=f"Target {coverage:.0%}")
        for coverage in COVERAGES:
            axes[0].axhline(coverage, color="0.55", linewidth=0.7, linestyle="--", alpha=0.65)
        axes[0].set_ylabel("Empirical coverage")
        axes[1].set_ylabel("Mean interval width")
        for axis in axes:
            axis.set_xticks(np.arange(len(METHODS)), [m.replace("_", "\n") for m in METHODS])
            axis.grid(axis="y", alpha=0.25)
            axis.legend(frameon=False)
        fig.suptitle(f"{subset}: coverage and interval width by method")
        fig.tight_layout()
        fig.savefig(OUT / f"{subset}_coverage_width.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=["all"], help="FD001..FD004 or all")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(42, 47)),
        help="Distinct random seeds; defaults to five seeds (42 through 46).",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-regimes", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = list(SUBSETS) if "all" in args.subsets else args.subsets
    if set(subsets) - set(SUBSETS):
        raise SystemExit(f"Unknown subset(s): {set(subsets) - set(SUBSETS)}")
    if len(set(args.seeds)) < 5:
        raise SystemExit("At least five distinct seeds are required")
    if args.n_regimes < 2:
        raise SystemExit("--n-regimes must be at least 2 for EARA-Conformal")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict] = []
    calibration_rows: list[dict] = []
    split_manifests = []
    for subset in subsets:
        for seed in args.seeds:
            print(f"Running {subset}, seed={seed} on {device}", flush=True)
            rows, calibration, manifest = run_seed(subset, seed, args, device)
            all_rows.extend(rows)
            calibration_rows.extend(calibration)
            split_manifests.append(manifest)
    seed_metrics = summarize(all_rows)
    summary = aggregate(seed_metrics)
    regime_metrics = regime_coverage(all_rows)
    calibration_summary = aggregate_calibration(calibration_rows)
    comparisons = paired_comparisons(seed_metrics)
    write_csv(OUT / "per_engine_intervals.csv", all_rows)
    write_csv(OUT / "per_seed_metrics.csv", seed_metrics)
    write_csv(OUT / "summary_mean_std.csv", summary)
    write_csv(OUT / "calibration_per_seed.csv", calibration_rows)
    write_csv(OUT / "calibration_summary_mean_std.csv", calibration_summary)
    write_csv(OUT / "regime_coverage_per_seed.csv", regime_metrics)
    write_csv(OUT / "paired_comparisons.csv", comparisons)
    (OUT / "split_manifest.json").write_text(json.dumps(split_manifests, indent=2), encoding="utf-8")
    (OUT / "run_config.json").write_text(json.dumps({
        "subsets": list(subsets), "seeds": args.seeds, "epochs": args.epochs,
        "batch_size": args.batch_size, "n_regimes": args.n_regimes,
        "coverage_levels": list(COVERAGES),
        "coverage_metrics": [
            "empirical_coverage", "signed_coverage_gap", "absolute_coverage_error",
            "mean_interval_width", "normalized_interval_width", "winkler_score",
            "engine_level_coverage", "per_regime_coverage",
        ],
        "calibration_diagnostics": [
            "calibration_score_count", "calibration_score_mean",
            "calibration_score_std", "calibrated_quantile", "global_fallback_count",
        ],
        "sign_flip_resamples": SIGN_FLIP_RESAMPLES,
        "methods": list(METHODS), "device": str(device),
        "calibration_unit": "One seeded random window per calibration engine as a censored-endpoint proxy; test unit is one final window per engine.",
    }, indent=2), encoding="utf-8")
    plot_summary(summary)
    print(f"Wrote benchmark outputs to {OUT}")


if __name__ == "__main__":
    main()
