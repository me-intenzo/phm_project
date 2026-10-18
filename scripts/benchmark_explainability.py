"""Benchmark ERI, explanation agreement, and attribution faithfulness."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.explainability_metrics import (
    correlation_statistics,
    explanation_agreement,
    feature_ablation_faithfulness,
)
from src.explainability.attention import explain_attention
from src.explainability.eri import compute_eri
from src.explainability.integrated_gradients import explain_with_integrated_gradients
from src.explainability.shap_explainer import explain_with_shap
from src.models.checkpoint import load_state_dict_compat
from src.models.gru import GRUPrognosticsModel
from src.models.gru_att_deg import GruAttDeg
from src.models.hybrid import HybridPrognosticsModel
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai"
DOCS_DIR = PROJECT_ROOT / "docs" / "explainability"
SEEDS_DEFAULT = (42,)
MODEL_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict[str, Any]]] = {
    "gru": (GRUPrognosticsModel, {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}),
    "gru_att_deg": (GruAttDeg, {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}),
    "lstm": (LSTMPrognosticsModel, {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}),
    "transformer": (
        TransformerPrognosticsModel,
        {"hidden_size": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.3},
    ),
    "hybrid": (
        HybridPrognosticsModel,
        {
            "hidden_size": 128,
            "num_gru_layers": 2,
            "num_transformer_layers": 2,
            "num_heads": 4,
            "dropout": 0.3,
        },
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--model", default="gru", choices=tuple(MODEL_REGISTRY))
    parser.add_argument("--target", default="rul", choices=("rul", "hi"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--shap-nsamples", type=int, default=128)
    parser.add_argument("--lime-samples", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-default-checkpoint",
        action="store_true",
        help="Use the standard checkpoint path instead of seed-specific checkpoints.",
    )
    return parser.parse_args()


def _load_model(
    model_name: str,
    subset: str,
    input_size: int,
    seed: int,
    device: torch.device,
    use_default_checkpoint: bool,
) -> torch.nn.Module:
    model_class, kwargs = MODEL_REGISTRY[model_name]
    model = model_class(input_size=input_size, **kwargs)
    checkpoint_dir = CHECKPOINT_DIR / model_name / subset
    if not use_default_checkpoint:
        checkpoint_dir /= f"seed_{seed}"
    checkpoint_path = checkpoint_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train this model/seed or pass --use-default-checkpoint."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    load_state_dict_compat(model, checkpoint["model_state_dict"])
    return model.to(device).eval()


def _select_prediction(model: torch.nn.Module, x: torch.Tensor, target: str) -> torch.Tensor:
    predictions = model(x)
    return predictions[0] if target == "rul" else predictions[1]


def _lime_masking_surrogate(
    model: torch.nn.Module,
    sample: torch.Tensor,
    target: str,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Fit a LIME-style weighted ridge surrogate over sensor masking coalitions."""
    n_features = sample.shape[1]
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(n_samples, n_features), dtype=np.int8)
    masks[0] = 1
    masks[1] = 0
    mask_tensor = torch.as_tensor(masks, dtype=sample.dtype, device=sample.device)
    masked = (
        mask_tensor[:, None, :] * sample[None, :, :]
    )
    with torch.no_grad():
        labels = _select_prediction(model, masked, target).detach().cpu().numpy().reshape(-1)

    distances = np.sqrt(np.mean((1 - masks) ** 2, axis=1))
    weights = np.exp(-(distances**2) / (0.75**2))
    surrogate = Ridge(alpha=1.0, fit_intercept=True)
    surrogate.fit(masks, labels, sample_weight=weights)
    return np.asarray(surrogate.coef_, dtype=np.float64)


def _per_sample_explanation(
    model: torch.nn.Module,
    sample: torch.Tensor,
    target: str,
    ig_steps: int,
    shap_nsamples: int,
    lime_samples: int,
    seed: int,
    top_k: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ig = explain_with_integrated_gradients(
        model, sample.unsqueeze(0), target=target, steps=ig_steps
    )
    shap = explain_with_shap(
        model, sample.unsqueeze(0), target=target, nsamples=shap_nsamples, seed=seed
    )
    attention = explain_attention(model, sample.unsqueeze(0), target=target)
    eri = compute_eri(ig, shap, attention, top_k=top_k)

    vectors = {
        "ig": ig["attributions"][0].detach().cpu().numpy().sum(axis=0),
        "shap": np.asarray(shap["sensor_shap_values"][0], dtype=np.float64),
        "lime": _lime_masking_surrogate(
            model, sample, target, n_samples=lime_samples, seed=seed
        ),
    }
    return vectors, {
        key: value
        for key, value in eri.items()
        if key in {"eri", "rho_ig_attn", "rho_ig_shap", "consensus_k"}
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty benchmark output: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explainers = ("ig", "shap", "lime")
    faithfulness_metrics = (
        "prediction_gap_top_50",
        "deletion_gap_auc",
        "insertion_auc",
    )
    predictors = ["eri"]
    predictors.extend(sorted(
        key for key in rows[0] if key.startswith("agreement_")
    ))
    summary: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = {"sample-seed": rows}

    def engine_means(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        engine_rows: dict[Any, list[dict[str, Any]]] = {}
        for row in records:
            engine_rows.setdefault(row["engine_id"], []).append(row)
        return [
            {
                "engine_id": engine_id,
                **{
                    key: float(np.nanmean([row[key] for row in engine_records]))
                    for key in predictors + [
                        f"{explainer}_{metric}"
                        for explainer in explainers
                        for metric in faithfulness_metrics
                    ]
                },
            }
            for engine_id, engine_records in engine_rows.items()
        ]

    grouped_rows["engine-mean"] = engine_means(rows)
    model_seeds = sorted(
        {row["model_seed"] for row in rows if row["model_seed"] is not None}
    )
    for model_seed in model_seeds:
        seed_rows = [row for row in rows if row["model_seed"] == model_seed]
        grouped_rows[f"engine-mean-seed-{model_seed}"] = engine_means(seed_rows)

    for analysis_level, analysis_rows in grouped_rows.items():
        for predictor in predictors:
            for explainer in explainers:
                for metric in faithfulness_metrics:
                    stats = correlation_statistics(
                        np.array([row[predictor] for row in analysis_rows], dtype=np.float64),
                        np.array(
                            [row[f"{explainer}_{metric}"] for row in analysis_rows],
                            dtype=np.float64,
                        ),
                    )
                    summary.append(
                        {
                            "analysis_level": analysis_level,
                            "predictor": predictor,
                            "explainer": explainer,
                            "faithfulness_metric": metric,
                            **stats,
                        }
                    )
    for analysis_level in grouped_rows:
        level_results = [
            result for result in summary if result["analysis_level"] == analysis_level
        ]
        for p_key, q_key in (
            ("pearson_p", "pearson_q_bh"),
            ("spearman_p", "spearman_q_bh"),
        ):
            p_values = np.array([result[p_key] for result in level_results], dtype=np.float64)
            finite = np.isfinite(p_values)
            q_values = np.full(p_values.shape, np.nan, dtype=np.float64)
            if finite.any():
                order = np.argsort(p_values[finite])
                sorted_p = p_values[finite][order]
                ranks = np.arange(1, len(sorted_p) + 1)
                sorted_q = np.minimum.accumulate(
                    (sorted_p * len(sorted_p) / ranks)[::-1]
                )[::-1]
                finite_indices = np.flatnonzero(finite)
                q_values[finite_indices[order]] = np.minimum(sorted_q, 1.0)
            for result, q_value in zip(level_results, q_values):
                result[q_key] = float(q_value)
    return summary


def _plot_correlations(rows: list[dict[str, Any]], summary: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(
        axes, ("deletion_gap_auc", "insertion_auc", "prediction_gap_top_50")
    ):
        for explainer, color in (("ig", "#1565c0"), ("shap", "#e65100"), ("lime", "#2e7d32")):
            axis.scatter(
                [row["eri"] for row in rows],
                [row[f"{explainer}_{metric}"] for row in rows],
                alpha=0.65,
                s=20,
                label=explainer.upper(),
                color=color,
            )
        axis.set_xlabel("ERI")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("ERI versus feature-ablation faithfulness")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    model_seeds = sorted(
        {row["model_seed"] for row in rows if row["model_seed"] is not None}
    )
    checkpoint_seed = (
        "default checkpoint (training seed unknown)"
        if args.use_default_checkpoint
        else ", ".join(map(str, args.seeds))
    )
    eri_primary = [
        result
        for result in summary
        if result["analysis_level"] == "engine-mean"
        and result["predictor"] == "eri"
    ]
    gap_support = sum(
        result["faithfulness_metric"] == "prediction_gap_top_50"
        and result["pearson_r"] > 0
        and result["pearson_q_bh"] < 0.05
        for result in eri_primary
    )
    insertion_support = sum(
        result["faithfulness_metric"] == "insertion_auc"
        and result["pearson_r"] > 0
        and result["pearson_q_bh"] < 0.05
        for result in eri_primary
    )
    deletion_opposition = sum(
        result["faithfulness_metric"] == "deletion_gap_auc"
        and result["pearson_r"] < 0
        and result["pearson_q_bh"] < 0.05
        for result in eri_primary
    )
    seed_level_primary = [
        result
        for result in summary
        if result["analysis_level"].startswith("engine-mean-seed-")
        and result["predictor"] == "eri"
    ]
    seed_gap_support = sum(
        result["faithfulness_metric"] == "prediction_gap_top_50"
        and result["pearson_r"] > 0
        and result["pearson_q_bh"] < 0.05
        for result in seed_level_primary
    )
    seed_insertion_support = sum(
        result["faithfulness_metric"] == "insertion_auc"
        and result["pearson_r"] > 0
        and result["pearson_q_bh"] < 0.05
        for result in seed_level_primary
    )
    seed_deletion_opposition = sum(
        result["faithfulness_metric"] == "deletion_gap_auc"
        and result["pearson_r"] < 0
        and result["pearson_q_bh"] < 0.05
        for result in seed_level_primary
    )
    lines = [
        f"# ERI Faithfulness Benchmark: {args.subset} / {args.model} / {args.target.upper()}",
        "",
        f"- Model checkpoint seed(s): {checkpoint_seed}",
        f"- Explanation/perturbation RNG seed(s): {', '.join(map(str, args.seeds))}",
        f"- Evaluated sample-seed rows: {len(rows)}",
        (
            "- Seed scope: only one default checkpoint was available; this run "
            "does not establish reproducibility across trained model seeds."
            if args.use_default_checkpoint
            else "- Seed scope: distinct seed-specific checkpoints were evaluated."
        ),
        "- The training seed controls both model initialization and the engine-wise train/validation split, so seed contrasts include both sources of variation.",
        (
            f"- Engine-mean Pearson outcomes after BH adjustment: positive ERI "
            f"association for top-50% prediction gap in {gap_support}/3 explainers, "
            f"for insertion AUC in {insertion_support}/3, and negative association "
            f"for deletion-gap AUC in {deletion_opposition}/3."
        ),
        *(
            [
                f"- Across seed-specific engine-mean tests (BH adjusted within each seed): "
                f"positive top-50% prediction-gap associations in {seed_gap_support}/"
                f"{len(model_seeds) * 3} explainer-seed tests, positive insertion-AUC "
                f"associations in {seed_insertion_support}/{len(model_seeds) * 3}, "
                f"and negative deletion-gap-AUC associations in "
                f"{seed_deletion_opposition}/{len(model_seeds) * 3}."
            ]
            if seed_level_primary
            else []
        ),
        f"- Attribution perturbation baseline: zero (standardized feature value)",
        "- Deletion-gap AUC: larger normalized prediction change after removing top-ranked sensors is better (opposite direction to the conventional residual-prediction deletion AUC).",
        "- Insertion AUC: higher fidelity to the full prediction while adding top-ranked sensors is better.",
        "- Prediction gap: absolute prediction change after ablating the top 50% sensors.",
        "- Agreement metrics are computed pairwise among IG, SHAP, and a LIME-style masking surrogate.",
        "- ERI remains an agreement score; its attention component reweights IG, so correlations are validation evidence, not proof of causal faithfulness.",
        "- The report includes correlations across sample-seed rows, after averaging within engine across all seeds, and separately after averaging within engine for each model seed. Use the per-seed engine means to assess consistency across checkpoints; the pooled engine mean can hide seed-specific differences.",
        "",
        f"![ERI versus faithfulness scatter plot](figures/{args.subset}_{args.model}_{args.target}_faithfulness.png)",
        *[
            f"![Seed {seed} ERI versus faithfulness](figures/{args.subset}_{args.model}_{args.target}_seed_{seed}_faithfulness.png)"
            for seed in model_seeds
        ],
        "",
        "## ERI-faithfulness correlations",
        "",
        "| Analysis level | Agreement predictor | Faithfulness explainer | Faithfulness metric | n | Pearson r | Pearson p | Pearson q (BH) | Spearman rho | Spearman p | Spearman q (BH) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary:
        lines.append(
            "| {analysis_level} | {predictor} | {explainer} | {faithfulness_metric} | {n} | {pearson_r:.4f} | "
            "{pearson_p:.4g} | {pearson_q_bh:.4g} | {spearman_rho:.4f} | {spearman_p:.4g} | "
            "{spearman_q_bh:.4g} |".format(**result)
        )
    lines.extend(
        [
            "",
            "## Agreement metric summary",
            "",
            "| Pair | Top-k rank agreement | Sign agreement | Spearman rank correlation |",
            "|---|---:|---:|---:|",
        ]
    )
    agreement_pairs = sorted(
        {
            name[len("agreement_"):].split("_rank_agreement_top_k")[0]
            for row in rows
            for name in row
            if name.startswith("agreement_") and name.endswith("_rank_agreement_top_k")
        }
    )
    for pair in agreement_pairs:
        rank = np.nanmean(
            [row[f"agreement_{pair}_rank_agreement_top_k"] for row in rows]
        )
        sign = np.nanmean(
            [row[f"agreement_{pair}_sign_agreement_top_k"] for row in rows]
        )
        rho = np.nanmean(
            [row[f"agreement_{pair}_rank_correlation_spearman"] for row in rows]
        )
        lines.append(f"| {pair.replace('__', ' / ')} | {rank:.4f} | {sign:.4f} | {rho:.4f} |")
    lines.extend(
        [
            "",
            "Agreement is reported per sample in the CSV; pairwise columns include top-k overlap, sign agreement among shared top-k features, and full-vector Spearman correlation.",
            "A positive, statistically significant coefficient is an empirical outcome, not an acceptance criterion imposed by the benchmark. Report null or negative outcomes as observed; do not tune on the test set to force significance.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.samples < 3:
        raise ValueError("--samples must be at least 3 for correlation analysis.")
    if args.lime_samples < 3:
        raise ValueError("--lime-samples must be at least 3.")
    if args.use_default_checkpoint and len(args.seeds) > 1:
        raise ValueError(
            "A shared default checkpoint cannot be counted as multiple model seeds."
        )
    device = torch.device(args.device)
    X = np.load(DATA_DIR / f"{args.subset}_test_X.npy")
    engine_ids = np.load(DATA_DIR / f"{args.subset}_test_engine_ids.npy")
    if X.ndim != 3:
        raise ValueError(f"Expected test data shaped (samples, time, features), got {X.shape}.")
    if len(engine_ids) != len(X):
        raise ValueError("Test engine IDs and feature windows must have equal lengths.")
    rng = np.random.default_rng(42)
    by_engine = {
        engine_id: rng.permutation(np.flatnonzero(engine_ids == engine_id)).tolist()
        for engine_id in np.unique(engine_ids)
    }
    engine_order = rng.permutation(np.unique(engine_ids)).tolist()
    sample_indices: list[int] = []
    while len(sample_indices) < min(args.samples, len(X)):
        for engine_id in engine_order:
            if by_engine[engine_id]:
                sample_indices.append(by_engine[engine_id].pop())
                if len(sample_indices) == min(args.samples, len(X)):
                    break
        if not any(by_engine.values()):
            break
    out_dir = (
        OUTPUT_DIR
        / args.subset
        / args.model
        / "faithfulness_benchmark"
        / args.target
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _load_model(
            args.model, args.subset, X.shape[-1], seed, device, args.use_default_checkpoint
        )
        for index in sample_indices:
            values = X[index]
            sample = torch.as_tensor(values, dtype=torch.float32, device=device)
            vectors, eri = _per_sample_explanation(
                model,
                sample,
                args.target,
                args.ig_steps,
                args.shap_nsamples,
                args.lime_samples,
                seed=seed * 100_003 + index,
                top_k=args.top_k,
            )
            row: dict[str, Any] = {
                "subset": args.subset,
                "model": args.model,
                "target": args.target,
                "model_seed": None if args.use_default_checkpoint else seed,
                "rng_seed": seed,
                "sample_index": index,
                "engine_id": int(engine_ids[index]),
                **eri,
            }
            agreement = explanation_agreement(vectors, top_k=args.top_k)
            for pair, metrics in agreement.items():
                row.update({f"agreement_{pair}_{key}": value for key, value in metrics.items()})
            for explainer, attribution in vectors.items():
                faithfulness = feature_ablation_faithfulness(
                    model, sample, attribution, target=args.target
                )
                row.update({f"{explainer}_{key}": value for key, value in faithfulness.items()})
            rows.append(row)

    summary = _summarize(rows)
    samples_path = out_dir / "per_sample.csv"
    summary_path = out_dir / "correlation_summary.csv"
    _write_csv(samples_path, rows)
    _write_csv(summary_path, summary)
    _plot_correlations(
        rows,
        summary,
        DOCS_DIR
        / "figures"
        / f"{args.subset}_{args.model}_{args.target}_faithfulness.png",
    )
    for seed in args.seeds:
        seed_rows = [row for row in rows if row["model_seed"] == seed]
        if seed_rows:
            _plot_correlations(
                seed_rows,
                summary,
                DOCS_DIR
                / "figures"
                / f"{args.subset}_{args.model}_{args.target}_seed_{seed}_faithfulness.png",
            )
    report_path = DOCS_DIR / f"{args.subset}_{args.model}_{args.target}_benchmark.md"
    _write_report(report_path, rows, summary, args)
    return samples_path, report_path


def main() -> None:
    args = _parse_args()
    samples_path, report_path = run_benchmark(args)
    print(f"Per-sample benchmark: {samples_path}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
