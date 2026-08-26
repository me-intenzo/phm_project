"""
PHM-XAI Explainability Pipeline

This script evaluates trained prognostic models by generating post-hoc
explanations for remaining useful life (RUL) and health index (HI) predictions.
It loads saved checkpoints, reconstructs the selected model architecture, and
computes attribution results for each engine sequence to explain which sensors
and time steps drive the model's decision.

Architecture
------------
The pipeline follows a simple three-layer design:

1. Data layer
   - Reads processed NASA C-MAPSS windows from data/processed for each subset
     (FD001-FD004) and target (rul / hi).
   - Loads engine metadata and windowed feature matrices used for inference.

2. Model layer
   - Uses the model registry in this script to construct architecture-specific
     networks such as GRU, GRU with attention degradation, LSTM, Transformer,
     and Hybrid models.
   - Loads the corresponding checkpoint from outputs/checkpoints/<model>/<subset>/.

3. Explainability layer
   - Runs SHAP, Integrated Gradients, attention-based attribution, and ERI
     analysis to explain predictions.
   - Saves plots and summaries under outputs/xai/ for downstream analysis.

Usage
-----
  # Single model (default: gru_att_deg, FD001, rul)
  python scripts/explain.py

  # All models, all subsets, both targets
  python scripts/explain.py --subset all --model all --target all

  # Specific combination
  python scripts/explain.py --subset FD001 --model transformer --target hi

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru import GRUPrognosticsModel
from src.models.gru_att_deg import GruAttDeg
from src.models.lstm import LSTMPrognosticsModel
from src.models.transformer import TransformerPrognosticsModel
from src.models.hybrid import HybridPrognosticsModel

from src.explainability.integrated_gradients import explain_with_integrated_gradients
from src.explainability.shap_explainer import explain_with_shap
from src.explainability.attention import explain_attention
from src.explainability.eri import compute_eri
from src.explainability.visualization import (
    plot_sensor_importance,
    plot_temporal_importance,
    plot_ig_heatmap,
)
from src.explainability.degradation_viz import (
    plot_engine_degradation,
    plot_multi_model_degradation,
    plot_eri_radar,
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

DATA_DIR       = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
XAI_DIR        = PROJECT_ROOT / "outputs" / "xai"
LOG_DIR        = PROJECT_ROOT / "outputs" / "logs" / "xai"

SEED            = 42
DEFAULT_SAMPLES = 32
IG_STEPS        = 50

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ALL_TARGETS = ["rul", "hi"]

# Model registry: name → (class, constructor kwargs)
MODEL_REGISTRY: dict[str, tuple] = {
    "gru": (
        GRUPrognosticsModel,
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3},
    ),
    "gru_att_deg": (
        GruAttDeg,
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3},
    ),
    "lstm": (
        LSTMPrognosticsModel,
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3},
    ),
    "transformer": (
        TransformerPrognosticsModel,
        {"hidden_size": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.3},
    ),
    "hybrid": (
        HybridPrognosticsModel,
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3},
    ),
}

SENSOR_NAMES_14 = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr",
    "Ps30", "phi", "NRf", "NRc",
]


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive XAI pipeline for PHM prognostics."
    )
    parser.add_argument(
        "--subset",
        default="FD001",
        help="Dataset subset or 'all'. Choices: FD001 FD002 FD003 FD004 all",
    )
    parser.add_argument(
        "--model",
        default="gru_att_deg",
        help=f"Model name or 'all'. Choices: {list(MODEL_REGISTRY)} all",
    )
    parser.add_argument(
        "--target",
        default="rul",
        choices=["rul", "hi", "all"],
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--ig-steps", type=int, default=IG_STEPS)
    parser.add_argument(
        "--engine-id",
        type=int,
        default=1,
        help="Engine index (0-based) for degradation trajectory plots.",
    )
    parser.add_argument(
        "--shap-nsamples",
        type=int,
        default=256,
        help="SHAP kernel samples (lower = faster, less accurate).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

def setup_logger(subset: str, model_name: str, target: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"xai_{subset}_{model_name}_{target}.log"

    logger = logging.getLogger(f"xai.{subset}.{model_name}.{target}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────

def load_data(subset: str):
    X = np.load(DATA_DIR / f"{subset}_test_X.npy")
    y_rul = np.load(DATA_DIR / f"{subset}_test_y_rul.npy")
    y_hi  = np.load(DATA_DIR / f"{subset}_test_y_hi.npy")
    ids   = np.load(DATA_DIR / f"{subset}_test_engine_ids.npy")
    return X, y_rul, y_hi, ids


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def load_model(model_name: str, input_size: int, subset: str, device: torch.device):
    ckpt_path = CHECKPOINT_DIR / model_name / subset / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train {model_name} on {subset} first."
        )
    cls, kwargs = MODEL_REGISTRY[model_name]
    model = cls(input_size=input_size, **kwargs)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────
# Per-engine degradation data
# ─────────────────────────────────────────────────────────────────────

def get_engine_sequences(
    X: np.ndarray,
    y_rul: np.ndarray,
    y_hi: np.ndarray,
    engine_ids: np.ndarray,
    engine_id: int,
):
    """Return all windows belonging to a specific engine (0-based index)."""
    unique = np.unique(engine_ids)
    if engine_id >= len(unique):
        engine_id = 0
    eid = unique[engine_id]
    mask = engine_ids == eid
    return X[mask], y_rul[mask], y_hi[mask], int(eid)


# ─────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────

def _img_tag(path: str, caption: str, width: int = 700) -> str:
    p = Path(path)
    return (
        f'<figure style="margin:20px 0">'
        f'<img src="{p.name}" width="{width}" style="border:1px solid #ddd;border-radius:4px">'
        f'<figcaption style="color:#555;font-size:13px;margin-top:6px">{caption}</figcaption>'
        f'</figure>'
    )


def generate_html_report(
    subset: str,
    target: str,
    model_results: dict,
    output_dir: Path,
) -> Path:
    """Generate an interactive HTML explanation report."""
    rows = ""
    for mname, res in model_results.items():
        eri = res.get("eri", {})
        eri_val  = f"{eri.get('eri', 0):.3f}"
        rho_attn = f"{eri.get('rho_ig_attn', 0):.3f}"
        rho_shap = f"{eri.get('rho_ig_shap', 0):.3f}"
        consensus = f"{eri.get('consensus_k', 0):.3f}"
        interp   = eri.get("interpretation", "—")
        color    = "#c8e6c9" if eri.get("eri", 0) > 0.8 else (
                   "#fff9c4" if eri.get("eri", 0) > 0.6 else "#ffcdd2")
        rows += (
            f'<tr style="background:{color}">'
            f'<td><b>{mname}</b></td>'
            f'<td>{eri_val}</td>'
            f'<td>{rho_attn}</td>'
            f'<td>{rho_shap}</td>'
            f'<td>{consensus}</td>'
            f'<td style="font-size:12px">{interp}</td>'
            f'</tr>\n'
        )

    figures_html = ""
    # ERI radar
    radar_path = output_dir / f"eri_radar_{target}.png"
    if radar_path.exists():
        figures_html += _img_tag(str(radar_path), "ERI Radar — Cross-model trustworthiness comparison")

    # Multi-model degradation
    for p in sorted(output_dir.glob("multi_model_degradation_engine*.png")):
        figures_html += _img_tag(str(p), f"Multi-model degradation trajectory — {p.stem}")

    # Per-model figures
    for mname in model_results:
        model_dir = output_dir / mname
        for fig_name, caption in [
            (f"sensor_importance_{target}.png", f"{mname} — Sensor Importance (IG vs SHAP)"),
            (f"temporal_importance_{target}.png", f"{mname} — Temporal Importance"),
            (f"ig_heatmap_{target}.png", f"{mname} — IG Attribution Heatmap"),
        ]:
            p = model_dir / fig_name
            if p.exists():
                figures_html += (
                    f'<figure style="margin:20px 0">'
                    f'<img src="{mname}/{p.name}" width="700" '
                    f'style="border:1px solid #ddd;border-radius:4px">'
                    f'<figcaption style="color:#555;font-size:13px;margin-top:6px">{caption}</figcaption>'
                    f'</figure>'
                )
        for p in sorted(model_dir.glob(f"degradation_engine*_{mname}.png")):
            figures_html += (
                f'<figure style="margin:20px 0">'
                f'<img src="{mname}/{p.name}" width="700" '
                f'style="border:1px solid #ddd;border-radius:4px">'
                f'<figcaption style="color:#555;font-size:13px;margin-top:6px">'
                f'{mname} — Engine degradation trajectory</figcaption>'
                f'</figure>'
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XAI Report — {subset} | {target.upper()}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #222; }}
  h1 {{ color: #1565C0; }} h2 {{ color: #37474F; border-bottom: 2px solid #eee; padding-bottom:6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
  th {{ background: #1565C0; color: white; }}
  .badge {{ display:inline-block; padding:3px 8px; border-radius:12px; font-size:12px; }}
</style>
</head>
<body>
<h1>PHM-XAI Explanation Report</h1>
<p><b>Subset:</b> {subset} &nbsp;|&nbsp; <b>Target:</b> {target.upper()} &nbsp;|&nbsp;
   <b>Models:</b> {", ".join(model_results.keys())}</p>

<h2>Explainability Reliability Index (ERI)</h2>
<p>ERI measures cross-method agreement between Integrated Gradients, SHAP, and Temporal Attention.
   Higher ERI → more trustworthy explanations.</p>
<table>
  <tr>
    <th>Model</th><th>ERI</th><th>ρ(IG,Attn)</th>
    <th>ρ(IG,SHAP)</th><th>Consensus@K</th><th>Interpretation</th>
  </tr>
  {rows}
</table>

<h2>Visualizations</h2>
{figures_html}

<hr>
<p style="font-size:11px;color:#999">
  Generated by PHM-XAI comprehensive explainability framework.
  Methods: Integrated Gradients · Kernel SHAP · Temporal Gradient Relevance · ERI.
</p>
</body>
</html>"""

    report_path = output_dir / f"xai_report_{subset}_{target}.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ─────────────────────────────────────────────────────────────────────
# Single model XAI run
# ─────────────────────────────────────────────────────────────────────

def run_model_xai(
    model_name: str,
    subset: str,
    target: str,
    X_explain: np.ndarray,
    X_engine: np.ndarray,
    y_rul_engine: np.ndarray,
    y_hi_engine: np.ndarray,
    engine_uid: int,
    device: torch.device,
    ig_steps: int,
    shap_nsamples: int,
    log: logging.Logger,
) -> dict:
    """Run full XAI pipeline for one model and return results dict."""
    input_size = X_explain.shape[-1]
    sensor_names = (
        SENSOR_NAMES_14 if input_size == 14
        else [f"S{i+1}" for i in range(input_size)]
    )

    model_out_dir = XAI_DIR / subset / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model checkpoint...")
    model, ckpt = load_model(model_name, input_size, subset, device)
    log.info("Loaded epoch %d", ckpt.get("epoch", "?"))

    x_tensor = torch.from_numpy(X_explain).float().to(device)

    # ── Integrated Gradients ────────────────────────────────────────
    log.info("Running Integrated Gradients (steps=%d)...", ig_steps)
    t0 = time.time()
    ig_result = explain_with_integrated_gradients(
        model=model, data=x_tensor, target=target, steps=ig_steps
    )
    log.info("IG complete in %.1fs", time.time() - t0)

    # ── SHAP ────────────────────────────────────────────────────────
    log.info("Running Kernel SHAP (nsamples=%d)...", shap_nsamples)
    t0 = time.time()
    shap_result = explain_with_shap(
        model=model, data=X_explain, target=target, nsamples=shap_nsamples
    )
    log.info(
        "SHAP complete in %.1fs | mean additivity error: %.4f",
        time.time() - t0,
        shap_result["mean_additivity_error"],
    )

    # ── Temporal relevance ──────────────────────────────────────────
    log.info("Running temporal gradient relevance...")
    attn_result = explain_attention(model=model, data=x_tensor, target=target)
    log.info("Temporal relevance complete.")

    # ── ERI ─────────────────────────────────────────────────────────
    log.info("Computing ERI...")
    eri_result = compute_eri(ig_result, shap_result, attn_result, top_k=5)
    log.info(
        "ERI=%.3f | ρ(IG,Attn)=%.3f | ρ(IG,SHAP)=%.3f | Consensus@5=%.3f",
        eri_result["eri"],
        eri_result["rho_ig_attn"],
        eri_result["rho_ig_shap"],
        eri_result["consensus_k"],
    )
    log.info("Interpretation: %s", eri_result["interpretation"])

    # ── Visualizations ──────────────────────────────────────────────
    log.info("Generating sensor importance plot...")
    plot_sensor_importance(
        ig_explanation=ig_result,
        shap_explanation=shap_result,
        sensor_names=sensor_names,
        output_dir=model_out_dir,
        target=target,
        top_k=10,
    )

    log.info("Generating temporal importance plot...")
    plot_temporal_importance(
        ig_explanation=ig_result,
        shap_explanation=shap_result,
        attention_explanation=attn_result,
        output_dir=model_out_dir,
        target=target,
    )

    log.info("Generating IG heatmap...")
    plot_ig_heatmap(
        ig_explanation=ig_result,
        sensor_names=sensor_names,
        output_dir=model_out_dir,
        target=target,
    )

    # ── Engine degradation trajectory ───────────────────────────────
    if len(X_engine) > 0:
        log.info("Generating engine degradation trajectory (engine_uid=%d)...", engine_uid)
        x_eng = torch.from_numpy(X_engine).float().to(device)
        with torch.no_grad():
            preds = model(x_eng)
            hi_pred  = preds[1].cpu().numpy()
            rul_pred = preds[0].cpu().numpy()

        eng_attn = explain_attention(model=model, data=x_eng, target=target)
        temporal_rel = eng_attn["temporal_relevance"].cpu().numpy().mean(axis=0)

        plot_engine_degradation(
            hi_pred=hi_pred,
            rul_pred=rul_pred,
            rul_true=y_rul_engine,
            temporal_relevance=temporal_rel,
            engine_id=engine_uid,
            model_name=model_name,
            output_dir=model_out_dir,
            subset=subset,
        )

    # ── Save raw arrays ─────────────────────────────────────────────
    np.savez_compressed(
        model_out_dir / f"xai_{target}.npz",
        ig_attributions=ig_result["attributions"].detach().cpu().numpy(),
        shap_sensor_values=shap_result["sensor_shap_values"],
        attention_temporal_relevance=attn_result["temporal_relevance"].detach().cpu().numpy(),
        eri=np.array([eri_result["eri"]]),
        rho_ig_attn=np.array([eri_result["rho_ig_attn"]]),
        rho_ig_shap=np.array([eri_result["rho_ig_shap"]]),
        consensus_k=np.array([eri_result["consensus_k"]]),
    )

    # ── Save ERI JSON ────────────────────────────────────────────────
    eri_json_path = model_out_dir / f"eri_{target}.json"
    with open(eri_json_path, "w") as f:
        json.dump(eri_result, f, indent=2)

    log.info("Outputs saved to: %s", model_out_dir)

    return {
        "eri": eri_result,
        "hi_pred": hi_pred if len(X_engine) > 0 else None,
        "output_dir": str(model_out_dir),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    models  = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    targets = ALL_TARGETS if args.target == "all" else [args.target]

    for subset in subsets:
        # Load data once per subset
        try:
            X, y_rul, y_hi, engine_ids = load_data(subset)
        except FileNotFoundError as e:
            print(f"[SKIP] {subset}: {e}")
            continue

        num_samples = min(args.samples, len(X))
        X_explain   = X[:num_samples]

        X_engine, y_rul_engine, y_hi_engine, engine_uid = get_engine_sequences(
            X, y_rul, y_hi, engine_ids, args.engine_id
        )

        for target in targets:
            model_results: dict[str, dict] = {}
            model_hi_preds: dict[str, np.ndarray] = {}

            for model_name in models:
                log = setup_logger(subset, model_name, target)
                log.info("=" * 60)
                log.info("XAI PIPELINE | %s | %s | %s", subset, model_name, target.upper())
                log.info("Device: %s | Samples: %d | IG steps: %d", device, num_samples, args.ig_steps)
                log.info("=" * 60)

                try:
                    result = run_model_xai(
                        model_name=model_name,
                        subset=subset,
                        target=target,
                        X_explain=X_explain,
                        X_engine=X_engine,
                        y_rul_engine=y_rul_engine,
                        y_hi_engine=y_hi_engine,
                        engine_uid=engine_uid,
                        device=device,
                        ig_steps=args.ig_steps,
                        shap_nsamples=args.shap_nsamples,
                        log=log,
                    )
                    model_results[model_name] = result
                    if result["hi_pred"] is not None:
                        model_hi_preds[model_name] = result["hi_pred"]

                    log.info("DONE — %s | %s | %s", subset, model_name, target.upper())

                except FileNotFoundError as e:
                    log.warning("SKIP %s/%s: %s", model_name, subset, e)
                except Exception as e:
                    log.error("ERROR %s/%s: %s", model_name, subset, e, exc_info=True)

            if not model_results:
                continue

            subset_out = XAI_DIR / subset
            subset_out.mkdir(parents=True, exist_ok=True)

            # Multi-model degradation overlay
            if len(model_hi_preds) > 1:
                plot_multi_model_degradation(
                    model_hi_dict=model_hi_preds,
                    rul_true=y_rul_engine,
                    engine_id=engine_uid,
                    output_dir=subset_out,
                    subset=subset,
                )

            # ERI radar
            eri_dict = {m: r["eri"] for m, r in model_results.items() if "eri" in r}
            if len(eri_dict) > 1:
                plot_eri_radar(
                    eri_results=eri_dict,
                    output_dir=subset_out,
                    subset=subset,
                    target=target,
                )

            # HTML report
            report_path = generate_html_report(
                subset=subset,
                target=target,
                model_results=model_results,
                output_dir=subset_out,
            )
            print(f"\n[REPORT] {report_path}")

    print("\n[XAI PIPELINE COMPLETE]")


if __name__ == "__main__":
    main()
