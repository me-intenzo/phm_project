"""
app.py
======

Streamlit dashboard for the PHM-XAI pipeline.

The dashboard is the operator-facing website of the project.  It reads the
artifacts the pipeline already writes under ``outputs/`` and presents them
as five connected views:

``Model``
    Architecture, accuracy metrics, prediction scatter, residual
    diagnostics, the training curves, and a live-inference panel that runs
    the stored checkpoint on the processed test windows.
``Uncertainty``
    Conformal prediction intervals with the matching coverage diagnostics
    and the calibration distribution the intervals were derived from.
``XAI``
    Integrated-gradients / SHAP / temporal attributions, the Explainability
    Reliability Index, and the rendered explanation figures.
``Decision``
    The O4 maintenance recommendations with their risk / urgency profile.
``HITL``
    The O5 review queue, the accumulated expert feedback, and a form that
    records a new approve / override response.

Design rule
-----------
The dashboard never recomputes the pipeline.  Every number it shows is read
from the artifacts the CLI stages wrote, so the website and the HTML reports
can never disagree.  The single exception is the expert-feedback append,
which goes through exactly the same ``src.hitl`` API that ``scripts/hitl.py``
uses, so a review recorded in the browser lands in the same append-only
``outputs/hitl/feedback.jsonl`` as a review recorded on the command line.

Usage
-----
    streamlit run app.py
    streamlit run app.py --server.port 8501

Prerequisites
-------------
The dashboard renders the artifacts of an existing run.  Produce them with::

    python scripts/demo.py --subset FD001 --model gru

author: me-intenzo
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    REPORTS_DIR,
    RESULTS_DIR,
)

# ------------------------------------------------------------------
# Page configuration
#
# ``set_page_config`` has to be the first Streamlit command of the run,
# so it is issued once every import is in place.
# ------------------------------------------------------------------

st.set_page_config(
    page_title="PHM-XAI Trustworthy Prognostics",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
XAI_DIR = OUTPUTS_DIR / "xai"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HITL_LOG_PATH = OUTPUTS_DIR / "hitl" / "feedback.jsonl"

#: The conformal variants ``scripts/uncertainty.py`` can produce.  The list
#: mirrors ``UNCERTAINTY_VARIANTS`` in ``scripts/decision.py`` so the
#: dashboard offers exactly the artifacts the decision stage can consume.
UNCERTAINTY_VARIANTS: tuple[str, ...] = (
    "eara_conformal",
    "conformal",
    "adaptive_conformal",
    "cqr",
    "engine_joint_conformal",
)

#: Nominal coverage levels calibrated by the uncertainty stage.
COVERAGE_LEVELS: tuple[int, ...] = (80, 90, 95)

#: XAI targets explained by ``scripts/explain.py``.
XAI_TARGETS: tuple[str, ...] = ("rul", "hi")

#: The pipeline stages the dashboard can show, and the label used for each.
STAGE_LABELS: dict[str, str] = {
    "predictions": "Predictions",
    "uncertainty": "Uncertainty",
    "xai": "Explanations",
    "decision": "Decisions",
    "hitl": "HITL review",
}


def _stage_commands(model: str, subset: str, tag: str) -> dict[str, str]:
    """The CLI command that produces each stage's artifact for one run.

    The dashboard never runs the pipeline, but it can always say which
    command is missing for the selected run.
    """

    suffix = f" --tag {tag}" if tag else ""

    return {
        "predictions": (
            f"python scripts/evaluate.py --model {model} --subset {subset}{suffix}"
        ),
        "uncertainty": (
            f"python scripts/uncertainty.py --model {model} --subset {subset}"
            f" --method engine_joint{suffix}"
        ),
        "xai": (
            f"python scripts/explain.py --model {model} --subset {subset}"
            f" --target all{suffix}"
        ),
        "decision": (
            f"python scripts/decision.py --subset {subset} --model {model}"
            f" --uncertainty-variant engine_joint_conformal{suffix}"
        ),
        "hitl": f"python scripts/hitl.py --subset {subset} --model {model}{suffix}",
    }


def _stage_coverage(subset: str, run: str, variant: str) -> dict[str, bool]:
    """Which pipeline stages produced an artifact for the selected run."""

    paths = _run_paths(subset, run)

    return {
        "predictions": paths["predictions"].exists(),
        "uncertainty": _uncertainty_path(subset, run, variant) is not None,
        "xai": (
            (paths["xai_dir"] / "xai_rul.npz").exists()
            or (paths["xai_dir"] / "xai_hi.npz").exists()
        ),
        "decision": paths["decision"].exists(),
        "hitl": paths["hitl"].exists(),
    }

# ------------------------------------------------------------------
# Colour language (shared with the HTML reports of scripts/decision.py)
# ------------------------------------------------------------------

ACTION_COLORS: dict[str, str] = {
    "CONTINUE_OPERATION": "#1e8449",
    "MONITOR": "#2471a3",
    "INSPECT": "#d68910",
    "SCHEDULE_MAINTENANCE": "#b9770e",
    "URGENT_MAINTENANCE": "#c0392b",
}

HEALTH_COLORS: dict[str, str] = {
    "HEALTHY": "#1e8449",
    "DEGRADING": "#2471a3",
    "AT_RISK": "#d68910",
    "CRITICAL": "#c0392b",
}

UNCERTAINTY_COLORS: dict[str, str] = {
    "LOW": "#1e8449",
    "MODERATE": "#d68910",
    "HIGH": "#c0392b",
}

PALETTE = px.colors.qualitative.Set2

#: True when the installed Streamlit supports the modern ``width``
#: parameter of ``st.plotly_chart`` / ``st.dataframe`` (Streamlit 1.49+).
_STREAMLIT_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


# ==================================================================
# Artifact discovery
# ==================================================================

@st.cache_data(show_spinner=False)
def _registered_models() -> tuple[str, ...]:
    """Architectures known to the model registry."""

    from src.models.registry import ALL_MODELS

    return tuple(ALL_MODELS)


@st.cache_data(show_spinner=False)
def _known_subsets() -> tuple[str, ...]:
    """C-MAPSS subsets known to the configuration layer."""

    from src.utils.config import ALL_SUBSETS

    return tuple(ALL_SUBSETS)


def _split_run_name(run_name: str, models: tuple[str, ...]) -> tuple[str, str] | None:
    """Split ``<model>[_<tag>]`` into its model and tag parts.

    The longest registered architecture wins, so ``gru_att_deg_v2_ms42``
    resolves to the model ``gru_att_deg_v2`` with the tag ``ms42`` and not
    to the model ``gru`` with the tag ``att_deg_v2_ms42``.
    """

    if run_name in models:
        return run_name, ""

    best: str | None = None
    for model in models:
        if run_name.startswith(f"{model}_"):
            if best is None or len(model) > len(best):
                best = model

    if best is None:
        return None

    return best, run_name[len(best) + 1:]


@st.cache_data(show_spinner=False)
def discover_runs() -> list[dict[str, str]]:
    """Every (subset, model, tag) triple that has evaluation artifacts.

    A run is discovered from its predictions file
    (``outputs/results/<subset>_<model>[_<tag>]_predictions.npz``), which
    every completed pipeline produces.  Runs whose name does not map onto a
    registered architecture (ensembles, for example) are ignored because
    their checkpoints are not addressable by the model registry.
    """

    models = _registered_models()
    subsets = _known_subsets()

    runs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for path in sorted(RESULTS_DIR.glob("*_predictions.npz")):
        stem = path.name[: -len("_predictions.npz")]
        subset, _, run_name = stem.partition("_")

        if subset not in subsets or not run_name:
            continue

        parsed = _split_run_name(run_name, models)
        if parsed is None:
            continue

        model, tag = parsed
        key = (subset, model, tag)
        if key in seen:
            continue
        seen.add(key)

        runs.append(
            {"subset": subset, "model": model, "tag": tag, "run": run_name}
        )

    return runs


def _run_paths(subset: str, run: str) -> dict[str, Path]:
    """All artifact paths of one run, whether or not they exist."""

    return {
        "predictions": RESULTS_DIR / f"{subset}_{run}_predictions.npz",
        "metrics": RESULTS_DIR / f"{subset}_{run}_metrics.json",
        "history": RESULTS_DIR / f"{subset}_{run}_history.npz",
        "decision": RESULTS_DIR / f"decision_{subset}_{run}.json",
        "hitl": RESULTS_DIR / f"hitl_{subset}_{run}.json",
        "xai_dir": XAI_DIR / subset / run,
        "decision_report": REPORTS_DIR / f"decision_{subset}_{run}.html",
        "hitl_report": REPORTS_DIR / f"hitl_{subset}_{run}.html",
    }


def _uncertainty_path(subset: str, run: str, variant: str) -> Path | None:
    """Resolve an uncertainty artifact, preferring the flat layout.

    The flat file is what ``scripts/uncertainty.py`` writes today; the
    structured tree is a legacy fallback.  The flat file must win for the
    same reason it wins in ``scripts/decision.py``: a stale structured
    artifact must never shadow the fresh flat one.
    """

    flat = RESULTS_DIR / f"{subset}_{run}_{variant}.npz"
    structured = RESULTS_DIR / "uncertainty" / variant / subset / f"{run}.npz"

    if flat.exists():
        return flat
    if structured.exists():
        return structured
    return None


# ==================================================================
# Loaders
# ==================================================================

def _load_npz(path: Path | None) -> dict[str, np.ndarray] | None:
    """Read an ``.npz`` into a plain dict of arrays."""

    if path is None or not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    except (OSError, ValueError):
        return None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON document."""

    if path is None or not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


@st.cache_data(show_spinner=False)
def load_predictions(subset: str, run: str) -> dict[str, np.ndarray] | None:
    """Point predictions and ground truth of one run."""

    return _load_npz(_run_paths(subset, run)["predictions"])


def _normalise_metrics(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise a metrics report to the canonical ``rul`` / ``hi`` keys.

    The report is written with the ``rul`` / ``hi`` keys of
    ``src.evaluation.metrics.predictive_accuracy``, but 41 of the artifacts
    in this repository were written by an earlier version of the pipeline
    and use ``RUL`` / ``HI``.  Both spellings are accepted here, otherwise
    every run that was evaluated with the older script would show ``n/a``
    for every metric.
    """

    if raw is None:
        return None

    block = raw.get("metrics") or {}

    rul = block.get("rul") or block.get("RUL") or {}
    hi = block.get("hi") or block.get("HI") or {}

    if not rul:
        return None

    return {
        "rul": rul,
        "hi": hi,
        "provenance": raw.get("provenance") or {},
        "epoch": raw.get("epoch"),
        "arch": raw.get("arch"),
        "source": "metrics report",
    }


@st.cache_data(show_spinner=False)
def load_metrics(subset: str, run: str) -> dict[str, Any] | None:
    """Accuracy metrics of one run, normalised to the canonical key names."""

    return _normalise_metrics(_load_json(_run_paths(subset, run)["metrics"]))


@st.cache_data(show_spinner=False)
def metrics_from_predictions(
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Recompute the accuracy metrics from the stored predictions.

    A run that was trained and evaluated always has a predictions file, so
    the metrics can always be shown — even for the runs whose metrics
    report was never written, or was written by an older pipeline version.
    The computation itself is the project's own
    ``src.evaluation.metrics.predictive_accuracy``, so the numbers are the
    same ones the CLI reports.
    """

    from src.evaluation.metrics import predictive_accuracy

    return predictive_accuracy(
        y_rul_true=np.asarray(predictions["y_rul_true"]).reshape(-1),
        y_rul_pred=np.asarray(predictions["y_rul_pred"]).reshape(-1),
        y_hi_true=np.asarray(predictions["y_hi_true"]).reshape(-1),
        y_hi_pred=np.asarray(predictions["y_hi_pred"]).reshape(-1),
    )


@st.cache_data(show_spinner=False)
def load_history(subset: str, run: str) -> dict[str, np.ndarray] | None:
    """Training history of one run."""

    return _load_npz(_run_paths(subset, run)["history"])


@st.cache_data(show_spinner=False)
def load_uncertainty(
    subset: str,
    run: str,
    variant: str,
) -> dict[str, np.ndarray] | None:
    """Calibrated prediction intervals of one run."""

    return _load_npz(_uncertainty_path(subset, run, variant))


@st.cache_data(show_spinner=False)
def load_xai(
    subset: str,
    run: str,
    target: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Attribution arrays and the ERI summary of one run / target."""

    directory = _run_paths(subset, run)["xai_dir"]
    arrays = _load_npz(directory / f"xai_{target}.npz")
    eri = _load_json(directory / f"eri_{target}.json")

    if arrays is None and eri is None:
        return None

    return arrays or {}, eri or {}


@st.cache_data(show_spinner=False)
def load_decision(subset: str, run: str) -> dict[str, Any] | None:
    """O4 decision output of one run."""

    return _load_json(_run_paths(subset, run)["decision"])


@st.cache_data(show_spinner=False)
def load_hitl(subset: str, run: str) -> dict[str, Any] | None:
    """O5 hitl output of one run."""

    return _load_json(_run_paths(subset, run)["hitl"])


@st.cache_data(show_spinner=False)
def load_feedback_log() -> list[dict[str, Any]]:
    """Every expert-feedback record ever appended to the JSONL log."""

    from src.hitl.logger import load_feedback

    try:
        return list(load_feedback(HITL_LOG_PATH))
    except (OSError, ValueError):
        return []


@st.cache_data(show_spinner=False)
def load_engine_ids(subset: str) -> np.ndarray | None:
    """Engine id of every test window."""

    path = PROCESSED_DIR / f"{subset}_test_engine_ids.npy"
    if not path.exists():
        return None
    return np.load(path)


@st.cache_data(show_spinner=False)
def load_metadata(subset: str) -> dict[str, Any]:
    """Preprocessing metadata of one subset (sensor names, HI breakpoints)."""

    return _load_json(PROCESSED_DIR / f"{subset}_test_metadata.json") or {}


@st.cache_resource(show_spinner=False)
def load_live_model(model: str, subset: str, tag: str) -> dict[str, Any]:
    """Load the stored checkpoint and run it on the processed test windows.

    The architecture is rebuilt from the ``model_config`` recorded inside
    the checkpoint, exactly like ``scripts/evaluate.py`` does, so the live
    predictions must reproduce the stored ones.
    """

    import torch

    from src.models.outputs import unpack_model_output
    from src.models.registry import load_model_from_checkpoint

    X = np.load(PROCESSED_DIR / f"{subset}_test_X.npy")
    y_rul = np.load(PROCESSED_DIR / f"{subset}_test_y_rul.npy")
    y_hi = np.load(PROCESSED_DIR / f"{subset}_test_y_hi.npy")

    net, checkpoint, arch = load_model_from_checkpoint(
        model,
        input_size=int(X.shape[-1]),
        device="cpu",
        subset=subset,
        tag=tag,
    )

    rul, hi = [], []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            batch = torch.from_numpy(X[start:start + 64]).float()
            pred_rul, pred_hi, _ = unpack_model_output(net(batch))
            rul.append(pred_rul.cpu().numpy().reshape(-1))
            if pred_hi is not None:
                hi.append(pred_hi.cpu().numpy().reshape(-1))

    return {
        "arch": arch,
        "epoch": checkpoint.get("epoch"),
        "seed": (checkpoint.get("run_metadata") or {}).get(
            "seed", checkpoint.get("seed")
        ),
        "model_config": checkpoint.get("model_config") or {},
        "y_rul_true": y_rul,
        "y_hi_true": y_hi,
        "y_rul_pred": np.concatenate(rul),
        "y_hi_pred": np.concatenate(hi) if hi else None,
    }


# ==================================================================
# Small formatting helpers
# ==================================================================

def _chart(figure: Any, **kwargs: Any) -> None:
    """Render a Plotly figure across the full container width.

    Streamlit renamed ``use_container_width`` to ``width`` in a later
    release and deprecated the old name.  The project still supports the
    Python versions that can only install the older Streamlit, so the
    modern parameter is used when it exists and the legacy one otherwise.
    """

    if _STREAMLIT_HAS_WIDTH:
        st.plotly_chart(figure, width="stretch", **kwargs)
    else:
        st.plotly_chart(figure, use_container_width=True, **kwargs)


def _table(data: Any, **kwargs: Any) -> None:
    """Render a table across the full container width.

    See :func:`_chart` for why the two parameter names are both handled.
    """

    if _STREAMLIT_HAS_WIDTH:
        st.dataframe(data, width="stretch", **kwargs)
    else:
        st.dataframe(data, use_container_width=True, **kwargs)


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a metric card."""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _colored_counts(counts: dict[str, int], palette: dict[str, str]) -> go.Figure:
    """Horizontal bar chart of a categorical distribution."""

    if not counts:
        return go.Figure()

    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = [palette.get(label, "#7f8c8d") for label in labels]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=values,
            textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    figure.update_layout(
        height=260,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(
            range=[0, max(values) * 1.25] if values else [0, 1],
            title="count",
        ),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return figure


def _first_present(
    data: dict[str, np.ndarray],
    *keys: str,
) -> np.ndarray | None:
    """First of *keys* that is present in *data*, as an array."""

    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _to_shares(values: Any) -> np.ndarray:
    """Normalise a vector of importances into shares that sum to one.

    The three explanation methods are reported on different scales (raw
    attribution magnitudes, SHAP values, attention weights), so they can
    only be compared once each of them is expressed as a share of its own
    total.
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    total = float(array.sum()) if array.size else 0.0

    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(array)

    return array / total


@st.cache_data(show_spinner=False)
def load_feature_names(subset: str) -> list[str]:
    """Feature names of one subset, in the order the model consumes them.

    The list comes from the feature-selection report the preprocessing
    stage writes.  FD002 and FD004 carry the three operational settings in
    addition to the sensors, so the count is subset dependent.
    """

    path = REPORTS_DIR / subset / "selected_features.csv"
    if not path.exists():
        return []

    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return []

    if frame.empty:
        return []

    column = "feature" if "feature" in frame.columns else frame.columns[0]
    return [str(name) for name in frame[column].tolist()]


def _sensor_labels(subset: str, count: int) -> list[str]:
    """Human readable feature names of one subset, exactly *count* long."""

    if count <= 0:
        return []

    names = load_feature_names(subset)
    if len(names) == count:
        return names

    metadata = load_metadata(subset)
    columns = (metadata.get("health_index") or {}).get("sensor_columns") or []
    columns = [str(name) for name in columns]
    if len(columns) == count:
        return columns

    return [f"feature {index + 1}" for index in range(count)]


def _decisions_frame(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the O4 decision records into a table."""

    rows = []
    for decision in decisions:
        inputs = decision.get("inputs") or {}
        rows.append(
            {
                "engine_id": decision.get("engine_id", decision.get("index")),
                "rul": inputs.get("rul"),
                "rul_lower": inputs.get("rul_lower"),
                "rul_upper": inputs.get("rul_upper"),
                "hi": inputs.get("hi"),
                "eri": inputs.get("eri"),
                "health_state": decision.get("health_state"),
                "uncertainty_level": decision.get("uncertainty_level"),
                "explanation_reliability": decision.get("explanation_reliability"),
                "risk_score": decision.get("risk_score"),
                "urgency_index": decision.get("urgency_index"),
                "recommended_action": decision.get("recommended_action"),
                "human_review": bool(decision.get("human_review", False)),
                "constraint_status": decision.get("constraint_status"),
                "top_k_sensors": ", ".join(
                    f"S{int(s)}" for s in decision.get("top_k_sensors", [])
                ),
            }
        )

    return pd.DataFrame(rows)


# ==================================================================
# Tab 1 -- Model
# ==================================================================

def render_model_tab(subset: str, run: str, model: str, tag: str) -> None:
    """Model performance, training behaviour and live inference."""

    paths = _run_paths(subset, run)

    st.subheader("Model performance")
    st.caption(
        f"Run `{run}` on `{subset}` — architecture `{model}`"
        + (f" (tag `{tag}`)" if tag else "")
    )

    predictions = load_predictions(subset, run)
    history = load_history(subset, run)

    metrics = load_metrics(subset, run)

    rul_metrics: dict[str, Any] = {}
    hi_metrics: dict[str, Any] = {}
    source = ""

    if metrics is not None:
        rul_metrics = metrics.get("rul") or {}
        hi_metrics = metrics.get("hi") or {}
        source = metrics.get("source", "metrics report")

    if not rul_metrics and predictions is not None:
        # The run has predictions but no usable metrics report (missing, or
        # written by an older pipeline version).  The metrics are recomputed
        # from the stored predictions with the project's own metric code.
        computed = metrics_from_predictions(predictions)
        rul_metrics = computed["rul"]
        hi_metrics = computed["hi"]
        source = "recomputed from the stored predictions"

    if not rul_metrics:
        st.info(
            "No predictions and no metrics for this run. "
            "Produce them with `python scripts/evaluate.py --model "
            f"{model} --subset {subset}`."
        )
        return

    st.caption(f"Metric source: **{source}**.")

    cards = [
        ("RUL MAE", _fmt(rul_metrics.get("MAE"), 2), "cycles"),
        ("RUL RMSE", _fmt(rul_metrics.get("RMSE"), 2), "cycles"),
        ("RUL R²", _fmt(rul_metrics.get("R2")), "1 - SSres/SStot"),
        ("NASA score", _fmt(rul_metrics.get("NASA_Score"), 1), "lower is better"),
        ("HI MAE", _fmt(hi_metrics.get("MAE")), "0-1 scale"),
        ("HI RMSE", _fmt(hi_metrics.get("RMSE")), "0-1 scale"),
        ("HI R²", _fmt(hi_metrics.get("R2")), "1 - SSres/SStot"),
    ]
    columns = st.columns(len(cards))
    for column, (label, value, hint) in zip(columns, cards):
        column.metric(label, value, help=hint)

    if metrics is not None and metrics.get("provenance"):
        with st.expander("Provenance of this run"):
            st.json(metrics["provenance"])

    if predictions is None:
        st.info("No predictions file for this run.")
        return

    y_true = np.asarray(predictions["y_rul_true"]).reshape(-1)
    y_pred = np.asarray(predictions["y_rul_pred"]).reshape(-1)
    hi_true = np.asarray(predictions["y_hi_true"]).reshape(-1)
    hi_pred = np.asarray(predictions["y_hi_pred"]).reshape(-1)

    left, right = st.columns(2)

    with left:
        st.markdown("**RUL: predicted vs. true**")
        scatter = go.Figure()
        scatter.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker=dict(
                    size=9,
                    color=np.abs(y_pred - y_true),
                    colorscale="Inferno",
                    showscale=True,
                    colorbar=dict(title="|error|"),
                ),
                name="engine window",
                hovertemplate=(
                    "true %{x:.1f}<br>predicted %{y:.1f}<extra></extra>"
                ),
            )
        )
        limit = float(max(y_true.max(), y_pred.max())) * 1.05
        scatter.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                line=dict(color="#95a5a6", dash="dash"),
                name="perfect prediction",
                hoverinfo="skip",
            )
        )
        scatter.update_layout(
            height=420,
            xaxis_title="true RUL (cycles)",
            yaxis_title="predicted RUL (cycles)",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        _chart(scatter)

    with right:
        st.markdown("**Residual diagnostics**")
        residual = y_pred - y_true

        residual_figure = px.histogram(
            x=residual,
            nbins=28,
            color_discrete_sequence=["#2471a3"],
            labels={"x": "residual (predicted - true) [cycles]"},
        )
        residual_figure.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        _chart(residual_figure)

        cumulative = np.sort(np.abs(residual))
        coverage = np.arange(1, len(cumulative) + 1) / len(cumulative)
        coverage_figure = go.Figure(
            go.Scatter(
                x=cumulative,
                y=coverage,
                mode="lines",
                line=dict(color="#1e8449"),
                name="empirical |error| CDF",
                hovertemplate="|error| %{x:.1f}<br>share %{y:.2f}<extra></extra>",
            )
        )
        coverage_figure.update_layout(
            height=180,
            xaxis_title="|residual| [cycles]",
            yaxis_title="fraction of windows",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        _chart(coverage_figure)

    with st.expander("Health index (HI) prediction"):
        hi_figure = go.Figure()
        hi_figure.add_trace(
            go.Scatter(
                x=hi_true,
                y=hi_pred,
                mode="markers",
                marker=dict(size=8, color="#2471a3", opacity=0.75),
                name="engine window",
                hovertemplate="true HI %{x:.3f}<br>predicted HI %{y:.3f}<extra></extra>",
            )
        )
        hi_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#95a5a6", dash="dash"),
                hoverinfo="skip",
                name="perfect prediction",
            )
        )
        hi_figure.update_layout(
            height=360,
            xaxis_title="true HI",
            yaxis_title="predicted HI",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        _chart(hi_figure)

    if history is not None:
        st.markdown("**Training history**")
        epochs = np.arange(1, len(history.get("train_loss", [])) + 1)

        history_figure = go.Figure()
        for key, colour in (
            ("train_loss", "#2471a3"),
            ("val_loss", "#c0392b"),
        ):
            if key in history:
                history_figure.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=np.asarray(history[key]).reshape(-1),
                        mode="lines",
                        name=key.replace("_", " "),
                        line=dict(color=colour),
                    )
                )
        for key, colour in (
            ("train_rul_loss", "#1e8449"),
            ("val_rul_loss", "#d68910"),
        ):
            if key in history:
                history_figure.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=np.asarray(history[key]).reshape(-1),
                        mode="lines",
                        name=key.replace("_", " "),
                        line=dict(color=colour, dash="dot"),
                    )
                )
        if "lr" in history:
            history_figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=np.asarray(history["lr"]).reshape(-1),
                    mode="lines",
                    name="learning rate",
                    line=dict(color="#7f8c8d", width=1),
                    yaxis="y2",
                )
            )
            history_figure.update_layout(
                yaxis2=dict(overlaying="y", side="right", showgrid=False)
            )

        history_figure.update_layout(
            height=380,
            xaxis_title="epoch",
            yaxis_title="loss",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        _chart(history_figure)

    with st.expander("Live inference from the stored checkpoint"):
        st.caption(
            "Runs the checkpoint through the model registry and compares the "
            "result with the stored predictions."
        )

        # The result is stored per run, so switching the run selector never
        # shows the live prediction of a different set of weights.
        live_key = f"live_model_result__{subset}__{run}"

        if st.button("Run inference", key=f"run_inference__{subset}__{run}"):
            with st.spinner("Loading checkpoint and running inference..."):
                try:
                    live = load_live_model(model, subset, tag)
                except Exception as error:  # noqa: BLE001 - surfaced in the UI
                    st.error(f"Live inference failed: {error}")
                else:
                    st.session_state[live_key] = live

        live = st.session_state.get(live_key)
        if live is not None:
            drift = float(
                np.max(np.abs(np.asarray(live["y_rul_pred"]).reshape(-1) - y_pred))
            )
            st.markdown(
                f"**Architecture** `{live['arch']}` &nbsp;|&nbsp; "
                f"**epoch** `{live['epoch']}` &nbsp;|&nbsp; "
                f"**seed** `{live['seed']}`"
            )
            st.json(live.get("model_config") or {})
            st.metric(
                "Max |live - stored| RUL difference",
                f"{drift:.2e}",
                help="Must be ~0: the dashboard and the CLI must agree.",
            )
            live_frame = pd.DataFrame(
                {
                    "engine": np.arange(1, len(y_true) + 1),
                    "true_rul": y_true,
                    "live_rul": np.asarray(live["y_rul_pred"]).reshape(-1),
                }
            )
            _table(live_frame.head(200))

    if paths["decision_report"].exists():
        st.link_button(
            "Open the full decision report (HTML)",
            paths["decision_report"].as_uri(),
        )


# ==================================================================
# Tab 2 -- Uncertainty
# ==================================================================

def render_uncertainty_tab(
    subset: str,
    run: str,
    model: str,
    tag: str,
    variant: str,
    level: int,
) -> None:
    """Conformal intervals, coverage diagnostics and the calibration set."""

    uncertainty = load_uncertainty(subset, run, variant)

    if uncertainty is None:
        st.info(
            f"This run has no `{variant}` artifact. Produce it with:\n\n"
            f"`{_stage_commands(model, subset, tag)['uncertainty']}`",
            icon="⚠️",
        )
        return

    lower_key, upper_key = f"lower_{level}", f"upper_{level}"
    if lower_key not in uncertainty:
        st.warning(
            f"This artifact has no {level}% interval. Available levels: "
            + ", ".join(
                sorted(
                    key.split("_")[1]
                    for key in uncertainty
                    if key.startswith("lower_")
                )
            )
        )
        return

    y_true = np.asarray(uncertainty.get("y_rul_test", np.nan)).reshape(-1)
    y_pred = np.asarray(uncertainty["y_rul_pred"]).reshape(-1)
    lower = np.asarray(uncertainty[lower_key]).reshape(-1)
    upper = np.asarray(uncertainty[upper_key]).reshape(-1)

    covered = (y_true >= lower) & (y_true <= upper)
    empirical = float(np.mean(covered)) if len(covered) else float("nan")
    width = float(np.mean(upper - lower)) if len(upper) else float("nan")

    # The current artifacts store ``q_hat_<level>``; the legacy EARA
    # artifacts use ``q_hats_<level>``.  Both are accepted so an older
    # artifact stays readable.
    quantile = _first_present(uncertainty, f"q_hat_{level}", f"q_hats_{level}")
    q_hat = (
        float(np.asarray(quantile).reshape(-1)[0]) if quantile is not None else float("nan")
    )

    engine_ids = load_engine_ids(subset)
    x_axis = (
        np.asarray(engine_ids).reshape(-1)
        if engine_ids is not None and len(engine_ids) == len(y_true)
        else np.arange(1, len(y_true) + 1)
    )

    cards = st.columns(5)
    cards[0].metric("Nominal coverage", f"{level}%")
    cards[1].metric("Empirical coverage", f"{empirical:.1%}")
    cards[2].metric("Coverage error", f"{abs(empirical - level / 100):.3f}")
    cards[3].metric("Mean interval width", f"{width:.1f}", help="MPIW in cycles")
    cards[4].metric("Conformal q̂", _fmt(q_hat, 4))

    st.markdown(f"**{level}% prediction intervals**")

    band = go.Figure()
    band.add_trace(
        go.Scatter(
            x=x_axis,
            y=upper,
            mode="lines",
            line=dict(color="rgba(41, 128, 185, 0.0)"),
            hoverinfo="skip",
            showlegend=False,
            name="upper",
        )
    )
    band.add_trace(
        go.Scatter(
            x=x_axis,
            y=lower,
            mode="lines",
            line=dict(color="rgba(41, 128, 185, 0.0)"),
            fill="tonexty",
            fillcolor="rgba(41, 128, 185, 0.18)",
            hoverinfo="skip",
            showlegend=False,
            name="interval",
        )
    )
    band.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_pred,
            mode="markers",
            marker=dict(size=6, color="#2471a3"),
            name="predicted RUL",
            hovertemplate="engine %{x}<br>predicted %{y:.1f}<extra></extra>",
        )
    )
    band.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_true,
            mode="markers",
            marker=dict(
                size=7,
                symbol="x",
                color=["#1e8449" if ok else "#c0392b" for ok in covered],
            ),
            name="true RUL (green = covered, red = miss)",
            hovertemplate="engine %{x}<br>true %{y:.1f}<extra></extra>",
        )
    )
    band.update_layout(
        height=460,
        xaxis_title="engine",
        yaxis_title="RUL (cycles)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    _chart(band)

    left, right = st.columns(2)

    with left:
        st.markdown("**Coverage across the calibrated levels**")

        rows = []
        for candidate in COVERAGE_LEVELS:
            low_key, high_key = f"lower_{candidate}", f"upper_{candidate}"
            if low_key not in uncertainty:
                continue
            low = np.asarray(uncertainty[low_key]).reshape(-1)
            high = np.asarray(uncertainty[high_key]).reshape(-1)
            hit = (y_true >= low) & (y_true <= high)
            rows.append(
                {
                    "nominal": f"{candidate}%",
                    "empirical": float(np.mean(hit)) if len(hit) else float("nan"),
                    "coverage_error": abs(
                        (float(np.mean(hit)) if len(hit) else float("nan"))
                        - candidate / 100
                    ),
                    "mpiw": float(np.mean(high - low)) if len(high) else float("nan"),
                }
            )

        if rows:
            table = pd.DataFrame(rows)
            _table(
                table.style.format(
                    {
                        "empirical": "{:.2%}",
                        "coverage_error": "{:.4f}",
                        "mpiw": "{:.2f}",
                    }
                ),
            )

            bar = go.Figure(
                go.Bar(
                    x=table["nominal"],
                    y=table["empirical"] * 100,
                    marker_color=[
                        "#1e8449" if error < 0.05 else "#d68910"
                        for error in table["coverage_error"]
                    ],
                    text=[f"{value:.1f}%" for value in table["empirical"] * 100],
                    textposition="outside",
                    hovertemplate="%{x}<br>empirical %{y:.2f}%<extra></extra>",
                )
            )
            bar.add_hline(
                y=table["empirical"].mean() * 100,
                line=dict(color="#7f8c8d", dash="dot"),
                annotation_text="mean",
            )
            bar.update_layout(
                height=300,
                yaxis_title="empirical coverage [%]",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            _chart(bar)

    with right:
        st.markdown("**Calibration and regime behaviour**")

        scores = _first_present(
            uncertainty,
            "normalized_calibration_scores",
            "cal_scores",
        )
        if scores is not None:
            scores = np.asarray(scores).reshape(-1)
            calibration = px.histogram(
                x=scores,
                nbins=40,
                color_discrete_sequence=["#8e44ad"],
                labels={"x": "normalised non-conformity score"},
            )
            if np.isfinite(q_hat):
                calibration.add_vline(
                    x=q_hat,
                    line=dict(color="#c0392b", dash="dash"),
                    annotation_text=f"q̂ = {q_hat:.3f}",
                )
            calibration.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            _chart(calibration)

        regimes = _first_present(uncertainty, "test_regimes", "regimes_test")
        if regimes is not None:
            regimes = np.asarray(regimes).reshape(-1)
            if len(regimes) == len(y_true):
                regime_rows = []
                for regime in sorted(set(regimes.tolist())):
                    mask = regimes == regime
                    hit = (y_true >= lower) & (y_true <= upper)
                    regime_rows.append(
                        {
                            "regime": f"regime {regime}",
                            "windows": int(mask.sum()),
                            "coverage": float(hit[mask].mean()) if mask.any() else float("nan"),
                            "mean_width": float((upper - lower)[mask].mean())
                            if mask.any()
                            else float("nan"),
                        }
                    )
                regime_frame = pd.DataFrame(regime_rows)
                _table(
                    regime_frame.style.format(
                        {"coverage": "{:.2%}", "mean_width": "{:.2f}"}
                    ),
                )
                regime_figure = px.bar(
                    regime_frame,
                    x="regime",
                    y="coverage",
                    color_discrete_sequence=["#16a085"],
                    labels={"coverage": "empirical coverage"},
                )
                regime_figure.update_layout(
                    height=260,
                    yaxis_tickformat=".0%",
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                _chart(regime_figure)

        scales = _first_present(uncertainty, "test_scales", "scale_pred")
        if scales is not None:
            scales = np.asarray(scales).reshape(-1)
            scale_figure = px.histogram(
                x=scales,
                nbins=32,
                color_discrete_sequence=["#16a085"],
                labels={"x": "regime-adaptive residual scale"},
            )
            scale_figure.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            _chart(scale_figure)

    with st.expander("Interval table"):
        interval_frame = pd.DataFrame(
            {
                "engine": x_axis,
                "true_rul": y_true,
                "predicted_rul": y_pred,
                f"lower_{level}": lower,
                f"upper_{level}": upper,
                "covered": covered,
                "error": y_pred - y_true,
            }
        )
        _table(interval_frame)
        st.download_button(
            "Download intervals (CSV)",
            interval_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{subset}_{run}_{variant}_{level}.csv",
            mime="text/csv",
        )


# ==================================================================
# Tab 3 -- XAI
# ==================================================================

def render_xai_tab(subset: str, run: str, model: str, tag: str, target: str) -> None:
    """Attributions, ERI and the rendered explanation figures."""

    loaded = load_xai(subset, run, target)
    directory = _run_paths(subset, run)["xai_dir"]

    if loaded is None:
        st.info(
            f"This run has no XAI artifact for the `{target}` target. "
            "Produce it with:\n\n"
            f"`{_stage_commands(model, subset, tag)['xai']}`",
            icon="⚠️",
        )
        return

    arrays, eri = loaded

    ig = arrays.get("ig_attributions")
    shap_values = arrays.get("shap_sensor_values")
    temporal = arrays.get("attention_temporal_relevance")

    # The number of input features differs per subset (FD002/FD004 carry the
    # three operational settings on top of the sensors), so the sensor
    # labels are resolved against the shape of the attribution arrays and
    # not against a hard-coded count.
    sensor_count = 0
    for candidate in (ig, shap_values):
        if candidate is not None:
            sensor_count = int(np.asarray(candidate).shape[-1])
            break

    labels = _sensor_labels(subset, sensor_count)

    st.subheader(f"Explainability — {target.upper()} target")
    st.caption(
        "Integrated gradients, SHAP and temporal attention attributions "
        "produced by the O3 explainability stage."
    )

    if eri:
        st.markdown("**Explainability Reliability Index (ERI)**")

        columns = st.columns(4)
        columns[0].metric(
            "ERI",
            _fmt(eri.get("eri")),
            help="0-1, higher is better",
        )
        columns[1].metric("IG ↔ attention ρ", _fmt(eri.get("rho_ig_attn")))
        columns[2].metric("IG ↔ SHAP ρ", _fmt(eri.get("rho_ig_shap")))
        columns[3].metric("Consensus k", _fmt(eri.get("consensus_k")))

        st.info(eri.get("interpretation", "No interpretation recorded."))

    ig = arrays.get("ig_attributions")
    shap_values = arrays.get("shap_sensor_values")
    temporal = arrays.get("attention_temporal_relevance")

    if ig is not None:
        ig = np.asarray(ig)
        n_samples, window, n_sensors = ig.shape
        st.markdown("**Integrated-gradients attribution**")

        sample = st.slider(
            "Window index",
            min_value=0,
            max_value=max(n_samples - 1, 0),
            value=0,
            key="xai_sample",
        )

        mean_ig = np.abs(ig).mean(axis=0)  # (window, sensors)

        # The ERI report stores the per-feature importance as a normalised
        # share (each method sums to one), which is the only form in which
        # integrated gradients, SHAP and temporal relevance can be compared
        # against each other.  When the report is missing, the raw
        # attribution arrays are normalised the same way.
        importance_source = (eri or {}).get("sensor_importance") or {}

        def _shares(
            key: str,
            fallback: np.ndarray,
        ) -> np.ndarray:
            values = importance_source.get(key)
            if values is not None and len(values) == n_sensors:
                return _to_shares(values)
            return _to_shares(fallback)

        shares = {
            "integrated gradients": _shares(
                "ig", np.abs(ig).mean(axis=(0, 1))
            ),
            "shap": _shares(
                "shap",
                (
                    np.abs(np.asarray(shap_values)).mean(axis=0)
                    if shap_values is not None
                    else np.zeros(n_sensors)
                ),
            ),
            "temporal relevance": _shares(
                "temporal_relevance", np.zeros(n_sensors)
            ),
        }

        left, right = st.columns([3, 2])

        with left:
            heatmap = go.Figure(
                go.Heatmap(
                    z=mean_ig.T,
                    x=[f"t-{window - 1 - i}" for i in range(window)],
                    y=labels,
                    colorscale="Viridis",
                    colorbar=dict(title="mean |IG|"),
                    hovertemplate="sensor %{y}<br>step %{x}<br>|IG| %{z:.4f}<extra></extra>",
                )
            )
            heatmap.update_layout(
                height=420,
                xaxis_title="time step (oldest → newest)",
                yaxis_title="sensor",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            _chart(heatmap)

        with right:
            st.markdown("**Sensor importance**")
            st.caption(
                "Normalised share per feature (each method sums to 1)."
            )

            importance_rows = []
            for index in range(n_sensors):
                importance_rows.append(
                    {
                        "sensor": labels[index],
                        "integrated gradients": float(
                            shares["integrated gradients"][index]
                        ),
                        "shap": float(shares["shap"][index]),
                        "temporal relevance": float(
                            shares["temporal relevance"][index]
                        ),
                    }
                )

            importance_frame = pd.DataFrame(importance_rows)
            long_frame = importance_frame.melt(
                id_vars="sensor",
                var_name="method",
                value_name="importance",
            )
            importance_figure = px.bar(
                long_frame,
                x="importance",
                y="sensor",
                color="method",
                orientation="h",
                barmode="group",
                color_discrete_sequence=["#2471a3", "#e67e22", "#16a085"],
            )
            importance_figure.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            _chart(importance_figure)

        st.markdown(f"**Attribution of window #{sample}**")
        sample_ig = np.abs(ig[sample])
        sample_figure = go.Figure(
            go.Heatmap(
                z=sample_ig.T,
                x=[f"t-{window - 1 - i}" for i in range(window)],
                y=labels,
                colorscale="Magma",
                colorbar=dict(title="|IG|"),
            )
        )
        sample_figure.update_layout(
            height=380,
            xaxis_title="time step (oldest → newest)",
            yaxis_title="sensor",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        _chart(sample_figure)

    if temporal is not None:
        temporal = np.asarray(temporal)
        st.markdown("**Temporal relevance (attention over the window)**")
        temporal_figure = go.Figure()
        temporal_figure.add_trace(
            go.Scatter(
                x=np.arange(temporal.shape[1]),
                y=temporal.mean(axis=0),
                mode="lines+markers",
                line=dict(color="#16a085"),
                name="mean relevance",
            )
        )
        temporal_figure.update_layout(
            height=300,
            xaxis_title="time step (0 = oldest)",
            yaxis_title="relevance",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        _chart(temporal_figure)

    figures = [
        ("ig_heatmap", f"ig_heatmap_{target}.png"),
        ("sensor importance", f"sensor_importance_{target}.png"),
        ("temporal importance", f"temporal_importance_{target}.png"),
    ]
    available = [
        (title, directory / name)
        for title, name in figures
        if (directory / name).exists()
    ]
    degradation = sorted(directory.glob("degradation_engine*.png"))
    if degradation:
        available.append(("degradation trajectory", degradation[0]))

    if available:
        st.markdown("**Rendered explanation figures**")
        columns = st.columns(min(2, len(available)))
        for index, (title, path) in enumerate(available):
            with columns[index % len(columns)]:
                st.image(str(path), caption=title)

    with st.expander("Attribution data"):
        if eri:
            st.json(eri)
        if arrays:
            st.write(
                {key: list(np.asarray(value).shape) for key, value in arrays.items()}
            )


# ==================================================================
# Tab 4 -- Decision
# ==================================================================

def render_decision_tab(subset: str, run: str, model: str, tag: str) -> None:
    """O4 maintenance recommendations."""

    decision = load_decision(subset, run)

    if decision is None:
        st.info(
            "This run has no decision output. Produce it with:\n\n"
            f"`{_stage_commands(model, subset, tag)['decision']}`",
            icon="⚠️",
        )
        return

    summary = decision.get("summary") or {}
    st.subheader("Maintenance decision intelligence (O4)")
    st.caption(
        f"Scenario `{decision.get('constraint_scenario', 'unconstrained')}` "
        f"· thresholds {len(decision.get('thresholds') or {})} · "
        f"{summary.get('num_decisions', 0)} decisions"
    )

    cards = st.columns(6)
    cards[0].metric("Decisions", summary.get("num_decisions", 0))
    cards[1].metric("Human review", summary.get("human_review_count", 0))
    cards[2].metric(
        "Constraint violations", summary.get("constraint_violation_count", 0)
    )
    cards[3].metric("Mean risk score", _fmt(summary.get("risk_score_mean")))
    cards[4].metric("Max risk score", _fmt(summary.get("risk_score_max")))
    cards[5].metric("Mean urgency", _fmt(summary.get("urgency_index_mean")))

    left, right, third = st.columns(3)

    with left:
        st.markdown("**Recommended action**")
        _chart(
            _colored_counts(
                summary.get("action_distribution") or {}, ACTION_COLORS
            ),
        )

    with right:
        st.markdown("**Health state**")
        _chart(
            _colored_counts(
                summary.get("health_state_distribution") or {}, HEALTH_COLORS
            ),
        )

    with third:
        st.markdown("**Uncertainty level**")
        _chart(
            _colored_counts(
                summary.get("uncertainty_distribution") or {}, UNCERTAINTY_COLORS
            ),
        )

    decisions = decision.get("decisions") or []
    frame = _decisions_frame(decisions)

    st.markdown("**Risk vs. urgency**")
    if not frame.empty:
        scatter = px.scatter(
            frame,
            x="risk_score",
            y="urgency_index",
            color="health_state",
            color_discrete_map=HEALTH_COLORS,
            hover_data=[
                "engine_id",
                "rul",
                "hi",
                "uncertainty_level",
                "recommended_action",
            ],
            labels={
                "risk_score": "risk score",
                "urgency_index": "urgency index",
            },
        )
        scatter.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        _chart(scatter)

    st.markdown("**Decision records**")
    if frame.empty:
        st.info("No decision records.")
        return

    action_filter = st.multiselect(
        "Filter by action",
        options=sorted(frame["recommended_action"].dropna().unique().tolist()),
        default=[],
    )
    review_only = st.checkbox("Only records flagged for human review", value=False)

    filtered = frame
    if action_filter:
        filtered = filtered[filtered["recommended_action"].isin(action_filter)]
    if review_only:
        filtered = filtered[filtered["human_review"]]

    _table(filtered)
    st.download_button(
        "Download decisions (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"decision_{subset}_{run}.csv",
        mime="text/csv",
    )

    report = _run_paths(subset, run)["decision_report"]
    if report.exists():
        st.link_button("Open the decision report (HTML)", report.as_uri())


# ==================================================================
# Tab 5 -- HITL
# ==================================================================

def render_hitl_tab(subset: str, run: str, model: str, tag: str) -> None:
    """O5 review queue, expert feedback and the review form."""

    from src.utils.config import VALID_ACTIONS

    hitl = load_hitl(subset, run)

    st.subheader("Human-in-the-loop review (O5)")

    if hitl is None:
        st.info(
            "This run has no HITL output. Produce it with:\n\n"
            f"`{_stage_commands(model, subset, tag)['hitl']}`",
            icon="⚠️",
        )
    else:
        review_summary = hitl.get("review_summary") or {}
        adaptation = hitl.get("adaptation_metrics") or {}

        cards = st.columns(5)
        cards[0].metric("Decisions", review_summary.get("decision_count", 0))
        cards[1].metric("Flagged for review", review_summary.get("human_review_count", 0))
        cards[2].metric("Pending", review_summary.get("pending_review_count", 0))
        cards[3].metric("Resolved", review_summary.get("resolved_review_count", 0))
        cards[4].metric(
            "Policy adaptations", adaptation.get("adaptations_applied", 0)
        )

        st.caption(
            "The prognostic model is never retrained by HITL — the feedback "
            "refines the **decision policy** only "
            f"(adaptation scope: `{hitl.get('adaptation_scope', 'n/a')}`)."
        )

        queue = hitl.get("review_queue") or []

        st.markdown("**Review queue**")
        if queue:
            queue_frame = pd.DataFrame(
                [
                    {
                        "review_id": item.get("review_id"),
                        "status": item.get("status"),
                        "engine_id": (item.get("decision") or {}).get("engine_id"),
                        "ai_action": (item.get("decision") or {}).get(
                            "recommended_action"
                        ),
                        "health_state": (item.get("decision") or {}).get("health_state"),
                        "risk_score": (item.get("decision") or {}).get("risk_score"),
                        "reviewer": item.get("reviewer"),
                        "final_action": item.get("final_action"),
                    }
                    for item in queue
                ]
            )
            _table(queue_frame)
        else:
            st.info("The review queue is empty for this run.")

        profile = hitl.get("policy_profile") or {}
        if profile:
            st.markdown("**Learned decision policy**")
            rows = []
            for state, entry in profile.items():
                policy_state = entry.get("policy_state") or {}
                rows.append(
                    {
                        "health_state": policy_state.get("health_state"),
                        "uncertainty": policy_state.get("uncertainty_level"),
                        "explanation": policy_state.get("explanation_reliability"),
                        "risk_band": policy_state.get("risk_band"),
                        "urgency_band": policy_state.get("urgency_band"),
                        "preferred_action": entry.get("preferred_action"),
                        "support": entry.get("support"),
                        "count": entry.get("count"),
                        "consensus": entry.get("consensus_ratio"),
                    }
                )
            _table(pd.DataFrame(rows))

        report = _run_paths(subset, run)["hitl_report"]
        if report.exists():
            st.link_button("Open the HITL report (HTML)", report.as_uri())

    # ------------------------------------------------------------------
    # Accumulated expert feedback
    # ------------------------------------------------------------------

    st.markdown("---")
    st.markdown("**Expert feedback log**")
    st.caption(
        "Append-only JSONL at `outputs/hitl/feedback.jsonl` — reviews "
        "recorded here and on the command line share the same log."
    )

    feedback = load_feedback_log()

    if feedback:
        feedback_frame = pd.DataFrame(feedback)
        leading = [
            column
            for column in (
                "timestamp",
                "subset",
                "model",
                "engine_id",
                "ai_action",
                "expert_action",
                "override",
                "expert_confidence",
                "reason",
            )
            if column in feedback_frame.columns
        ]
        _table(feedback_frame[leading])

        override_flags = (
            feedback_frame.get("override", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
        )
        overrides = int(override_flags.sum())

        columns = st.columns(4)
        columns[0].metric("Feedback records", len(feedback_frame))
        columns[1].metric("Overrides", overrides)
        columns[2].metric(
            "Override rate",
            f"{overrides / max(len(feedback_frame), 1):.1%}",
        )
        columns[3].metric(
            "Mean expert confidence",
            _fmt(
                pd.to_numeric(
                    feedback_frame.get("expert_confidence", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean()
            ),
        )

        if "ai_action" in feedback_frame.columns and overrides:
            counts = (
                feedback_frame.loc[override_flags, "ai_action"]
                .value_counts()
                .to_dict()
            )
            _chart(
                _colored_counts(
                    {str(key): int(value) for key, value in counts.items()},
                    ACTION_COLORS,
                ),
            )
    else:
        st.info("No expert feedback has been recorded yet.")

    # ------------------------------------------------------------------
    # Review form
    # ------------------------------------------------------------------

    st.markdown("---")
    st.markdown("**Record a review response**")

    if hitl is None:
        st.info(
            "A review cannot be recorded until this run has a HITL output. "
            "Produce it with:\n\n"
            f"`{_stage_commands(model, subset, tag)['hitl']}`"
        )
        return

    queue = hitl.get("review_queue") or []
    if not queue:
        st.info("The review queue is empty, so there is nothing to review.")
        return

    labels: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(queue):
        decision = item.get("decision") or {}
        label = (
            f"{item.get('review_id')}  ·  engine {decision.get('engine_id')}"
            f"  ·  {decision.get('recommended_action')}"
        )
        # Two queue entries can share a review id, so the position is part
        # of the label and the mapping can never lose an entry.
        labels[f"{label}  (#{index + 1})"] = item
    label_list = list(labels)

    selected_label = st.selectbox(
        "Review item",
        options=label_list,
        index=0,
        key="hitl_review_item",
    )
    selected = labels[selected_label]
    decision = selected.get("decision") or {}
    ai_action = str(decision.get("recommended_action", ""))
    inputs = decision.get("inputs") or {}

    columns = st.columns(4)
    columns[0].metric("AI recommendation", ai_action or "n/a")
    columns[1].metric("RUL", _fmt(inputs.get("rul"), 1))
    columns[2].metric("Health index", _fmt(inputs.get("hi")))
    columns[3].metric("Risk score", _fmt(decision.get("risk_score")))

    with st.expander("Why the engine was flagged"):
        for reason in decision.get("reasons", []) or []:
            st.markdown(f"- {reason}")
        if decision.get("constraint_violations"):
            st.warning(
                "Constraint violations: "
                + ", ".join(decision["constraint_violations"])
            )

    override_options = [
        action for action in sorted(VALID_ACTIONS) if action != ai_action
    ]

    with st.form("hitl_review_form"):
        reviewer = st.text_input("Reviewer", value="operator-1")
        review_action = st.radio(
            "Review action",
            options=["APPROVE", "OVERRIDE"],
            horizontal=True,
        )
        override_action = st.selectbox(
            "Override action (used when the action is OVERRIDE)",
            options=override_options,
        )
        confidence = st.slider(
            "Expert confidence", min_value=0.0, max_value=1.0, value=0.8, step=0.05
        )
        comment = st.text_area(
            "Comment",
            value="",
            placeholder="Why is the AI recommendation accepted or changed?",
        )
        submitted = st.form_submit_button("Submit review", type="primary")

    if submitted:
        if not reviewer.strip():
            st.error("A reviewer name is required.")
        else:
            try:
                from src.hitl.logger import log_feedback
                from src.hitl.workflow import resolve_review, review_to_feedback

                review = dict(selected)
                resolve_review(
                    review,
                    reviewer=reviewer,
                    review_action=review_action,
                    expert_confidence=confidence,
                    comment=comment,
                    override_action=(
                        override_action if review_action == "OVERRIDE" else None
                    ),
                )
                feedback_record = review_to_feedback(
                    review,
                    subset=subset,
                    model=model,
                    source_decision_output=str(
                        _run_paths(subset, run)["decision"]
                    ),
                )
                log_feedback(feedback_record, HITL_LOG_PATH)
            except Exception as error:  # noqa: BLE001 - surfaced in the UI
                st.error(f"Could not record the review: {error}")
            else:
                st.success(
                    f"Review recorded: **{review_action}** → "
                    f"**{review.get('final_action')}** "
                    f"(confidence {confidence:.2f}). "
                    "The feedback log was updated."
                )
                st.cache_data.clear()
                st.rerun()


# ==================================================================
# Sidebar + main
# ==================================================================

def render_sidebar() -> dict[str, Any] | None:
    """Run selector and view options."""

    st.sidebar.title("🛠️ PHM-XAI")
    st.sidebar.caption(
        "Trustworthy prognostics for NASA C-MAPSS: RUL + health index, "
        "calibrated uncertainty, explainable AI and human-in-the-loop "
        "maintenance decisions."
    )

    runs = discover_runs()

    if not runs:
        st.sidebar.error("No evaluated runs found.")
        st.info(
            "The dashboard renders the artifacts of an existing pipeline run.\n\n"
            "Produce them with:\n\n"
            "```bash\n"
            "python scripts/demo.py --subset FD001 --model gru\n"
            "```"
        )
        return None

    subset_options = sorted({run["subset"] for run in runs})
    subset = st.sidebar.selectbox("C-MAPSS subset", subset_options, index=0)

    model_options = sorted(
        {run["model"] for run in runs if run["subset"] == subset}
    )
    model = st.sidebar.selectbox(
        "Model", model_options, index=0 if model_options else None
    )

    tag_options = sorted(
        {
            run["tag"]
            for run in runs
            if run["subset"] == subset and run["model"] == model
        }
    )
    # The untagged run and an empty tag are the same run, so the label is
    # built as a unique list: an option may never appear twice.
    tag_labels: list[str] = []
    for tag in tag_options:
        label = tag if tag else "(untagged)"
        if label not in tag_labels:
            tag_labels.append(label)
    if not tag_labels:
        tag_labels = ["(untagged)"]

    tag_label = st.sidebar.selectbox(
        "Run tag (seed)",
        tag_labels,
        index=0,
    )
    tag = "" if tag_label == "(untagged)" else tag_label

    match = [
        run
        for run in runs
        if run["subset"] == subset and run["model"] == model and run["tag"] == tag
    ]
    if not match:
        st.sidebar.error("No run matches the selection.")
        return None

    run = match[0]["run"]

    st.sidebar.divider()
    st.sidebar.markdown("#### View options")

    available_variants = [
        variant
        for variant in UNCERTAINTY_VARIANTS
        if _uncertainty_path(subset, run, variant) is not None
    ]
    variant_options = available_variants or list(UNCERTAINTY_VARIANTS)
    variant = st.sidebar.selectbox(
        "Uncertainty variant",
        variant_options,
        index=variant_options.index("engine_joint_conformal")
        if "engine_joint_conformal" in variant_options
        else 0,
    )

    level = st.sidebar.radio(
        "Nominal coverage",
        [str(item) for item in COVERAGE_LEVELS],
        index=1,
        horizontal=True,
    )

    target = st.sidebar.radio(
        "XAI target",
        list(XAI_TARGETS),
        horizontal=True,
    )

    st.sidebar.divider()
    if st.sidebar.button("↻  Refresh data"):
        st.cache_data.clear()
        st.rerun()

    # ------------------------------------------------------------------
    # Pipeline coverage: which stages actually produced an artifact for
    # the selected run.  Without it a run that was only trained looks
    # broken, because four of the five tabs have nothing to show.
    # ------------------------------------------------------------------

    coverage = _stage_coverage(subset, run, variant)
    commands = _stage_commands(model, subset, tag)

    st.sidebar.markdown("#### Pipeline coverage")
    for stage, label in STAGE_LABELS.items():
        icon = "✅" if coverage.get(stage) else "❌"
        st.sidebar.markdown(f"{icon} &nbsp;{label}")

    with st.sidebar.expander("Artifacts of this run"):
        paths = _run_paths(subset, run)
        st.write(
            {
                "predictions": str(paths["predictions"].name),
                "metrics": str(paths["metrics"].name),
                "decision": str(paths["decision"].name),
                "hitl": str(paths["hitl"].name),
                "xai": str(paths["xai_dir"].relative_to(OUTPUTS_DIR)),
            }
        )

    return {
        "subset": subset,
        "model": model,
        "tag": tag,
        "run": run,
        "variant": variant,
        "level": int(level),
        "target": target,
        "coverage": coverage,
        "commands": commands,
    }


def main() -> None:
    """Entry point of the dashboard."""

    st.title("PHM-XAI — Trustworthy Engine Prognostics")
    st.caption(
        "Remaining useful life and health-index estimation with calibrated "
        "conformal uncertainty, explainable attributions and "
        "human-in-the-loop maintenance decisions."
    )

    selection = render_sidebar()
    if selection is None:
        return

    subset = selection["subset"]
    run = selection["run"]
    model = selection["model"]
    tag = selection["tag"]

    # A run that was only trained has predictions but no uncertainty, no
    # explanations, no decisions and no review.  Saying so once, up front,
    # is the difference between "this run is incomplete" and "the app is
    # broken".
    coverage = selection.get("coverage") or {}
    commands = selection.get("commands") or {}
    missing = [stage for stage in STAGE_LABELS if not coverage.get(stage)]

    if missing:
        st.warning(
            "This run has **"
            + ", ".join(STAGE_LABELS[stage] for stage in missing)
            + "** missing — the corresponding tab stays empty until the "
            "stage is run:\n\n"
            + "\n".join(f"* `{commands[stage]}`" for stage in missing),
            icon="⚠️",
        )

    model_tab, uncertainty_tab, xai_tab, decision_tab, hitl_tab = st.tabs(
        [
            "🧠 Model",
            "📊 Uncertainty",
            "🔍 XAI",
            "🛠️ Decision",
            "👤 HITL",
        ]
    )

    with model_tab:
        render_model_tab(subset, run, model, tag)

    with uncertainty_tab:
        render_uncertainty_tab(
            subset,
            run,
            model,
            tag,
            selection["variant"],
            selection["level"],
        )

    with xai_tab:
        render_xai_tab(subset, run, model, tag, selection["target"])

    with decision_tab:
        render_decision_tab(subset, run, model, tag)

    with hitl_tab:
        render_hitl_tab(subset, run, model, tag)


# Streamlit executes the script with ``__name__ == "__main__"``, so the
# guard below is what actually renders the dashboard.
if __name__ == "__main__":
    main()
