"""
O4 Decision Intelligence Engine
================================

Entry point for the maintenance decision pipeline.

This script:
1. Loads prediction/uncertainty/XAI outputs.
2. Constructs the decision state (fusing RUL, HI, uncertainty, ERI).
3. Passes the state to the O4 decision engine.
4. Applies operational constraints.
5. Saves per-input results to outputs/results/ and reports to outputs/reports/.
6. Writes structured logs to outputs/logs/decision/.

Decision logic is implemented in:
    src/decision_engine/rules.py
    src/decision_engine/constraints.py
    src/decision_engine/recommender.py

Expected decision state:
    {
        "rul":       float,
        "hi":        float,
        "rul_lower": float,
        "rul_upper": float,
        "eri":       float,
        "top_k_sensors": [int, ...]   # optional, from O3 XAI output
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision_engine.recommender import recommend_action
from src.decision_engine.rules import DEFAULT_THRESHOLDS
from src.decision_engine.scenarios import (
    describe_scenario,
    get_scenario,
    scenario_names,
)
from src.models.registry import ALL_MODELS as REGISTERED_MODELS
from src.utils.config import ALL_SUBSETS as CONFIG_SUBSETS
from src.utils.logger import get_script_logger
from src.utils.paths import run_dir_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ALL_SUBSETS  = tuple(CONFIG_SUBSETS)
ALL_MODELS   = tuple(REGISTERED_MODELS)
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
XAI_DIR      = PROJECT_ROOT / "outputs" / "xai"
DEFAULT_MODEL  = "gru"
INTERVAL_LEVEL = 90
UNCERTAINTY_VARIANTS = (
    "eara_conformal",
    "conformal",
    "adaptive_conformal",
    "cqr",
    "engine_joint_conformal",
)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{name}': {value!r}") from exc


# ---------------------------------------------------------------------
# Decision-state construction
# ---------------------------------------------------------------------

def build_decision_state(
    rul: Any,
    hi: Any,
    rul_lower: Any,
    rul_upper: Any,
    eri: Any,
    top_k_sensors: list[int] | None = None,
) -> dict[str, Any]:
    """Construct and validate the state consumed by the O4 engine."""
    state: dict[str, Any] = {
        "rul":       to_float(rul,       "rul"),
        "hi":        to_float(hi,        "hi"),
        "rul_lower": to_float(rul_lower, "rul_lower"),
        "rul_upper": to_float(rul_upper, "rul_upper"),
        "eri":       to_float(eri,       "eri"),
    }

    if state["rul_lower"] > state["rul_upper"]:
        raise ValueError("Invalid RUL interval: rul_lower must be <= rul_upper.")
    if not 0.0 <= state["hi"] <= 1.0:
        raise ValueError(f"HI must be in [0, 1], received {state['hi']}.")
    if not 0.0 <= state["eri"] <= 1.0:
        raise ValueError(f"ERI must be in [0, 1], received {state['eri']}.")

    if top_k_sensors:
        state["top_k_sensors"] = [int(s) for s in top_k_sensors]

    return state


# ---------------------------------------------------------------------
# Single decision
# ---------------------------------------------------------------------

def make_decision(
    rul: Any,
    hi: Any,
    rul_lower: Any,
    rul_upper: Any,
    eri: Any,
    top_k_sensors: list[int] | None = None,
    constraints: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate one maintenance recommendation."""
    state = build_decision_state(rul, hi, rul_lower, rul_upper, eri, top_k_sensors)
    return recommend_action(state=state, constraints=constraints, thresholds=thresholds)


# ---------------------------------------------------------------------
# Batch decisions
# ---------------------------------------------------------------------

def process_records(
    records: list[dict[str, Any]],
    constraints: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Process multiple prediction/XAI records.

    Each record must contain: rul, hi, rul_lower, rul_upper, eri.
    Optional: id, engine_id, unit_id, timestamp, top_k_sensors.

    The **whole** record is carried into the decision record: only a
    hand-written list of four identifiers used to be forwarded, so any
    other field of the upstream record was silently dropped.
    """
    decisions = []
    for index, record in enumerate(records):
        decision = make_decision(
            rul=record["rul"],
            hi=record["hi"],
            rul_lower=record["rul_lower"],
            rul_upper=record["rul_upper"],
            eri=record["eri"],
            top_k_sensors=record.get("top_k_sensors"),
            constraints=constraints,
            thresholds=thresholds,
        )
        decisions.append({"index": index, **record, **decision})
    return decisions


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a compact summary of decision outcomes."""
    action_counts:      dict[str, int] = {}
    health_counts:      dict[str, int] = {}
    uncertainty_counts: dict[str, int] = {}
    review_count        = 0
    constraint_violations = 0
    constraint_violated_records = 0
    risk_scores:    list[float] = []
    urgency_scores: list[float] = []

    for d in decisions:
        for key, counter in (
            ("recommended_action", action_counts),
            ("health_state",       health_counts),
            ("uncertainty_level",  uncertainty_counts),
        ):
            val = d.get(key)
            if val:
                counter[val] = counter.get(val, 0) + 1

        if d.get("human_review", False):
            review_count += 1
        violations = d.get("constraint_violations", [])
        constraint_violations += len(violations)
        if violations:
            constraint_violated_records += 1

        if "risk_score" in d:
            risk_scores.append(float(d["risk_score"]))
        if "urgency_index" in d:
            urgency_scores.append(float(d["urgency_index"]))

    summary: dict[str, Any] = {
        "num_decisions":              len(decisions),
        "action_distribution":        action_counts,
        "health_state_distribution":  health_counts,
        "uncertainty_distribution":   uncertainty_counts,
        "human_review_count":         review_count,
        "constraint_violation_count": constraint_violations,
        "constraint_violated_records": constraint_violated_records,
    }

    if risk_scores:
        summary["risk_score_mean"] = round(sum(risk_scores) / len(risk_scores), 4)
        summary["risk_score_max"]  = round(max(risk_scores), 4)
    if urgency_scores:
        summary["urgency_index_mean"] = round(sum(urgency_scores) / len(urgency_scores), 4)
        summary["urgency_index_max"]  = round(max(urgency_scores), 4)

    return summary


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def load_input_records(path: str | Path) -> list[dict[str, Any]]:
    """Load decision input records.

    Supported JSON structures:
    1. A list:           [{...}, {...}]
    2. Object with key:  {"records": [{...}, ...]}
    3. Single record:    {"rul": ..., "hi": ..., ...}
    """
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "records" in data:
            records = data["records"]
            if not isinstance(records, list):
                raise ValueError("'records' must contain a list.")
            return records

        required = {"rul", "hi", "rul_lower", "rul_upper", "eri"}
        if required.issubset(data.keys()):
            return [data]

    raise ValueError(
        "Unsupported input format. Expected a JSON list, "
        "an object containing 'records', or a single decision record."
    )


def resolve_input_path(subset: str) -> Path:
    """Resolve the legacy JSON input for compatibility."""
    subset = subset.upper()
    subset_path = EXTERNAL_DIR / f"decision_input_{subset}.json"
    default_path = EXTERNAL_DIR / "decision_input.json"
    return subset_path if subset_path.exists() else default_path


def load_thresholds(subset: str) -> dict[str, float]:
    """Policy thresholds for a subset, with the calibrated HI breakpoints.

    The default thresholds of ``src.decision_engine.rules`` belong to the
    legacy ``HI = RUL / max_rul`` label.  When the pipeline builds the
    sensor-derived health index, preprocessing calibrates the three HI
    breakpoints on the training split and stores them in the dataset
    metadata; they are merged in here so that the HI criterion fires at the
    same degradation levels as the RUL criterion.
    """

    thresholds = dict(DEFAULT_THRESHOLDS)

    metadata_path = PROCESSED_DIR / f"{subset.upper()}_test_metadata.json"
    if not metadata_path.exists():
        return thresholds

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return thresholds

    calibrated = metadata.get("hi_thresholds") or {}
    for key in ("hi_critical", "hi_at_risk", "hi_degrading"):
        value = calibrated.get(key)
        if value is not None:
            thresholds[key] = float(value)

    return thresholds


def resolve_uncertainty_path(
    subset: str,
    run_name: str,
    uncertainty_variant: str,
) -> tuple[Path, str]:
    """Choose the interval artifact to read, preferring the flat layout.

    Two layouts exist for the same artifact:

    * the **flat** file ``outputs/results/<subset>_<run>_<variant>.npz``,
      which is what ``scripts/uncertainty.py`` writes today;
    * the **structured** file
      ``outputs/results/uncertainty/<variant>/<subset>/<run>.npz``, a
      legacy read-path fallback that no current script writes.

    The flat file must win.  Preferring the structured tree let a
    month-old artifact silently shadow the fresh one: the two files for
    FD001/``gru``/``eara_conformal`` disagreed by up to 16 RUL cycles in
    their point predictions, so a maintenance decision could have been
    produced from a different model's intervals than the ones on disk.

    Returns
    -------
    (path, note)
        The resolved path and a short note describing which layout was
        used (empty when there is nothing worth reporting).
    """

    structured_path = (
        RESULTS_DIR / "uncertainty" / uncertainty_variant / subset / f"{run_name}.npz"
    )
    flat_path = RESULTS_DIR / f"{subset}_{run_name}_{uncertainty_variant}.npz"

    flat_exists = flat_path.exists()
    structured_exists = structured_path.exists()

    if flat_exists:
        note = ""
        if structured_exists:
            note = (
                f"both the flat and the legacy structured artifact exist for "
                f"{subset}/{run_name}/{uncertainty_variant}; the flat file "
                f"was used"
            )
        return flat_path, note

    if structured_exists:
        return structured_path, (
            f"the flat artifact for {subset}/{run_name}/{uncertainty_variant} "
            f"is missing; the legacy structured file was used instead"
        )

    return flat_path, ""


def load_upstream_records(
    subset: str,
    model_name: str = DEFAULT_MODEL,
    uncertainty_variant: str = "eara_conformal",
    tag: str = "",
) -> list[dict[str, Any]]:
    """Build decision records from uncertainty and explainability outputs.

    ``tag`` selects a specific run of the model (e.g. ``ms42``).  When it is
    given, the predictions, the calibrated intervals and the XAI report are
    all read from that run, so a maintenance decision is never made from a
    mixture of two different sets of weights.
    """
    subset = subset.upper()
    model_name = model_name.lower()
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unsupported decision model: {model_name}")

    run_name = run_dir_name(model_name, tag)

    prediction_path = RESULTS_DIR / f"{subset}_{run_name}_predictions.npz"
    if uncertainty_variant not in UNCERTAINTY_VARIANTS:
        raise ValueError(f"Unsupported uncertainty variant: {uncertainty_variant}")
    uncertainty_path, _ = resolve_uncertainty_path(
        subset,
        run_name,
        uncertainty_variant,
    )
    eri_path = XAI_DIR / subset / run_name / "eri_rul.json"

    missing = [
        path for path in (prediction_path, uncertainty_path, eri_path)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing upstream decision artifact(s) for {subset}: {missing_text}. "
            f"Run uncertainty.py and explain.py with --subset {subset} "
            f"--model {model_name}"
            + (f" --tag {tag}" if tag else "")
            + "."
        )

    with np.load(prediction_path) as predictions, np.load(uncertainty_path) as uncertainty:
        rul = np.asarray(uncertainty["y_rul_pred"]).reshape(-1)
        hi = np.clip(np.asarray(predictions["y_hi_pred"]).reshape(-1), 0.0, 1.0)
        lower = np.asarray(uncertainty[f"lower_{INTERVAL_LEVEL}"]).reshape(-1)
        upper = np.asarray(uncertainty[f"upper_{INTERVAL_LEVEL}"]).reshape(-1)

        count = min(len(rul), len(hi), len(lower), len(upper))
        records = [
            {
                "engine_id": index + 1,
                "rul": rul[index],
                "hi": hi[index],
                "rul_lower": lower[index],
                "rul_upper": upper[index],
            }
            for index in range(count)
        ]

    eri_data = load_json(eri_path)
    eri = eri_data.get("eri")
    if eri is None:
        raise ValueError(f"XAI output does not contain 'eri': {eri_path}")
    importance = eri_data.get("sensor_importance", {}).get("ig", [])
    top_k_sensors = [
        index + 1
        for index in np.argsort(np.asarray(importance))[::-1][:5]
    ]
    for record in records:
        record["eri"] = eri
        record["top_k_sensors"] = top_k_sensors
    return records


# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------

_ACTION_BADGE = {
    "URGENT_MAINTENANCE": ("🔴", "#c0392b"),
    "INSPECT":            ("🟡", "#d68910"),
    "CONTINUE_OPERATION": ("🟢", "#1e8449"),
}

_HEALTH_BADGE = {
    "CRITICAL":  "#c0392b",
    "AT_RISK":   "#d68910",
    "DEGRADING": "#2471a3",
    "HEALTHY":   "#1e8449",
}


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600">{text}</span>'
    )


def _decision_card(d: dict[str, Any]) -> str:
    action  = d.get("recommended_action", "—")
    health  = d.get("health_state", "—")
    icon, ac = _ACTION_BADGE.get(action, ("⚪", "#555"))
    hc       = _HEALTH_BADGE.get(health, "#555")
    eng_id   = d.get("engine_id", d.get("index", "—"))
    sensors  = ", ".join(f"S{s}" for s in d.get("top_k_sensors", []))
    reasons  = "".join(f"<li>{r}</li>" for r in d.get("reasons", []))
    violations = d.get("constraint_violations", [])
    viol_html  = (
        "".join(f"<li style='color:#c0392b'>{v}</li>" for v in violations)
        if violations else "<li style='color:#1e8449'>None</li>"
    )
    inp = d.get("inputs", {})

    return f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px;
                background:#fafafa;box-shadow:0 1px 3px rgba(0,0,0,.08)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <h3 style="margin:0">Engine {eng_id}</h3>
        <div>{_badge(f"{icon} {action}", ac)}&nbsp;{_badge(health, hc)}</div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.9em">
        <tr style="background:#eef2f7">
          <th style="padding:6px 10px;text-align:left">Input</th>
          <th style="padding:6px 10px;text-align:left">Value</th>
          <th style="padding:6px 10px;text-align:left">Metric</th>
          <th style="padding:6px 10px;text-align:left">Value</th>
        </tr>
        <tr>
          <td style="padding:5px 10px">RUL</td>
          <td style="padding:5px 10px"><b>{inp.get('rul', '—')}</b></td>
          <td style="padding:5px 10px">Risk Score</td>
          <td style="padding:5px 10px"><b>{d.get('risk_score', '—'):.4f}</b></td>
        </tr>
        <tr style="background:#f5f5f5">
          <td style="padding:5px 10px">HI</td>
          <td style="padding:5px 10px"><b>{inp.get('hi', '—')}</b></td>
          <td style="padding:5px 10px">Urgency Index</td>
          <td style="padding:5px 10px"><b>{d.get('urgency_index', '—'):.4f}</b></td>
        </tr>
        <tr>
          <td style="padding:5px 10px">RUL Interval</td>
          <td style="padding:5px 10px">[{inp.get('rul_lower', '—')}, {inp.get('rul_upper', '—')}]</td>
          <td style="padding:5px 10px">Uncertainty</td>
          <td style="padding:5px 10px">{d.get('uncertainty_level', '—')}</td>
        </tr>
        <tr style="background:#f5f5f5">
          <td style="padding:5px 10px">ERI</td>
          <td style="padding:5px 10px"><b>{inp.get('eri', '—')}</b></td>
          <td style="padding:5px 10px">Explanation Reliability</td>
          <td style="padding:5px 10px">{d.get('explanation_reliability', '—')}</td>
        </tr>
        <tr>
          <td style="padding:5px 10px">Top Sensors</td>
          <td style="padding:5px 10px" colspan="3">{sensors or '—'}</td>
        </tr>
      </table>
      <div style="margin-top:10px;display:flex;gap:24px">
        <div style="flex:1">
          <b>Reasons:</b><ul style="margin:4px 0 0 16px;padding:0">{reasons}</ul>
        </div>
        <div style="flex:1">
          <b>Constraint Violations:</b>
          <ul style="margin:4px 0 0 16px;padding:0">{viol_html}</ul>
        </div>
      </div>
      {"<div style='margin-top:8px;color:#d68910;font-weight:600'>⚠ Human Review Required</div>" if d.get('human_review') else ""}
    </div>"""


def build_html_report(output: dict[str, Any], input_stem: str) -> str:
    summary   = output["summary"]
    decisions = output["decisions"]

    action_rows = "".join(
        f"<tr><td style='padding:4px 12px'>{k}</td>"
        f"<td style='padding:4px 12px;font-weight:600'>{v}</td></tr>"
        for k, v in summary["action_distribution"].items()
    )
    health_rows = "".join(
        f"<tr><td style='padding:4px 12px'>{k}</td>"
        f"<td style='padding:4px 12px;font-weight:600'>{v}</td></tr>"
        for k, v in summary["health_state_distribution"].items()
    )
    cards = "".join(_decision_card(d) for d in decisions)

    risk_line = (
        f"Risk Score — mean: <b>{summary.get('risk_score_mean', '—')}</b> &nbsp;|&nbsp; "
        f"max: <b>{summary.get('risk_score_max', '—')}</b><br>"
        f"Urgency Index — mean: <b>{summary.get('urgency_index_mean', '—')}</b> &nbsp;|&nbsp; "
        f"max: <b>{summary.get('urgency_index_max', '—')}</b>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>O4 Decision Report — {input_stem}</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 24px 32px;
            background: #f0f2f5; color: #222; }}
    h1   {{ color: #1a252f; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; }}
    h2   {{ color: #2c3e50; margin-top: 28px; }}
    table {{ border-collapse: collapse; background: #fff; border-radius: 6px;
             overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    th   {{ background: #2c3e50; color: #fff; padding: 8px 14px; text-align: left; }}
    td   {{ border-bottom: 1px solid #eee; }}
  </style>
</head>
<body>
  <h1>O4 Explainable Decision Intelligence Engine</h1>
  <p><b>Input:</b> {input_stem} &nbsp;|&nbsp;
     <b>Decisions:</b> {summary['num_decisions']} &nbsp;|&nbsp;
     <b>Human Reviews:</b> {summary['human_review_count']} &nbsp;|&nbsp;
     <b>Constraint Violations:</b> {summary['constraint_violation_count']}</p>
  <p>{risk_line}</p>

  <h2>Action Distribution</h2>
  <table><tr><th>Action</th><th>Count</th></tr>{action_rows}</table>

  <h2>Health State Distribution</h2>
  <table><tr><th>Health State</th><th>Count</th></tr>{health_rows}</table>

  <h2>Individual Decisions</h2>
  {cards}
</body>
</html>"""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="O4 Explainable Decision Intelligence Engine"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=[*ALL_SUBSETS, "all", *[subset.lower() for subset in ALL_SUBSETS]],
        help="C-MAPSS subset to process, or 'all'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=[*ALL_MODELS, "all"],
        help="Prognostics model to process, or 'all'.",
    )
    parser.add_argument(
        "--uncertainty-variant",
        choices=UNCERTAINTY_VARIANTS,
        default="eara_conformal",
        help="Calibrated uncertainty artifact to use.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Checkpoint tag (e.g. 'ms42'). Decides using the predictions, "
            "intervals and XAI report of that specific run, so all upstream "
            "evidence comes from the same set of weights."
        ),
    )
    parser.add_argument("--xai-context", type=str, default=None,
                        help="Optional JSON file with top_k_sensors per record from O3.")
    parser.add_argument("--maintenance-window",    action="store_true", default=False)
    parser.add_argument("--no-maintenance-window", action="store_true", default=False)
    parser.add_argument("--minimum-safe-rul",      type=float, default=None)
    parser.add_argument("--maintenance-lead-time", type=float, default=None)
    parser.add_argument("--max-allowable-delay",   type=float, default=None)
    parser.add_argument("--resource-availability", type=lambda x: x.lower() == "true",
                        default=None,
                        help="true/false — whether maintenance resources are available.")
    parser.add_argument(
        "--constraint-scenario",
        type=str,
        default="",
        choices=["", *scenario_names()],
        help=(
            "Named operational-constraint scenario to apply.  Explicit "
            "constraint flags override the scenario's values.  Use "
            "'unconstrained' for the reference reading with no constraints."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help=(
            "Optional JSON file with policy threshold overrides, e.g. "
            '{"w_rul": 0.5, "rul_critical": 8.0}.'
        ),
    )
    return parser.parse_args()


def build_constraints(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the named scenario with the explicit constraint flags.

    The explicit flags win: a scenario provides the baseline operational
    context, and a flag overrides one of its values.
    """

    constraints: dict[str, Any] = {}

    if args.constraint_scenario:
        constraints.update(get_scenario(args.constraint_scenario))

    if args.maintenance_window:
        constraints["maintenance_window_available"] = True
    if args.no_maintenance_window:
        constraints["maintenance_window_available"] = False
    if args.minimum_safe_rul is not None:
        constraints["minimum_safe_rul"] = args.minimum_safe_rul
    if args.maintenance_lead_time is not None:
        constraints["maintenance_lead_time"] = args.maintenance_lead_time
    if args.max_allowable_delay is not None:
        constraints["max_allowable_delay"] = args.max_allowable_delay
    if args.resource_availability is not None:
        constraints["resource_availability"] = args.resource_availability
    return constraints


def build_thresholds(args: argparse.Namespace, subset: str) -> dict[str, float]:
    """Subset thresholds (with the calibrated HI breakpoints) plus overrides."""

    thresholds = load_thresholds(subset)

    if args.thresholds:
        overrides = load_json(args.thresholds)
        if not isinstance(overrides, dict):
            raise ValueError(
                f"The thresholds file {args.thresholds} must contain a JSON object."
            )
        thresholds.update({key: float(value) for key, value in overrides.items()})

    return thresholds


def _merge_xai_context(
    records: list[dict[str, Any]],
    xai_path: str | None,
) -> list[dict[str, Any]]:
    """Merge top_k_sensors from a separate XAI context file if provided."""
    if xai_path is None:
        return records
    xai_data = load_json(xai_path)
    if not isinstance(xai_data, list):
        xai_data = [xai_data]
    for i, record in enumerate(records):
        if i < len(xai_data) and "top_k_sensors" in xai_data[i]:
            record.setdefault("top_k_sensors", xai_data[i]["top_k_sensors"])
    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_subset(
    subset: str,
    model_name: str,
    args: argparse.Namespace,
    constraints: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    stem = f"{subset}_{run_dir_name(model_name, args.tag)}"
    log = get_script_logger("decision", f"decision_{stem}")
    result_path = RESULTS_DIR / f"decision_{stem}.json"
    report_path = REPORTS_DIR / f"decision_{stem}.html"

    log.info("=" * 60)
    log.info("O4 EXPLAINABLE DECISION INTELLIGENCE ENGINE")
    log.info("=" * 60)
    log.info("Subset : %s", subset)
    log.info("Model  : %s", model_name)
    if args.tag:
        log.info("Tag    : %s", args.tag)
    if args.constraint_scenario:
        log.info(
            "Scenario: %s (%s)",
            args.constraint_scenario,
            describe_scenario(args.constraint_scenario),
        )
    log.info(
        "Input  : outputs/results/%s_%s_* + outputs/xai/%s/%s",
        subset, run_dir_name(model_name, args.tag), subset,
        run_dir_name(model_name, args.tag),
    )
    log.info("Result : %s", result_path)
    log.info("Report : %s", report_path)

    records = load_upstream_records(
        subset,
        model_name,
        args.uncertainty_variant,
        tag=args.tag,
    )
    _, path_note = resolve_uncertainty_path(
        subset,
        run_dir_name(model_name, args.tag),
        args.uncertainty_variant,
    )
    if path_note:
        log.warning("Interval artifact: %s", path_note)
    log.info(
        "Loaded %d decision record(s) from %s uncertainty and %s XAI outputs.",
        len(records), model_name, model_name,
    )
    records = _merge_xai_context(records, args.xai_context)

    if constraints:
        log.info("Operational constraints:")
        for key, value in constraints.items():
            log.info("  %s: %s", key, value)
    else:
        log.info("No additional operational constraints supplied.")

    log.info("Generating maintenance recommendations...")
    decisions = process_records(
        records=records,
        constraints=constraints,
        thresholds=thresholds,
    )
    summary = summarize_decisions(decisions)

    log.info("Decisions             : %d", summary["num_decisions"])
    log.info("Human review required : %d", summary["human_review_count"])
    log.info("Constraint violations : %d", summary["constraint_violation_count"])
    log.info(
        "Records with a violation: %d",
        summary["constraint_violated_records"],
    )

    output = {
        "objective": "O4",
        "description": (
            "Explainable Decision Intelligence Engine "
            "for maintenance recommendation."
        ),
        "subset": subset,
        "model": model_name,
        "tag": args.tag,
        "constraint_scenario": args.constraint_scenario or "unconstrained",
        "thresholds": thresholds,
        "inputs": ["RUL", "HI", "RUL uncertainty interval", "ERI",
                   "top_k_sensors", "operational constraints"],
        "summary": summary,
        "decisions": decisions,
        "xai_context": args.xai_context,
        "operational_constraints": constraints,
    }

    save_json(output, result_path)
    log.info("Result saved  : %s", result_path)

    html = build_html_report(output, stem)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    log.info("Report saved  : %s", report_path)
    log.info("DECISION ENGINE COMPLETE FOR %s / %s", subset, model_name)


def main() -> None:
    args = parse_args()
    subsets = ALL_SUBSETS if args.subset.lower() == "all" else (args.subset.upper(),)
    models = ALL_MODELS if args.model.lower() == "all" else (args.model.lower(),)
    constraints = build_constraints(args)

    for subset in subsets:
        thresholds = build_thresholds(args, subset)
        for model_name in models:
            run_subset(
                subset=subset,
                model_name=model_name,
                args=args,
                constraints=constraints,
                thresholds=thresholds,
            )


if __name__ == "__main__":
    main()
