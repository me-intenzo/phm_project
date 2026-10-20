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

Usage (PowerShell)
------------------
Activate the environment, then merge the seed list from each EARA run_config.json:
    ./.venv/Scripts/Activate.ps1
    python scripts/decision.py --subset all --model gru

To explicitly provide five or more distinct seeds:
    python scripts/decision.py --subset all --model gru --seeds 42 43 44 45 46

This writes one merged decision output per subset/model, not one output per seed.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision_engine.recommender import recommend_action
from src.utils.logger import get_script_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR  = PROJECT_RESULTS_DIR / "uncertainty"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "reports"
ALL_SUBSETS  = ("FD001", "FD002", "FD003", "FD004")
ALL_MODELS   = ("hybrid", "lstm", "gru", "transformer")
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
XAI_DIR      = PROJECT_ROOT / "outputs" / "xai"
DEFAULT_MODEL  = "gru"
INTERVAL_LEVEL = 90
UNCERTAINTY_VARIANTS = (
    "eara_conformal",
    "conformal",
    "adaptive_conformal",
    "cqr",
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
        result = {"index": index, **decision}
        for key in ("id", "engine_id", "unit_id", "timestamp"):
            if key in record:
                result[key] = record[key]
        for key in (
            "seed_count", "seed_action_counts", "seed_action_agreement", "seed_actions"
        ):
            if key in record:
                result[key] = record[key]
        if record.get("human_review"):
            result["human_review"] = True
            result.setdefault("reasons", []).append(
                "Seed-specific recommendations disagree; review the merged recommendation."
            )
        decisions.append(result)
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
        constraint_violations += len(d.get("constraint_violations", []))

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


def load_upstream_records(
    subset: str,
    model_name: str = DEFAULT_MODEL,
    uncertainty_variant: str = "eara_conformal",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build decision records from uncertainty and explainability outputs."""
    subset = subset.upper()
    model_name = model_name.lower()
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unsupported decision model: {model_name}")

    if uncertainty_variant not in UNCERTAINTY_VARIANTS:
        raise ValueError(f"Unsupported uncertainty variant: {uncertainty_variant}")
    method_dir, artifact_suffix = {
        "eara_conformal": ("eara", "eara_conformal"),
        "conformal": ("split", "conformal"),
        "adaptive_conformal": ("adaptive", "adaptive_conformal"),
        "cqr": ("cqr", "cqr"),
    }[uncertainty_variant]
    seed_root = (
        RESULTS_DIR
        / model_name
        / subset
        / "eara"
        / f"seed_{seed}"
    )
    prediction_path = seed_root / "results" / f"{subset}_{model_name}_predictions.npz"
    multiseed_path = (
        RESULTS_DIR
        / model_name
        / subset
        / method_dir
        / f"seed_{seed}"
        / "results"
        / f"{subset}_{model_name}_{artifact_suffix}.npz"
    )
    structured_path = RESULTS_DIR / uncertainty_variant / subset / f"{model_name}.npz"
    legacy_path = PROJECT_RESULTS_DIR / f"{subset}_{model_name}_{artifact_suffix}.npz"
    uncertainty_path = next(
        (path for path in (multiseed_path, structured_path, legacy_path) if path.exists()),
        multiseed_path,
    )
    eri_path = seed_root / "xai" / subset / model_name / "eri_rul.json"

    missing = [
        path for path in (prediction_path, uncertainty_path, eri_path)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing upstream decision artifact(s) for {subset}: {missing_text}. "
            f"Run uncertainty.py and explain.py with --subset {subset} "
            f"--model {model_name}; uncertainty seed {seed} was not found."
        )

    with np.load(prediction_path) as predictions, np.load(uncertainty_path) as uncertainty:
        rul = np.asarray(uncertainty["y_rul_pred"]).reshape(-1)
        hi = np.clip(np.asarray(predictions["y_hi_pred"]).reshape(-1), 0.0, 1.0)
        lower = np.asarray(uncertainty[f"lower_{INTERVAL_LEVEL}"]).reshape(-1)
        upper = np.asarray(uncertainty[f"upper_{INTERVAL_LEVEL}"]).reshape(-1)
        engine_ids = np.asarray(predictions["engine_ids"]).reshape(-1)
        if not np.allclose(
            np.asarray(uncertainty["y_rul_test"]).reshape(-1),
            np.asarray(predictions["y_rul_true"]).reshape(-1),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                f"Seed {seed} prediction and conformal artifacts have different "
                "test-target order/content. Recreate them from the same subset data."
            )
        lengths = {len(rul), len(hi), len(lower), len(upper), len(engine_ids)}
        if len(lengths) != 1:
            raise ValueError(
                f"Seed {seed} has misaligned predictions/intervals/engine IDs: "
                f"{sorted(lengths)} lengths. No records were dropped."
            )
        records = [
            {
                "engine_id": int(engine_ids[index]),
                "rul": rul[index],
                "hi": hi[index],
                "rul_lower": lower[index],
                "rul_upper": upper[index],
            }
            for index in range(len(engine_ids))
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
        record["seed"] = seed
    return records


def resolve_seeds(subset: str, model_name: str, requested: list[int] | None) -> list[int]:
    if requested:
        seeds = requested
    else:
        config_path = (
            RESULTS_DIR / model_name / subset / "eara" / "run_config.json"
        )
        if config_path.exists():
            configured = load_json(config_path).get("seeds", [])
            seeds = [int(seed) for seed in configured]
        else:
            seed_root = RESULTS_DIR / model_name / subset / "eara"
            seeds = sorted(
                int(match.group(1))
                for path in seed_root.glob("seed_*")
                if (match := re.fullmatch(r"seed_(\d+)", path.name))
            )
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError(
            f"Expected at least five distinct seed artifacts for {subset}/{model_name}; "
            f"found {seeds}. Complete the multi-seed uncertainty run first."
        )
    return list(seeds)


def ensure_seed_evaluation(subset: str, model_name: str, seeds: list[int]) -> None:
    """Create missing prediction/XAI artifacts for the complete seed list."""
    seed_roots = [
        RESULTS_DIR / model_name / subset / "eara" / f"seed_{seed}"
        for seed in seeds
    ]
    predictions = [
        root / "results" / f"{subset}_{model_name}_predictions.npz"
        for root in seed_roots
    ]
    eri_files = [
        root / "xai" / subset / model_name / "eri_rul.json"
        for root in seed_roots
    ]
    if any(not path.exists() for path in predictions):
        subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--model", model_name, "--subset", subset, "--seeds",
                *[str(seed) for seed in seeds],
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    if any(not path.exists() for path in eri_files):
        subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "scripts" / "explain.py"),
                "--model", model_name, "--subset", subset, "--target", "rul",
                "--seeds", *[str(seed) for seed in seeds],
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    missing = [path for path in (*predictions, *eri_files) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Seed preparation did not produce required artifacts: "
            + ", ".join(str(path) for path in missing)
        )


def merge_seed_records(
    seed_records: dict[int, list[dict[str, Any]]],
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Merge aligned per-seed estimates into one auditable record per engine."""
    seeds = sorted(seed_records)
    engine_sets = {
        seed: {int(record["engine_id"]) for record in records}
        for seed, records in seed_records.items()
    }
    duplicate_ids = {
        seed: len(records) - len(engine_sets[seed])
        for seed, records in seed_records.items()
        if len(records) != len(engine_sets[seed])
    }
    if duplicate_ids:
        raise ValueError(f"Duplicate engine records found by seed: {duplicate_ids}")
    reference = engine_sets[seeds[0]]
    mismatched = {
        seed: sorted(reference.symmetric_difference(engine_ids))
        for seed, engine_ids in engine_sets.items()
        if engine_ids != reference
    }
    if mismatched:
        raise ValueError(f"Engine IDs differ across seeds; refusing lossy merge: {mismatched}")

    by_seed = {
        seed: {int(record["engine_id"]): record for record in records}
        for seed, records in seed_records.items()
    }
    merged = []
    for engine_id in sorted(reference):
        rows = [by_seed[seed][engine_id] for seed in seeds]
        for key in ("rul", "hi", "rul_lower", "rul_upper", "eri"):
            if any(not np.isfinite(float(row[key])) for row in rows):
                raise ValueError(f"Non-finite {key} for engine {engine_id} across seeds.")
        if any(row["rul_lower"] > row["rul_upper"] for row in rows):
            raise ValueError(f"Invalid seed interval for engine {engine_id}.")
        sensor_votes = collections.Counter(
            sensor for row in rows for sensor in row.get("top_k_sensors", [])
        )
        aggregate = {
            "engine_id": engine_id,
            "rul": float(np.median([row["rul"] for row in rows])),
            "hi": float(np.median([row["hi"] for row in rows])),
            "rul_lower": float(min(row["rul_lower"] for row in rows)),
            "rul_upper": float(max(row["rul_upper"] for row in rows)),
            "eri": float(np.median([row["eri"] for row in rows])),
            "top_k_sensors": [
                int(sensor) for sensor, _ in sensor_votes.most_common(5)
            ],
        }
        seed_decisions = {
            seed: make_decision(
                rows[index]["rul"], rows[index]["hi"],
                rows[index]["rul_lower"], rows[index]["rul_upper"],
                rows[index]["eri"], rows[index].get("top_k_sensors"),
                constraints=constraints,
            )
            for index, seed in enumerate(seeds)
        }
        actions = [d.get("recommended_action", "UNKNOWN") for d in seed_decisions.values()]
        counts = collections.Counter(actions)
        aggregate["seed_count"] = len(seeds)
        aggregate["seed_action_counts"] = dict(counts)
        aggregate["seed_action_agreement"] = max(counts.values()) / len(seeds)
        aggregate["seed_actions"] = {str(seed): action for seed, action in zip(seeds, actions)}
        aggregate["human_review"] = len(counts) > 1
        merged.append(aggregate)
    return merged


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
    seed_count = d.get("seed_count")
    agreement = d.get("seed_action_agreement")
    seed_line = (
        f"<small>{seed_count} seeds · action agreement {agreement:.0%}</small>"
        if seed_count and agreement is not None else ""
    )
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
        <div><h3 style="margin:0">Engine {eng_id}</h3>{seed_line}</div>
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
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "At least five distinct seeds to merge; by default read the EARA "
            "run_config.json seed list."
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
    return parser.parse_args()


def build_constraints(args: argparse.Namespace) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
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
) -> None:
    stem = f"{subset}_{model_name}"
    log = get_script_logger("decision", f"decision_{stem}")
    result_path = PROJECT_RESULTS_DIR / f"decision_{stem}.json"
    report_path = REPORTS_DIR / f"decision_{stem}.html"

    log.info("=" * 60)
    log.info("O4 EXPLAINABLE DECISION INTELLIGENCE ENGINE")
    log.info("=" * 60)
    log.info("Subset : %s", subset)
    log.info("Model  : %s", model_name)
    log.info(
        "Input  : outputs/results/%s_%s_* + outputs/xai/%s/%s",
        subset, model_name, subset, model_name,
    )
    log.info("Result : %s", result_path)
    log.info("Report : %s", report_path)

    seeds = resolve_seeds(subset, model_name, args.seeds)
    ensure_seed_evaluation(subset, model_name, seeds)
    seed_records = {
        seed: load_upstream_records(
            subset, model_name, args.uncertainty_variant, seed=seed
        )
        for seed in seeds
    }
    records = merge_seed_records(seed_records, constraints=constraints)
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
    decisions = process_records(records=records, constraints=constraints)
    summary = summarize_decisions(decisions)

    log.info("Decisions             : %d", summary["num_decisions"])
    log.info("Human review required : %d", summary["human_review_count"])
    log.info("Constraint violations : %d", summary["constraint_violation_count"])

    output = {
        "objective": "O4",
        "description": (
            "Explainable Decision Intelligence Engine "
            "for maintenance recommendation."
        ),
        "subset": subset,
        "model": model_name,
        "seed_aggregation": {
            "seeds": seeds,
            "seed_count": len(seeds),
            "point_estimate": "median across seed predictions",
            "interval": "envelope across seed-specific conformal intervals",
            "conformal_guarantee": (
                "No additional formal coverage guarantee is claimed for the "
                "merged envelope; evaluate its empirical coverage separately."
            ),
        },
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
        for model_name in models:
            run_subset(
                subset=subset,
                model_name=model_name,
                args=args,
                constraints=constraints,
            )


if __name__ == "__main__":
    main()
