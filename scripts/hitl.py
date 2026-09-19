"""Human-in-the-loop review workflow for O4 decision outputs.

Examples::

    python scripts/hitl.py --subset FD004 --model gru
    python scripts/hitl.py --input outputs/results/decision_FD004_gru.json
    python scripts/hitl.py --input outputs/results/decision_FD004_gru.json --review-id decision-0-0 --reviewer operator-1 --action APPROVE
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hitl.feedback import store_feedback
from src.hitl.logger import log_feedback
from src.hitl.workflow import create_review_queue, resolve_review, summarize_review_queue
from src.utils.logger import get_script_logger

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FEEDBACK_DIR = RESULTS_DIR / "hitl_feedback"
ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ALL_MODELS = ["lstm", "gru", "transformer", "hybrid", "gru_att_deg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review human-flagged O4 decisions and persist HITL feedback."
    )
    parser.add_argument("--subset", default="FD001", help="FD001-FD004 or all")
    parser.add_argument("--model", default="gru", help="Model name or all")
    parser.add_argument("--input", default=None, help="decision.py JSON output")
    parser.add_argument("--review-id", help="Review item to resolve")
    parser.add_argument("--reviewer", help="Human reviewer identifier")
    parser.add_argument("--action", choices=["APPROVE", "OVERRIDE", "REJECT"])
    parser.add_argument("--override-action", help="Final action for OVERRIDE")
    parser.add_argument("--comment", default="", help="Reviewer comment")
    return parser.parse_args()


def _validate_choice(value: str, choices: list[str], name: str) -> None:
    if value != "all" and value not in choices:
        raise ValueError(f"Unsupported {name} '{value}'. Choose {choices} or all.")


def _decision_input_path(subset: str, model: str) -> Path:
    candidates = [
        RESULTS_DIR / f"decision_{subset}_{model}.json",
        RESULTS_DIR / f"decision_{subset}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Decision output not found: " + ", ".join(map(str, candidates)))


def load_decision_output(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Decision output not found: {input_path}")
    output = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(output, dict) or not isinstance(output.get("decisions"), list):
        raise ValueError("Decision output must contain a 'decisions' list.")
    return output


def _load_or_create_queue(
    decision_output: dict[str, Any],
    queue_path: Path,
    input_path: Path,
) -> list[dict[str, Any]]:
    if queue_path.exists():
        saved = json.loads(queue_path.read_text(encoding="utf-8"))
        saved_source = saved.get("source_decision_output") if isinstance(saved, dict) else None
        same_source = saved_source and Path(saved_source).resolve() == input_path.resolve()
        if same_source and isinstance(saved.get("review_queue"), list):
            return saved["review_queue"]
    return create_review_queue(decision_output["decisions"])


def build_output(decision_output: dict[str, Any], queue: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    subset = decision_output.get("subset", input_path.stem.removeprefix("decision_"))
    return {
        "objective": "HITL",
        "source_decision_output": str(input_path),
        "subset": subset,
        "model": decision_output.get("model"),
        "decision_summary": decision_output.get("summary", {}),
        "summary": summarize_review_queue(decision_output["decisions"], queue),
        "review_queue": queue,
    }


def build_html_report(output: dict[str, Any]) -> str:
    summary = output["summary"]
    rows = []
    for item in output["review_queue"]:
        decision = item.get("decision", {})
        status_class = "resolved" if item.get("status") == "RESOLVED" else "pending"
        rows.append(
            f"<tr class='{status_class}'>"
            f"<td>{html.escape(str(item.get('review_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('status', '')))}</td>"
            f"<td>{html.escape(str(decision.get('recommended_action', '')))}</td>"
            f"<td>{html.escape(str(item.get('final_action') or ''))}</td>"
            f"<td>{html.escape(str(decision.get('health_state', '')))}</td>"
            f"<td>{html.escape(str(decision.get('risk_score', '')))}</td>"
            f"<td>{html.escape(str(item.get('reviewer') or ''))}</td>"
            f"<td>{html.escape(str(item.get('comment') or ''))}</td></tr>"
        )
    table = "".join(rows) or "<tr><td colspan='8'>No human review required.</td></tr>"
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>HITL Review {html.escape(str(output.get('subset', '')))}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#222}}
header{{border-bottom:3px solid #2c3e50;margin-bottom:20px}}
.metric{{display:inline-block;margin-right:24px;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #d8d8d8;padding:8px;text-align:left;vertical-align:top}}
th{{background:#2c3e50;color:white}}.pending{{background:#fff8e1}}
.resolved{{background:#eef8ee}}</style></head><body>
<header><h1>Human-in-the-loop decision review</h1>
<p><b>Source:</b> {html.escape(str(output['source_decision_output']))}<br>
<b>Subset:</b> {html.escape(str(output.get('subset', '')))} &nbsp;
<b>Model:</b> {html.escape(str(output.get('model') or 'not specified'))}</p></header>
<p><span class='metric'>Decisions: {summary['decision_count']}</span>
<span class='metric'>Human review: {summary['human_review_count']}</span>
<span class='metric'>Pending: {summary['pending_review_count']}</span>
<span class='metric'>Resolved: {summary['resolved_review_count']}</span></p>
<p>Resolve reviews with <code>--review-id</code>, <code>--reviewer</code>, and
<code>--action</code>. The report is regenerated after each resolution.</p>
<table><tr><th>Review ID</th><th>Status</th><th>Recommended</th>
<th>Final action</th><th>Health</th><th>Risk</th><th>Reviewer</th><th>Comment</th></tr>
{table}</table></body></html>"""


def run(input_path: Path, reviewer: str | None = None, review_id: str | None = None,
        action: str | None = None, override_action: str | None = None,
        comment: str = "") -> None:
    subset = input_path.stem.removeprefix("decision_")
    output_path = RESULTS_DIR / f"hitl_{subset}.json"
    report_path = REPORTS_DIR / f"hitl_{subset}.html"
    log = get_script_logger("hitl", f"hitl_{subset}")
    decision_output = load_decision_output(input_path)
    queue = _load_or_create_queue(decision_output, output_path, input_path)

    supplied_resolution = any(value is not None for value in (review_id, reviewer, action, override_action)) or bool(comment)
    if supplied_resolution:
        if not all((review_id, reviewer, action)):
            raise ValueError("Resolving a review requires --review-id, --reviewer, and --action")
        matches = [item for item in queue if item.get("review_id") == review_id]
        if not matches:
            raise ValueError(f"Review ID not found: {review_id}")
        resolved = resolve_review(matches[0], reviewer, action, comment, override_action)
        store_feedback(resolved, FEEDBACK_DIR / f"feedback_{subset}.jsonl")
        log_feedback(log, resolved)

    output = build_output(decision_output, queue, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    report_path.write_text(build_html_report(output), encoding="utf-8")
    log.info("Decision source: %s", input_path)
    log.info("Decisions=%d | pending=%d | resolved=%d", output["summary"]["decision_count"], output["summary"]["pending_review_count"], output["summary"]["resolved_review_count"])
    log.info("Result saved: %s", output_path)
    log.info("Report saved: %s", report_path)


def main() -> None:
    args = parse_args()
    _validate_choice(args.subset, ALL_SUBSETS, "subset")
    _validate_choice(args.model, ALL_MODELS, "model")
    if args.input:
        run(Path(args.input), args.reviewer, args.review_id, args.action, args.override_action, args.comment)
        return

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    models = ALL_MODELS if args.model == "all" else [args.model]
    failures = 0
    for subset in subsets:
        for model in models:
            try:
                run(_decision_input_path(subset, model), args.reviewer, args.review_id, args.action, args.override_action, args.comment)
            except (FileNotFoundError, ValueError) as exc:
                failures += 1
                print(f"[SKIP] {subset}/{model}: {exc}")
    if failures and failures == len(subsets) * len(models):
        raise SystemExit("No HITL workflows completed.")


if __name__ == "__main__":
    main()
