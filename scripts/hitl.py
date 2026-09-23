"""
Human-in-the-Loop Decision Refinement
========================================

Workflow:
     decision output
        ↓
    review queue
        ↓
    APPROVE / OVERRIDE
        ↓
    ExpertFeedback
        ↓
    append-only JSONL log
        ↓
    accumulated policy evidence
        ↓
    conservative policy refinement
        ↓
    operational constraint re-check
        ↓
       output

The prognostic GRU/model is never retrained by Human-in-the-Loop.

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

from src.hitl.feedback import VALID_ACTIONS
from src.hitl.logger import (
    DEFAULT_LOG_PATH,
    feedback_statistics,
    load_feedback,
    log_feedback,
)
from src.hitl.updater import (
    build_policy_profile,
    refine_recommendation,
)
from src.hitl.workflow import (
    create_review_queue,
    resolve_review,
    review_to_feedback,
    summarize_review_queue,
)
from src.utils.logger import get_script_logger


RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

ALL_SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]

ALL_MODELS = [
    "lstm",
    "gru",
    "transformer",
    "hybrid",
    "gru_att_deg",
]

CONTROLLED_BASELINE_MODEL = "gru"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "O5 Human-in-the-Loop review and "
            "adaptive decision-policy refinement."
        )
    )

    parser.add_argument(
        "--subset",
        default="FD001",
        help="FD001-FD004 or all.",
    )

    parser.add_argument(
        "--model",
        default="gru",
        help="Model name or all.",
    )

    parser.add_argument(
        "--input",
        default=None,
        help="O4 decision JSON output.",
    )

    parser.add_argument(
        "--review-id",
        help="Review item to resolve.",
    )

    parser.add_argument(
        "--reviewer",
        help="Human reviewer identifier.",
    )

    parser.add_argument(
        "--action",
        choices=[
            "APPROVE",
            "OVERRIDE",
        ],
        help="Human review action.",
    )

    parser.add_argument(
        "--override-action",
        choices=sorted(VALID_ACTIONS),
        help="Alternative maintenance action for OVERRIDE.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        help="Expert confidence in [0,1].",
    )

    parser.add_argument(
        "--reason",
        default="",
        help="Reason/comment supporting the expert decision.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_choice(
    value: str,
    choices: list[str],
    name: str,
) -> None:

    value = value.upper() if name == "subset" else value.lower()

    normalized = [
        choice.upper()
        if name == "subset"
        else choice.lower()
        for choice in choices
    ]

    if value != "all" and value not in normalized:
        raise ValueError(
            f"Unsupported {name} '{value}'. "
            f"Choose {choices} or all."
        )


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _decision_input_path(
    subset: str,
    model: str,
) -> Path:

    candidates = [
        RESULTS_DIR
        / f"decision_{subset}_{model}.json",

        RESULTS_DIR
        / f"decision_{subset}.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Decision output not found: "
        + ", ".join(
            map(str, candidates)
        )
    )


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

def load_json(
    path: str | Path,
) -> Any:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json(
    data: Any,
    path: str | Path,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Decision loading
# ---------------------------------------------------------------------

def load_decision_output(
    path: str | Path,
) -> dict[str, Any]:

    output = load_json(path)

    if not isinstance(output, dict):
        raise ValueError(
            "O4 output must be a JSON object."
        )

    if not isinstance(
        output.get("decisions"),
        list,
    ):
        raise ValueError(
            "O4 output must contain a "
            "'decisions' list."
        )

    return output


# ---------------------------------------------------------------------
# Review queue persistence
# ---------------------------------------------------------------------

def _load_or_create_queue(
    decision_output: dict[str, Any],
    queue_path: Path,
    input_path: Path,
) -> list[dict[str, Any]]:

    if queue_path.exists():

        try:
            saved = load_json(queue_path)
        except Exception:
            saved = None

        if isinstance(saved, dict):

            saved_source = saved.get(
                "source_decision_output"
            )

            if (
                saved_source
                and Path(
                    saved_source
                ).resolve()
                == input_path.resolve()
                and isinstance(
                    saved.get(
                        "review_queue"
                    ),
                    list,
                )
            ):
                return saved[
                    "review_queue"
                ]

    return create_review_queue(
        decision_output["decisions"]
    )


# ---------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------

def _extract_constraints(
    decision_output: dict[str, Any],
) -> dict[str, Any]:

    """
    Retrieve operational constraints if O4 stored them.

    Older O4 result files may not contain this metadata.
    In that case, adaptation proceeds without reapplying
    configurable constraints.
    """

    constraints = decision_output.get(
        "operational_constraints",
        {},
    )

    if not isinstance(
        constraints,
        dict,
    ):
        return {}

    return constraints


# ---------------------------------------------------------------------
# O5 adaptation
# ---------------------------------------------------------------------

def apply_policy_refinement(
    decisions: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Apply historical expert policy refinement.

    Current-run newly submitted feedback is intentionally excluded
    from this adaptation pass. It becomes evidence for subsequent
    recommendations, preventing same-run feedback leakage.
    """

    refined_decisions: list[
        dict[str, Any]
    ] = []

    adaptation_count = 0

    for decision in decisions:

        refined = refine_recommendation(
            decision=decision,
            feedback_records=feedback_records,
            constraints=constraints,
        )

        if refined.get(
            "adaptation_applied",
            False,
        ):
            adaptation_count += 1

        refined_decisions.append(
            refined
        )

    total = len(decisions)

    metrics = {
        "total_decisions": total,
        "adaptation_count": adaptation_count,
        "adaptations_applied": adaptation_count,
        "adaptation_rate": (
            round(
                adaptation_count / total,
                4,
            )
            if total
            else 0.0
        ),
        "recommendation_change_rate": (
            round(
                sum(
                    decision.get("recommended_action")
                    != decision.get("original_action")
                    for decision in refined_decisions
                ) / total,
                4,
            )
            if total
            else 0.0
        ),
        "constraint_violations_after_adaptation": sum(
            len(decision.get("constraint_violations", []))
            for decision in refined_decisions
        ),
    }

    return (
        refined_decisions,
        metrics,
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def build_output(
    decision_output: dict[str, Any],
    decisions: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    feedback_stats: dict[str, Any],
    adaptation_metrics: dict[str, Any],
    policy_profile: dict[str, Any],
    input_path: Path,
    historical_feedback_count: int = 0,
    current_run_feedback_count: int = 0,
) -> dict[str, Any]:

    subset = decision_output.get(
        "subset",
        input_path.stem.removeprefix(
            "decision_"
        ),
    )

    model = decision_output.get(
        "model"
    )

    return {
        "objective": "O5",

        "description": (
            "Human-in-the-Loop feedback "
            "and adaptive decision-policy "
            "refinement."
        ),

        "source_decision_output": str(
            input_path
        ),

        "subset": subset,

        "model": model,

        "model_role": (
            "controlled_baseline"
            if str(model).lower() == CONTROLLED_BASELINE_MODEL
            else "comparison_experiment"
        ),

        "controlled_baseline_model": CONTROLLED_BASELINE_MODEL,

        "adaptation_scope": (
            "decision_policy_only"
        ),

        "prognostic_model_updated": False,

        "review_summary": summarize_review_queue(
            decision_output["decisions"],
            queue,
        ),

        "feedback_statistics": (
            feedback_stats
        ),

        "feedback_provenance": {
            "historical_feedback_used_for_adaptation": historical_feedback_count,
            "current_run_feedback_recorded": current_run_feedback_count,
            "total_accumulated_feedback_in_profile": feedback_stats["total_feedback"],
            "policy_profile_uses_total_accumulated_feedback": True,
        },

        "adaptation_metrics": (
            adaptation_metrics
        ),

        "policy_profile": (
            policy_profile
        ),

        "decisions": decisions,

        "review_queue": queue,
    }


# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------

def build_html_report(
    output: dict[str, Any],
) -> str:

    feedback = output[
        "feedback_statistics"
    ]

    adaptation = output[
        "adaptation_metrics"
    ]

    review = output[
        "review_summary"
    ]

    rows = []

    for item in output[
        "review_queue"
    ]:

        decision = item.get(
            "decision",
            {},
        )

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('review_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('status', '')))}</td>"
            f"<td>{html.escape(str(decision.get('recommended_action', '')))}</td>"
            f"<td>{html.escape(str(item.get('expert_action') or ''))}</td>"
            f"<td>{html.escape(str(item.get('final_action') or ''))}</td>"
            f"<td>{html.escape(str(item.get('expert_confidence') or ''))}</td>"
            f"<td>{html.escape(str(item.get('reviewer') or ''))}</td>"
            f"<td>{html.escape(str(item.get('comment') or ''))}</td>"
            "</tr>"
        )

    table = "".join(rows)

    if not table:
        table = (
            "<tr>"
            "<td colspan='8'>"
            "No human review required."
            "</td>"
            "</tr>"
        )

    adaptation_rows = ""

    for decision in output[
        "decisions"
    ]:

        if not decision.get(
            "adaptation_applied",
            False,
        ):
            continue

        adaptation_rows += (
            "<tr>"
            f"<td>{html.escape(str(decision.get('engine_id', decision.get('index', ''))))}</td>"
            f"<td>{html.escape(str(decision.get('original_action', '')))}</td>"
            f"<td>{html.escape(str(decision.get('recommended_action', '')))}</td>"
            f"<td>{html.escape(str(decision.get('adaptation_reason', '')))}</td>"
            "</tr>"
        )

    if not adaptation_rows:
        adaptation_rows = (
            "<tr>"
            "<td colspan='4'>"
            "No policy adaptations applied."
            "</td>"
            "</tr>"
        )

    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">

<title>O5 HITL Report</title>

<style>

body {{
    font-family: Segoe UI, Arial, sans-serif;
    margin: 32px;
    color: #222;
}}

header {{
    border-bottom: 3px solid #2c3e50;
    margin-bottom: 24px;
}}

.metric {{
    display: inline-block;
    margin-right: 24px;
    font-weight: 600;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 12px;
    font-size: 14px;
}}

th,
td {{
    border: 1px solid #d8d8d8;
    padding: 8px;
    text-align: left;
    vertical-align: top;
}}

th {{
    background: #2c3e50;
    color: white;
}}

.section {{
    margin-top: 32px;
}}

</style>
</head>

<body>

<header>

<h1>O5 Human-in-the-Loop Decision Refinement</h1>

<p>
<b>Source:</b>
{html.escape(str(output["source_decision_output"]))}
</p>

<p>
<b>Subset:</b>
{html.escape(str(output.get("subset", "")))}
&nbsp;&nbsp;

<b>Model:</b>
{html.escape(str(output.get("model", "")))}

&nbsp;&nbsp;

<b>Model role:</b>
{html.escape(str(output.get("model_role", "")))}
</p>

</header>

<div class="section">

<h2>Review Metrics</h2>

<p>
<span class="metric">
Total Decisions: {review["decision_count"]}
</span>

<span class="metric">
Human Reviews: {review["human_review_count"]}
</span>

<span class="metric">
Pending: {review["pending_review_count"]}
</span>

<span class="metric">
Resolved: {review["resolved_review_count"]}
</span>
</p>

</div>


<div class="section">

<h2>Expert Feedback</h2>

<p>
<span class="metric">
Feedback: {feedback["total_feedback"]}
</span>

<span class="metric">
Accepted: {feedback["accepted"]}
</span>

<span class="metric">
Overrides: {feedback["overrides"]}
</span>

<span class="metric">
Override Rate: {feedback["override_rate"]}
</span>

<span class="metric">
Mean Confidence: {feedback["mean_expert_confidence"]}
</span>
</p>

<p>
Historical Feedback Used for Adaptation:
{output["feedback_provenance"]["historical_feedback_used_for_adaptation"]}
&nbsp;&nbsp;
Total Accumulated Feedback in Profile:
{output["feedback_provenance"]["total_accumulated_feedback_in_profile"]}
</p>

</div>


<div class="section">

<h2>Policy Adaptation</h2>

<p>
<span class="metric">
Adaptations: {adaptation["adaptation_count"]}
</span>

<span class="metric">
Adaptation Rate: {adaptation["adaptation_rate"]}
</span>

<span class="metric">
Recommendation Changes: {adaptation["recommendation_change_rate"]}
</span>

<span class="metric">
Constraint Violations After Adaptation: {adaptation["constraint_violations_after_adaptation"]}
</span>
</p>

<table>

<tr>
<th>Engine</th>
<th>Original Action</th>
<th>Final Action</th>
<th>Reason</th>
</tr>

{adaptation_rows}

</table>

</div>


<div class="section">

<h2>Human Review Queue</h2>

<table>

<tr>
<th>Review ID</th>
<th>Status</th>
<th>AI Action</th>
<th>Expert Action</th>
<th>Final Action</th>
<th>Confidence</th>
<th>Reviewer</th>
<th>Reason</th>
</tr>

{table}

</table>

</div>

</body>
</html>
"""


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run(
    input_path: Path,
    *,
    reviewer: str | None = None,
    review_id: str | None = None,
    action: str | None = None,
    override_action: str | None = None,
    confidence: float | None = None,
    reason: str = "",
) -> None:

    decision_output = load_decision_output(
        input_path
    )

    subset = str(
        decision_output.get(
            "subset",
            input_path.stem,
        )
    ).upper()

    model = str(
        decision_output.get(
            "model",
            "unknown",
        )
    ).lower()

    output_path = (
        RESULTS_DIR
        / f"hitl_{subset}_{model}.json"
    )

    report_path = (
        REPORTS_DIR
        / f"hitl_{subset}_{model}.html"
    )

    logger = get_script_logger(
        "hitl",
        f"hitl_{subset}_{model}",
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Load historical feedback BEFORE recording new feedback.
    #
    # This prevents same-run feedback leakage.
    # ---------------------------------------------------------------

    historical_feedback = load_feedback(
        DEFAULT_LOG_PATH
    )

    historical_feedback_count = len(historical_feedback)

    queue = _load_or_create_queue(
        decision_output,
        output_path,
        input_path,
    )

    # ---------------------------------------------------------------
    # Resolve human review if requested
    # ---------------------------------------------------------------

    supplied_resolution = any(
        value is not None
        for value in (
            review_id,
            reviewer,
            action,
            override_action,
            confidence,
        )
    ) or bool(reason)

    if supplied_resolution:

        if not all(
            (
                review_id,
                reviewer,
                action,
                confidence is not None,
            )
        ):
            raise ValueError(
                "Resolving a review requires "
                "--review-id, --reviewer, "
                "--action and --confidence."
            )

        matches = [
            item
            for item in queue
            if item.get(
                "review_id"
            ) == review_id
        ]

        if not matches:
            raise ValueError(
                f"Review ID not found: {review_id}"
            )

        review = matches[0]

        if review.get(
            "status"
        ) == "RESOLVED":
            raise ValueError(
                f"Review already resolved: "
                f"{review_id}"
            )

        resolved = resolve_review(
            review=review,
            reviewer=reviewer,
            review_action=action,
            expert_confidence=confidence,
            comment=reason,
            override_action=override_action,
        )

        feedback = review_to_feedback(
            resolved,
            subset=subset,
            model=model,
            source_decision_output=str(
                input_path
            ),
        )

        log_feedback(
            feedback,
            DEFAULT_LOG_PATH,
        )

        current_run_feedback_count = 1

        logger.info(
            "HITL feedback recorded: "
            "review=%s action=%s expert_action=%s confidence=%.3f",
            review_id,
            action,
            feedback["expert_action"],
            confidence,
        )

    if not supplied_resolution:
        current_run_feedback_count = 0

    # ---------------------------------------------------------------
    # Historical policy adaptation
    # ---------------------------------------------------------------

    constraints = _extract_constraints(
        decision_output
    )

    refined_decisions, adaptation_metrics = (
        apply_policy_refinement(
            decisions=decision_output[
                "decisions"
            ],
            feedback_records=historical_feedback,
            constraints=constraints,
        )
    )

    # ---------------------------------------------------------------
    # Updated statistics include newly logged feedback.
    # ---------------------------------------------------------------

    feedback_stats = feedback_statistics(
        DEFAULT_LOG_PATH
    )

    all_feedback = load_feedback(
        DEFAULT_LOG_PATH
    )

    adaptation_metrics.update({
        "feedback_count": feedback_stats["total_feedback"],
        "acceptance_rate": feedback_stats["acceptance_rate"],
        "override_rate": feedback_stats["override_rate"],
        "mean_expert_confidence": feedback_stats["mean_expert_confidence"],
    })

    policy_profile = build_policy_profile(
        all_feedback
    )

    output = build_output(
        decision_output=decision_output,
        decisions=refined_decisions,
        queue=queue,
        feedback_stats=feedback_stats,
        adaptation_metrics=adaptation_metrics,
        policy_profile=policy_profile,
        input_path=input_path,
        historical_feedback_count=historical_feedback_count,
        current_run_feedback_count=current_run_feedback_count,
    )

    # ---------------------------------------------------------------
    # Save O5 outputs
    # ---------------------------------------------------------------

    save_json(
        output,
        output_path,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        build_html_report(output),
        encoding="utf-8",
    )

    logger.info(
        "O5 source: %s",
        input_path,
    )

    logger.info(
        "Feedback: total=%d accepted=%d overrides=%d",
        feedback_stats[
            "total_feedback"
        ],
        feedback_stats[
            "accepted"
        ],
        feedback_stats[
            "overrides"
        ],
    )

    logger.info(
        "Adaptations: %d / %d",
        adaptation_metrics[
            "adaptation_count"
        ],
        adaptation_metrics[
            "total_decisions"
        ],
    )

    logger.info(
        "Result saved: %s",
        output_path,
    )

    logger.info(
        "Report saved: %s",
        report_path,
    )


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    _validate_choice(
        args.subset,
        ALL_SUBSETS,
        "subset",
    )

    _validate_choice(
        args.model,
        ALL_MODELS,
        "model",
    )

    if args.input:

        run(
            Path(args.input),
            reviewer=args.reviewer,
            review_id=args.review_id,
            action=args.action,
            override_action=args.override_action,
            confidence=args.confidence,
            reason=args.reason,
        )

        return

    subsets = (
        ALL_SUBSETS
        if args.subset.lower() == "all"
        else [args.subset.upper()]
    )

    models = (
        ALL_MODELS
        if args.model.lower() == "all"
        else [args.model.lower()]
    )

    failures = 0

    for subset in subsets:

        for model in models:

            try:

                run(
                    _decision_input_path(
                        subset,
                        model,
                    ),
                    reviewer=args.reviewer,
                    review_id=args.review_id,
                    action=args.action,
                    override_action=args.override_action,
                    confidence=args.confidence,
                    reason=args.reason,
                )

            except (
                FileNotFoundError,
                ValueError,
            ) as exc:

                failures += 1

                print(
                    f"[SKIP] "
                    f"{subset}/{model}: "
                    f"{exc}"
                )

    if (
        failures
        and failures
        == len(subsets) * len(models)
    ):
        raise SystemExit(
            "No HITL workflows completed."
        )


if __name__ == "__main__":
    main()