"""Decision policy for maintenance recommendations.

O4 Decision Intelligence Engine
--------------------------------
Multi-factor fusion of RUL, HI, conformal uncertainty, and ERI.

Risk Score
----------
A continuous risk index R ∈ [0, 1] that fuses all four signals:

    R = w_rul * r_rul  +  w_hi * r_hi
        + w_u * r_uncertainty  +  w_e * r_eri_penalty

where each component is normalised to [0, 1]:

    r_rul  = 1 - clip(rul / rul_max, 0, 1)
    r_hi   = 1 - hi                          (HI=1 → healthy)
    r_u    = clip(interval_width / rul_max, 0, 1)
    r_e    = 1 - eri                         (low ERI → high penalty)

Urgency Index
-------------
    U = R * (1 + uncertainty_amplifier * r_u)

capped at 1.0.  High uncertainty amplifies urgency when the engine
is already degraded.

author: me-intenzo
"""

from __future__ import annotations

from typing import Any

from src.utils.config import (
    DEFAULT_CONFIG,
    ERI_HIGH_THRESHOLD,
    ERI_LOW_THRESHOLD,
)

# ---------------------------------------------------------------------
# Severity ladder
# ---------------------------------------------------------------------
# Every action sits on an explicit severity ladder:
#
#     CONTINUE_OPERATION < MONITOR < INSPECT
#                         < SCHEDULE_MAINTENANCE < URGENT_MAINTENANCE
#
# The ladder is the *commitment* ordering: the further right, the more of
# the maintenance action is committed to and the sooner the operator has to
# act.  It is defined once, here, and used by the rule layer, by the
# constraint layer and by the sensitivity analysis, so that "more severe"
# means the same thing everywhere in the repository.

SEVERITY_LADDER: tuple[str, ...] = (
    "CONTINUE_OPERATION",
    "MONITOR",
    "INSPECT",
    "SCHEDULE_MAINTENANCE",
    "URGENT_MAINTENANCE",
)

SEVERITY: dict[str, int] = {
    action: index for index, action in enumerate(SEVERITY_LADDER)
}

#: The three uncertainty classes, in increasing order of distrust.
UNCERTAINTY_LEVELS: tuple[str, ...] = ("LOW", "MODERATE", "HIGH")

#: Base severity of each health state, before any confidence escalation.
HEALTH_STATE_BASE_SEVERITY: dict[str, int] = {
    "HEALTHY":   0,   # CONTINUE_OPERATION
    "DEGRADING": 1,   # MONITOR
    "AT_RISK":   3,   # SCHEDULE_MAINTENANCE
    "CRITICAL":  4,   # URGENT_MAINTENANCE
}

#: Escalation step applied on top of the base severity when the explanation
#: cannot be trusted.  The *uncertainty* is not a separate step: it already
#: enters the action through the planning state, which is derived from the
#: conservative end of the calibrated interval.  Counting it twice would
#: double-count the same evidence.
ERI_ESCALATION: dict[str, int] = {
    "HIGH":     0,
    "MODERATE": 0,
    "LOW":      1,
}


def severity_of(action: str) -> int:
    """Position of *action* on the severity ladder (0 = least severe)."""

    return SEVERITY.get(str(action).upper(), 0)


def action_at_severity(index: int) -> str:
    """Action at a given position on the severity ladder."""

    index = max(0, min(int(index), len(SEVERITY_LADDER) - 1))
    return SEVERITY_LADDER[index]

# ---------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------
# Calibrated against C-MAPSS FD001–FD004 piecewise-linear RUL cap (125).
# Adjust via --thresholds JSON for cross-dataset experiments.

DEFAULT_THRESHOLDS: dict[str, float] = {
    # RUL breakpoints (cycles)
    "rul_critical":  10.0,
    "rul_at_risk":   25.0,
    "rul_degrading": 50.0,
    # C-MAPSS piecewise cap.  Kept in sync with the ``max_rul`` key in
    # ``configs/<subset>.yaml`` (see ``src/utils/config.py``).
    "rul_max":      float(DEFAULT_CONFIG.get("max_rul", 125.0)),

    # HI breakpoints (normalised, 1 = new, 0 = failed).  These defaults
    # belong to the legacy HI = RUL / max_rul label.  When the pipeline
    # builds the sensor-derived health index, preprocessing calibrates the
    # three breakpoints on the training split and the decision CLI merges
    # them in (see ``scripts/decision.py``).
    "hi_critical":  0.20,
    "hi_at_risk":   0.40,
    "hi_degrading": 0.70,

    # Uncertainty: relative interval width thresholds
    "uncertainty_moderate_relative": 0.25,
    "uncertainty_high_relative":     0.50,

    # ERI trustworthiness thresholds.  Shared with the explainability report
    # so that both classify the same ERI value identically.
    "eri_low":      ERI_LOW_THRESHOLD,
    "eri_high":     ERI_HIGH_THRESHOLD,

    # Risk-score fusion weights (must sum to 1.0)
    "w_rul":  0.40,
    "w_hi":   0.30,
    "w_u":    0.15,
    "w_eri":  0.15,

    # Urgency amplification factor for high uncertainty
    "uncertainty_amplifier": 0.30,
}


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def _get_thresholds(thresholds: dict[str, float] | None) -> dict[str, float]:
    result = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        result.update(thresholds)
    return result


# ---------------------------------------------------------------------
# Quantitative risk and urgency
# ---------------------------------------------------------------------

def compute_risk_score(
    rul: float,
    hi: float,
    rul_lower: float,
    rul_upper: float,
    eri: float,
    thresholds: dict[str, float] | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Compute a continuous risk score R ∈ [0, 1] and urgency index U ∈ [0, 1].

    Returns
    -------
    risk_score, urgency_index, components
    """
    t = _get_thresholds(thresholds)
    rul_max = max(t["rul_max"], 1e-6)

    r_rul = 1.0 - min(max(rul / rul_max, 0.0), 1.0)
    r_hi  = 1.0 - min(max(hi, 0.0), 1.0)
    r_u   = min(max((rul_upper - rul_lower) / rul_max, 0.0), 1.0)
    r_eri = 1.0 - min(max(eri, 0.0), 1.0)

    risk = (
        t["w_rul"] * r_rul
        + t["w_hi"]  * r_hi
        + t["w_u"]   * r_u
        + t["w_eri"] * r_eri
    )
    risk = min(max(risk, 0.0), 1.0)

    urgency = risk * (1.0 + t["uncertainty_amplifier"] * r_u)
    urgency = min(urgency, 1.0)

    components = {
        "r_rul": round(r_rul, 4),
        "r_hi":  round(r_hi,  4),
        "r_uncertainty": round(r_u, 4),
        "r_eri_penalty": round(r_eri, 4),
    }
    return round(risk, 4), round(urgency, 4), components


# ---------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------

def classify_uncertainty(
    rul_lower: float,
    rul_upper: float,
    rul_prediction: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Classify prediction uncertainty as LOW, MODERATE, or HIGH.

    Relative interval width:
        U_rel = (upper - lower) / max(|prediction|, ε)
    """
    t = _get_thresholds(thresholds)
    width = max(0.0, rul_upper - rul_lower)
    denom = max(abs(rul_prediction), 1e-6)
    rel = width / denom

    if rel >= t["uncertainty_high_relative"]:
        return "HIGH"
    if rel >= t["uncertainty_moderate_relative"]:
        return "MODERATE"
    return "LOW"


def classify_explanation_reliability(
    eri: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map ERI scalar to LOW / MODERATE / HIGH."""
    t = _get_thresholds(thresholds)
    if eri >= t["eri_high"]:
        return "HIGH"
    if eri >= t["eri_low"]:
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------
# Health-state policy
# ---------------------------------------------------------------------

def determine_health_state(
    rul_prediction: float,
    hi_prediction: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Determine health state from RUL (primary) and HI (secondary).

    Returns HEALTHY | DEGRADING | AT_RISK | CRITICAL.
    """
    t = _get_thresholds(thresholds)
    rul = float(rul_prediction)
    hi  = float(hi_prediction)

    if rul <= t["rul_critical"] or hi <= t["hi_critical"]:
        return "CRITICAL"
    if rul <= t["rul_at_risk"] or hi <= t["hi_at_risk"]:
        return "AT_RISK"
    if rul <= t["rul_degrading"] or hi <= t["hi_degrading"]:
        return "DEGRADING"
    return "HEALTHY"


# ---------------------------------------------------------------------
# Maintenance action policy
# ---------------------------------------------------------------------

def determine_candidate_action(
    health_state: str,
    uncertainty_level: str,
    explanation_reliability: str,
    risk_score: float,
) -> tuple[str, bool, list[str]]:
    """Select a candidate maintenance action.

    The action is computed from the severity ladder rather than read off a
    hand-written table::

        severity = min(base(planning_state) + eri_step, top)

    which makes the policy **monotone by construction**: the action can
    never be less severe than the state the engine is planned for, and a
    less reliable explanation can only escalate it.  It also makes the
    ``AT_RISK`` row consistent, which the previous table was not: it
    produced ``INSPECT`` for ``HIGH`` uncertainty and
    ``SCHEDULE_MAINTENANCE`` for ``MODERATE`` uncertainty, so more
    uncertainty produced a *less* committed action
    (see ``docs/ARCHITECTURE.md`` §7.4).

    ``health_state`` is the *planning state* — the state the engine would be
    in if the prediction were at the conservative end of its calibrated
    interval (see :func:`apply_rules`) — not the state of the point
    estimate.  The uncertainty enters the action through that planning
    state, and through the human-review flag; it is deliberately *not* a
    second escalation step, which would count the same evidence twice.

    Risk score provides a continuous override: if R > 0.85 the action
    is always escalated to URGENT_MAINTENANCE regardless of state.

    Returns
    -------
    action, human_review, reasons
    """

    reasons: list[str] = []
    human_review = False

    hs = health_state.upper()
    ul = uncertainty_level.upper()
    er = explanation_reliability.upper()

    if hs not in HEALTH_STATE_BASE_SEVERITY:
        raise ValueError(f"Unknown health state: {health_state!r}")
    if ul not in UNCERTAINTY_LEVELS:
        raise ValueError(f"Unknown uncertainty level: {uncertainty_level!r}")
    if er not in ERI_ESCALATION:
        raise ValueError(f"Unknown explanation reliability: {explanation_reliability!r}")

    base = HEALTH_STATE_BASE_SEVERITY[hs]
    eri_step = ERI_ESCALATION[er]

    severity = min(base + eri_step, len(SEVERITY_LADDER) - 1)
    action = action_at_severity(severity)

    # -- reasons ----------------------------------------------------
    reasons.append(
        f"Planning state {hs} sets the base action "
        f"{action_at_severity(base)}."
    )

    if eri_step:
        reasons.append(
            f"Explanation reliability is {er} — escalating one step on the "
            f"severity ladder."
        )

    # -- human review ------------------------------------------------
    # A degraded confidence signal always means a human has to look at the
    # record, even when the action itself did not change.
    if ul == "HIGH":
        human_review = True
        reasons.append("Human review required: the prediction interval is wide.")
    if er == "LOW":
        human_review = True
        reasons.append("Human review required: the explanation is not reliable.")

    # Hard override: extreme risk
    if risk_score > 0.85:
        reasons.append(
            f"Risk score {risk_score:.3f} exceeds critical threshold (0.85)."
        )
        return "URGENT_MAINTENANCE", True, reasons

    return action, human_review, reasons


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_rules(
    state: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the O4 decision policy to a prognostic state.

    Expected state fields
    ---------------------
    rul, hi, rul_lower, rul_upper, eri

    Optional
    --------
    top_k_sensors : list[int]
        Indices of the most influential sensors from XAI (O3 output).
        Passed through to the decision output for auditability.

    Returns
    -------
    dict
        Enriched state with health state, uncertainty, explanation
        reliability, risk score, urgency index, candidate action,
        human review flag, and reasons.
    """
    required = {"rul", "hi", "rul_lower", "rul_upper", "eri"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"Missing required decision state fields: {sorted(missing)}")

    rul   = float(state["rul"])
    hi    = float(state["hi"])
    lower = float(state["rul_lower"])
    upper = float(state["rul_upper"])
    eri   = float(state["eri"])

    if lower > upper:
        raise ValueError(f"Invalid RUL interval: lower={lower} > upper={upper}")

    health_state = determine_health_state(rul, hi, thresholds)

    # The *planning state* is the state the engine would be in if the
    # prediction were at the conservative end of its calibrated interval.
    # The action is planned on that state, not on the point estimate: a
    # maintenance work order has to be scheduled so that it is carried out
    # before the worst credible case, not before the most likely one.  It is
    # also what makes the policy monotone in the uncertainty — a wider
    # interval can only lower the planning state, never raise it.
    planning_state = determine_health_state(lower, hi, thresholds)

    uncertainty_level = classify_uncertainty(lower, upper, rul, thresholds)

    explanation_reliability = classify_explanation_reliability(eri, thresholds)

    risk_score, urgency_index, risk_components = compute_risk_score(
        rul=rul, hi=hi,
        rul_lower=lower, rul_upper=upper,
        eri=eri, thresholds=thresholds,
    )

    action, human_review, reasons = determine_candidate_action(
        health_state=planning_state,
        uncertainty_level=uncertainty_level,
        explanation_reliability=explanation_reliability,
        risk_score=risk_score,
    )

    if planning_state != health_state:
        reasons.append(
            f"The point estimate is in state {health_state}, but the lower "
            f"bound of the interval ({lower:.1f} cycles) puts the engine in "
            f"state {planning_state} — the action is planned on the latter."
        )

    result = dict(state)
    result.update({
        "health_state":           health_state,
        "planning_state":         planning_state,
        "uncertainty_level":      uncertainty_level,
        "explanation_reliability": explanation_reliability,
        "risk_score":             risk_score,
        "urgency_index":          urgency_index,
        "risk_components":        risk_components,
        "candidate_action":       action,
        "action_severity":        severity_of(action),
        "human_review":           human_review,
        "reasons":                reasons,
    })
    return result
