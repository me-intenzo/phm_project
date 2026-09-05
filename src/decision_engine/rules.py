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
    "rul_max":      125.0,   # C-MAPSS piecewise cap

    # HI breakpoints (normalised, 1 = new, 0 = failed)
    "hi_critical":  0.20,
    "hi_at_risk":   0.40,
    "hi_degrading": 0.70,

    # Uncertainty: relative interval width thresholds
    "uncertainty_moderate_relative": 0.25,
    "uncertainty_high_relative":     0.50,

    # ERI trustworthiness thresholds
    "eri_low":      0.60,
    "eri_high":     0.80,

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

    Uncertainty and ERI modulate the action for a given health state.
    Risk score provides a continuous override: if R > 0.85 the action
    is always escalated to URGENT_MAINTENANCE regardless of state.

    Returns
    -------
    action, human_review, reasons
    """
    reasons: list[str] = []
    human_review = False

    hs  = health_state.upper()
    ul  = uncertainty_level.upper()
    er  = explanation_reliability.upper()

    # Hard override: extreme risk
    if risk_score > 0.85:
        reasons.append(f"Risk score {risk_score:.3f} exceeds critical threshold (0.85).")
        return "URGENT_MAINTENANCE", True, reasons

    if hs == "HEALTHY":
        reasons.append("RUL and HI indicate a healthy operating state.")
        if ul == "HIGH":
            reasons.append("Prediction interval is wide — increased monitoring warranted.")
            human_review = er == "LOW"
            return "MONITOR", human_review, reasons
        if ul == "MODERATE" and er == "LOW":
            reasons.append("Moderate uncertainty with low explanation reliability.")
            return "MONITOR", True, reasons
        return "CONTINUE_OPERATION", False, reasons

    if hs == "DEGRADING":
        reasons.append("RUL or HI indicates an emerging degradation trend.")
        if ul == "HIGH" or er == "LOW":
            if ul == "HIGH":
                reasons.append("Prediction uncertainty is high.")
            if er == "LOW":
                reasons.append("Explanation reliability is low.")
            return "INSPECT", True, reasons
        if ul == "MODERATE":
            reasons.append("Moderate uncertainty — proactive monitoring recommended.")
            return "MONITOR", False, reasons
        return "MONITOR", False, reasons

    if hs == "AT_RISK":
        reasons.append("RUL or HI indicates an at-risk condition.")
        if ul == "HIGH" or er == "LOW":
            if ul == "HIGH":
                reasons.append("Prediction interval is wide.")
            if er == "LOW":
                reasons.append("Explanation reliability is low.")
            return "INSPECT", True, reasons
        return "SCHEDULE_MAINTENANCE", False, reasons

    if hs == "CRITICAL":
        reasons.append("RUL or HI indicates a critical condition.")
        if er == "LOW":
            reasons.append("Explanation reliability is low — human review required.")
            return "URGENT_MAINTENANCE", True, reasons
        if ul == "HIGH":
            reasons.append("Prediction interval is wide.")
            return "URGENT_MAINTENANCE", True, reasons
        return "URGENT_MAINTENANCE", False, reasons

    raise ValueError(f"Unknown health state: {health_state!r}")


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

    uncertainty_level = classify_uncertainty(lower, upper, rul, thresholds)

    explanation_reliability = classify_explanation_reliability(eri, thresholds)

    risk_score, urgency_index, risk_components = compute_risk_score(
        rul=rul, hi=hi,
        rul_lower=lower, rul_upper=upper,
        eri=eri, thresholds=thresholds,
    )

    action, human_review, reasons = determine_candidate_action(
        health_state=health_state,
        uncertainty_level=uncertainty_level,
        explanation_reliability=explanation_reliability,
        risk_score=risk_score,
    )

    result = dict(state)
    result.update({
        "health_state":           health_state,
        "uncertainty_level":      uncertainty_level,
        "explanation_reliability": explanation_reliability,
        "risk_score":             risk_score,
        "urgency_index":          urgency_index,
        "risk_components":        risk_components,
        "candidate_action":       action,
        "human_review":           human_review,
        "reasons":                reasons,
    })
    return result
