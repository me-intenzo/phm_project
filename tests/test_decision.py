"""Tests for the O4 Decision Intelligence Engine."""

import pytest

from src.decision_engine.rules import (
    apply_rules,
    classify_uncertainty,
    classify_explanation_reliability,
    compute_risk_score,
    determine_health_state,
)
from src.decision_engine.constraints import check_constraints
from src.decision_engine.recommender import recommend_action
from scripts.decision import build_decision_state, make_decision, summarize_decisions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _state(rul=80.0, hi=0.85, lower=70.0, upper=90.0, eri=0.90, **kw):
    s = {"rul": rul, "hi": hi, "rul_lower": lower, "rul_upper": upper, "eri": eri}
    s.update(kw)
    return s


# ---------------------------------------------------------------------------
# Health state classification
# ---------------------------------------------------------------------------

class TestHealthState:
    def test_healthy(self):
        assert determine_health_state(80.0, 0.85) == "HEALTHY"

    def test_degrading_by_rul(self):
        assert determine_health_state(40.0, 0.85) == "DEGRADING"

    def test_degrading_by_hi(self):
        assert determine_health_state(80.0, 0.60) == "DEGRADING"

    def test_at_risk_by_rul(self):
        assert determine_health_state(20.0, 0.85) == "AT_RISK"

    def test_at_risk_by_hi(self):
        assert determine_health_state(80.0, 0.35) == "AT_RISK"

    def test_critical_by_rul(self):
        assert determine_health_state(5.0, 0.85) == "CRITICAL"

    def test_critical_by_hi(self):
        assert determine_health_state(80.0, 0.10) == "CRITICAL"

    def test_boundary_rul_critical(self):
        assert determine_health_state(10.0, 0.85) == "CRITICAL"

    def test_boundary_rul_at_risk(self):
        assert determine_health_state(25.0, 0.85) == "AT_RISK"


# ---------------------------------------------------------------------------
# Uncertainty classification
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_low(self):
        assert classify_uncertainty(75.0, 85.0, 80.0) == "LOW"   # rel=0.125

    def test_moderate(self):
        # rel = 40/80 = 0.5 → HIGH (at the high threshold)
        assert classify_uncertainty(60.0, 100.0, 80.0) == "HIGH"
        # rel = 30/80 = 0.375 → MODERATE (between 0.25 and 0.50)
        assert classify_uncertainty(68.0, 98.0, 80.0) == "MODERATE"

    def test_high(self):
        assert classify_uncertainty(40.0, 120.0, 80.0) == "HIGH"  # rel=1.0

    def test_moderate_boundary(self):
        # rel = 20/80 = 0.25 → exactly at moderate threshold
        result = classify_uncertainty(70.0, 90.0, 80.0)
        assert result in ("LOW", "MODERATE")  # boundary inclusive

    def test_zero_prediction(self):
        # Should not raise ZeroDivisionError
        result = classify_uncertainty(0.0, 10.0, 0.0)
        assert result in ("LOW", "MODERATE", "HIGH")


# ---------------------------------------------------------------------------
# ERI classification
# ---------------------------------------------------------------------------

class TestERI:
    def test_high(self):
        assert classify_explanation_reliability(0.90) == "HIGH"

    def test_moderate(self):
        assert classify_explanation_reliability(0.70) == "MODERATE"

    def test_low(self):
        assert classify_explanation_reliability(0.40) == "LOW"

    def test_boundary_high(self):
        assert classify_explanation_reliability(0.80) == "HIGH"

    def test_boundary_low(self):
        assert classify_explanation_reliability(0.60) == "MODERATE"


# ---------------------------------------------------------------------------
# Risk score and urgency index
# ---------------------------------------------------------------------------

class TestRiskScore:
    def test_healthy_low_risk(self):
        risk, urgency, _ = compute_risk_score(80.0, 0.85, 75.0, 85.0, 0.90)
        assert 0.0 <= risk <= 0.4
        assert urgency >= risk

    def test_critical_high_risk(self):
        risk, urgency, _ = compute_risk_score(5.0, 0.10, 0.0, 15.0, 0.30)
        assert risk > 0.6

    def test_risk_in_unit_interval(self):
        for rul, hi, eri in [(0, 0, 0), (125, 1, 1), (50, 0.5, 0.5)]:
            risk, urgency, _ = compute_risk_score(rul, hi, 0, 125, eri)
            assert 0.0 <= risk <= 1.0
            assert 0.0 <= urgency <= 1.0

    def test_urgency_amplified_by_uncertainty(self):
        # Wide interval should amplify urgency relative to risk
        risk_narrow, urgency_narrow, _ = compute_risk_score(30.0, 0.50, 28.0, 32.0, 0.70)
        risk_wide,   urgency_wide,   _ = compute_risk_score(30.0, 0.50, 0.0,  80.0, 0.70)
        assert urgency_wide > urgency_narrow

    def test_components_keys(self):
        _, _, components = compute_risk_score(50.0, 0.60, 40.0, 60.0, 0.75)
        assert set(components) == {"r_rul", "r_hi", "r_uncertainty", "r_eri_penalty"}


# ---------------------------------------------------------------------------
# apply_rules
# ---------------------------------------------------------------------------

class TestApplyRules:
    def test_healthy_continue(self):
        result = apply_rules(_state(rul=80, hi=0.85, lower=75, upper=85, eri=0.90))
        assert result["health_state"] == "HEALTHY"
        assert result["candidate_action"] == "CONTINUE_OPERATION"
        assert result["human_review"] is False

    def test_critical_urgent(self):
        result = apply_rules(_state(rul=5, hi=0.10, lower=0, upper=15, eri=0.85))
        assert result["health_state"] == "CRITICAL"
        assert result["candidate_action"] == "URGENT_MAINTENANCE"

    def test_risk_score_present(self):
        result = apply_rules(_state())
        assert "risk_score" in result
        assert "urgency_index" in result
        assert "risk_components" in result

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="Missing required"):
            apply_rules({"rul": 50.0, "hi": 0.5})

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError, match="Invalid RUL interval"):
            apply_rules(_state(lower=90, upper=70))

    def test_high_uncertainty_healthy_monitor(self):
        # Wide interval on healthy engine → MONITOR
        result = apply_rules(_state(rul=80, hi=0.85, lower=10, upper=120, eri=0.90))
        assert result["candidate_action"] == "MONITOR"

    def test_low_eri_degrading_inspect(self):
        result = apply_rules(_state(rul=40, hi=0.65, lower=35, upper=45, eri=0.30))
        assert result["candidate_action"] == "INSPECT"
        assert result["human_review"] is True

    def test_extreme_risk_override(self):
        # risk_score > 0.85 should force URGENT_MAINTENANCE
        result = apply_rules(_state(rul=2, hi=0.05, lower=0, upper=100, eri=0.10))
        assert result["candidate_action"] == "URGENT_MAINTENANCE"
        assert result["human_review"] is True

    def test_top_k_sensors_passthrough(self):
        state = _state()
        state["top_k_sensors"] = [2, 11, 4]
        result = apply_rules(state)
        assert result["top_k_sensors"] == [2, 11, 4]


# ---------------------------------------------------------------------------
# check_constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def _base_decision(self, action="SCHEDULE_MAINTENANCE", rul=20.0):
        return {
            "rul": rul, "hi": 0.40, "rul_lower": 15.0, "rul_upper": 25.0,
            "eri": 0.75, "health_state": "AT_RISK",
            "uncertainty_level": "LOW", "explanation_reliability": "MODERATE",
            "risk_score": 0.55, "urgency_index": 0.58,
            "risk_components": {}, "candidate_action": action,
            "human_review": False, "reasons": [],
        }

    def test_no_window_downgrades_schedule(self):
        d = self._base_decision("SCHEDULE_MAINTENANCE")
        result = check_constraints(d, {"maintenance_window_available": False})
        assert result["candidate_action"] == "INSPECT"
        assert result["human_review"] is True
        assert result["constraint_status"] == "VIOLATION"

    def test_window_available_no_change(self):
        d = self._base_decision("SCHEDULE_MAINTENANCE")
        result = check_constraints(d, {"maintenance_window_available": True})
        assert result["candidate_action"] == "SCHEDULE_MAINTENANCE"
        assert result["constraint_status"] == "SATISFIED"

    def test_minimum_safe_rul_escalates(self):
        d = self._base_decision("MONITOR", rul=8.0)
        result = check_constraints(d, {"minimum_safe_rul": 10.0})
        assert result["candidate_action"] == "URGENT_MAINTENANCE"
        assert "minimum safe operating limit" in result["constraint_violations"][0]

    def test_lead_time_escalates(self):
        d = self._base_decision("SCHEDULE_MAINTENANCE", rul=5.0)
        result = check_constraints(d, {"maintenance_lead_time": 10.0})
        assert result["candidate_action"] == "URGENT_MAINTENANCE"

    def test_max_allowable_delay_escalates(self):
        d = self._base_decision("CONTINUE_OPERATION", rul=15.0)
        result = check_constraints(d, {"max_allowable_delay": 20.0})
        assert result["candidate_action"] == "INSPECT"
        assert "within the maximum allowable delay" in result["constraint_violations"][0]

    def test_resource_unavailable_flags_review(self):
        d = self._base_decision("URGENT_MAINTENANCE", rul=5.0)
        result = check_constraints(d, {"resource_availability": False})
        assert result["human_review"] is True
        assert result["constraint_status"] == "VIOLATION"

    def test_no_constraints_satisfied(self):
        d = self._base_decision("SCHEDULE_MAINTENANCE")
        result = check_constraints(d, {})
        assert result["constraint_status"] == "SATISFIED"
        assert result["constraint_violations"] == []

    def test_invalid_action_raises(self):
        d = self._base_decision("UNKNOWN_ACTION")
        with pytest.raises(ValueError, match="Unknown candidate action"):
            check_constraints(d)


# ---------------------------------------------------------------------------
# recommend_action (integration)
# ---------------------------------------------------------------------------

class TestRecommendAction:
    def test_output_keys(self):
        result = recommend_action(_state())
        expected = {
            "inputs", "health_state", "uncertainty_level",
            "explanation_reliability", "risk_score", "urgency_index",
            "risk_components", "recommended_action", "human_review",
            "reasons", "constraint_status", "constraint_violations",
            "constraint_adjustments", "top_k_sensors",
        }
        assert expected.issubset(result.keys())

    def test_healthy_no_constraints(self):
        result = recommend_action(_state(rul=80, hi=0.85, lower=75, upper=85, eri=0.90))
        assert result["recommended_action"] == "CONTINUE_OPERATION"
        assert result["constraint_status"] == "SATISFIED"

    def test_critical_urgent_maintenance(self):
        result = recommend_action(_state(rul=5, hi=0.10, lower=0, upper=15, eri=0.85))
        assert result["recommended_action"] == "URGENT_MAINTENANCE"

    def test_constraint_escalation_in_pipeline(self):
        # RUL=60 → HEALTHY → CONTINUE_OPERATION by rules (passive);
        # minimum_safe_rul=65 fires because RUL(60) <= 65 → URGENT.
        result = recommend_action(
            _state(rul=60, hi=0.85, lower=58, upper=62, eri=0.90),
            constraints={"minimum_safe_rul": 65.0},
        )
        assert result["recommended_action"] == "URGENT_MAINTENANCE"
        assert result["constraint_status"] == "VIOLATION"

    def test_top_k_sensors_in_output(self):
        state = _state()
        state["top_k_sensors"] = [2, 11, 4]
        result = recommend_action(state)
        assert result["top_k_sensors"] == [2, 11, 4]

    def test_risk_score_range(self):
        result = recommend_action(_state())
        assert 0.0 <= result["risk_score"] <= 1.0
        assert 0.0 <= result["urgency_index"] <= 1.0

    def test_custom_thresholds(self):
        # Lower rul_critical threshold → engine at RUL=15 should be CRITICAL
        result = recommend_action(
            _state(rul=15, hi=0.50, lower=10, upper=20, eri=0.85),
            thresholds={"rul_critical": 20.0},
        )
        assert result["health_state"] == "CRITICAL"


# ---------------------------------------------------------------------------
# build_decision_state validation
# ---------------------------------------------------------------------------

class TestBuildDecisionState:
    def test_valid(self):
        s = build_decision_state(50.0, 0.6, 40.0, 60.0, 0.75)
        assert s["rul"] == 50.0

    def test_invalid_hi(self):
        with pytest.raises(ValueError, match="HI must be in"):
            build_decision_state(50.0, 1.5, 40.0, 60.0, 0.75)

    def test_invalid_eri(self):
        with pytest.raises(ValueError, match="ERI must be in"):
            build_decision_state(50.0, 0.6, 40.0, 60.0, -0.1)

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="rul_lower must be <= rul_upper"):
            build_decision_state(50.0, 0.6, 70.0, 60.0, 0.75)

    def test_top_k_sensors_stored(self):
        s = build_decision_state(50.0, 0.6, 40.0, 60.0, 0.75, top_k_sensors=[2, 11])
        assert s["top_k_sensors"] == [2, 11]


# ---------------------------------------------------------------------------
# summarize_decisions
# ---------------------------------------------------------------------------

class TestSummarizeDecisions:
    def _make_decisions(self):
        return [
            make_decision(80, 0.85, 75, 85, 0.90),
            make_decision(5,  0.10, 0,  15, 0.85),
            make_decision(20, 0.35, 15, 25, 0.70),
        ]

    def test_summary_keys(self):
        summary = summarize_decisions(self._make_decisions())
        assert "num_decisions" in summary
        assert "risk_score_mean" in summary
        assert "urgency_index_mean" in summary
        assert "action_distribution" in summary

    def test_num_decisions(self):
        decisions = self._make_decisions()
        summary = summarize_decisions(decisions)
        assert summary["num_decisions"] == 3

    def test_risk_score_mean_in_range(self):
        summary = summarize_decisions(self._make_decisions())
        assert 0.0 <= summary["risk_score_mean"] <= 1.0
        assert summary["risk_score_max"] >= summary["risk_score_mean"]

    def test_empty_decisions(self):
        summary = summarize_decisions([])
        assert summary["num_decisions"] == 0
        assert "risk_score_mean" not in summary
