"""Tests for the decision policy: severity ladder, monotonicity, scenarios."""

from __future__ import annotations

import itertools

import pytest

from src.decision_engine.constraints import check_constraints
from src.decision_engine.recommender import recommend_action
from src.decision_engine.rules import (
    ERI_ESCALATION,
    HEALTH_STATE_BASE_SEVERITY,
    SEVERITY_LADDER,
    UNCERTAINTY_LEVELS,
    action_at_severity,
    apply_rules,
    classify_explanation_reliability,
    determine_candidate_action,
    severity_of,
)
from src.decision_engine.scenarios import (
    CONSTRAINT_SCENARIOS,
    describe_scenario,
    get_scenario,
    scenario_names,
)
from src.utils.config import ERI_HIGH_THRESHOLD, ERI_LOW_THRESHOLD

HEALTH_STATES = ("HEALTHY", "DEGRADING", "AT_RISK", "CRITICAL")
ERI_LEVELS = ("LOW", "MODERATE", "HIGH")


def _state(rul=80.0, hi=0.85, lower=70.0, upper=90.0, eri=0.90, **kw):
    s = {"rul": rul, "hi": hi, "rul_lower": lower, "rul_upper": upper, "eri": eri}
    s.update(kw)
    return s


# ---------------------------------------------------------------------------
# Severity ladder
# ---------------------------------------------------------------------------

class TestSeverityLadder:
    def test_ladder_order(self):
        assert SEVERITY_LADDER == (
            "CONTINUE_OPERATION",
            "MONITOR",
            "INSPECT",
            "SCHEDULE_MAINTENANCE",
            "URGENT_MAINTENANCE",
        )

    def test_severity_is_strictly_increasing(self):
        values = [severity_of(action) for action in SEVERITY_LADDER]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_action_at_severity_round_trips(self):
        for index, action in enumerate(SEVERITY_LADDER):
            assert action_at_severity(index) == action

    def test_action_at_severity_clamps(self):
        assert action_at_severity(-5) == SEVERITY_LADDER[0]
        assert action_at_severity(99) == SEVERITY_LADDER[-1]

    def test_every_health_state_has_a_base_severity(self):
        assert set(HEALTH_STATE_BASE_SEVERITY) == set(HEALTH_STATES)

    def test_escalation_maps_are_complete(self):
        assert set(ERI_ESCALATION) == set(ERI_LEVELS)

    def test_uncertainty_levels_are_declared(self):
        assert UNCERTAINTY_LEVELS == ("LOW", "MODERATE", "HIGH")


# ---------------------------------------------------------------------------
# Monotonicity — the §7.4 fix
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """The policy must never become *less* severe when the evidence worsens.

    This is the regression test for the AT_RISK row that used to produce
    ``INSPECT`` for HIGH uncertainty and ``SCHEDULE_MAINTENANCE`` for
    MODERATE uncertainty (docs/ARCHITECTURE.md §7.4).
    """

    @staticmethod
    def _severity(health_state, uncertainty, eri, risk=0.0):
        action, _, _ = determine_candidate_action(
            health_state=health_state,
            uncertainty_level=uncertainty,
            explanation_reliability=eri,
            risk_score=risk,
        )
        return severity_of(action)

    def test_action_is_monotone_in_explanation_reliability(self):
        for health_state, uncertainty in itertools.product(
            HEALTH_STATES, UNCERTAINTY_LEVELS
        ):
            severities = [
                self._severity(health_state, uncertainty, level)
                for level in reversed(ERI_LEVELS)
            ]
            assert severities == sorted(severities), (
                f"{health_state} / uncertainty {uncertainty}: severity fell "
                f"as the explanation reliability fell: {severities}"
            )

    def test_action_is_monotone_in_the_interval(self):
        """A wider interval must never produce a less severe action.

        The action is planned on the conservative end of the calibrated
        interval, so widening the interval downwards can only make the
        planning state more severe.
        """

        # Same point estimate, same HI, progressively wider interval.
        for rul, hi in ((90.0, 0.9), (40.0, 0.6), (20.0, 0.35), (5.0, 0.1)):
            severities = []
            for lower in (rul, rul * 0.6, rul * 0.3, 0.0):
                result = apply_rules(
                    _state(
                        rul=rul, hi=hi,
                        lower=lower, upper=rul + 10.0,
                        eri=0.9,
                    )
                )
                severities.append(result["action_severity"])
            assert severities == sorted(severities), (
                f"rul={rul}: severity fell as the interval widened: {severities}"
            )

    def test_action_is_never_below_the_planning_state(self):
        for health_state, uncertainty, eri in itertools.product(
            HEALTH_STATES, UNCERTAINTY_LEVELS, ERI_LEVELS
        ):
            severity = self._severity(health_state, uncertainty, eri)
            base = HEALTH_STATE_BASE_SEVERITY[health_state]
            assert severity >= base

    def test_at_risk_never_falls_below_schedule_maintenance(self):
        """The §7.4 property: an at-risk engine is never *less* committed.

        Under the old table a high-uncertainty at-risk engine produced
        ``INSPECT`` while a moderate-uncertainty one produced
        ``SCHEDULE_MAINTENANCE``.  With the planning state that is
        structurally impossible: ``INSPECT`` is below
        ``SCHEDULE_MAINTENANCE`` on the ladder and is only reachable from
        the DEGRADING band.
        """

        for lower in (25.0, 20.0, 12.0, 5.0, 0.0):
            result = apply_rules(
                _state(rul=20.0, hi=0.40, lower=lower, upper=30.0, eri=0.9)
            )
            assert result["action_severity"] >= severity_of(
                "SCHEDULE_MAINTENANCE"
            ), (
                f"lower={lower}: an at-risk engine produced a less committed "
                f"action than SCHEDULE_MAINTENANCE"
            )

    def test_at_risk_confident_case_is_schedule_maintenance(self):
        result = apply_rules(
            _state(rul=20.0, hi=0.40, lower=18.0, upper=22.0, eri=0.9)
        )
        assert result["planning_state"] == "AT_RISK"
        assert result["candidate_action"] == "SCHEDULE_MAINTENANCE"

    def test_at_risk_with_a_critical_lower_bound_is_urgent(self):
        """If the interval reaches into the critical band, act now."""

        result = apply_rules(
            _state(rul=20.0, hi=0.40, lower=4.0, upper=36.0, eri=0.9)
        )
        assert result["planning_state"] == "CRITICAL"
        assert result["candidate_action"] == "URGENT_MAINTENANCE"

    def test_planning_state_is_reported_alongside_the_point_state(self):
        result = apply_rules(
            _state(rul=20.0, hi=0.40, lower=4.0, upper=36.0, eri=0.9)
        )
        assert result["health_state"] == "AT_RISK"
        assert result["planning_state"] == "CRITICAL"
        assert any("planning" in reason.lower() for reason in result["reasons"])

    def test_degrading_table_is_unchanged(self):
        """The DEGRADING and CRITICAL rows must keep their old behaviour."""

        assert determine_candidate_action(
            "DEGRADING", "LOW", "HIGH", 0.3
        )[0] == "MONITOR"
        assert determine_candidate_action(
            "DEGRADING", "HIGH", "HIGH", 0.3
        )[0] == "MONITOR"
        assert determine_candidate_action(
            "DEGRADING", "LOW", "LOW", 0.3
        )[0] == "INSPECT"
        assert determine_candidate_action(
            "CRITICAL", "LOW", "HIGH", 0.3
        )[0] == "URGENT_MAINTENANCE"

    def test_healthy_table(self):
        assert determine_candidate_action(
            "HEALTHY", "LOW", "HIGH", 0.1
        )[0] == "CONTINUE_OPERATION"
        assert determine_candidate_action(
            "HEALTHY", "HIGH", "HIGH", 0.1
        )[0] == "CONTINUE_OPERATION"
        assert determine_candidate_action(
            "HEALTHY", "LOW", "LOW", 0.1
        )[0] == "MONITOR"

    def test_risk_override_still_wins(self):
        action, review, reasons = determine_candidate_action(
            "HEALTHY", "LOW", "HIGH", 0.9
        )
        assert action == "URGENT_MAINTENANCE"
        assert review is True
        assert any("Risk score" in reason for reason in reasons)

    def test_unknown_inputs_raise(self):
        with pytest.raises(ValueError, match="Unknown health state"):
            determine_candidate_action("NOPE", "LOW", "HIGH", 0.1)
        with pytest.raises(ValueError, match="Unknown uncertainty level"):
            determine_candidate_action("HEALTHY", "NOPE", "HIGH", 0.1)
        with pytest.raises(ValueError, match="Unknown explanation reliability"):
            determine_candidate_action("HEALTHY", "LOW", "NOPE", 0.1)

    def test_high_uncertainty_always_requires_review(self):
        _, review, _ = determine_candidate_action(
            "HEALTHY", "HIGH", "HIGH", 0.1
        )
        assert review is True

    def test_low_eri_always_requires_review(self):
        _, review, _ = determine_candidate_action(
            "HEALTHY", "LOW", "LOW", 0.1
        )
        assert review is True


# ---------------------------------------------------------------------------
# Constraint layer
# ---------------------------------------------------------------------------

class TestConstraintSeverity:
    def _decision(self, action="MONITOR", rul=20.0):
        return {
            "rul": rul, "hi": 0.40, "rul_lower": 15.0, "rul_upper": 25.0,
            "eri": 0.75, "health_state": "AT_RISK",
            "uncertainty_level": "LOW", "explanation_reliability": "MODERATE",
            "risk_score": 0.55, "urgency_index": 0.58,
            "risk_components": {}, "candidate_action": action,
            "human_review": False, "reasons": [],
        }

    def test_minimum_safe_rul_escalates_severity(self):
        result = check_constraints(
            self._decision("MONITOR", rul=8.0), {"minimum_safe_rul": 10.0}
        )
        assert result["action_severity"] > severity_of("MONITOR")
        assert result["severity_delta"] > 0

    def test_max_allowable_delay_escalates_severity(self):
        result = check_constraints(
            self._decision("CONTINUE_OPERATION", rul=15.0),
            {"max_allowable_delay": 20.0},
        )
        assert result["action_severity"] > severity_of("CONTINUE_OPERATION")

    def test_lead_time_escalates_severity(self):
        result = check_constraints(
            self._decision("SCHEDULE_MAINTENANCE", rul=5.0),
            {"maintenance_lead_time": 10.0},
        )
        assert result["action_severity"] > severity_of("SCHEDULE_MAINTENANCE")

    def test_maintenance_window_is_recorded_as_a_substitution(self):
        result = check_constraints(
            self._decision("SCHEDULE_MAINTENANCE"),
            {"maintenance_window_available": False},
        )
        assert result["candidate_action"] == "INSPECT"
        assert result["constraint_substitutions"]
        assert result["human_review"] is True

    def test_no_violations_leaves_the_action_untouched(self):
        result = check_constraints(self._decision("MONITOR", rul=50.0), {})
        assert result["candidate_action"] == "MONITOR"
        assert result["severity_delta"] == 0
        assert result["constraint_status"] == "SATISFIED"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_every_scenario_is_retrievable(self):
        for name in scenario_names():
            constraints = get_scenario(name)
            assert isinstance(constraints, dict)

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown constraint scenario"):
            get_scenario("does_not_exist")

    def test_unconstrained_is_empty(self):
        assert get_scenario("unconstrained") == {}

    def test_degraded_operations_covers_every_constraint(self):
        constraints = get_scenario("degraded_operations")
        for key in (
            "maintenance_window_available",
            "minimum_safe_rul",
            "maintenance_lead_time",
            "max_allowable_delay",
            "resource_availability",
        ):
            assert key in constraints

    def test_scenario_returns_a_copy(self):
        first = get_scenario("degraded_operations")
        first["minimum_safe_rul"] = 999.0
        assert get_scenario("degraded_operations")["minimum_safe_rul"] != 999.0

    def test_describe_is_non_empty(self):
        assert describe_scenario("unconstrained")
        assert describe_scenario("degraded_operations")

    def test_scenarios_produce_constraint_violations(self):
        """The scenarios must actually exercise the constraint layer."""

        state = _state(rul=20.0, hi=0.40, lower=15.0, upper=25.0, eri=0.90)
        result = recommend_action(
            state,
            constraints=CONSTRAINT_SCENARIOS["degraded_operations"],
        )
        assert result["constraint_status"] == "VIOLATION"
        assert result["constraint_violations"]

    def test_scenarios_only_ever_escalate_or_substitute(self):
        """No scenario may make a decision less severe than the base policy."""

        states = [
            _state(rul=80.0, hi=0.85, lower=70.0, upper=90.0, eri=0.90),
            _state(rul=40.0, hi=0.65, lower=35.0, upper=45.0, eri=0.90),
            _state(rul=20.0, hi=0.40, lower=15.0, upper=25.0, eri=0.90),
            _state(rul=5.0, hi=0.10, lower=0.0, upper=15.0, eri=0.90),
        ]
        for state in states:
            base = recommend_action(state)
            base_severity = base["action_severity"]
            for name in scenario_names():
                result = recommend_action(state, constraints=get_scenario(name))
                # Only the maintenance-window substitution may reduce the
                # commitment; every other rule escalates.
                if result.get("constraint_substitutions"):
                    continue
                assert result["action_severity"] >= base_severity, (
                    f"scenario {name} de-escalated a decision"
                )


# ---------------------------------------------------------------------------
# apply_rules / recommend_action integration
# ---------------------------------------------------------------------------

class TestApplyRulesSeverity:
    def test_severity_is_reported(self):
        result = apply_rules(_state())
        assert result["action_severity"] == severity_of(result["candidate_action"])

    def test_recommend_action_reports_severity(self):
        result = recommend_action(_state())
        assert result["action_severity"] == severity_of(result["recommended_action"])

    def test_reasons_mention_the_escalation(self):
        result = apply_rules(
            _state(rul=20.0, hi=0.40, lower=18.0, upper=22.0, eri=0.90)
        )
        assert result["candidate_action"] == "SCHEDULE_MAINTENANCE"

        # A lower bound inside the critical band plans the action as urgent.
        escalated = apply_rules(
            _state(rul=20.0, hi=0.40, lower=5.0, upper=45.0, eri=0.90)
        )
        assert escalated["action_severity"] > result["action_severity"]
        assert any("planning" in reason.lower() for reason in escalated["reasons"])

        # A low ERI escalates by one step on the ladder.
        weak_explanation = apply_rules(
            _state(rul=80.0, hi=0.85, lower=78.0, upper=82.0, eri=0.30)
        )
        assert weak_explanation["action_severity"] > severity_of("CONTINUE_OPERATION")
        assert any("escalating" in reason for reason in weak_explanation["reasons"])


# ---------------------------------------------------------------------------
# The ERI bands must be classified identically by the report and the policy
# ---------------------------------------------------------------------------

class TestEriBandConsistency:
    """The report and the policy must agree on the ERI bands.

    They used to disagree on the boundary values: the report used ``>`` and
    the policy ``>=``, so an ERI of exactly 0.80 was ``HIGH`` for the
    policy and ``MODERATE`` for the report.
    """

    @staticmethod
    def _report_band(eri: float) -> str:
        if eri >= ERI_HIGH_THRESHOLD:
            return "HIGH"
        if eri >= ERI_LOW_THRESHOLD:
            return "MODERATE"
        return "LOW"

    @pytest.mark.parametrize(
        "eri",
        [0.0, 0.3, 0.59, 0.6, 0.61, 0.79, 0.8, 0.81, 0.95, 1.0],
    )
    def test_bands_agree(self, eri: float):
        assert self._report_band(eri) == classify_explanation_reliability(eri)

    def test_thresholds_come_from_the_shared_constants(self):
        from src.decision_engine.rules import DEFAULT_THRESHOLDS

        assert DEFAULT_THRESHOLDS["eri_low"] == ERI_LOW_THRESHOLD
        assert DEFAULT_THRESHOLDS["eri_high"] == ERI_HIGH_THRESHOLD
