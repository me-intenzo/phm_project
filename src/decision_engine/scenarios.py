"""Named operational-constraint scenarios.

Why this module exists
----------------------
The constraint layer used to be exercised only in unit tests: the certified
multi-seed runs produced their decisions with **no** operational constraints,
so every published ``constraint_violation_count`` was zero and the subsystem
was unvalidated by any recorded result.

A *scenario* is a named, reproducible set of operational constraints.  The
decision CLI can apply one (``--constraint-scenario``), the multi-seed
pipeline records which scenario produced each decision artifact, and the
sensitivity analysis compares the decision mix under each scenario.  The
scenarios are the ones an operator actually faces:

``unconstrained``
    No operational constraints.  The reference reading.

``no_maintenance_window``
    The engine is not in a shop right now, so scheduled maintenance cannot
    be booked.  Scheduled maintenance is substituted by an inspection and the
    record is flagged for human review.

``tight_safety_margin``
    A short minimum safe RUL and a short maximum allowable delay: passive
    operation is no longer acceptable close to end of life.

``lead_time_pressure``
    Maintenance needs a long lead time (parts, crews, access), so a
    scheduled maintenance that cannot start in time has to become urgent.

``resource_shortage``
    Crew and parts are unavailable, so every active action has to go
    through a human before it can be executed.

``degraded_operations``
    All of the above at once — the worst realistic case, and the one the
    certified pipeline uses.

The numeric limits are chosen so that every rule actually binds on the
C-MAPSS test engines: a passive action only ever sits on an engine with more
than 45 cycles of life left, so a 30-cycle minimum safe RUL would never
fire, and a 20-cycle lead time would only touch the most degraded
at-risk engines.

author: me-intenzo
"""

from __future__ import annotations

from typing import Any

#: Scenario name -> operational constraints handed to the decision engine.
CONSTRAINT_SCENARIOS: dict[str, dict[str, Any]] = {
    "unconstrained": {},

    "no_maintenance_window": {
        "maintenance_window_available": False,
    },

    "tight_safety_margin": {
        "minimum_safe_rul": 50.0,
        "max_allowable_delay": 60.0,
    },

    "lead_time_pressure": {
        "maintenance_lead_time": 30.0,
    },

    "resource_shortage": {
        "resource_availability": False,
    },

    "degraded_operations": {
        "maintenance_window_available": False,
        "minimum_safe_rul": 50.0,
        "maintenance_lead_time": 30.0,
        "max_allowable_delay": 60.0,
        "resource_availability": False,
    },
}

DEFAULT_SCENARIO = "degraded_operations"


def scenario_names() -> list[str]:
    """Names of every available scenario, in declaration order."""

    return list(CONSTRAINT_SCENARIOS)


def get_scenario(name: str) -> dict[str, Any]:
    """Return a copy of the constraints of the named scenario."""

    key = str(name).strip().lower()
    if key not in CONSTRAINT_SCENARIOS:
        raise ValueError(
            f"Unknown constraint scenario {name!r}. "
            f"Available: {', '.join(scenario_names())}"
        )
    return dict(CONSTRAINT_SCENARIOS[key])


def describe_scenario(name: str) -> str:
    """One-line human-readable summary of a scenario."""

    constraints = get_scenario(name)
    if not constraints:
        return "no operational constraints"
    return ", ".join(f"{key}={value}" for key, value in sorted(constraints.items()))


__all__ = [
    "CONSTRAINT_SCENARIOS",
    "DEFAULT_SCENARIO",
    "describe_scenario",
    "get_scenario",
    "scenario_names",
]
