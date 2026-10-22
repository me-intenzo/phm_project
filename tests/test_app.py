"""
Smoke tests for the Streamlit dashboard (``app.py``).

The dashboard is exercised with Streamlit's own ``AppTest`` harness, so
the script is really executed and every tab is really rendered.  The tests
are skipped when Streamlit is not installed, because the dashboard is an
optional extra (``pip install -e ".[dashboard]"``).

author: me-intenzo
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1",
    reason="streamlit is not installed; the dashboard is an optional extra.",
)

AppTest = streamlit_testing.AppTest

APP_PATH = ROOT / "app.py"


def _has_artifacts() -> bool:
    """True when at least one evaluated run exists to render."""

    results = ROOT / "outputs" / "results"
    if not results.exists():
        return False
    return any(results.glob("*_predictions.npz"))


@pytest.fixture(scope="module")
def app() -> "AppTest":
    """A rendered instance of the dashboard."""

    instance = AppTest.from_file(str(APP_PATH), default_timeout=300)
    instance.run()
    return instance


@pytest.mark.skipif(not _has_artifacts(), reason="no pipeline artifacts to render")
def test_dashboard_renders_without_errors(app: "AppTest") -> None:

    assert not app.exception, [str(error.value) for error in app.exception]
    assert not app.error, [str(error.value) for error in app.error]


@pytest.mark.skipif(not _has_artifacts(), reason="no pipeline artifacts to render")
def test_dashboard_exposes_every_view(app: "AppTest") -> None:

    labels = [tab.label for tab in app.tabs]

    assert len(labels) == 5
    for expected in ("Model", "Uncertainty", "XAI", "Decision", "HITL"):
        assert any(expected in label for label in labels), labels

    # The model view reports its accuracy metrics, the uncertainty view
    # reports the calibrated coverage and the XAI view reports the ERI.
    metric_labels = {metric.label for metric in app.metric}

    assert "RUL MAE" in metric_labels
    assert "Empirical coverage" in metric_labels
    assert "ERI" in metric_labels
    assert "Decisions" in metric_labels


@pytest.mark.skipif(not _has_artifacts(), reason="no pipeline artifacts to render")
def test_dashboard_renders_charts(app: "AppTest") -> None:

    assert len(app.get("plotly_chart")) > 0
    assert len(app.dataframe) > 0


def test_run_discovery_ignores_unknown_architectures() -> None:
    """Only registered architectures may become selectable runs."""

    sys.path.insert(0, str(ROOT))
    import app as dashboard  # noqa: PLC0415 - imported inside the test on purpose

    models = dashboard._registered_models()

    assert dashboard._split_run_name("gru", models) == ("gru", "")
    assert dashboard._split_run_name("gru_att_deg_v2_ms42", models) == (
        "gru_att_deg_v2",
        "ms42",
    )
    assert dashboard._split_run_name("ensemble", models) is None


def test_metrics_report_accepts_both_key_spellings() -> None:
    """A report written by an older pipeline must still show its numbers.

    41 of the metrics artifacts in this repository use the legacy
    ``RUL`` / ``HI`` keys; only the current ``rul`` / ``hi`` spelling was
    understood before, which left every metric of those runs on "n/a".
    """

    sys.path.insert(0, str(ROOT))
    import app as dashboard  # noqa: PLC0415 - imported inside the test on purpose

    legacy = dashboard._normalise_metrics(
        {"metrics": {"RUL": {"MAE": 8.8}, "HI": {"MAE": 0.08}}}
    )
    current = dashboard._normalise_metrics(
        {"metrics": {"rul": {"MAE": 8.8}, "hi": {"MAE": 0.08}}}
    )

    assert legacy is not None and current is not None
    assert legacy["rul"]["MAE"] == 8.8
    assert legacy["hi"]["MAE"] == 0.08
    assert current["rul"]["MAE"] == 8.8
    assert current["hi"]["MAE"] == 0.08

    # A report without any RUL block is not usable.
    assert dashboard._normalise_metrics({"metrics": {}}) is None
    assert dashboard._normalise_metrics(None) is None


def test_metrics_can_be_recomputed_from_the_predictions() -> None:
    """A run without a metrics report still shows real numbers."""

    sys.path.insert(0, str(ROOT))
    import numpy as np

    import app as dashboard  # noqa: PLC0415 - imported inside the test on purpose

    truth = np.array([100.0, 50.0, 0.0])
    guess = np.array([90.0, 55.0, 5.0])

    computed = dashboard.metrics_from_predictions(
        {
            "y_rul_true": truth,
            "y_rul_pred": guess,
            "y_hi_true": np.array([0.9, 0.5, 0.1]),
            "y_hi_pred": np.array([0.8, 0.5, 0.2]),
        }
    )

    assert set(computed) == {"rul", "hi"}
    # Errors are -10, +5, +5 cycles.
    assert computed["rul"]["MAE"] == pytest.approx(20.0 / 3.0)
    assert computed["rul"]["RMSE"] == pytest.approx(np.sqrt(50.0))
    assert "R2" in computed["rul"]
    assert "NASA_Score" in computed["rul"]


def test_stage_commands_carry_the_seed_tag() -> None:
    """The command that produces a missing artifact must be exact."""

    sys.path.insert(0, str(ROOT))
    import app as dashboard  # noqa: PLC0415 - imported inside the test on purpose

    untagged = dashboard._stage_commands("gru_att_deg_v2", "FD001", "")
    tagged = dashboard._stage_commands("gru_att_deg_v2", "FD001", "ms42")

    assert "--tag" not in untagged["decision"]
    assert "--tag ms42" in tagged["decision"]
    assert "--tag ms42" in tagged["hitl"]
    assert "--tag ms42" in tagged["uncertainty"]

    for command in tagged.values():
        assert command.startswith("python scripts/")
