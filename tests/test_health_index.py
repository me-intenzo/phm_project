"""Tests for the sensor-derived health index.

The index replaces the legacy ``HI = RUL / max_rul`` label, which was a
deterministic affine function of the RUL label (docs/ARCHITECTURE.md §7.2).
These tests use synthetic frames so the suite stays green without the
(uncommitted) C-MAPSS dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.health_index import (
    HealthIndexModel,
    calibrate_hi_thresholds,
)

WINDOW_SIZE = 10
BASELINE = 5


def _synthetic_frame(
    n_engines: int = 6,
    cycles: int = 60,
    sensors: int = 4,
    seed: int = 0,
    with_conditions: bool = True,
) -> pd.DataFrame:
    """Engines whose sensors degrade linearly with the remaining life.

    The operating settings are constant per engine (one operating condition
    per engine) so the condition-removal step has something to remove.
    """

    rng = np.random.default_rng(seed)
    rows = []

    for engine in range(1, n_engines + 1):
        condition = float(engine % 3)
        for cycle in range(1, cycles + 1):
            wear = 1.0 - (cycle / cycles)
            row = {
                "unit_number": engine,
                "time_in_cycles": cycle,
                "operational_setting_1": condition,
                "operational_setting_2": 0.5 * condition,
                "operational_setting_3": 1.0,
                "RUL": float(cycles - cycle),
            }
            for s in range(1, sensors + 1):
                # Each sensor degrades at its own rate, plus noise.
                row[f"sensor_{s}"] = (
                    10.0 * s
                    + 20.0 * s * wear
                    + condition * 100.0 * s
                    + rng.normal(0.0, 0.05)
                )
            rows.append(row)

    return pd.DataFrame(rows)


def _split(df: pd.DataFrame, n_engines: int):
    train = df[df["unit_number"] <= n_engines].copy()
    test = df[df["unit_number"] > n_engines].copy()
    return train, test


def _fit(df: pd.DataFrame, n_engines: int = 4):
    train, test = _split(df, n_engines)
    sensors = [c for c in df.columns if c.startswith("sensor_")]
    model = HealthIndexModel(
        baseline_cycles=BASELINE,
        window_size=WINDOW_SIZE,
    )
    model.fit(
        raw_df=train,
        scaled_df=train,
        sensor_columns=sensors,
        target_rul=train["RUL"],
    )
    return model, train, test, sensors


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestHealthIndexModel:
    def test_index_is_in_the_unit_interval(self):
        df = _synthetic_frame()
        model, train, _, sensors = _fit(df)
        hi = model.transform(raw_df=train, scaled_df=train, sensor_columns=sensors)
        assert hi.min() >= 0.0
        assert hi.max() <= 1.0

    def test_index_falls_with_degradation(self):
        """A higher RUL (healthier) must give a higher index."""

        df = _synthetic_frame()
        model, train, _, sensors = _fit(df)
        hi = model.transform(raw_df=train, scaled_df=train, sensor_columns=sensors)
        correlation = float(np.corrcoef(hi, train["RUL"].to_numpy())[0, 1])
        assert correlation > 0.5, (
            f"the index must track the degradation state, got {correlation:.3f}"
        )

    def test_index_is_not_a_deterministic_function_of_rul(self):
        """The whole point: the index must not be a re-scaled RUL."""

        df = _synthetic_frame()
        model, train, _, sensors = _fit(df)
        hi = model.transform(raw_df=train, scaled_df=train, sensor_columns=sensors)
        correlation = float(np.corrcoef(hi, train["RUL"].to_numpy())[0, 1])
        assert correlation < 0.999, (
            "the index is a deterministic function of RUL — the legacy "
            "degeneracy is back"
        )

    def test_index_generalises_to_unseen_engines(self):
        df = _synthetic_frame(n_engines=8)
        model, train, test, sensors = _fit(df, n_engines=4)
        hi = model.transform(raw_df=test, scaled_df=test, sensor_columns=sensors)
        correlation = float(np.corrcoef(hi, test["RUL"].to_numpy())[0, 1])
        assert correlation > 0.3

    def test_transform_requires_fit(self):
        model = HealthIndexModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.transform(raw_df=None, scaled_df=None)

    def test_fit_requires_a_target_or_falls_back(self):
        df = _synthetic_frame()
        train, _ = _split(df, 4)
        sensors = [c for c in df.columns if c.startswith("sensor_")]
        model = HealthIndexModel(baseline_cycles=BASELINE, window_size=WINDOW_SIZE)
        model.fit(train, train, sensors)
        assert model.fitted
        assert model.direction is not None

    def test_empty_sensor_list_raises(self):
        df = _synthetic_frame()
        train, _ = _split(df, 4)
        model = HealthIndexModel(baseline_cycles=BASELINE, window_size=WINDOW_SIZE)
        with pytest.raises(ValueError, match="sensor column"):
            model.fit(train, train, [])

    def test_save_and_load_round_trip(self, tmp_path):
        df = _synthetic_frame()
        model, train, _, sensors = _fit(df)
        path = tmp_path / "health_index.pkl"
        model.save(path)
        restored = HealthIndexModel.load(path)

        original = model.transform(raw_df=train, scaled_df=train, sensor_columns=sensors)
        reloaded = restored.transform(
            raw_df=train, scaled_df=train, sensor_columns=sensors
        )
        assert np.allclose(original, reloaded)

    def test_summary_describes_the_fitted_index(self):
        df = _synthetic_frame()
        model, _, _, _ = _fit(df)
        summary = model.summary()
        assert summary["mode"] == "sensor"
        assert summary["baseline_cycles"] == BASELINE
        assert summary["window_size"] == WINDOW_SIZE
        assert summary["reference_min"] < summary["reference_max"]

    def test_row_order_is_preserved(self):
        df = _synthetic_frame()
        model, train, _, sensors = _fit(df)
        shuffled = train.sample(frac=1.0, random_state=3)
        hi = model.transform(raw_df=shuffled, scaled_df=shuffled, sensor_columns=sensors)
        reference = model.transform(
            raw_df=train, scaled_df=train, sensor_columns=sensors
        )
        assert np.allclose(hi, reference[shuffled.index.to_numpy()])


# ---------------------------------------------------------------------------
# Breakpoint calibration
# ---------------------------------------------------------------------------

class TestHiThresholdCalibration:
    def test_breakpoints_are_ordered(self):
        # The index is 1 for a healthy engine and 0 for a failed one, so the
        # breakpoints must increase as the health state improves.
        rul = np.linspace(125.0, 0.0, 1000)
        hi = rul / 125.0
        thresholds = calibrate_hi_thresholds(hi, rul)
        assert thresholds["hi_critical"] < thresholds["hi_at_risk"]
        assert thresholds["hi_at_risk"] < thresholds["hi_degrading"]

    def test_breakpoints_are_in_the_unit_interval(self):
        rul = np.linspace(125.0, 0.0, 1000)
        hi = rul / 125.0
        thresholds = calibrate_hi_thresholds(hi, rul)
        for value in thresholds.values():
            assert 0.0 <= value <= 1.0

    def test_missing_band_raises(self):
        hi = np.ones(10)
        rul = np.full(10, 100.0)     # no window in any RUL band
        with pytest.raises(ValueError, match="No training windows"):
            calibrate_hi_thresholds(hi, rul)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calibrate_hi_thresholds([0.1, 0.2], [1.0])

    def test_breakpoints_are_the_band_medians(self):
        """Each breakpoint is the median index of its RUL band."""

        rul = np.concatenate([np.linspace(0, 10, 100), np.linspace(10, 25, 150),
                              np.linspace(25, 50, 250), np.linspace(50, 125, 500)])
        hi = rul / 125.0
        thresholds = calibrate_hi_thresholds(hi, rul)

        assert thresholds["hi_critical"] == pytest.approx(
            float(np.median(hi[(rul > -np.inf) & (rul <= 10)]))
        )
        assert thresholds["hi_at_risk"] == pytest.approx(
            float(np.median(hi[(rul > 10) & (rul <= 25)]))
        )
        assert thresholds["hi_degrading"] == pytest.approx(
            float(np.median(hi[(rul > 25) & (rul <= 50)]))
        )

    def test_breakpoints_move_with_the_index_scale(self):
        """Rescaling the index must move the breakpoints with it."""

        rul = np.concatenate([np.linspace(0, 10, 100), np.linspace(10, 25, 150),
                              np.linspace(25, 50, 250), np.linspace(50, 125, 500)])
        plain = calibrate_hi_thresholds(rul / 125.0, rul)
        compressed = calibrate_hi_thresholds((rul / 125.0) * 0.5, rul)
        assert compressed["hi_critical"] == pytest.approx(
            plain["hi_critical"] * 0.5
        )
