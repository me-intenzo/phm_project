"""
health_index.py

Sensor-derived health index (SDHI) for the C-MAPSS subsets.

Why this module exists
----------------------
The original health-index label was ``HI = RUL / max_rul`` — a *deterministic
affine function of the RUL label*.  On FD001-FD004 the two labels therefore
correlated at 1.000000 to machine precision, the HI head had nothing to learn
that the RUL head had not already learned, and the reported HI metrics were a
by-product of the shared trunk rather than a measurement of anything
(see ``docs/ARCHITECTURE.md`` §7.2).

This module builds a health index from the *sensor signal* instead, so that:

* it is strongly — but not perfectly — related to the true degradation state
  (it is a *health indicator*, not a re-scaled RUL), and
* it is a genuinely different learning problem for the multi-task trunk, so
  the HI head contributes real gradient to the shared representation.

Construction
------------
1. **Condition removal.**  Each sensor is regressed on the three operational
   settings (least squares, fitted on the *training* split only) and the
   residual is kept.  On FD002/FD004 this is what makes the index usable at
   all: without it the dominant variance direction of the sensors is the
   operating condition rather than the degradation.

2. **Per-engine baseline.**  Each engine's first ``baseline_cycles`` cycles
   define its own healthy reference.  Subtracting it removes the
   engine-to-engine offset (different units wear differently) without ever
   looking at the engine's end of life.

3. **Degradation direction.**  A one-component PLS direction is fitted on the
   *training* windows against the training RUL label.  PLS is used instead of
   PCA because the direction we want is the one that tracks degradation, not
   the one that explains the most variance.

4. **Projection and mapping.**  The projection of a window's centred feature
   vector is mapped linearly onto ``[0, 1]`` with reference quantiles taken
   from the training windows: ``HI = 1 - clip((d - lo) / (hi - lo), 0, 1)``.

Window alignment
----------------
The index of a row is computed from the **same trailing window the model
sees** (left-padded with the engine's first cycle when the engine is shorter
than ``window_size``, matching
``src.preprocessing.windowing.WindowGenerator.create_test_windows``).  The
label attached to a training window is therefore a function of exactly that
window's features — no information from the future, and no mismatch between
what the label says and what the input contains.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OPERATIONAL_SETTINGS = (
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
)

_EPS = 1e-12


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sorted_by_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted by engine and cycle, with a positional index."""

    return df.sort_values(["unit_number", "time_in_cycles"], kind="mergesort")


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Least-squares design matrix for the operating conditions.

    Constant operating settings carry no information and are dropped, which
    keeps the matrix well conditioned on FD001/FD003 (where all three
    settings are constant).
    """

    columns: list[np.ndarray] = [np.ones(len(df), dtype=np.float64)]

    for name in OPERATIONAL_SETTINGS:
        if name not in df.columns:
            continue
        values = np.asarray(df[name], dtype=np.float64)
        if values.std() <= _EPS:
            continue
        columns.append(values)

    return np.column_stack(columns)


def _trailing_window_means(
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Rolling mean over the trailing ``window_size`` rows of one engine.

    Rows before the first full window are left-padded with the engine's first
    row, which is the same padding convention the test-window builder uses.
    """

    n = len(values)
    if n == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float64)

    cumulative = np.cumsum(values, axis=0)
    first = values[0]

    out = np.empty((n, values.shape[1]), dtype=np.float64)
    for index in range(n):
        start = index - window_size + 1
        if start >= 0:
            window_sum = cumulative[index] - (
                cumulative[start - 1] if start > 0 else 0.0
            )
        else:
            pad = window_size - (index + 1)
            window_sum = cumulative[index] + pad * first
        out[index] = window_sum / window_size
    return out


def _pls_direction(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One-component PLS direction (NIPALS-style closed form).

    Returns a unit-norm vector ``w`` such that ``features @ w`` is the
    projection that best tracks ``target`` in a single component.
    """

    x_centered = features - features.mean(axis=0)
    y_centered = target - target.mean()

    direction = x_centered.T @ y_centered
    norm = np.linalg.norm(direction)
    if norm <= _EPS:
        # Degenerate response: fall back to the first principal component so
        # the index still has a well-defined direction.
        _, vectors = np.linalg.eigh(np.cov(x_centered, rowvar=False))
        return vectors[:, -1]

    return direction / norm


# ---------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------

class HealthIndexModel:
    """Sensor-derived health index.

    Parameters
    ----------
    baseline_cycles:
        Number of leading cycles per engine used as that engine's healthy
        reference.
    window_size:
        Length of the trailing window the index is computed from.  It must
        match the window size used for the model input.
    reference_low / reference_high:
        Percentiles of the training projection that map to HI = 1 and HI = 0.
    """

    def __init__(
        self,
        baseline_cycles: int = 30,
        window_size: int = 40,
        reference_low: float = 1.0,
        reference_high: float = 99.0,
    ) -> None:
        self.baseline_cycles = int(baseline_cycles)
        self.window_size = int(window_size)
        self.reference_low = float(reference_low)
        self.reference_high = float(reference_high)

        # Fitted state
        self.sensor_columns: list[str] = []
        self.condition_coefficients: dict[str, np.ndarray] = {}
        self.direction: np.ndarray | None = None
        self.reference_min: float = float("nan")
        self.reference_max: float = float("nan")
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _condition_residuals(
        self,
        raw_df: pd.DataFrame,
        scaled_df: pd.DataFrame,
        columns: Sequence[str],
    ) -> np.ndarray:
        """Remove the operating conditions from every sensor."""

        design = _design_matrix(raw_df)
        residuals = np.empty((len(scaled_df), len(columns)), dtype=np.float64)
        for index, name in enumerate(columns):
            values = np.asarray(scaled_df[name], dtype=np.float64)
            coefficients = self.condition_coefficients[name]
            residuals[:, index] = values - design @ coefficients
        return residuals

    def _engine_baselines(
        self,
        residuals: np.ndarray,
        ordered: pd.DataFrame,
    ) -> dict[int, np.ndarray]:
        """Per-engine healthy reference (mean of the leading cycles)."""

        baselines: dict[int, np.ndarray] = {}
        units = np.asarray(ordered["unit_number"])
        for unit in np.unique(units):
            mask = units == unit
            head = residuals[mask][: self.baseline_cycles]
            baselines[int(unit)] = head.mean(axis=0)
        return baselines

    def _window_features(
        self,
        residuals: np.ndarray,
        ordered: pd.DataFrame,
        baselines: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Trailing-window mean of the centred residuals, per engine."""

        units = np.asarray(ordered["unit_number"])
        features = np.empty_like(residuals)
        for unit in np.unique(units):
            mask = units == unit
            engine_values = residuals[mask] - baselines[int(unit)]
            engine_features = _trailing_window_means(engine_values, self.window_size)
            features[mask] = engine_features
        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        raw_df: pd.DataFrame,
        scaled_df: pd.DataFrame,
        sensor_columns: Sequence[str],
        target_rul: pd.Series | Sequence[float] | None = None,
    ) -> "HealthIndexModel":
        """Fit the index on the **training** split.

        Parameters
        ----------
        raw_df:
            Frame that still carries the operating settings (unscaled is fine).
        scaled_df:
            Frame with the selected, standardised sensor columns.
        sensor_columns:
            Sensor columns used for the index.
        target_rul:
            Training RUL label, used only to orient the PLS direction.  A
            ``pd.Series`` is aligned by index; a plain sequence is assumed to
            be in the row order of ``raw_df``.  When omitted, the direction
            falls back to the leading principal component.
        """

        self.sensor_columns = [str(name) for name in sensor_columns]
        if not self.sensor_columns:
            raise ValueError("At least one sensor column is required.")

        ordered_raw = _sorted_by_cycle(raw_df)
        ordered_scaled = _sorted_by_cycle(scaled_df)

        # ---- condition model (training split only) -----------------
        design = _design_matrix(ordered_raw)
        for name in self.sensor_columns:
            values = np.asarray(ordered_scaled[name], dtype=np.float64)
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            self.condition_coefficients[name] = coefficients

        residuals = self._condition_residuals(
            ordered_raw, ordered_scaled, self.sensor_columns
        )

        baselines = self._engine_baselines(residuals, ordered_raw)
        features = self._window_features(residuals, ordered_raw, baselines)

        # ---- degradation direction --------------------------------
        units = np.asarray(ordered_raw["unit_number"])
        # Only rows that close a full window are usable training samples:
        # the model never sees a shorter history than that.
        eligible = np.zeros(len(ordered_raw), dtype=bool)
        for unit in np.unique(units):
            position = np.flatnonzero(units == unit)
            eligible[position[self.window_size - 1:]] = True

        rul_ordered = self._aligned_target(target_rul, raw_df, ordered_raw)

        if rul_ordered is None:
            self.direction = _pls_direction(
                features[eligible], np.zeros(int(eligible.sum()))
            )
        else:
            self.direction = _pls_direction(features[eligible], rul_ordered[eligible])

        projection = features[eligible] @ self.direction
        if rul_ordered is not None:
            correlation = np.corrcoef(projection, rul_ordered[eligible])[0, 1]
            if np.isfinite(correlation) and correlation > 0:
                # A larger projection must mean *more* degradation, i.e. a
                # smaller RUL.  Flip when the fitted direction says the
                # opposite.
                self.direction = -self.direction
                projection = -projection

        self.reference_min = float(
            np.percentile(projection, self.reference_low)
        )
        self.reference_max = float(
            np.percentile(projection, self.reference_high)
        )
        if self.reference_max - self.reference_min <= _EPS:
            raise ValueError(
                "The health-index projection is degenerate on the training "
                "split; the reference range cannot be established."
            )

        self.fitted = True

        logger.info(
            "Health index fitted | sensors=%d | baseline_cycles=%d | "
            "window=%d | reference=[%.4f, %.4f]",
            len(self.sensor_columns),
            self.baseline_cycles,
            self.window_size,
            self.reference_min,
            self.reference_max,
        )
        return self

    @staticmethod
    def _aligned_target(
        target: pd.Series | Sequence[float] | None,
        raw_df: pd.DataFrame,
        ordered_raw: pd.DataFrame,
    ) -> np.ndarray | None:
        """Bring the RUL target into the sorted row order."""

        if target is None:
            return None
        if isinstance(target, pd.Series):
            return np.asarray(target.loc[ordered_raw.index], dtype=np.float64)
        values = np.asarray(target, dtype=np.float64)
        if len(values) != len(raw_df):
            raise ValueError(
                "target_rul must have one entry per row of the fitted frame."
            )
        order = _sort_index(raw_df)
        return values[order]

    def transform(
        self,
        raw_df: pd.DataFrame,
        scaled_df: pd.DataFrame,
        sensor_columns: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Compute the health index for every row of *raw_df*.

        The rows are returned in the order of *raw_df*.
        """

        if not self.fitted:
            raise RuntimeError("HealthIndexModel has not been fitted.")

        columns = [str(name) for name in (sensor_columns or self.sensor_columns)]
        missing = [name for name in columns if name not in self.condition_coefficients]
        if missing:
            raise ValueError(
                f"Sensor columns were not present during fit: {missing}"
            )

        ordered_raw = _sorted_by_cycle(raw_df)
        ordered_scaled = _sorted_by_cycle(scaled_df)

        residuals = self._condition_residuals(
            ordered_raw, ordered_scaled, columns
        )
        baselines = self._engine_baselines(residuals, ordered_raw)
        features = self._window_features(residuals, ordered_raw, baselines)

        projection = features @ self.direction
        span = self.reference_max - self.reference_min
        hi = 1.0 - np.clip((projection - self.reference_min) / span, 0.0, 1.0)

        # Undo the sort so the returned array matches the caller's row order.
        sorted_positions = _sort_index(raw_df)
        out = np.empty_like(hi)
        out[sorted_positions] = hi
        return out

    def project(
        self,
        raw_df: pd.DataFrame,
        scaled_df: pd.DataFrame,
        sensor_columns: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return the raw degradation projection (before the [0, 1] mapping)."""

        if not self.fitted:
            raise RuntimeError("HealthIndexModel has not been fitted.")

        columns = [str(name) for name in (sensor_columns or self.sensor_columns)]
        ordered_raw = _sorted_by_cycle(raw_df)
        ordered_scaled = _sorted_by_cycle(scaled_df)
        residuals = self._condition_residuals(ordered_raw, ordered_scaled, columns)
        baselines = self._engine_baselines(residuals, ordered_raw)
        features = self._window_features(residuals, ordered_raw, baselines)
        projection = features @ self.direction
        sorted_positions = _sort_index(raw_df)
        out = np.empty_like(projection)
        out[sorted_positions] = projection
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "baseline_cycles": self.baseline_cycles,
                "window_size": self.window_size,
                "reference_low": self.reference_low,
                "reference_high": self.reference_high,
                "sensor_columns": self.sensor_columns,
                "condition_coefficients": self.condition_coefficients,
                "direction": self.direction,
                "reference_min": self.reference_min,
                "reference_max": self.reference_max,
                "fitted": self.fitted,
            },
            path,
        )
        logger.info("Health-index model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HealthIndexModel":
        payload = joblib.load(Path(path))
        model = cls(
            baseline_cycles=payload["baseline_cycles"],
            window_size=payload["window_size"],
            reference_low=payload["reference_low"],
            reference_high=payload["reference_high"],
        )
        model.sensor_columns = list(payload["sensor_columns"])
        model.condition_coefficients = dict(payload["condition_coefficients"])
        model.direction = payload["direction"]
        model.reference_min = float(payload["reference_min"])
        model.reference_max = float(payload["reference_max"])
        model.fitted = bool(payload["fitted"])
        return model

    def summary(self) -> dict[str, Any]:
        """Machine-readable description of the fitted index."""

        return {
            "mode": "sensor",
            "baseline_cycles": self.baseline_cycles,
            "window_size": self.window_size,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
            "reference_min": self.reference_min,
            "reference_max": self.reference_max,
            "n_sensors": len(self.sensor_columns),
            "sensor_columns": list(self.sensor_columns),
        }


def _sort_index(df: pd.DataFrame) -> np.ndarray:
    """Positional index that sorts *df* by engine and cycle."""

    keys = np.lexsort(
        (
            np.asarray(df["time_in_cycles"], dtype=np.float64),
            np.asarray(df["unit_number"], dtype=np.float64),
        )
    )
    return keys


def calibrate_hi_thresholds(
    hi: Sequence[float],
    rul: Sequence[float],
    rul_critical: float = 10.0,
    rul_at_risk: float = 25.0,
    rul_degrading: float = 50.0,
) -> dict[str, float]:
    """Calibrate the health-index breakpoints of the decision policy.

    The decision engine decides a health state from *either* the RUL or the
    HI criterion.  With the legacy label the two were consistent by
    construction; with the sensor-derived index they are not, because the
    index has its own scale.

    The three breakpoints are therefore calibrated on the **training** split
    so that the HI criterion fires at the same degradation levels as the RUL
    criterion: each breakpoint is the median index of the training windows
    whose RUL lies in the corresponding band.

    Parameters
    ----------
    hi, rul:
        The training-split index and RUL label.
    rul_critical, rul_at_risk, rul_degrading:
        The RUL breakpoints of the decision policy.

    Returns
    -------
    dict
        ``{"hi_critical": ..., "hi_at_risk": ..., "hi_degrading": ...}``
    """

    hi_values = np.asarray(hi, dtype=np.float64).reshape(-1)
    rul_values = np.asarray(rul, dtype=np.float64).reshape(-1)

    if hi_values.shape != rul_values.shape:
        raise ValueError("hi and rul must have the same length.")

    bands = (
        ("hi_critical", rul_critical, -np.inf, rul_critical),
        ("hi_at_risk", rul_at_risk, rul_critical, rul_at_risk),
        ("hi_degrading", rul_degrading, rul_at_risk, rul_degrading),
    )

    thresholds: dict[str, float] = {}

    for name, _breakpoint, low, high in bands:
        mask = (rul_values > low) & (rul_values <= high)
        if not np.any(mask):
            raise ValueError(
                f"No training windows in the RUL band ({low}, {high}]; "
                "the health-index breakpoints cannot be calibrated."
            )
        thresholds[name] = float(np.median(hi_values[mask]))

    # The bands are ordered by degradation, so the breakpoints have to be
    # non-decreasing for the policy to be well defined.  A noisy index can
    # invert two band medians; the later one is then lifted to the earlier
    # value rather than the pair being swapped.
    names = ("hi_critical", "hi_at_risk", "hi_degrading")
    values = [
        float(min(max(thresholds[name], 0.0), 1.0)) for name in names
    ]
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1])
    for name, value in zip(names, values):
        thresholds[name] = value

    return thresholds


__all__ = [
    "HealthIndexModel",
    "OPERATIONAL_SETTINGS",
    "calibrate_hi_thresholds",
]
