"""
preprocess.py

Full preprocessing pipeline for NASA C-MAPSS.

Pipeline
--------
Load -> Validate -> Summary -> Visualize -> Generate RUL/HI
-> Feature Selection -> Feature Scaling -> Sliding Windows
-> Save -> Logging -> Done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.feature_selection import FeatureSelector
from src.preprocessing.health_index import HealthIndexModel, calibrate_hi_thresholds
from src.preprocessing.labeling import LabelGenerator
from src.preprocessing.loader import CMAPSSLoader
from src.preprocessing.scaling import FeatureScaler
from src.preprocessing.validator import DatasetValidator
from src.preprocessing.visualization import DatasetVisualizer
from src.preprocessing.windowing import WindowGenerator
from src.utils.config import ALL_SUBSETS, load_config
from src.utils.logger import get_script_logger

DEFAULT_SUBSET = "FD001"

# ------------------------------------------------------------------ #
# Paths (anchored to the project root, not the current directory, so
# the script behaves the same however it is invoked)
# ------------------------------------------------------------------ #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a NASA C-MAPSS dataset subset."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help="C-MAPSS subset to preprocess, or 'all'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Explicit YAML config path (default: configs/<subset>.yaml). "
            "Supplies window_size and max_rul."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help=(
            "Sliding window length in cycles. "
            "Defaults to the value in the subset's YAML config."
        ),
    )
    parser.add_argument(
        "--max-rul",
        type=int,
        default=None,
        help=(
            "RUL cap used for the piecewise-linear RUL label.  Also applied "
            "to the test labels so the train/test targets stay consistent. "
            "Defaults to the value in the subset's YAML config."
        ),
    )
    parser.add_argument(
        "--hi-mode",
        type=str,
        default=None,
        choices=["sensor", "rul"],
        help=(
            "Health-index construction.  'sensor' (default) builds the "
            "sensor-derived index from the sensor signal; 'rul' reproduces "
            "the legacy HI = RUL / max_rul label.  Defaults to the value in "
            "the subset's YAML config."
        ),
    )
    parser.add_argument(
        "--hi-baseline-cycles",
        type=int,
        default=None,
        help=(
            "Leading cycles per engine used as that engine's healthy "
            "reference when building the sensor-derived health index. "
            "Defaults to the value in the subset's YAML config."
        ),
    )
    return parser.parse_args()


def setup_logging(subset: str):
    return get_script_logger("preprocessing", f"preprocess_{subset}")


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

def run(
    subset: str,
    window_size: int | None = None,
    max_rul: int | None = None,
    config_path: str | None = None,
    hi_mode: str | None = None,
    hi_baseline_cycles: int | None = None,
) -> None:
    """Run the full preprocessing pipeline for a single subset.

    ``window_size`` and ``max_rul`` fall back to the subset's YAML config
    (``configs/<subset>.yaml``) when they are not supplied explicitly.  This
    keeps preprocessing and training on the same label cap — ``train.py``
    refuses to run when the stored ``max_rul`` differs from its config.

    ``hi_mode`` selects the health-index construction:

    ``sensor`` (default)
        The sensor-derived index of
        :class:`src.preprocessing.health_index.HealthIndexModel`.  It is
        built from the sensor signal, so it is strongly — but not perfectly —
        related to the degradation state, and it is a genuinely different
        learning problem from the RUL head.
    ``rul``
        The legacy ``HI = RUL / max_rul`` label, kept for comparison.
    """

    config = load_config(path=config_path, subset=subset)

    if window_size is None:
        window_size = int(config["window_size"])
    if max_rul is None:
        max_rul = int(config["max_rul"])
    if hi_mode is None:
        hi_mode = str(config.get("hi_mode", "sensor"))
    if hi_baseline_cycles is None:
        hi_baseline_cycles = int(config.get("hi_baseline_cycles", 30))

    if hi_mode not in {"sensor", "rul"}:
        raise ValueError(
            f"Unknown hi_mode {hi_mode!r}; expected 'sensor' or 'rul'."
        )

    log = setup_logging(subset)

    def _step(name: str) -> None:
        log.info("=" * 50)
        log.info("  %s", name)
        log.info("=" * 50)

    figures_dir = FIGURES_DIR / subset
    figures_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = REPORTS_DIR / subset
    reports_dir.mkdir(parents=True, exist_ok=True)

    # subset-scoped scaler path — prevents subsets overwriting each other
    scaler_path = MODELS_DIR / f"scaler_{subset}.pkl"

    log.info("Subset: %s  |  window_size: %d  |  max_rul: %d  |  hi_mode: %s",
             subset, window_size, max_rul, hi_mode)

    # ── Load Dataset ──────────────────────────────────────────────
    _step("Load Dataset")
    loader = CMAPSSLoader()
    train_df, test_df, rul_df = loader.load_dataset(subset)
    log.info("Train: %s  |  Test: %s  |  RUL: %s",
             train_df.shape, test_df.shape, rul_df.shape)

    # ── Validate Dataset ──────────────────────────────────────────
    _step("Validate Dataset")
    validator = DatasetValidator(reports_dir)
    validator.validate(train_df, f"{subset} Train")
    validator.validate(test_df, f"{subset} Test")

    # ── Dataset Summary ───────────────────────────────────────────
    _step("Dataset Summary")
    summary = validator.summarize(train_df, f"{subset} Train")
    validator.save_report(summary, f"{subset.lower()}_train_summary.csv")
    log.info("\n%s", summary.T.to_string(header=False))

    # ── Visualizations ────────────────────────────────────────────
    _step("Visualizations")
    viz = DatasetVisualizer(figures_dir)
    viz.plot_engine_lifetime(train_df)
    viz.plot_sensor_trend(train_df, sensor="sensor_2", engines=[1, 2, 3])
    viz.plot_sensor_trends_grid(train_df)
    viz.plot_correlation(train_df)
    viz.sensor_variance(train_df)
    viz.plot_sensor_variance(train_df)
    viz.plot_sensor_distributions_grid(train_df)
    viz.plot_operating_conditions(train_df)
    viz.plot_sensor_distribution(train_df, sensor="sensor_2")

    # ── Generate Labels ───────────────────────────────────────────
    _step("Generate Labels")
    label_gen = LabelGenerator(max_rul=max_rul)
    train_df = label_gen.generate_rul(train_df)
    viz.plot_label_distributions(train_df)
    log.info(
        "\n%s",
        train_df[["unit_number", "time_in_cycles", "RUL"]].head(10),
    )

    # Assign RUL to the test set using the ground-truth RUL file.
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    test_df["RUL"] = (
        max_cycles - test_df["time_in_cycles"]
        + rul_df["RUL"].reindex(test_df["unit_number"] - 1).values
    ).clip(upper=max_rul)

    # The raw frames are kept: the health-index construction needs the
    # operating settings, which the variance-based feature selector drops on
    # the single-condition subsets.
    train_raw = train_df.copy()
    test_raw = test_df.copy()

    # ── Feature Selection ─────────────────────────────────────────
    _step("Feature Selection")
    selector = FeatureSelector(method="variance", threshold=1e-4)
    train_df = selector.fit_transform(train_df)
    test_df = selector.transform(test_df)
    selector.save(reports_dir / "selected_features.csv")
    feature_columns = list(selector.selected_features)
    log.info("Retained %d features.", len(feature_columns))

    # ── Feature Scaling ───────────────────────────────────────────
    _step("Feature Scaling")
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    scaler.save(scaler_path)
    log.info("Scaler saved to: %s", scaler_path)

    # ── Health Index ──────────────────────────────────────────────
    _step("Health Index")
    sensor_columns = [
        column for column in train_df.columns if column.startswith("sensor_")
    ]
    if not sensor_columns:
        raise ValueError(
            "No sensor columns survived feature selection; the health index "
            "cannot be built."
        )

    if hi_mode == "sensor":
        hi_model = HealthIndexModel(
            baseline_cycles=hi_baseline_cycles,
            window_size=window_size,
        )
        hi_model.fit(
            raw_df=train_raw,
            scaled_df=train_df,
            sensor_columns=sensor_columns,
            target_rul=train_df["RUL"],
        )
        train_df["HI"] = hi_model.transform(
            raw_df=train_raw, scaled_df=train_df, sensor_columns=sensor_columns
        )
        test_df["HI"] = hi_model.transform(
            raw_df=test_raw, scaled_df=test_df, sensor_columns=sensor_columns
        )
        hi_path = MODELS_DIR / f"health_index_{subset}.pkl"
        hi_model.save(hi_path)
        log.info("Health-index model saved to: %s", hi_path)
        hi_metadata = hi_model.summary()
    else:
        # Legacy label: HI = RUL / max(RUL per engine).
        train_df = label_gen.generate_health_index(train_df)
        test_df["HI"] = (
            test_df.groupby("unit_number")["RUL"]
            .transform(lambda x: x / x.max())
            .fillna(0)
        )
        hi_metadata = {"mode": "rul"}

    correlation = float(
        np.corrcoef(
            np.asarray(train_df["HI"], dtype=np.float64),
            np.asarray(train_df["RUL"], dtype=np.float64),
        )[0, 1]
    )
    log.info(
        "Health index built | mode=%s | corr(HI, RUL) on the training split = %.6f",
        hi_mode,
        correlation,
    )
    log.info(
        "HI distribution: min=%.4f  median=%.4f  max=%.4f",
        float(np.min(train_df["HI"])),
        float(np.median(train_df["HI"])),
        float(np.max(train_df["HI"])),
    )

    # The decision policy decides a health state from the RUL *or* the HI
    # criterion, so the HI breakpoints are calibrated on the training split
    # to fire at the same degradation levels as the RUL criterion.
    hi_thresholds = calibrate_hi_thresholds(
        hi=train_df["HI"].to_numpy(),
        rul=train_df["RUL"].to_numpy(),
        rul_critical=10.0,
        rul_at_risk=25.0,
        rul_degrading=50.0,
    )
    log.info(
        "Calibrated HI breakpoints: critical=%.4f at_risk=%.4f degrading=%.4f",
        hi_thresholds["hi_critical"],
        hi_thresholds["hi_at_risk"],
        hi_thresholds["hi_degrading"],
    )

    # ── Sliding Windows ───────────────────────────────────────────
    _step("Sliding Windows")
    win_gen = WindowGenerator(window_size=window_size, stride=1)

    X_train, y_rul_train, y_hi_train, train_engine_ids = win_gen.create_windows(train_df)
    log.info("X_train shape    : %s", X_train.shape)
    log.info("y_rul shape      : %s", y_rul_train.shape)
    log.info("y_hi shape       : %s", y_hi_train.shape)
    log.info("Training engines : %d", len(set(train_engine_ids)))

    X_test, y_rul_test, y_hi_test, test_engine_ids = win_gen.create_test_windows(test_df)
    log.info("X_test shape     : %s", X_test.shape)
    log.info("Test engines     : %d", len(set(test_engine_ids)))

    # ── Save Processed Dataset ────────────────────────────────────
    _step("Save Processed Dataset")

    extra_metadata = {
        "max_rul": max_rul,
        "subset": subset,
        "hi_mode": hi_mode,
        "hi_baseline_cycles": hi_baseline_cycles if hi_mode == "sensor" else None,
        "health_index": hi_metadata,
        "hi_thresholds": hi_thresholds,
    }

    win_gen.save_dataset(
        X_train, y_rul_train, y_hi_train,
        PROCESSED_DIR, f"{subset}_train", engine_ids=train_engine_ids,
        extra_metadata=extra_metadata,
    )
    win_gen.save_dataset(
        X_test, y_rul_test, y_hi_test,
        PROCESSED_DIR, f"{subset}_test", engine_ids=test_engine_ids,
        extra_metadata=extra_metadata,
    )

    # ── Save the processed test frame ──────────────────────────────
    #
    # The engine-level degradation plots of the XAI stage need *every*
    # window of every test engine, not only the final window per engine
    # that the evaluation set contains.  Saving the processed frame
    # (feature-selected, scaled and labelled) lets that stage build the
    # windows from the artifact instead of re-running the whole
    # preprocessing pipeline — which used to produce a *different* health
    # index than the one the model was trained on.
    # ───────────────────────────────────────────────────────────────

    frame_path = PROCESSED_DIR / f"{subset}_test_frame.npz"

    np.savez_compressed(
        frame_path,
        features=test_df[feature_columns].to_numpy(dtype=np.float32),
        y_rul=test_df["RUL"].to_numpy(dtype=np.float32),
        y_hi=test_df["HI"].to_numpy(dtype=np.float32),
        engine_ids=test_df["unit_number"].to_numpy(dtype=np.int64),
        time_in_cycles=test_df["time_in_cycles"].to_numpy(dtype=np.int64),
        feature_names=np.asarray(feature_columns, dtype="U"),
    )

    log.info("Processed test frame saved to: %s", frame_path)

    # ── Done ──────────────────────────────────────────────────────
    _step("Done")
    log.info("Preprocessing completed successfully.")
    log.info("Outputs saved to: %s", PROCESSED_DIR.resolve())


def main() -> None:
    args = parse_args()
    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    for subset in subsets:
        run(
            subset,
            window_size=args.window_size,
            max_rul=args.max_rul,
            config_path=args.config,
            hi_mode=args.hi_mode,
            hi_baseline_cycles=args.hi_baseline_cycles,
        )


if __name__ == "__main__":
    main()
