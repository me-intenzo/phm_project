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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.feature_selection import FeatureSelector
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
) -> None:
    """Run the full preprocessing pipeline for a single subset.

    ``window_size`` and ``max_rul`` fall back to the subset's YAML config
    (``configs/<subset>.yaml``) when they are not supplied explicitly.  This
    keeps preprocessing and training on the same label cap — ``train.py``
    refuses to run when the stored ``max_rul`` differs from its config.
    """

    config = load_config(path=config_path, subset=subset)

    if window_size is None:
        window_size = int(config["window_size"])
    if max_rul is None:
        max_rul = int(config["max_rul"])

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

    log.info("Subset: %s  |  window_size: %d  |  max_rul: %d", subset, window_size, max_rul)

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
    train_df = label_gen.generate_labels(train_df)
    viz.plot_label_distributions(train_df)
    log.info(
        "\n%s",
        train_df[["unit_number", "time_in_cycles", "RUL", "HI"]].head(10),
    )

    # Assign RUL/HI to test set using ground-truth RUL file
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    test_df["RUL"] = (
        max_cycles - test_df["time_in_cycles"]
        + rul_df["RUL"].reindex(test_df["unit_number"] - 1).values
    ).clip(upper=max_rul)
    test_df["HI"] = (
        test_df.groupby("unit_number")["RUL"]
        .transform(lambda x: x / x.max())
        .fillna(0)
    )

    # ── Feature Selection ─────────────────────────────────────────
    _step("Feature Selection")
    selector = FeatureSelector(method="variance", threshold=1e-4)
    train_df = selector.fit_transform(train_df)
    test_df = selector.transform(test_df)
    selector.save(reports_dir / "selected_features.csv")
    log.info("Retained %d features.", len(selector.selected_features))

    # ── Feature Scaling ───────────────────────────────────────────
    _step("Feature Scaling")
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    scaler.save(scaler_path)
    log.info("Scaler saved to: %s", scaler_path)

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
        )


if __name__ == "__main__":
    main()
