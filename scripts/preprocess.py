"""
preprocess.py

Full preprocessing pipeline for NASA C-MAPSS.

Pipeline
--------
Load → Validate → Summary → Visualize → Generate RUL/HI
→ Feature Scaling → Feature Selection → Sliding Windows
→ Save → Logging → Done
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_script_logger

from src.preprocessing.feature_selection import FeatureSelector
from src.preprocessing.labeling import LabelGenerator
from src.preprocessing.loader import CMAPSSLoader
from src.preprocessing.scaling import FeatureScaler
from src.preprocessing.validator import DatasetValidator
from src.preprocessing.visualization import DatasetVisualizer
from src.preprocessing.windowing import WindowGenerator

DEFAULT_SUBSET = "FD001"
ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("outputs/models")


def parse_args() -> argparse.Namespace:
    """
    Parse preprocessing configuration from the command line.
    """

    parser = argparse.ArgumentParser(
        description="Preprocess a NASA C-MAPSS dataset subset."
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help="C-MAPSS subset to preprocess, or 'all'.",
    )

    return parser.parse_args()

def setup_logging(subset: str):
    return get_script_logger("preprocessing", f"preprocess_{subset}")

# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

def run(subset: str) -> None:
    """Run the full preprocessing pipeline for a single subset."""
    log = setup_logging(subset)
    _step = lambda name: (log.info("=" * 50), log.info("  %s", name), log.info("=" * 50))
    _step = lambda name: (log.info("=" * 50), log.info("  %s", name), log.info("=" * 50))

    figures_dir = Path("outputs") / "figures" / subset
    figures_dir.mkdir(parents=True, exist_ok=True)
    viz = DatasetVisualizer(figures_dir)

    reports_dir = Path("outputs") / "reports" / subset
    reports_dir.mkdir(parents=True, exist_ok=True)

    log.info("Selected subset for C-MAPSS subset: %s", subset)

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

    label_gen = LabelGenerator(max_rul=125)

    train_df = label_gen.generate_labels(train_df)

    viz.plot_label_distributions(train_df)

    log.info(
        "\n%s",
        train_df[
            [
                "unit_number",
                "time_in_cycles",
                "RUL",
                "HI",
            ]
        ].head(10),
    )

    # Assign RUL to test set using ground-truth RUL file
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    test_df["RUL"] = (
        max_cycles - test_df["time_in_cycles"]
        + rul_df["RUL"].reindex(
            test_df["unit_number"] - 1
        ).values
    ).clip(upper=125)
    test_df["HI"] = (
        test_df.groupby("unit_number")["RUL"]
        .transform(lambda x: x / x.max())
        .fillna(0)
    )

    # ── Feature Selection ─────────────────────────────────────────
    _step("Feature Selection")

    selector = FeatureSelector(
        method="variance",
        threshold=1e-4,
    )

    train_df = selector.fit_transform(train_df)
    test_df = selector.transform(test_df)

    selector.save(
        reports_dir / "selected_features.csv"
    )

    log.info(
        "Retained %d features.",
        len(selector.selected_features),
    )

    # ── Feature Scaling ───────────────────────────────────────────
    _step("Feature Scaling")

    scaler = FeatureScaler()

    train_df = scaler.fit_transform(train_df)

    test_df = scaler.transform(test_df)

    scaler.save(
        MODELS_DIR / "scaler.pkl"
    )

    # ── Sliding Windows ───────────────────────────────────────────
    _step("Sliding Windows")

    win_gen = WindowGenerator(
        window_size=40,
        stride=1,
    )

    (
        X_train,
        y_rul_train,
        y_hi_train,
        train_engine_ids,
    ) = win_gen.create_windows(
        train_df
    )

    log.info(
        "X_train shape      : %s",
        X_train.shape,
    )

    log.info(
        "y_rul shape        : %s",
        y_rul_train.shape,
    )

    log.info(
        "y_hi shape         : %s",
        y_hi_train.shape,
    )

    log.info(
        "Engine IDs shape   : %s",
        train_engine_ids.shape,
    )

    log.info(
        "Training engines   : %d",
        len(
            set(
                train_engine_ids
            )
        ),
    )

    # ── Test Windows ──────────────────────────────────────────────

    (
        X_test,
        y_rul_test,
        y_hi_test,
        test_engine_ids,
    ) = win_gen.create_test_windows(
        test_df
    )

    log.info(
        "X_test shape       : %s",
        X_test.shape,
    )

    log.info(
        "y_rul_test shape   : %s",
        y_rul_test.shape,
    )

    log.info(
        "y_hi_test shape    : %s",
        y_hi_test.shape,
    )

    log.info(
        "Test engines       : %d",
        len(
            set(
                test_engine_ids
            )
        ),
    )

    # ── Save Processed Dataset ────────────────────────────────────
    _step("Save Processed Dataset")

    # Training dataset
    win_gen.save_dataset(
        X_train,
        y_rul_train,
        y_hi_train,
        PROCESSED_DIR,
        f"{subset}_train",
        engine_ids=train_engine_ids,
    )

    # Test dataset
    win_gen.save_dataset(
        X_test,
        y_rul_test,
        y_hi_test,
        PROCESSED_DIR,
        f"{subset}_test",
        engine_ids=test_engine_ids,
    )

    # ── Done ──────────────────────────────────────────────────────
    _step("Done")
    log.info("Preprocessing completed successfully.")
    log.info("Outputs saved to: %s", PROCESSED_DIR.resolve())


def main() -> None:
    args = parse_args()
    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
    for subset in subsets:
        run(subset)


if __name__ == "__main__":
    main()
