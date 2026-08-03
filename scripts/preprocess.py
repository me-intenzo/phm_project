"""
preprocess.py

Full preprocessing pipeline for NASA C-MAPSS.

Pipeline
--------
Load → Validate → Summary → Visualize → Generate RUL/HI
→ Feature Scaling → Feature Selection → Sliding Windows
→ Save → Logging → Done
"""

import logging
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

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "preprocess.log", mode="w"),
    ],
)

log = logging.getLogger(__name__)

SUBSET = "FD001"
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("outputs/reports")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")


def _step(name: str) -> None:
    log.info("=" * 50)
    log.info("  %s", name)
    log.info("=" * 50)


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

def main() -> None:

    # ── Load Dataset ──────────────────────────────────────────────
    _step("Load Dataset")
    loader = CMAPSSLoader()
    train_df, test_df, rul_df = loader.load_dataset(SUBSET)
    log.info("Train: %s  |  Test: %s  |  RUL: %s",
             train_df.shape, test_df.shape, rul_df.shape)

    # ── Validate Dataset ──────────────────────────────────────────
    _step("Validate Dataset")
    validator = DatasetValidator(REPORTS_DIR)
    validator.validate(train_df, f"{SUBSET} Train")
    validator.validate(test_df, f"{SUBSET} Test")

    # ── Dataset Summary ───────────────────────────────────────────
    _step("Dataset Summary")
    summary = validator.summarize(train_df, f"{SUBSET} Train")
    validator.save_report(summary, f"{SUBSET.lower()}_train_summary.csv")
    log.info("\n%s", summary.T.to_string(header=False))

    # ── Visualizations ────────────────────────────────────────────
    _step("Visualizations")
    viz = DatasetVisualizer(FIGURES_DIR)
    viz.plot_engine_lifetime(train_df)
    viz.plot_sensor_trend(train_df, sensor="sensor_2", engines=[1, 2, 3])
    viz.plot_correlation(train_df)
    viz.sensor_variance(train_df)
    viz.plot_sensor_distribution(train_df, sensor="sensor_2")

    # ── Generate Labels ───────────────────────────────────────────
    _step("Generate Labels")

    label_gen = LabelGenerator(max_rul=None)

    train_df = label_gen.generate_labels(train_df)

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
        REPORTS_DIR / "selected_features.csv"
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
    win_gen = WindowGenerator(window_size=30, stride=1)
    X_train, y_rul_train, y_hi_train = win_gen.create_windows(train_df)

    log.info("X_train shape : %s", X_train.shape)
    log.info("y_rul shape  : %s", y_rul_train.shape)
    log.info("y_hi shape   : %s", y_hi_train.shape)

    X_test, y_rul_test, y_hi_test = win_gen.create_windows(test_df)
    log.info("Test windows  : %s", X_test.shape)

    # ── Save Processed Dataset ────────────────────────────────────
    _step("Save Processed Dataset")
    win_gen.save_dataset(X_train, y_rul_train, y_hi_train,
                         PROCESSED_DIR, f"{SUBSET}_train")
    win_gen.save_dataset(X_test, y_rul_test, y_hi_test,
                         PROCESSED_DIR, f"{SUBSET}_test")

    # ── Done ──────────────────────────────────────────────────────
    _step("Done")
    log.info("Preprocessing completed successfully.")
    log.info("Outputs saved to: %s", PROCESSED_DIR.resolve())


if __name__ == "__main__":
    main()
