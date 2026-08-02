import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.loader import CMAPSSLoader
from src.preprocessing.validator import DatasetValidator
from src.preprocessing.visualization import DatasetVisualizer


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    subset = "FD001"

    # ------------------------------
    # Load Dataset
    # ------------------------------

    loader = CMAPSSLoader()

    train_df, test_df, rul_df = loader.load_dataset(
        subset
    )

    # ------------------------------
    # Validate Dataset
    # ------------------------------

    validator = DatasetValidator(
        Path("outputs/reports")
    )

    validator.validate(
        train_df,
        f"{subset} Train"
    )

    summary = validator.summarize(
        train_df,
        f"{subset} Train"
    )

    validator.save_report(
        summary,
        f"{subset.lower()}_train_summary.csv"
    )

    # ------------------------------
    # Visualization
    # ------------------------------

    visualizer = DatasetVisualizer(
        Path("outputs/figures")
    )

    visualizer.plot_engine_lifetime(train_df)

    visualizer.plot_sensor_trend(
        train_df,
        sensor="sensor_2",
        engines=[1, 2, 3],
    )

    visualizer.plot_correlation(train_df)

    visualizer.sensor_variance(train_df)

    visualizer.plot_sensor_distribution(
        train_df,
        sensor="sensor_2",
    )

    logging.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()