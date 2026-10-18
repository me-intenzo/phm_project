"""
main.py

Exploratory Data Analysis entry point for NASA C-MAPSS FD001.

Runs dataset validation, summary statistics, and sensor visualisations.
All outputs are written to outputs/reports/EDA/ and outputs/figures/EDA/.

Usage
-----
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.loader import CMAPSSLoader
from src.preprocessing.validator import DatasetValidator
from src.preprocessing.visualization import DatasetVisualizer
from src.utils.logger import get_script_logger

REPORT_DIR  = Path("outputs/reports/EDA")
FIGURES_DIR = Path("outputs/figures/EDA")
SUBSET      = "FD001"


def run_eda() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log = get_script_logger("eda", f"eda_{SUBSET}")
    log.info("Starting EDA for %s...", SUBSET)

    train, _, _ = CMAPSSLoader().load_dataset(SUBSET)

    validator = DatasetValidator(output_dir=REPORT_DIR)
    validator.validate(train, "Training")

    summary = validator.summarize(train, "Training")
    validator.save_report(summary, "training_summary.csv")
    log.info("\n%s", summary.to_string(index=False))

    viz = DatasetVisualizer(output_dir=FIGURES_DIR)
    viz.plot_engine_lifetime(train)
    viz.plot_sensor_trend(train, sensor="sensor_2")
    viz.plot_sensor_trend(train, sensor="sensor_3")
    viz.plot_sensor_trend(train, sensor="sensor_4")
    viz.plot_sensor_trend(train, sensor="sensor_11")
    viz.plot_sensor_trend(train, sensor="sensor_15")
    viz.plot_correlation(train)
    viz.sensor_variance(train)

    log.info(
        "EDA complete.  Figures -> %s  |  Reports -> %s",
        FIGURES_DIR.resolve(),
        REPORT_DIR.resolve(),
    )


if __name__ == "__main__":
    run_eda()
