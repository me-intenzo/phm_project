"""
visualization.py

Visualization utilities for NASA C-MAPSS dataset.

Responsibilities
----------------
1. Engine lifetime visualization.
2. Sensor trend visualization.
3. Correlation heatmap.
4. Sensor variance analysis.
5. Sensor distribution plots.

Author: me-intenzo
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class DatasetVisualizer:

    def __init__(self, output_dir: Path):

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- #
    # Engine Lifetime
    # ---------------------------------------------------- #

    def plot_engine_lifetime(
        self,
        df: pd.DataFrame,
        filename: str = "engine_lifetime.png",
    ) -> None:

        logger.info("Generating engine lifetime plot...")

        lifetime = (
            df.groupby("unit_number")["time_in_cycles"]
            .max()
        )

        plt.figure(figsize=(12, 6))

        plt.bar(
            lifetime.index,
            lifetime.to_numpy(),
        )

        plt.title("Engine Lifetime Distribution")

        plt.xlabel("Engine Number")

        plt.ylabel("Maximum Cycle")

        plt.tight_layout()

        plt.savefig(self.output_dir / filename)

        plt.close()

    # ---------------------------------------------------- #
    # Sensor Trend
    # ---------------------------------------------------- #

    def plot_sensor_trend(
        self,
        df: pd.DataFrame,
        sensor: str,
        engines: list[int] | None = None,
    ) -> None:

        if engines is None:
            engines = sorted(df["unit_number"].unique())[:5]

        logger.info("Plotting %s...", sensor)

        plt.figure(figsize=(12, 6))

        for engine in engines:

            subset = df[
                df["unit_number"] == engine
            ]

            plt.plot(
                subset["time_in_cycles"],
                subset[sensor],
                label=f"Engine {engine}",
            )

        plt.title(f"{sensor} Trend")

        plt.xlabel("Cycle")

        plt.ylabel(sensor)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            f"{sensor}_trend.png"
        )

        plt.close()

    # ---------------------------------------------------- #
    # Correlation
    # ---------------------------------------------------- #

    def plot_correlation(
        self,
        df: pd.DataFrame,
        filename: str = "sensor_correlation.png",
    ) -> None:

        logger.info("Generating correlation heatmap...")

        sensor_cols = [
            c for c in df.columns
            if c.startswith("sensor_")
        ]

        corr = df[sensor_cols].corr()

        plt.figure(figsize=(14, 10))

        sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
)

        plt.title("Sensor Correlation")

        plt.tight_layout()

        plt.savefig(
            self.output_dir / filename
        )

        plt.close()


    # ---------------------------------------------------- #
    # Sensor vs RUL Correlation
    # ---------------------------------------------------- #

    def plot_sensor_rul_correlation(
        self,
        df: pd.DataFrame,
        filename: str = "sensor_rul_correlation.png",
    ) -> None:

        logger.info(
            "Generating sensor vs RUL correlation heatmap..."
        )

        sensor_cols = [
            c for c in df.columns
            if c.startswith("sensor_")
        ]

        # Pearson correlation between every sensor
        # and the RUL target.
        correlations = (
            df[sensor_cols + ["RUL"]]
            .corr()["RUL"]
            .drop("RUL")
            .to_frame()
        )

        plt.figure(figsize=(6, 10))

        sns.heatmap(
            correlations,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
        )

        plt.title(
            "Pearson Correlation Between "
            "Sensor Readings and RUL"
        )

        plt.xlabel("Target")
        plt.ylabel("Sensor")

        plt.tight_layout()

        plt.savefig(
            self.output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------------- #
    # Sensor vs HI Correlation
    # ---------------------------------------------------- #

    def plot_sensor_hi_correlation(
        self,
        df: pd.DataFrame,
        filename: str = "sensor_hi_correlation.png",
    ) -> None:

        logger.info(
            "Generating sensor vs HI correlation heatmap..."
        )

        sensor_cols = [
            c for c in df.columns
            if c.startswith("sensor_")
        ]

        correlations = (
            df[sensor_cols + ["HI"]]
            .corr()["HI"]
            .drop("HI")
            .to_frame()
        )

        plt.figure(figsize=(6, 10))

        sns.heatmap(
            correlations,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
        )

        plt.title(
            "Pearson Correlation Between "
            "Sensor Readings and HI"
        )

        plt.xlabel("Target")
        plt.ylabel("Sensor")

        plt.tight_layout()

        plt.savefig(
            self.output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
    # ---------------------------------------------------- #
    # Variance
    # ---------------------------------------------------- #

    def sensor_variance(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("Computing sensor variance...")

        sensor_cols = [
            c for c in df.columns
            if c.startswith("sensor_")
        ]

        variance = (
            df[sensor_cols]
            .var()
            .sort_values(ascending=False)
        )

        variance_df = variance.reset_index()

        variance_df.columns = [
            "Sensor",
            "Variance",
        ]

        variance_df.to_csv(
            self.output_dir /
            "sensor_variance.csv",
            index=False,
        )

        return variance_df

    # ---------------------------------------------------- #
    # Histogram
    # ---------------------------------------------------- #

    def plot_sensor_distribution(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> None:

        plt.figure(figsize=(8,5))

        sns.histplot(
            data=df,
            x=sensor,
            kde=True,
        )

        plt.title(
            f"{sensor} Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            f"{sensor}_distribution.png"
        )

        plt.close()