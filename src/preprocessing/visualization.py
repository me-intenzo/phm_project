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

    def plot_sensor_trends_grid(
        self,
        df: pd.DataFrame,
        engines: list[int] | None = None,
        filename: str = "sensor_trends_all.png",
    ) -> None:
        """Plot every sensor in one consistently labelled figure."""
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        if not sensor_cols:
            raise ValueError("Dataframe contains no sensor columns.")
        if engines is None:
            engines = sorted(df["unit_number"].unique())[:5]
        ncols = 3
        nrows = (len(sensor_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
        axes = axes.ravel()
        for ax, sensor in zip(axes, sensor_cols):
            for engine in engines:
                subset = df[df["unit_number"] == engine]
                if not subset.empty:
                    ax.plot(subset["time_in_cycles"], subset[sensor], lw=1.1, label=f"Engine {engine}")
            ax.set_title(f"{sensor.replace('_', ' ').title()} vs cycle")
            ax.set_xlabel("Operational cycle")
            ax.set_ylabel(sensor)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc="best")
        for ax in axes[len(sensor_cols):]:
            ax.set_visible(False)
        fig.suptitle("C-MAPSS Sensor Trajectories", fontsize=15)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

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
        )

        plt.title("Sensor Correlation")

        plt.tight_layout()

        plt.savefig(
            self.output_dir / filename
        )

        plt.close()

    def plot_operating_conditions(
        self,
        df: pd.DataFrame,
        filename: str = "operating_conditions.png",
    ) -> None:
        """Show the distribution of the three operating settings."""
        setting_cols = [c for c in df.columns if c.startswith("operational_setting_")]
        if not setting_cols:
            return
        fig, axes = plt.subplots(1, len(setting_cols), figsize=(15, 4))
        axes = [axes] if len(setting_cols) == 1 else axes.ravel()
        for ax, setting in zip(axes, setting_cols):
            sns.histplot(df[setting].dropna(), kde=True, ax=ax, color="#1565C0")
            ax.set_title(setting.replace("_", " ").title())
            ax.set_xlabel("Setting value")
            ax.set_ylabel("Observations")
        fig.suptitle("Operating Condition Distributions", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

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

    def plot_sensor_variance(
        self,
        df: pd.DataFrame,
        filename: str = "sensor_variance.png",
    ) -> None:
        """Rank sensor variance so informative and constant sensors are visible."""
        variance = df[[c for c in df.columns if c.startswith("sensor_")]].var().sort_values()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(variance.index, variance.to_numpy(), color="#00897B")
        ax.set_title("Sensor Variance Ranking")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Sensor")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_sensor_distributions_grid(
        self,
        df: pd.DataFrame,
        filename: str = "sensor_distributions_all.png",
    ) -> None:
        """Plot distributions for all sensors with one label per panel."""
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        if not sensor_cols:
            raise ValueError("Dataframe contains no sensor columns.")
        ncols = 3
        nrows = (len(sensor_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows))
        axes = axes.ravel()
        for ax, sensor in zip(axes, sensor_cols):
            sns.histplot(df[sensor].dropna(), kde=True, ax=ax, color="#EF6C00")
            ax.set_title(sensor.replace("_", " ").title())
            ax.set_xlabel(sensor)
            ax.set_ylabel("Observations")
        for ax in axes[len(sensor_cols):]:
            ax.set_visible(False)
        fig.suptitle("Sensor Value Distributions", fontsize=15)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_label_distributions(
        self,
        df: pd.DataFrame,
        filename: str = "label_distributions.png",
    ) -> None:
        """Plot generated RUL and HI labels when they are present."""
        labels = [c for c in ("RUL", "HI") if c in df.columns]
        if not labels:
            return
        fig, axes = plt.subplots(1, len(labels), figsize=(12, 4))
        axes = [axes] if len(labels) == 1 else axes.ravel()
        for ax, label in zip(axes, labels):
            sns.histplot(df[label].dropna(), kde=True, ax=ax, color="#6A1B9A")
            ax.set_title(f"{label} label distribution")
            ax.set_xlabel(label)
            ax.set_ylabel("Observations")
        fig.suptitle("Training Target Distributions", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

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