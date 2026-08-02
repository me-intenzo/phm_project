"""
validator.py

Dataset validation utilities for the NASA C-MAPSS dataset.

Responsibilities
----------------
1. Validate dataset integrity.
2. Generate summary statistics.
3. Save validation reports.

Author: me-intenzo
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Performs integrity checks and generates summary
    statistics for a C-MAPSS dataset.
    """

    EXPECTED_COLUMNS = 26

    def __init__(self, output_dir: Path):

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ #
    # Validation
    # ------------------------------------------------ #

    def validate(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Validate dataset integrity.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.

        dataset_name : str
            Name of dataset.
        """

        logger.info("Validating %s...", dataset_name)

        self._check_columns(df)
        self._check_missing(df)
        self._check_duplicates(df)

        logger.info("%s validation completed.", dataset_name)

    # ------------------------------------------------ #

    def _check_columns(self, df: pd.DataFrame) -> None:

        if df.shape[1] != self.EXPECTED_COLUMNS:
            raise ValueError(
                f"Expected {self.EXPECTED_COLUMNS} columns "
                f"but found {df.shape[1]}."
            )

    # ------------------------------------------------ #

    def _check_missing(self, df: pd.DataFrame) -> None:

        missing = df.isnull().sum().sum()

        if missing != 0:
            raise ValueError(
                f"Dataset contains {missing} missing values."
            )

    # ------------------------------------------------ #

    def _check_duplicates(self, df: pd.DataFrame) -> None:

        duplicates = df.duplicated().sum()

        if duplicates > 0:
            logger.warning(
                "Dataset contains %d duplicate rows.",
                duplicates,
            )

    # ------------------------------------------------ #
    # Summary
    # ------------------------------------------------ #

    def summarize(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Generate dataset summary.
        """

        logger.info("Generating dataset summary...")

        summary = {
            "Dataset": dataset_name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Missing Values": df.isnull().sum().sum(),
            "Duplicate Rows": df.duplicated().sum(),
            "Number of Engines": df["unit_number"].nunique(),
            "Min Cycle": df["time_in_cycles"].min(),
            "Max Cycle": df["time_in_cycles"].max(),
        }

        summary_df = pd.DataFrame([summary])

        return summary_df

    # ------------------------------------------------ #
    # Save
    # ------------------------------------------------ #

    def save_report(
        self,
        summary_df: pd.DataFrame,
        filename: str,
    ) -> Path:
        """
        Save validation report.
        """

        output_path = self.output_dir / filename

        summary_df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Validation report saved to %s",
            output_path,
        )

        return output_path