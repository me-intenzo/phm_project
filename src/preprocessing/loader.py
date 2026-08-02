"""
loader.py

Data loading utilities for the NASA C-MAPSS dataset.

Responsibilities
----------------
1. Load training, testing and RUL files.
2. Validate dataset files.
3. Assign proper column names.
4. Return clean pandas DataFrames.

Author: me-intenzo
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# -----------------------------------------------------
# Dataset Columns
# -----------------------------------------------------

COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


class CMAPSSLoader:
    """
    Loader for NASA C-MAPSS datasets.

    Supports:
        FD001
        FD002
        FD003
        FD004
    """

    def __init__(self, data_dir: Path | None = None):

        self.data_dir = data_dir or DATA_DIR

    # -------------------------------------------------

    def _check_exists(self, filepath: Path) -> None:
        """
        Ensure dataset file exists.
        """

        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found:\n{filepath}\n\n"
                "Download the NASA C-MAPSS dataset and place "
                "the files inside:\n"
                "data/raw/"
            )

    # -------------------------------------------------

    def _load_txt(self, filename: str) -> pd.DataFrame:
        """
        Load a CMAPSS txt file.
        """

        filepath = self.data_dir / filename

        self._check_exists(filepath)

        logger.info("Loading %s", filename)

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            engine="python",
        )

        # Remove trailing empty columns
        df = df.dropna(axis=1, how="all")

        # Train/Test files have 26 columns
        if df.shape[1] == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES

        return df

    # -------------------------------------------------

    def load_train(
        self,
        subset: str = "FD001",
    ) -> pd.DataFrame:
        """
        Load training dataset.
        """

        return self._load_txt(
            f"train_{subset}.txt"
        )

    # -------------------------------------------------

    def load_test(
        self,
        subset: str = "FD001",
    ) -> pd.DataFrame:
        """
        Load testing dataset.
        """

        return self._load_txt(
            f"test_{subset}.txt"
        )

    # -------------------------------------------------

    def load_rul(
        self,
        subset: str = "FD001",
    ) -> pd.DataFrame:
        """
        Load RUL labels.
        """

        filename = f"RUL_{subset}.txt"

        filepath = self.data_dir / filename

        self._check_exists(filepath)

        logger.info("Loading %s", filename)

        return pd.read_csv(
            filepath,
            header=None,
            names=["RUL"],
        )

    # -------------------------------------------------

    def load_dataset(
        self,
        subset: str = "FD001",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset.

        Parameters
        ----------
        subset : str
            FD001 / FD002 / FD003 / FD004

        Returns
        -------
        train_df
        test_df
        rul_df
        """

        subset = subset.upper()

        if subset not in {
            "FD001",
            "FD002",
            "FD003",
            "FD004",
        }:
            raise ValueError(
                f"Unknown subset '{subset}'."
            )

        train_df = self.load_train(subset)
        test_df = self.load_test(subset)
        rul_df = self.load_rul(subset)

        logger.info(
            "%s loaded successfully.",
            subset,
        )

        return train_df, test_df, rul_df


# -----------------------------------------------------
# Example
# -----------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    loader = CMAPSSLoader()

    train_df, test_df, rul_df = loader.load_dataset(
        "FD001"
    )

    print("\nTraining :", train_df.shape)
    print("Testing  :", test_df.shape)
    print("RUL      :", rul_df.shape)

    print("\nColumns")
    print(train_df.columns.tolist())