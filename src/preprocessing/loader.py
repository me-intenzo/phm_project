"""
loader.py

Data loading utilities for the NASA C-MAPSS dataset.

Responsibilities
----------------
1. Load training, testing and RUL files.
2. Validate dataset files.
3. Assign proper column names.
4. Return clean pandas DataFrames.
5. Name the feature columns (operational settings + sensors).

Author: me-intenzo
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "raw"

# -----------------------------------------------------
# Dataset Columns
# -----------------------------------------------------

COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

#: Prefixes of the columns that are used as model features.
FEATURE_PREFIXES: tuple[str, ...] = ("operational_setting", "sensor_")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature columns of *df*.

    The feature columns are the operational settings and the sensor
    readings; the identifiers (``unit_number``, ``time_in_cycles``) and
    the labels (``RUL``, ``HI``) are not features.

    This function is the **single definition** of that selection.  The
    window generator, the feature selector and the feature scaler each
    used to carry their own copy of the same list comprehension.
    """

    return [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)
    ]


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
