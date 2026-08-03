"""
labeling.py

Generate Remaining Useful Life (RUL) and Health Index (HI)
labels for NASA C-MAPSS datasets.

Responsibilities
----------------
1. Generate RUL labels.
2. Generate Health Index labels.
3. Return labeled dataframe.

Author: me-intenzo
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class LabelGenerator:
    def __init__(self, max_rul: int | None = 125):
        """
        Parameters
        ----------
        max_rul : int | None
            Maximum RUL cap.
            If None, no capping is applied.
        """
        self.max_rul = max_rul

    # -------------------------------------------------- #
    # Remaining Useful Life
    # -------------------------------------------------- #

    def generate_rul(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate Remaining Useful Life (RUL).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        logger.info("Generating Remaining Useful Life...")

        data = df.copy()

        max_cycles = (
            data.groupby("unit_number")["time_in_cycles"]
            .transform("max")
        )

        data["RUL"] = (
            max_cycles
            - data["time_in_cycles"]
        )

        if self.max_rul is not None:
            data["RUL"] = data["RUL"].clip(upper=self.max_rul)

        logger.info("RUL generation completed.")

        return data

    # -------------------------------------------------- #
    # Health Index
    # -------------------------------------------------- #

    def generate_health_index(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate normalized Health Index.

        HI = RUL / max(RUL per engine)
        """

        logger.info("Generating Health Index...")

        data = df.copy()

        max_rul = (
            data.groupby("unit_number")["RUL"]
            .transform("max")
        )

        data["HI"] = (
            data["RUL"] / max_rul
        ).fillna(0)

        logger.info("Health Index generation completed.")

        return data

    # -------------------------------------------------- #
    # Combined Pipeline
    # -------------------------------------------------- #

    def generate_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate all labels.

        Returns
        -------
        DataFrame
        """

        data = self.generate_rul(df)

        data = self.generate_health_index(data)

        return data