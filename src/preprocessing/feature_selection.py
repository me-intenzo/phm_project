"""
feature_selection.py

Feature selection utilities for the NASA C-MAPSS dataset.

Responsibilities
----------------
1. Identify informative features.
2. Remove low-variance features.
3. Save selected feature list.
4. Apply feature selection consistently.

Author: me-intenzo
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection using configurable strategies.

    Currently Supported
    -------------------
    - variance
    """

    def __init__(
        self,
        method: str = "variance",
        threshold: float = 1e-4,
    ):

        self.method = method
        self.threshold = threshold

        self.selected_features: list[str] = []

    # -------------------------------------------------- #

    @staticmethod
    def get_feature_columns(
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Return operational settings and sensor columns.
        """

        return [

            col

            for col in df.columns

            if col.startswith("operational_setting")
            or col.startswith("sensor_")
        ]

    # -------------------------------------------------- #

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> None:
        """
        Learn which features to keep.
        """

        logger.info(
            "Running feature selection (%s)...",
            self.method,
        )

        features = self.get_feature_columns(train_df)

        if self.method == "variance":

            selector = VarianceThreshold(
                threshold=self.threshold
            )

            selector.fit(train_df[features])

            self.selected_features = [

                feature

                for feature, keep in zip(
                    features,
                    selector.get_support(),
                )

                if keep

            ]

        else:

            raise ValueError(
                f"Unknown method: {self.method}"
            )

        logger.info(
            "Selected %d/%d features.",
            len(self.selected_features),
            len(features),
        )

    # -------------------------------------------------- #

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Keep only selected features while
        preserving identifiers and labels.
        """

        if not self.selected_features:

            raise RuntimeError(
                "FeatureSelector has not been fitted."
            )

        required_columns = [

            "unit_number",
            "time_in_cycles",

        ]

        optional_columns = [

            col

            for col in ["RUL", "HI"]

            if col in df.columns

        ]

        keep_columns = (

            required_columns
            + self.selected_features
            + optional_columns

        )

        return df[keep_columns].copy()

    # -------------------------------------------------- #

    def fit_transform(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(train_df)

        return self.transform(train_df)

    # -------------------------------------------------- #

    def save(
        self,
        output_path: Path,
    ) -> None:
        """
        Save selected features.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.Series(
            self.selected_features,
            name="feature",
        ).to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Selected features saved to %s",
            output_path,
        )