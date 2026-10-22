"""
scaling.py

Feature scaling utilities for NASA C-MAPSS.

Responsibilities
----------------
1. Fit scaler on training data.
2. Transform train/test datasets.
3. Save and load fitted scaler.

Author: me-intenzo
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.loader import get_feature_columns

logger = logging.getLogger(__name__)


class FeatureScaler:

    def __init__(self):

        self.scaler = StandardScaler()

    # ------------------------------------------------ #

    @staticmethod
    def get_feature_columns(
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Return operational settings and sensor columns.

        See ``src.preprocessing.loader.get_feature_columns`` — the single
        definition of the feature selection.
        """

        return get_feature_columns(df)

    # ------------------------------------------------ #

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> None:

        logger.info("Fitting StandardScaler...")

        features = self.get_feature_columns(train_df)

        self.scaler.fit(
            train_df[features]
        )

    # ------------------------------------------------ #

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("Scaling dataset...")

        data = df.copy()

        features = self.get_feature_columns(data)

        data[features] = self.scaler.transform(
            data[features]
        )

        return data

    # ------------------------------------------------ #

    def fit_transform(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(train_df)

        return self.transform(train_df)

    # ------------------------------------------------ #

    def save(
        self,
        path: Path,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.scaler,
            path,
        )

        logger.info(
            "Scaler saved to %s",
            path,
        )

    # ------------------------------------------------ #

    @staticmethod
    def load(
        path: Path,
    ) -> StandardScaler:

        logger.info(
            "Loading scaler..."
        )

        return joblib.load(path)
