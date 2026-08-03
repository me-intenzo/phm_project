"""
windowing.py

Sliding window generation for NASA C-MAPSS dataset.

Responsibilities
----------------
1. Create fixed-length sequences.
2. Generate RUL and HI targets.
3. Save processed datasets.

Author: me-intenzo
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WindowGenerator:

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 1,
    ):

        self.window_size = window_size
        self.stride = stride

    # ------------------------------------------------ #

    @staticmethod
    def get_feature_columns(
        df: pd.DataFrame,
    ) -> list[str]:

        return [

            col

            for col in df.columns

            if col.startswith("operational_setting")
            or col.startswith("sensor_")

        ]

    # ------------------------------------------------ #

    def create_windows(
        self,
        df: pd.DataFrame,
    ):

        logger.info("Generating sliding windows...")

        feature_columns = self.get_feature_columns(df)

        X = []
        y_rul = []
        y_hi = []

        engine_ids = df["unit_number"].unique()

        for engine in engine_ids:

            engine_df = (

                df[df["unit_number"] == engine]

                .sort_values("time_in_cycles")

            )

            features = engine_df[
                feature_columns
            ].values

            rul = engine_df["RUL"].values
            hi = engine_df["HI"].values

            for i in range(
                0,
                len(engine_df) - self.window_size + 1,
                self.stride,
            ):

                end = i + self.window_size

                X.append(
                    features[i:end]
                )

                y_rul.append(
                    rul[end - 1]
                )

                y_hi.append(
                    hi[end - 1]
                )

        X = np.asarray(X, dtype=np.float32)

        y_rul = np.asarray(
            y_rul,
            dtype=np.float32,
        )

        y_hi = np.asarray(
            y_hi,
            dtype=np.float32,
        )

        logger.info(
            "Generated %d windows.",
            len(X),
        )

        return X, y_rul, y_hi

    # ------------------------------------------------ #

    def save_dataset(
        self,
        X,
        y_rul,
        y_hi,
        output_dir: Path,
        prefix: str,
    ):

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_dir / f"{prefix}_X.npy",
            X,
        )

        np.save(
            output_dir / f"{prefix}_y_rul.npy",
            y_rul,
        )

        np.save(
            output_dir / f"{prefix}_y_hi.npy",
            y_hi,
        )

        metadata = {

            "window_size": self.window_size,

            "stride": self.stride,

            "samples": len(X),

            "features": X.shape[2],

        }

        with open(
            output_dir / f"{prefix}_metadata.json",
            "w",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        logger.info(
            "Processed dataset saved."
        )