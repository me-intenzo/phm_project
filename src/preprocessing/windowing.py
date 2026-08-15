"""
windowing.py

Sliding window generation for NASA C-MAPSS dataset.

Responsibilities
----------------
1. Create fixed-length sequences.
2. Generate RUL and HI targets.
3. Preserve engine IDs for group-wise validation.
4. Create final test windows.
5. Save processed datasets and metadata.

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
    # Feature Columns
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
    # Training Windows
    # ------------------------------------------------ #

    def create_windows(
        self,
        df: pd.DataFrame,
    ):
        """
        Generate sliding windows for training.

        Returns
        -------
        X : np.ndarray
            Shape:
                (samples, window_size, features)

        y_rul : np.ndarray
            RUL target for each window.

        y_hi : np.ndarray
            HI target for each window.

        window_engine_ids : np.ndarray
            Engine ID associated with each window.

        Notes
        -----
        The engine ID is preserved so that training/validation
        splitting can be performed at engine level rather than
        randomly across correlated windows.
        """

        logger.info(
            "Generating sliding windows..."
        )

        feature_columns = self.get_feature_columns(df)

        X = []
        y_rul = []
        y_hi = []
        window_engine_ids = []

        engine_ids = sorted(
            df["unit_number"].unique()
        )

        for engine in engine_ids:

            engine_df = (
                df[df["unit_number"] == engine]
                .sort_values("time_in_cycles")
            )

            features = engine_df[
                feature_columns
            ].values

            rul = engine_df[
                "RUL"
            ].values

            hi = engine_df[
                "HI"
            ].values

            # --------------------------------------------------
            # Sliding windows
            # --------------------------------------------------

            for i in range(
                0,
                len(engine_df)
                - self.window_size
                + 1,
                self.stride,
            ):

                end = (
                    i
                    + self.window_size
                )

                X.append(
                    features[i:end]
                )

                # Target corresponds to the final
                # timestep of the window.
                y_rul.append(
                    rul[end - 1]
                )

                y_hi.append(
                    hi[end - 1]
                )

                # Preserve source engine.
                window_engine_ids.append(
                    engine
                )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y_rul = np.asarray(
            y_rul,
            dtype=np.float32,
        )

        y_hi = np.asarray(
            y_hi,
            dtype=np.float32,
        )

        window_engine_ids = np.asarray(
            window_engine_ids,
            dtype=np.int64,
        )

        logger.info(
            "Generated %d windows.",
            len(X),
        )

        logger.info(
            "Windows from %d engines.",
            len(
                np.unique(
                    window_engine_ids
                )
            ),
        )

        return (
            X,
            y_rul,
            y_hi,
            window_engine_ids,
        )

    # ------------------------------------------------ #
    # Save Dataset
    # ------------------------------------------------ #

    def save_dataset(
        self,
        X,
        y_rul,
        y_hi,
        output_dir: Path,
        prefix: str,
        engine_ids=None,
    ):
        """
        Save processed dataset.

        Parameters
        ----------
        X : np.ndarray
            Input windows.

        y_rul : np.ndarray
            RUL targets.

        y_hi : np.ndarray
            HI targets.

        output_dir : Path
            Destination directory.

        prefix : str
            Dataset filename prefix.

        engine_ids : np.ndarray, optional
            Engine ID corresponding to each window.
        """

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Save arrays
        # --------------------------------------------------

        np.save(
            output_dir
            / f"{prefix}_X.npy",
            X,
        )

        np.save(
            output_dir
            / f"{prefix}_y_rul.npy",
            y_rul,
        )

        np.save(
            output_dir
            / f"{prefix}_y_hi.npy",
            y_hi,
        )

        # --------------------------------------------------
        # Save engine IDs
        # --------------------------------------------------

        if engine_ids is not None:

            engine_ids = np.asarray(
                engine_ids,
                dtype=np.int64,
            )

            if len(engine_ids) != len(X):

                raise ValueError(
                    "Number of engine IDs "
                    "must match number of windows."
                )

            np.save(
                output_dir
                / f"{prefix}_engine_ids.npy",
                engine_ids,
            )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = {
            "window_size": self.window_size,
            "stride": self.stride,
            "samples": len(X),
            "features": X.shape[2],
            "engines": (
                int(
                    len(
                        np.unique(
                            engine_ids
                        )
                    )
                )
                if engine_ids is not None
                else None
            ),
            "has_engine_ids": (
                engine_ids is not None
            ),
        }

        with open(
            output_dir
            / f"{prefix}_metadata.json",
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

    # ------------------------------------------------ #
    # Test Windows
    # ------------------------------------------------ #

    def create_test_windows(
        self,
        df: pd.DataFrame,
    ):
        """
        Create one final window for each test engine.

        NASA C-MAPSS test evaluation uses the final
        available observation window from each engine.

        Returns
        -------
        X : np.ndarray
            Shape:
                (num_engines, window_size, num_features)

        y_rul : np.ndarray
            Official RUL associated with each engine.

        y_hi : np.ndarray
            HI associated with each engine.

        engine_ids : np.ndarray
            Engine ID associated with each test window.
        """

        logger.info(
            "Generating final test windows..."
        )

        feature_columns = (
            self.get_feature_columns(df)
        )

        X = []
        y_rul = []
        y_hi = []
        engine_ids = []

        sorted_engine_ids = sorted(
            df["unit_number"].unique()
        )

        for engine in sorted_engine_ids:

            engine_df = (
                df[df["unit_number"] == engine]
                .sort_values("time_in_cycles")
            )

            if len(engine_df) < self.window_size:

                logger.warning(
                    "Engine %s has only %d cycles. "
                    "Skipping test window.",
                    engine,
                    len(engine_df),
                )

                continue

            # --------------------------------------------------
            # Final available sequence
            # --------------------------------------------------

            final_window = (
                engine_df[
                    feature_columns
                ].values[
                    -self.window_size:
                ]
            )

            X.append(
                final_window
            )

            # --------------------------------------------------
            # Ground-truth RUL
            # --------------------------------------------------

            y_rul.append(
                engine_df[
                    "RUL"
                ].iloc[-1]
            )

            # --------------------------------------------------
            # HI
            # --------------------------------------------------

            y_hi.append(
                engine_df[
                    "HI"
                ].iloc[-1]
            )

            # --------------------------------------------------
            # Engine ID
            # --------------------------------------------------

            engine_ids.append(
                engine
            )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y_rul = np.asarray(
            y_rul,
            dtype=np.float32,
        )

        y_hi = np.asarray(
            y_hi,
            dtype=np.float32,
        )

        engine_ids = np.asarray(
            engine_ids,
            dtype=np.int64,
        )

        logger.info(
            "Generated %d final test windows.",
            len(X),
        )

        return (
            X,
            y_rul,
            y_hi,
            engine_ids,
        )