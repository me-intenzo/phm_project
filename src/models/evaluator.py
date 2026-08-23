"""
evaluator.py

Evaluation metrics for NASA C-MAPSS prognostics models.

Supported metrics
-----------------
- MAE
- RMSE
- R2
- NASA C-MAPSS scoring function

Supports
--------
- RUL prediction
- Health Index prediction

Author: me-intenzo
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


class PrognosticsEvaluator:
    """
    Evaluate RUL and Health Index predictions.
    """

    # --------------------------------------------------
    # Basic Regression Metrics
    # --------------------------------------------------

    @staticmethod
    def mae(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate Mean Absolute Error."""

        return float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

    # --------------------------------------------------

    @staticmethod
    def rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate Root Mean Squared Error."""

        return float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

    # --------------------------------------------------

    @staticmethod
    def r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate R² score."""

        return float(
            r2_score(
                y_true,
                y_pred,
            )
        )

    # --------------------------------------------------
    # NASA C-MAPSS Score
    # --------------------------------------------------

    @staticmethod
    def nasa_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate the NASA C-MAPSS scoring function.

        Earlier predictions are penalized differently from
        later predictions.

        Negative error:
            Prediction < Actual

        Positive error:
            Prediction > Actual
        """

        errors = y_pred - y_true

        score = 0.0

        for error in errors:

            # Ensure error is handled regardless of array dimensionality or size
            err_val = float(np.mean(error))

            if err_val < 0:

                score += np.exp(
                    -err_val / 13.0
                ) - 1.0

            else:

                score += np.exp(
                    err_val / 10.0
                ) - 1.0

        return float(score)

    # --------------------------------------------------
    # RUL Evaluation
    # --------------------------------------------------

    def evaluate_rul(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate RUL predictions.
        """

        y_true = np.asarray(
            y_true,
            dtype=np.float64,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=np.float64,
        )

        metrics = {
            "MAE": self.mae(
                y_true,
                y_pred,
            ),
            "RMSE": self.rmse(
                y_true,
                y_pred,
            ),
            "R2": self.r2(
                y_true,
                y_pred,
            ),
            "NASA_Score": self.nasa_score(
                y_true,
                y_pred,
            ),
        }

        logger.info(
            "RUL | MAE: %.4f | RMSE: %.4f | R²: %.4f | NASA: %.4f",
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"],
            metrics["NASA_Score"],
        )

        return metrics

    # --------------------------------------------------
    # Health Index Evaluation
    # --------------------------------------------------

    def evaluate_hi(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate Health Index predictions.
        """

        y_true = np.asarray(
            y_true,
            dtype=np.float64,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=np.float64,
        )

        metrics = {
            "MAE": self.mae(
                y_true,
                y_pred,
            ),
            "RMSE": self.rmse(
                y_true,
                y_pred,
            ),
            "R2": self.r2(
                y_true,
                y_pred,
            ),
        }

        logger.info(
            "HI | MAE: %.4f | RMSE: %.4f | R²: %.4f",
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"],
        )

        return metrics

    # --------------------------------------------------
    # Joint Evaluation
    # --------------------------------------------------

    def evaluate(
        self,
        y_rul_true: np.ndarray,
        y_rul_pred: np.ndarray,
        y_hi_true: np.ndarray,
        y_hi_pred: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate both RUL and HI predictions.
        """

        logger.info(
            "Starting joint prognostics evaluation."
        )

        rul_metrics = self.evaluate_rul(
            y_rul_true,
            y_rul_pred,
        )

        hi_metrics = self.evaluate_hi(
            y_hi_true,
            y_hi_pred,
        )

        return {
            "RUL": rul_metrics,
            "HI": hi_metrics,
        }