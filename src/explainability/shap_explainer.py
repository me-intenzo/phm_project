"""
shap_explainer.py

SHAP-based explainability for the GRU prognostics model.

Explains which sensor features contribute to RUL predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import shap


def explain_with_shap(
    model: torch.nn.Module,
    data: np.ndarray | torch.Tensor,
    background_size: int = 50,
    sample_size: int = 10,
) -> dict[str, Any]:
    """
    Generate SHAP explanations for RUL predictions.

    Parameters
    ----------
    model:
        Trained prognostics model.

    data:
        Input sequences with shape:
        (samples, sequence_length, features)

    background_size:
        Number of background samples used by SHAP.

    sample_size:
        Number of samples to explain.

    Returns
    -------
    dict
        SHAP values and feature-level importance.
    """

    model.eval()

    # --------------------------------------------------
    # Convert input to NumPy
    # --------------------------------------------------

    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)

    if data_np.ndim != 3:
        raise ValueError(
            "Expected data with shape "
            "(samples, sequence_length, features)."
        )

    # --------------------------------------------------
    # Limit samples
    # --------------------------------------------------

    background = data_np[:background_size]
    samples = data_np[:sample_size]

    # --------------------------------------------------
    # Prediction function
    # --------------------------------------------------

    device = next(model.parameters()).device

    def predict_rul(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            predictions = model(x_tensor)[0]

        return predictions.detach().cpu().numpy()

    # --------------------------------------------------
    # SHAP Kernel Explainer
    # --------------------------------------------------

    # Flatten the sequence so SHAP can work with
    # individual sensor/time-step inputs.
    background_flat = background.reshape(
        background.shape[0],
        -1,
    )

    samples_flat = samples.reshape(
        samples.shape[0],
        -1,
    )

    sequence_length = data_np.shape[1]
    num_features = data_np.shape[2]

    def predict_flat(x_flat: np.ndarray) -> np.ndarray:
        x_sequence = x_flat.reshape(
            -1,
            sequence_length,
            num_features,
        )

        return predict_rul(x_sequence)

    explainer = shap.KernelExplainer(
        predict_flat,
        background_flat,
    )

    shap_values = explainer.shap_values(
        samples_flat,
    )

    # SHAP versions may return either an ndarray
    # or a list depending on the model/output format.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # --------------------------------------------------
    # Reshape back to:
    #
    # samples × sequence × features
    # --------------------------------------------------

    shap_values = shap_values.reshape(
        samples.shape
    )

    # --------------------------------------------------
    # Global sensor importance
    #
    # Average absolute SHAP contribution across
    # samples and time steps.
    # --------------------------------------------------

    sensor_importance = np.mean(
        np.abs(shap_values),
        axis=(0, 1),
    )

    # --------------------------------------------------
    # Temporal importance
    #
    # Average absolute contribution for each time step.
    # --------------------------------------------------

    temporal_importance = np.mean(
        np.abs(shap_values),
        axis=(0, 2),
    )

    return {
        "shap_values": shap_values,
        "sensor_importance": sensor_importance,
        "temporal_importance": temporal_importance,
        "samples": samples,
        "predictions": predict_rul(samples),
    }