"""
Genuine Kernel SHAP explanation for the GRU prognostics model.

Each sensor is treated as one explanatory feature whose value is
its complete temporal trajectory.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np
import torch
import shap


def _select_output(
    model,
    x: torch.Tensor,
    target: str,
) -> torch.Tensor:
    """Select RUL or HI from the multitask GRU output."""
    pred_rul, pred_hi, *_ = model(x)
    target = target.lower()
    if target == "rul":
        return pred_rul
    if target == "hi":
        return pred_hi
    raise ValueError("target must be either 'rul' or 'hi'.")


def _predict_from_sensor_masks(
    masks: np.ndarray,
    sample: np.ndarray,
    baseline: np.ndarray,
    model,
    device: torch.device,
) -> np.ndarray:
    """Convert sensor-level coalition masks into real GRU inputs."""
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)

    n_coalitions = masks.shape[0]
    sample_batch   = np.broadcast_to(sample,   (n_coalitions, *sample.shape)).copy()
    baseline_batch = np.broadcast_to(baseline, (n_coalitions, *baseline.shape))

    mask_3d = masks[:, np.newaxis, :]
    coalition_batch = mask_3d * sample_batch + (1.0 - mask_3d) * baseline_batch

    x_tensor = torch.from_numpy(coalition_batch).float().to(device)
    with torch.no_grad():
        predictions = (
            _select_output(model=model, x=x_tensor, target=_CURRENT_TARGET)
            .detach().cpu().numpy().reshape(-1)
        )
    return predictions


def _extract_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError("Expected a single model output.")
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    return np.asarray(shap_values, dtype=np.float32)


def explain_with_shap(
    model,
    data,
    target: str = "rul",
    background=None,
    baseline=None,
    nsamples: int | str = 1024,
    seed: int = 42,
):
    """Generate genuine Kernel SHAP sensor-level explanations."""
    global _CURRENT_TARGET
    _CURRENT_TARGET = target.lower()

    if _CURRENT_TARGET not in {"rul", "hi"}:
        raise ValueError("target must be either 'rul' or 'hi'.")

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"data must have shape (batch, timesteps, sensors). Got {data.shape}.")

    batch_size, timesteps, n_sensors = data.shape

    # Determine baseline
    if baseline is not None:
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.detach().cpu().numpy()
        baseline = np.asarray(baseline, dtype=np.float32)
    elif background is not None:
        if isinstance(background, torch.Tensor):
            background = background.detach().cpu().numpy()
        background = np.asarray(background, dtype=np.float32)
        baseline = np.median(background, axis=0) if background.ndim == 3 else background
    else:
        baseline = np.zeros((timesteps, n_sensors), dtype=np.float32)

    if baseline.shape != (timesteps, n_sensors):
        raise ValueError(f"baseline must have shape ({timesteps}, {n_sensors}). Got {baseline.shape}.")

    model.eval()
    device = next(model.parameters()).device

    sensor_shap_values = np.zeros((batch_size, n_sensors), dtype=np.float32)
    base_values        = np.zeros(batch_size, dtype=np.float32)
    predictions        = np.zeros(batch_size, dtype=np.float32)
    additivity_errors  = np.zeros(batch_size, dtype=np.float32)

    for sample_index in range(batch_size):
        sample = data[sample_index]

        def model_function(masks):
            return _predict_from_sensor_masks(
                masks=masks, sample=sample, baseline=baseline,
                model=model, device=device,
            )

        background_mask = np.zeros((1, n_sensors), dtype=np.float32)
        explainer = shap.KernelExplainer(
            model_function, background_mask,
            feature_names=[f"S{i+1}" for i in range(n_sensors)],
            seed=seed,
        )

        instance_mask = np.ones((1, n_sensors), dtype=np.float32)
        shap_vals = _extract_shap_values(explainer.shap_values(instance_mask, nsamples=nsamples))

        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)
        if shap_vals.shape[-1] != n_sensors:
            raise ValueError(f"Unexpected SHAP output shape: {shap_vals.shape}")

        sensor_values = shap_vals[0]
        sensor_shap_values[sample_index] = sensor_values

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            expected_value = np.asarray(expected_value).reshape(-1)[0]
        base_values[sample_index] = float(expected_value)

        sample_tensor = torch.from_numpy(
            sample.copy().reshape(1, timesteps, n_sensors)
        ).float().to(device)
        with torch.no_grad():
            prediction = (
                _select_output(model=model, x=sample_tensor, target=target)
                .detach().cpu().numpy().reshape(-1)[0]
            )
        predictions[sample_index] = float(prediction)
        additivity_errors[sample_index] = abs(base_values[sample_index] + sensor_values.sum() - prediction)

    return {
        "method": "kernel_shap",
        "target": target,
        "sensor_shap_values": sensor_shap_values,
        "temporal_shap_values": None,
        "base_value": base_values,
        "prediction": predictions,
        "additivity_error": additivity_errors,
        "mean_additivity_error": float(additivity_errors.mean()),
        "max_additivity_error": float(additivity_errors.max()),
        "feature_names": [f"S{i+1}" for i in range(n_sensors)],
    }


# Module-level target used only by the SHAP model-function callback.
_CURRENT_TARGET = "rul"
