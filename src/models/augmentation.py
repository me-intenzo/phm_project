"""
augmentation.py

Training-time augmentation for C-MAPSS sliding windows.

Why this exists
---------------
The training curves in ``outputs/results/*_history.npz`` show a clear
overfitting signature: for ``gru_att_deg`` on FD001 the training RUL loss
falls to ~85 while the validation RUL loss sits at ~166, and the
validation loss oscillates by 40 % between consecutive epochs.  With only
80–260 *engines* per subset there are very few independent degradation
trajectories, so a network with ~180 k parameters memorises the training
engines instead of learning the degradation law.

The augmentations below are physically motivated for turbofan prognostics
and, crucially, all of them preserve the semantics of the window: the last
timestep is always the current state and the target is unchanged.

* ``bias``     — per-sample, per-sensor constant offset (sensor drift).
* ``gain``     — per-sample, per-sensor multiplicative jitter (sensor gain).
* ``noise``    — per-timestep Gaussian noise (measurement noise).
* ``dropout``  — masks whole sensor channels (sensor faults).
* ``time_mask``— masks a contiguous span of timesteps (transient glitches).
* ``history_crop`` — drops the oldest steps of the window and left-pads
  with the first retained frame, simulating a shorter observation history.

Author: me-intenzo
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationConfig:
    """Magnitudes of every augmentation, on the standardised feature scale.

    A magnitude of ``0`` (or a probability of ``0``) disables that
    augmentation.
    """

    noise_std: float = 0.0
    bias_std: float = 0.0
    gain_std: float = 0.0
    sensor_dropout: float = 0.0
    time_mask_prob: float = 0.0
    time_mask_len: int = 0
    history_crop: int = 0

    @property
    def enabled(self) -> bool:
        """True when at least one augmentation is active."""

        return (
            self.noise_std > 0
            or self.bias_std > 0
            or self.gain_std > 0
            or self.sensor_dropout > 0
            or self.time_mask_prob > 0
            or self.history_crop > 0
        )

    @classmethod
    def from_config(cls, config: dict) -> "AugmentationConfig":
        """Build from a flat mapping (e.g. the YAML experiment config)."""

        return cls(
            noise_std=float(config.get("aug_noise_std", 0.0)),
            bias_std=float(config.get("aug_bias_std", 0.0)),
            gain_std=float(config.get("aug_gain_std", 0.0)),
            sensor_dropout=float(config.get("aug_sensor_dropout", 0.0)),
            time_mask_prob=float(config.get("aug_time_mask_prob", 0.0)),
            time_mask_len=int(config.get("aug_time_mask_len", 0)),
            history_crop=int(config.get("aug_history_crop", 0)),
        )


class WindowAugmenter:
    """Apply C-MAPSS augmentation transforms to a batch of windows.

    Parameters
    ----------
    config:
        Magnitudes of the individual transforms.
    seed:
        Base seed for the internal RNG.  Call :meth:`reset` between epochs
        to get a different stream of augmentations while staying
        reproducible for a given run.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------
    # RNG control
    # ------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Restart the augmentation RNG stream."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------

    def _apply_bias_gain_noise(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        B, T, Fdim = x.shape
        cfg = self.config

        if cfg.gain_std > 0:
            gain = 1.0 + self.rng.normal(
                0.0, cfg.gain_std, size=(B, 1, Fdim)
            ).astype(np.float32)
            x = x * gain

        if cfg.bias_std > 0:
            bias = self.rng.normal(
                0.0, cfg.bias_std, size=(B, 1, Fdim)
            ).astype(np.float32)
            x = x + bias

        if cfg.noise_std > 0:
            x = x + self.rng.normal(
                0.0, cfg.noise_std, size=x.shape
            ).astype(np.float32)

        return x

    def _apply_sensor_dropout(self, x: np.ndarray) -> np.ndarray:
        p = self.config.sensor_dropout
        if p <= 0:
            return x

        B, T, Fdim = x.shape
        mask = self.rng.random(size=(B, 1, Fdim)) >= p
        # Keep at least one sensor alive per sample.
        dead_all = ~mask.any(axis=2, keepdims=True)
        mask = mask | dead_all
        return x * mask.astype(np.float32)

    def _apply_time_mask(self, x: np.ndarray) -> np.ndarray:
        p = self.config.time_mask_prob
        if p <= 0 or self.config.time_mask_len <= 0:
            return x

        B, T, Fdim = x.shape
        max_len = min(self.config.time_mask_len, T)

        for b in range(B):
            if self.rng.random() >= p:
                continue
            length = int(self.rng.integers(1, max_len + 1))
            start = int(self.rng.integers(0, max(1, T - length + 1)))
            x[b, start:start + length, :] = 0.0

        return x

    def _apply_history_crop(self, x: np.ndarray) -> np.ndarray:
        max_crop = self.config.history_crop
        if max_crop <= 0:
            return x

        B, T, Fdim = x.shape
        max_crop = min(max_crop, T - 1)

        for b in range(B):
            crop = int(self.rng.integers(0, max_crop + 1))
            if crop == 0:
                continue
            # Keep the most recent (T - crop) steps, left-pad with the
            # oldest retained frame so the window length is preserved.
            kept = x[b, T - crop:, :]
            x[b, :crop, :] = kept[0:1, :]

        return x

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Augment a batch of windows.

        Parameters
        ----------
        x : np.ndarray
            Shape (B, T, F), already standardised.

        Returns
        -------
        np.ndarray
            A new float32 array of the same shape.  The input is not
            modified.
        """

        if not self.config.enabled:
            return x

        out = np.array(x, dtype=np.float32, copy=True)

        out = self._apply_bias_gain_noise(out)
        out = self._apply_sensor_dropout(out)
        out = self._apply_time_mask(out)
        out = self._apply_history_crop(out)

        return out


__all__ = ["AugmentationConfig", "WindowAugmenter"]
