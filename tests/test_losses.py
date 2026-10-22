"""Tests for the multi-task loss weighting.

The HI term used to be numerically negligible: the RUL loss is an MSE in
cycles squared (order 1e2) while the HI loss is a bounded MSE (order 1e-2),
so ``hi_loss_weight`` moved the total loss by ~1e-5 in relative terms
(docs/ARCHITECTURE.md §7.3).  ``hi_scale`` puts the two on the same scale.
"""

from __future__ import annotations

import pytest
import torch

from src.models.losses import MultiTaskLoss


def _batch(n: int = 16):
    pred_rul = torch.full((n,), 60.0)
    target_rul = torch.full((n,), 70.0)
    pred_hi = torch.full((n,), 0.6)
    target_hi = torch.full((n,), 0.5)
    pred_scale = torch.full((n,), 5.0)
    return pred_rul, pred_hi, target_rul, target_hi, pred_scale


class TestHiScale:
    def test_scaled_term_is_the_raw_term_times_the_factor(self):
        pred_rul, pred_hi, target_rul, target_hi, pred_scale = _batch()
        raw = MultiTaskLoss(beta=1.0, hi_scale=1.0)(
            pred_rul, pred_hi, target_rul, target_hi, pred_scale
        )
        scaled = MultiTaskLoss(beta=1.0, hi_scale=125.0 ** 2)(
            pred_rul, pred_hi, target_rul, target_hi, pred_scale
        )
        assert scaled["hi_loss"] == pytest.approx(float(raw["hi_loss"]))
        assert scaled["hi_loss_scaled"] == pytest.approx(
            float(raw["hi_loss"]) * 125.0 ** 2
        )

    def test_the_hi_term_is_no_longer_negligible(self):
        """With the scale factor the HI term must be a real share of the loss."""

        pred_rul, pred_hi, target_rul, target_hi, pred_scale = _batch()
        losses = MultiTaskLoss(alpha=1.0, beta=0.4, hi_scale=125.0 ** 2)(
            pred_rul, pred_hi, target_rul, target_hi, pred_scale
        )
        hi_share = 0.4 * float(losses["hi_loss_scaled"])
        rul_share = float(losses["rul_loss"])
        assert hi_share / rul_share > 0.01, (
            "the HI term still moves the total loss by less than 1%"
        )

    def test_unscaled_variant_reproduces_the_old_behaviour(self):
        pred_rul, pred_hi, target_rul, target_hi, pred_scale = _batch()
        losses = MultiTaskLoss(alpha=1.0, beta=0.4, hi_scale=1.0)(
            pred_rul, pred_hi, target_rul, target_hi, pred_scale
        )
        hi_share = 0.4 * float(losses["hi_loss"])
        rul_share = float(losses["rul_loss"])
        assert hi_share / rul_share < 0.001

    def test_total_loss_is_the_weighted_sum(self):
        pred_rul, pred_hi, target_rul, target_hi, pred_scale = _batch()
        losses = MultiTaskLoss(alpha=1.0, beta=0.4, gamma=0.1, hi_scale=100.0)(
            pred_rul, pred_hi, target_rul, target_hi, pred_scale
        )
        expected = (
            1.0 * float(losses["rul_loss"])
            + 0.4 * float(losses["hi_loss_scaled"])
            + 0.1 * float(losses["scale_loss"])
        )
        assert float(losses["total_loss"]) == pytest.approx(expected, rel=1e-5)

    def test_missing_scale_head_is_tolerated(self):
        pred_rul, pred_hi, target_rul, target_hi, _ = _batch()
        losses = MultiTaskLoss(hi_scale=125.0 ** 2)(
            pred_rul, pred_hi, target_rul, target_hi, None
        )
        assert float(losses["scale_loss"]) == 0.0
