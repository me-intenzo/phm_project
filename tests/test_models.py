"""Tests for the prognostics models, augmentation and config loader."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from src.models.augmentation import AugmentationConfig, WindowAugmenter
from src.models.gru_att_deg_v2 import GruAttDegV2
from src.models.losses import MultiTaskLoss
from src.models.registry import ALL_MODELS, build_model, model_config
from src.utils.config import CONFIG_DIR, DEFAULT_CONFIG, load_config

INPUT_SIZE = 6
SEQ_LEN = 20


# ------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------


@pytest.mark.parametrize("arch", ALL_MODELS)
def test_every_arch_builds_and_forwards(arch):
    model = build_model(
        arch,
        input_size=INPUT_SIZE,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(3, SEQ_LEN, INPUT_SIZE)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, tuple)
    assert len(out) >= 2

    pred_rul, pred_hi = out[0], out[1]
    assert pred_rul.shape == (3,)
    assert pred_hi.shape == (3,)
    assert torch.isfinite(pred_rul).all()
    assert torch.isfinite(pred_hi).all()


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("not_a_model", input_size=INPUT_SIZE)


def test_model_config_round_trips():
    cfg = model_config("gru_att_deg_v2", hidden_size=32, num_layers=3, dropout=0.2, num_heads=4)

    assert cfg["arch"] == "gru_att_deg_v2"
    assert cfg["hidden_size"] == 32
    assert cfg["num_layers"] == 3
    assert cfg["dropout"] == 0.2

    built = build_model(cfg.pop("arch"), input_size=INPUT_SIZE, **cfg)
    assert isinstance(built, GruAttDegV2)


def test_v2_attention_disabled_falls_back_to_recency():
    model = build_model(
        "gru_att_deg_v2",
        input_size=INPUT_SIZE,
        hidden_size=16,
        num_layers=1,
        use_attention=False,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        rul, hi, scale = model(torch.randn(2, SEQ_LEN, INPUT_SIZE))

    assert rul.shape == (2,)
    assert hi.shape == (2,)
    assert (scale > 0).all()


def test_v2_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        GruAttDegV2(input_size=INPUT_SIZE, hidden_size=10, num_heads=4)


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------


def test_augmenter_disabled_is_identity():
    x = np.random.default_rng(0).normal(size=(4, SEQ_LEN, INPUT_SIZE)).astype(np.float32)

    augmenter = WindowAugmenter(AugmentationConfig())
    out = augmenter(x)

    assert out is x


def test_augmenter_preserves_shape_and_input():
    x = np.random.default_rng(0).normal(size=(4, SEQ_LEN, INPUT_SIZE)).astype(np.float32)
    before = x.copy()

    config = AugmentationConfig(
        noise_std=0.1,
        bias_std=0.1,
        gain_std=0.1,
        sensor_dropout=0.2,
        time_mask_prob=0.5,
        time_mask_len=4,
        history_crop=4,
    )
    out = WindowAugmenter(config, seed=1)(x)

    assert out.shape == x.shape
    assert out.dtype == np.float32
    np.testing.assert_array_equal(x, before)
    assert not np.allclose(out, before)


def test_history_crop_leaves_last_timestep_untouched():
    x = np.random.default_rng(3).normal(size=(8, SEQ_LEN, INPUT_SIZE)).astype(np.float32)

    config = AugmentationConfig(history_crop=5)
    out = WindowAugmenter(config, seed=2)(x)

    np.testing.assert_array_equal(out[:, -1, :], x[:, -1, :])


def test_augmenter_is_reproducible_with_same_seed():
    x = np.random.default_rng(4).normal(size=(4, SEQ_LEN, INPUT_SIZE)).astype(np.float32)
    config = AugmentationConfig(noise_std=0.1, bias_std=0.05)

    a = WindowAugmenter(config, seed=11)(x)
    b = WindowAugmenter(config, seed=11)(x)

    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------


@pytest.mark.parametrize("loss_type", ["mse", "huber", "smooth_l1"])
def test_multitask_loss_supported_types(loss_type):
    criterion = MultiTaskLoss(rul_loss_type=loss_type)

    pred_rul = torch.tensor([10.0, 20.0])
    pred_hi = torch.tensor([0.2, 0.8])
    true_rul = torch.tensor([12.0, 18.0])
    true_hi = torch.tensor([0.3, 0.7])
    scale = torch.tensor([1.0, 1.0])

    losses = criterion(pred_rul, pred_hi, true_rul, true_hi, scale)

    assert losses["total_loss"].item() > 0
    assert torch.isfinite(losses["total_loss"])


def test_multitask_loss_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown rul_loss_type"):
        MultiTaskLoss(rul_loss_type="mae")


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


def test_default_config_has_every_documented_key():
    for key in ("max_rul", "hidden_size", "num_layers", "dropout", "scheduler",
                "grad_clip", "ema_decay", "augment"):
        assert key in DEFAULT_CONFIG


def test_load_config_merges_subset_and_overrides():
    cfg = load_config(subset="FD004", overrides={"learning_rate": 1e-3, "epochs": 2})

    assert cfg["learning_rate"] == 1e-3
    assert cfg["epochs"] == 2
    # FD004's own YAML turns augmentation on.
    assert cfg["augment"] is True


def test_subset_yaml_only_uses_known_keys():
    """Every YAML key must exist in :data:`DEFAULT_CONFIG`.

    The subset files list deviations from the defaults, so a key that the
    defaults do not know about would be inert — the config would look
    authoritative while nothing reads it.
    """

    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))

        assert not unknown, (
            f"{path.name} declares keys that are not in DEFAULT_CONFIG "
            f"(and are therefore never read): {unknown}"
        )


def test_load_config_does_not_mutate_defaults():
    load_config(subset="FD001", overrides={"dropout": 0.99})
    assert DEFAULT_CONFIG["dropout"] != 0.99


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(path="does/not/exist.yaml")
