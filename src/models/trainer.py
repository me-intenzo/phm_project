"""
trainer.py

Generic training pipeline for prognostics models.

Supports:
- LSTM
- GRU
- Transformer
- Hybrid

Author: me-intenzo
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from .checkpoint import load_state_dict_compat
from .outputs import unpack_model_output

logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device: torch.device | str = "cpu",
        batch_size=64,
        checkpoint_dir="outputs/checkpoints",
        augmenter=None,
        grad_clip: float = 0.0,
        ema_decay: float = 0.0,
        seed: int | None = None,
    ):

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.batch_size = batch_size

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_loss = float("inf")

        self.input_size: int | None = None
        self.window_size: int | None = None
        self.subset: str | None = None

        # --------------------------------------------------
        # Regularisation / stabilisation options
        # --------------------------------------------------

        self.augmenter = augmenter
        self.grad_clip = float(grad_clip or 0.0)

        self.ema_decay = float(ema_decay or 0.0)
        self.ema_model: torch.nn.Module | None = None
        if self.ema_decay > 0:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        # Seed used for dataloader shuffling.  The RNG streams themselves are
        # seeded by ``src.utils.seed.set_seed`` before the trainer is built;
        # this generator makes the batch order depend on the run seed alone
        # and not on how much randomness the model construction consumed.
        self.seed = None if seed is None else int(seed)

        # Architecture description stored inside the checkpoint so the
        # model can be rebuilt exactly at evaluation time.
        self.model_config: dict | None = None

        # Run provenance (seed, tag, determinism settings) stored in the
        # checkpoint so a published number can be traced back to its run.
        self.run_metadata: dict | None = None

        logger.info(
            "Using device: %s",
            self.device,
        )

        logger.info(
            "Regularisation: augmenter=%s | grad_clip=%.3f | ema_decay=%.4f",
            "on" if self.augmenter is not None else "off",
            self.grad_clip,
            self.ema_decay,
        )

    # ----------------------------------------------------
    # DataLoader
    # ----------------------------------------------------

    def create_dataloaders(
        self,
        X_train,
        y_rul_train,
        y_hi_train,
        X_val,
        y_rul_val,
        y_hi_val,
        ):

        train_dataset = TensorDataset(

            torch.from_numpy(X_train).float(),

            torch.from_numpy(y_rul_train).float(),

            torch.from_numpy(y_hi_train).float(),

        )

        val_dataset = TensorDataset(

            torch.from_numpy(X_val).float(),

            torch.from_numpy(y_rul_val).float(),

            torch.from_numpy(y_hi_val).float(),

        )

        # An explicit generator decouples the batch order from the global
        # RNG state, so the order is a pure function of the run seed.
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        train_loader = DataLoader(

            train_dataset,

            batch_size=self.batch_size,

            shuffle=True,

            generator=generator,

        )

        val_loader = DataLoader(

            val_dataset,

            batch_size=self.batch_size,

            shuffle=False,

        )

        return train_loader, val_loader

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------

    def save_checkpoint(
        self,
        epoch,
        loss,
        input_size: int | None = None,
        ):

        # The EMA weights are the ones that generalise best, so they are
        # stored as the primary ``model_state_dict``.  The raw weights
        # are kept separately so training can be resumed.
        primary_state = (
            self.ema_model.state_dict()
            if self.ema_model is not None
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": primary_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "loss": loss,
        }

        if self.ema_model is not None:
            checkpoint["raw_model_state_dict"] = self.model.state_dict()
            checkpoint["ema_decay"] = self.ema_decay

        if self.model_config is not None:
            checkpoint["model_config"] = self.model_config

        if self.run_metadata is not None:
            checkpoint["run_metadata"] = self.run_metadata
            checkpoint["seed"] = self.run_metadata.get("seed")

        if input_size is not None:
            checkpoint["input_size"] = input_size
        if self.window_size is not None:
            checkpoint["window_size"] = self.window_size
        if self.subset is not None:
            checkpoint["subset"] = self.subset

        torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
        logger.info("Checkpoint saved.")

    # ----------------------------------------------------
    # EMA helpers
    # ----------------------------------------------------

    @torch.no_grad()
    def update_ema(self) -> None:
        """Fold the current weights into the exponential moving average."""

        if self.ema_model is None:
            return

        decay = self.ema_decay

        ema_params = dict(self.ema_model.named_parameters())
        live_params = dict(self.model.named_parameters())

        for name, ema_param in ema_params.items():
            live_param = live_params.get(name)
            if live_param is None:
                continue
            if not ema_param.dtype.is_floating_point:
                ema_param.copy_(live_param)
                continue
            ema_param.mul_(decay).add_(live_param, alpha=1.0 - decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        live_buffers = dict(self.model.named_buffers())

        for name, ema_buffer in ema_buffers.items():
            live_buffer = live_buffers.get(name)
            if live_buffer is None:
                continue
            if ema_buffer.dtype.is_floating_point:
                ema_buffer.mul_(decay).add_(live_buffer, alpha=1.0 - decay)
            else:
                ema_buffer.copy_(live_buffer)

    # ----------------------------------------------------
    # Load Model
    # ----------------------------------------------------

    def load_checkpoint(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if (
            self.scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
        )

        load_state_dict_compat(
            self.model,
            checkpoint["model_state_dict"],
            logger=logger,
        )


        self.optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        logger.info(
            "Checkpoint loaded."
        )

        return checkpoint["epoch"]

    # ----------------------------------------------------
    # One epoch (train / validate)
    # ----------------------------------------------------

    def _run_epoch(
        self,
        loader,
        *,
        training: bool,
    ) -> dict[str, float]:
        """
        Run one epoch over *loader*.

        The training and the validation pass used to be two copies of the
        same block — the forward pass, the loss computation and the
        accumulation of the four running sums were written out twice and
        had to be kept in sync by hand.  There is one implementation now;
        the *mode* selects the differences:

        * ``training=True``  — augmentation, gradient clipping, the
          optimiser step and the EMA update;
        * ``training=False`` — no gradients, and the EMA weights are
          evaluated when the EMA is enabled, so the checkpoint that is
          selected is the one that is actually deployed.
        """

        if training:
            self.model.train()
            evaluated_model = self.model
        else:
            # The live model is switched to eval as well, so that a caller
            # that only validates does not leave the model in train mode.
            self.model.eval()

            # The EMA copy is created while the live model is still in
            # train mode, so it must be switched to eval explicitly —
            # otherwise dropout stays active and inflates the validation
            # loss.
            evaluated_model = self.ema_model or self.model
            evaluated_model.eval()

        running_loss = 0.0
        running_rul = 0.0
        running_hi = 0.0
        running_scale = 0.0

        with torch.set_grad_enabled(training):

            for x, rul, hi in loader:

                # --------------------------------------------------
                # Augmentation (CPU, on the standardised features)
                # --------------------------------------------------

                if training and self.augmenter is not None:
                    x_aug = self.augmenter(x.numpy())
                    x = torch.from_numpy(
                        np.ascontiguousarray(x_aug)
                    ).float()
                else:
                    x = x.float()

                x = x.to(self.device)

                rul = rul.to(self.device)

                hi = hi.to(self.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                pred_rul, pred_hi, pred_scale = unpack_model_output(
                    evaluated_model(x)
                )

                losses = self.criterion(
                    pred_rul,
                    pred_hi,
                    rul,
                    hi,
                    pred_scale,
                )

                loss = losses["total_loss"]

                if training:

                    loss.backward()

                    # --------------------------------------------------
                    # Gradient clipping
                    # --------------------------------------------------

                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip,
                        )

                    self.optimizer.step()

                    # --------------------------------------------------
                    # EMA update
                    # --------------------------------------------------

                    self.update_ema()

                running_loss += loss.item()

                running_rul += losses["rul_loss"].item()

                running_hi += losses["hi_loss"].item()

                running_scale += losses["scale_loss"].item()

        n_batches = len(loader)

        return {

            "loss": running_loss / n_batches,

            "rul_loss": running_rul / n_batches,

            "hi_loss": running_hi / n_batches,

            "scale_loss": running_scale / n_batches,

        }

    # ----------------------------------------------------
    # Train One Epoch
    # ----------------------------------------------------

    def train_epoch(
        self,
        train_loader,
    ):

        return self._run_epoch(train_loader, training=True)

    # ----------------------------------------------------
    # Validate One Epoch
    # ----------------------------------------------------

    def validate_epoch(
        self,
        val_loader,
    ):

        return self._run_epoch(val_loader, training=False)

    # ----------------------------------------------------
    # Complete Training Pipeline
    # ----------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=300,
        patience=20,
        min_delta=1e-5,
        warmup_epochs: int = 0,
        base_lr: float | None = None,
    ):

        history = {

            "train_loss": [],
            "val_loss": [],

            "train_rul_loss": [],
            "val_rul_loss": [],

            "train_hi_loss": [],
            "val_hi_loss": [],

            "train_scale_loss": [],
            "val_scale_loss": [],

            "lr": [],

        }

        early_stop_counter = 0

        logger.info("Starting training...")

        for epoch in range(1, epochs + 1):

            # The augmenter draws fresh randomness for every batch, so
            # each epoch sees a different set of transforms.
            if self.augmenter is not None:
                self.augmenter.reset()

            # ----------------------------
            # Linear LR warm-up
            # ----------------------------

            if (
                warmup_epochs > 0
                and base_lr is not None
                and epoch <= warmup_epochs
            ):
                warm_lr = base_lr * (epoch / max(1, warmup_epochs))
                for group in self.optimizer.param_groups:
                    group["lr"] = warm_lr
                logger.info("Warm-up epoch %d | lr=%.6f", epoch, warm_lr)

            train_metrics = self.train_epoch(
                train_loader
            )

            val_metrics = self.validate_epoch(
                val_loader
            )

            history["train_loss"].append(
                train_metrics["loss"]
            )

            history["val_loss"].append(
                val_metrics["loss"]
            )

            history["train_rul_loss"].append(
                train_metrics["rul_loss"]
            )

            history["val_rul_loss"].append(
                val_metrics["rul_loss"]
            )

            history["train_hi_loss"].append(
                train_metrics["hi_loss"]
            )

            history["val_hi_loss"].append(
                val_metrics["hi_loss"]
            )

            history["train_scale_loss"].append(
                train_metrics["scale_loss"]
            )

            history["val_scale_loss"].append(
                val_metrics["scale_loss"]
            )

            history["lr"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            logger.info(

                "Epoch %3d/%3d | "
                "Train %.4f | "
                "Val %.4f | "
                "lr %.2e",

                epoch,

                epochs,

                train_metrics["loss"],

                val_metrics["loss"],

                self.optimizer.param_groups[0]["lr"],

            )

            # ----------------------------
            # Save Best Model
            # ----------------------------

            if val_metrics["loss"] < self.best_loss - min_delta:

                self.best_loss = val_metrics["loss"]

                self.save_checkpoint(

                    epoch,

                    self.best_loss,

                    self.input_size,

                )

                early_stop_counter = 0

            else:

                early_stop_counter += 1

            # ----------------------------
            # LR Scheduler
            # ----------------------------

            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(
                        val_metrics["loss"]
                    )
                else:
                    # Step-based schedulers (cosine, one-cycle, ...)
                    # must not be stepped during the warm-up phase,
                    # otherwise the cosine curve is misaligned.
                    if not (warmup_epochs > 0 and epoch <= warmup_epochs):
                        self.scheduler.step()

            # ----------------------------
            # Early Stopping
            # ----------------------------

            if early_stop_counter >= patience:

                logger.info(
                    "Early stopping triggered."
                )

                break

        logger.info(
            "Training completed."
        )

        return history
