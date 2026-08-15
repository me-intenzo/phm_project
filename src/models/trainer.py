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

import logging
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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

        logger.info(
            "Using device: %s",
            self.device,
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

        train_loader = DataLoader(

            train_dataset,

            batch_size=self.batch_size,

            shuffle=True,

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
        ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None,

            "loss": loss,

        }

        torch.save(

            checkpoint,

            self.checkpoint_dir /
            "best_model.pt",

        )

        logger.info(
            "Checkpoint saved."
        )

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

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )


        self.optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        logger.info(
            "Checkpoint loaded."
        )

        return checkpoint["epoch"]

    # ----------------------------------------------------
    # Train One Epoch
    # ----------------------------------------------------

    def train_epoch(
        self,
        train_loader,
    ):

        self.model.train()

        running_loss = 0.0
        running_rul = 0.0
        running_hi = 0.0

        for x, rul, hi in train_loader:

            x = x.to(self.device)

            rul = rul.to(self.device)

            hi = hi.to(self.device)

            self.optimizer.zero_grad()

            pred_rul, pred_hi = self.model(x)

            losses = self.criterion(
                pred_rul,
                pred_hi,
                rul,
                hi,
            )

            loss = losses["total_loss"]

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            running_rul += losses["rul_loss"].item()

            running_hi += losses["hi_loss"].item()

        n_batches = len(train_loader)

        return {

            "loss": running_loss / n_batches,

            "rul_loss": running_rul / n_batches,

            "hi_loss": running_hi / n_batches,

        }

    # ----------------------------------------------------
    # Validate One Epoch
    # ----------------------------------------------------

    @torch.no_grad()

    def validate_epoch(
        self,
        val_loader,
    ):

        self.model.eval()

        running_loss = 0.0
        running_rul = 0.0
        running_hi = 0.0

        for x, rul, hi in val_loader:

            x = x.to(self.device)

            rul = rul.to(self.device)

            hi = hi.to(self.device)

            pred_rul, pred_hi = self.model(x)

            losses = self.criterion(
                pred_rul,
                pred_hi,
                rul,
                hi,
            )

            running_loss += losses["total_loss"].item()

            running_rul += losses["rul_loss"].item()

            running_hi += losses["hi_loss"].item()

        n_batches = len(val_loader)

        return {

            "loss": running_loss / n_batches,

            "rul_loss": running_rul / n_batches,

            "hi_loss": running_hi / n_batches,

        }

    # ----------------------------------------------------
    # Complete Training Pipeline
    # ----------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=10,
        min_delta=1e-4,
    ):

        history = {

            "train_loss": [],
            "val_loss": [],

            "train_rul_loss": [],
            "val_rul_loss": [],

            "train_hi_loss": [],
            "val_hi_loss": [],

        }

        early_stop_counter = 0

        logger.info("Starting training...")

        for epoch in range(1, epochs + 1):

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

            logger.info(

                "Epoch %3d/%3d | "
                "Train %.4f | "
                "Val %.4f",

                epoch,

                epochs,

                train_metrics["loss"],

                val_metrics["loss"],

            )

            # ----------------------------
            # Save Best Model
            # ----------------------------

            if val_metrics["loss"] < self.best_loss - min_delta:

                self.best_loss = val_metrics["loss"]

                self.save_checkpoint(

                    epoch,

                    self.best_loss,

                )

                early_stop_counter = 0

            else:

                early_stop_counter += 1

            # ----------------------------
            # LR Scheduler
            # ----------------------------

            if self.scheduler is not None:

                self.scheduler.step(
                    val_metrics["loss"]
                )

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