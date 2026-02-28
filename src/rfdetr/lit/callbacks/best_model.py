# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Best-model checkpointing and early stopping callbacks for RF-DETR Lightning training."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from rfdetr.util.logger import get_logger
from rfdetr.util.misc import strip_checkpoint

logger = get_logger()


class BestModelCallback(Callback):
    """Track best validation mAP and save best checkpoints during training.

    Mirrors the legacy ``main.py`` best-model logic: saves
    ``checkpoint_best_regular.pth`` and optionally ``checkpoint_best_ema.pth``
    whenever the respective mAP improves.  At the end of training, copies the
    overall winner to ``checkpoint_best_total.pth`` and strips it down to
    ``{"model": ..., "args": ...}`` via :func:`rfdetr.util.misc.strip_checkpoint`.

    Args:
        output_dir: Directory where checkpoint files are written.
        monitor_regular: Metric key for the regular model mAP.
        monitor_ema: Metric key for the EMA model mAP.  ``None`` disables
            EMA tracking.
        run_test: If ``True``, run ``trainer.test()`` on the best model at
            the end of training.
    """

    def __init__(
        self,
        output_dir: str,
        monitor_regular: str = "val/mAP_50_95",
        monitor_ema: Optional[str] = None,
        run_test: bool = True,
    ) -> None:
        super().__init__()
        self._output_dir = Path(output_dir)
        self._monitor_regular = monitor_regular
        self._monitor_ema = monitor_ema
        self._run_test = run_test

        self._best_regular: float = 0.0
        self._best_ema: float = 0.0

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save best regular/EMA checkpoints when validation mAP improves.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModule`` being trained.
        """
        if not trainer.is_global_zero:
            return

        # --- Regular model ---
        regular_map = trainer.callback_metrics.get(self._monitor_regular, torch.tensor(0.0)).item()
        if regular_map > self._best_regular:
            self._best_regular = regular_map
            self._output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": pl_module.model.state_dict(),
                    "args": pl_module.train_config,
                    "epoch": trainer.current_epoch,
                },
                self._output_dir / "checkpoint_best_regular.pth",
            )
            logger.info(
                "Best regular mAP improved to %.4f (epoch %d)",
                regular_map,
                trainer.current_epoch,
            )

        # --- EMA model ---
        if self._monitor_ema is not None:
            ema_map = trainer.callback_metrics.get(self._monitor_ema, torch.tensor(0.0)).item()
            if ema_map > self._best_ema:
                self._best_ema = ema_map
                self._output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": pl_module.model.state_dict(),
                        "args": pl_module.train_config,
                        "epoch": trainer.current_epoch,
                    },
                    self._output_dir / "checkpoint_best_ema.pth",
                )
                logger.info(
                    "Best EMA mAP improved to %.4f (epoch %d)",
                    ema_map,
                    trainer.current_epoch,
                )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Select the overall best model and optionally run test evaluation.

        Copies the winner (regular vs EMA, strict ``>`` for EMA) to
        ``checkpoint_best_total.pth``, strips optimizer/scheduler state, then
        optionally runs ``trainer.test()``.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModule`` being trained.
        """
        if not trainer.is_global_zero:
            return

        regular_path = self._output_dir / "checkpoint_best_regular.pth"
        ema_path = self._output_dir / "checkpoint_best_ema.pth"
        total_path = self._output_dir / "checkpoint_best_total.pth"

        # Strict > for EMA to win (matches legacy behaviour)
        best_is_ema = self._best_ema > self._best_regular
        best_path = ema_path if (best_is_ema and ema_path.exists()) else regular_path

        if best_path.exists():
            shutil.copy2(best_path, total_path)
            strip_checkpoint(total_path)
            logger.info(
                "Best total checkpoint saved from %s (regular=%.4f, ema=%.4f)",
                "EMA" if best_is_ema else "regular",
                self._best_regular,
                self._best_ema,
            )

        if self._run_test:
            trainer.test(pl_module, datamodule=trainer.datamodule)


class RFDETREarlyStopping(Callback):
    """Early stopping callback monitoring validation mAP for RF-DETR.

    Mirrors the legacy :class:`rfdetr.util.early_stopping.EarlyStoppingCallback`
    but uses PTL's ``trainer.should_stop`` mechanism instead of
    ``model.request_early_stop()``.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum mAP improvement to reset the patience counter.
        use_ema: When ``True`` and both regular and EMA metrics are available,
            monitor only the EMA metric.  When ``False``, monitor the
            ``max(regular, ema)``.
        monitor_regular: Metric key for the regular model mAP.
        monitor_ema: Metric key for the EMA model mAP.
        verbose: If ``True``, log early stopping status each epoch.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        use_ema: bool = False,
        monitor_regular: str = "val/mAP_50_95",
        monitor_ema: str = "val/ema_mAP_50_95",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._patience = patience
        self._min_delta = min_delta
        self._use_ema = use_ema
        self._monitor_regular = monitor_regular
        self._monitor_ema = monitor_ema
        self._verbose = verbose

        self._best_map: float = 0.0
        self._counter: int = 0

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Check whether training should stop due to stalled mAP improvement.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModule`` being trained.
        """
        metrics = trainer.callback_metrics

        # Read available metrics
        regular_tensor = metrics.get(self._monitor_regular, None)
        ema_tensor = metrics.get(self._monitor_ema, None)

        regular_map: Optional[float] = regular_tensor.item() if regular_tensor is not None else None
        ema_map: Optional[float] = ema_tensor.item() if ema_tensor is not None else None

        # Determine current_map following legacy logic
        current_map: Optional[float] = None
        if regular_map is not None and ema_map is not None:
            current_map = ema_map if self._use_ema else max(regular_map, ema_map)
        elif ema_map is not None:
            current_map = ema_map
        elif regular_map is not None:
            current_map = regular_map
        else:
            # Neither metric available -- nothing to do
            return

        if current_map > self._best_map + self._min_delta:
            if self._verbose:
                logger.info(
                    "Early stopping: mAP improved %.4f -> %.4f",
                    self._best_map,
                    current_map,
                )
            self._best_map = current_map
            self._counter = 0
        else:
            self._counter += 1
            if self._verbose:
                logger.info(
                    "Early stopping: no improvement for %d/%d epochs (best=%.4f, current=%.4f)",
                    self._counter,
                    self._patience,
                    self._best_map,
                    current_map,
                )
            if self._counter >= self._patience:
                if self._verbose:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement above min_delta=%.4f",
                        self._patience,
                        self._min_delta,
                    )
                trainer.should_stop = True
