# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Metrics plotting callback for RF-DETR Lightning training.

Replaces the legacy ``MetricsPlotSink`` from ``rfdetr.util.metrics`` with a
proper Lightning callback that reads from ``trainer.callback_metrics`` and
produces the same 2x2 matplotlib figure at the end of training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer

from rfdetr.util.logger import get_logger

logger = get_logger()

PLOT_FILE_NAME = "metrics_plot.png"

# Mapping from Lightning metric keys to internal history keys.
_METRIC_KEY_MAP: list[tuple[str, str]] = [
    ("train/loss", "train_loss"),
    ("val/loss", "val_loss"),
    ("val/mAP_50_95", "ap50_95"),
    ("val/mAP_50", "ap50"),
    ("val/mAR", "ar"),
    ("val/ema_mAP_50_95", "ema_ap50_95"),
    ("val/ema_mAP_50", "ema_ap50"),
    ("val/ema_mAR", "ema_ar"),
]


class MetricsPlotCallback(Callback):
    """Record per-epoch metrics and save a 2x2 summary plot at end of training.

    The callback listens to ``on_validation_end`` (which fires after all
    ``on_validation_epoch_end`` callbacks have run, so COCO metrics are
    available) and snapshots the current ``trainer.callback_metrics`` into
    an internal history list.  At ``on_fit_end`` the accumulated data is
    rendered into a matplotlib figure matching the legacy layout:

    - (0,0) Training and Validation Loss
    - (0,1) Average Precision @0.50
    - (1,0) Average Precision @0.50:0.95
    - (1,1) Average Recall @0.50:0.95

    Args:
        output_dir: Directory where ``metrics_plot.png`` will be saved.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self._output_dir: Path = Path(output_dir)
        self._history: list[dict[str, float]] = []

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Snapshot current epoch metrics into ``self._history``.

        Skips recording when running on a non-main process or during the
        sanity-check validation epoch.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The current LightningModule.
        """
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics

        def _get(key: str) -> Optional[float]:
            v = metrics.get(key)
            return float(v) if v is not None else None

        entry: dict[str, float] = {"epoch": float(trainer.current_epoch)}
        for src_key, dest_key in _METRIC_KEY_MAP:
            v = _get(src_key)
            if v is not None:
                entry[dest_key] = v

        self._history.append(entry)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save the metrics plot at the end of training.

        Only executes on the main process (``trainer.is_global_zero``).

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The current LightningModule.
        """
        if not trainer.is_global_zero:
            return
        self._save()

    def _save(self) -> None:
        """Render and write ``metrics_plot.png``.

        Produces a 2x2 matplotlib figure mirroring the legacy
        ``MetricsPlotSink`` layout.  Skips gracefully (with a warning log)
        if no data has been recorded.
        """
        if not self._history:
            logger.warning("No metrics data available to generate plot. Skipping.")
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.ioff()

        def _arr(key: str) -> np.ndarray:
            return np.array([h[key] for h in self._history if key in h], dtype=np.float32)

        epochs = _arr("epoch")
        train_loss = _arr("train_loss")
        val_loss = _arr("val_loss")
        ap50_95 = _arr("ap50_95")
        ap50 = _arr("ap50")
        ar = _arr("ar")
        ema_ap50_95 = _arr("ema_ap50_95")
        ema_ap50 = _arr("ema_ap50")
        ema_ar = _arr("ema_ar")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # (0,0) Loss
        if len(epochs) > 0:
            if train_loss.size > 0:
                axes[0][0].plot(
                    epochs[: len(train_loss)],
                    train_loss,
                    label="Training Loss",
                    marker="o",
                    linestyle="-",
                )
            if val_loss.size > 0:
                axes[0][0].plot(
                    epochs[: len(val_loss)],
                    val_loss,
                    label="Validation Loss",
                    marker="o",
                    linestyle="--",
                )
            axes[0][0].set_title("Training and Validation Loss")
            axes[0][0].set_xlabel("Epoch Number")
            axes[0][0].set_ylabel("Loss Value")
            axes[0][0].legend()
            axes[0][0].grid(True)

        # (0,1) AP50
        if ap50.size > 0 or ema_ap50.size > 0:
            if ap50.size > 0:
                axes[0][1].plot(
                    epochs[: len(ap50)],
                    ap50,
                    label="Base Model",
                    marker="o",
                    linestyle="-",
                )
            if ema_ap50.size > 0:
                axes[0][1].plot(
                    epochs[: len(ema_ap50)],
                    ema_ap50,
                    label="EMA Model",
                    marker="o",
                    linestyle="--",
                )
            axes[0][1].set_title("Average Precision @0.50")
            axes[0][1].set_xlabel("Epoch Number")
            axes[0][1].set_ylabel("AP50")
            axes[0][1].legend()
            axes[0][1].grid(True)

        # (1,0) AP50-95
        if ap50_95.size > 0 or ema_ap50_95.size > 0:
            if ap50_95.size > 0:
                axes[1][0].plot(
                    epochs[: len(ap50_95)],
                    ap50_95,
                    label="Base Model",
                    marker="o",
                    linestyle="-",
                )
            if ema_ap50_95.size > 0:
                axes[1][0].plot(
                    epochs[: len(ema_ap50_95)],
                    ema_ap50_95,
                    label="EMA Model",
                    marker="o",
                    linestyle="--",
                )
            axes[1][0].set_title("Average Precision @0.50:0.95")
            axes[1][0].set_xlabel("Epoch Number")
            axes[1][0].set_ylabel("AP")
            axes[1][0].legend()
            axes[1][0].grid(True)

        # (1,1) AR
        if ar.size > 0 or ema_ar.size > 0:
            if ar.size > 0:
                axes[1][1].plot(
                    epochs[: len(ar)],
                    ar,
                    label="Base Model",
                    marker="o",
                    linestyle="-",
                )
            if ema_ar.size > 0:
                axes[1][1].plot(
                    epochs[: len(ema_ar)],
                    ema_ar,
                    label="EMA Model",
                    marker="o",
                    linestyle="--",
                )
            axes[1][1].set_title("Average Recall @0.50:0.95")
            axes[1][1].set_xlabel("Epoch Number")
            axes[1][1].set_ylabel("AR")
            axes[1][1].legend()
            axes[1][1].grid(True)

        plt.tight_layout()
        out_path = self._output_dir / PLOT_FILE_NAME
        self._output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
        logger.info("Results saved to %s", out_path)
