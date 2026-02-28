# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for MetricsPlotCallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

from rfdetr.lit.callbacks.metrics import PLOT_FILE_NAME, MetricsPlotCallback


def _make_trainer(
    metrics: dict[str, float],
    epoch: int = 0,
    is_global_zero: bool = True,
    sanity_checking: bool = False,
) -> MagicMock:
    """Build a minimal mock Trainer with the given callback_metrics.

    Args:
        metrics: Dict of metric name to float value.
        epoch: Current epoch number.
        is_global_zero: Whether this is the main process.
        sanity_checking: Whether the trainer is in sanity-check mode.

    Returns:
        A MagicMock configured to behave like a Trainer.
    """
    trainer = MagicMock()
    trainer.callback_metrics = {k: torch.tensor(float(v)) for k, v in metrics.items()}
    trainer.current_epoch = epoch
    trainer.is_global_zero = is_global_zero
    trainer.sanity_checking = sanity_checking
    return trainer


class TestMetricsPlotCallbackHistory:
    """Tests for on_validation_end history accumulation."""

    def test_history_empty_on_init(self, tmp_path: Path) -> None:
        """History list is empty before any hook fires."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        assert cb._history == []

    def test_on_validation_end_appends_entry(self, tmp_path: Path) -> None:
        """A single on_validation_end call appends exactly one entry."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50_95": 0.5}, epoch=0)
        cb.on_validation_end(trainer, MagicMock())
        assert len(cb._history) == 1

    def test_entry_contains_epoch(self, tmp_path: Path) -> None:
        """The recorded entry contains an 'epoch' key matching current_epoch."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50": 0.6}, epoch=3)
        cb.on_validation_end(trainer, MagicMock())
        assert cb._history[0]["epoch"] == 3.0

    def test_optional_keys_absent_when_not_in_metrics(self, tmp_path: Path) -> None:
        """EMA keys are absent from the entry when not in callback_metrics."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50": 0.4}, epoch=0)
        cb.on_validation_end(trainer, MagicMock())
        assert "ema_ap50_95" not in cb._history[0]

    def test_sanity_check_epoch_skipped(self, tmp_path: Path) -> None:
        """History stays empty when trainer.sanity_checking is True."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50": 0.5}, epoch=0, sanity_checking=True)
        cb.on_validation_end(trainer, MagicMock())
        assert cb._history == []

    def test_not_global_zero_skipped(self, tmp_path: Path) -> None:
        """History stays empty when is_global_zero is False."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50": 0.5}, epoch=0, is_global_zero=False)
        cb.on_validation_end(trainer, MagicMock())
        assert cb._history == []


class TestMetricsPlotCallbackSave:
    """Tests for on_fit_end plot saving."""

    def test_plot_file_created(self, tmp_path: Path) -> None:
        """After on_fit_end, metrics_plot.png exists in output_dir."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer(
            {
                "train/loss": 1.0,
                "val/loss": 0.8,
                "val/mAP_50_95": 0.3,
                "val/mAP_50": 0.5,
                "val/mAR": 0.4,
            },
            epoch=0,
        )
        cb.on_validation_end(trainer, MagicMock())
        cb.on_fit_end(trainer, MagicMock())
        assert (tmp_path / PLOT_FILE_NAME).exists()

    def test_save_skips_when_history_empty(self, tmp_path: Path) -> None:
        """No file created and no error when history is empty."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({}, epoch=0)
        cb.on_fit_end(trainer, MagicMock())
        assert not (tmp_path / PLOT_FILE_NAME).exists()

    def test_save_skips_when_not_global_zero(self, tmp_path: Path) -> None:
        """No file created when is_global_zero is False at on_fit_end."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        # Record data on main process first
        trainer_main = _make_trainer({"val/mAP_50": 0.5}, epoch=0, is_global_zero=True)
        cb.on_validation_end(trainer_main, MagicMock())
        # Then call on_fit_end from a non-main process
        trainer_worker = _make_trainer({}, epoch=0, is_global_zero=False)
        cb.on_fit_end(trainer_worker, MagicMock())
        assert not (tmp_path / PLOT_FILE_NAME).exists()

    def test_multiple_epochs_recorded(self, tmp_path: Path) -> None:
        """Three on_validation_end calls produce three history entries."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        for epoch in range(3):
            trainer = _make_trainer({"val/mAP_50": 0.3 + epoch * 0.1}, epoch=epoch)
            cb.on_validation_end(trainer, MagicMock())
        assert len(cb._history) == 3

    def test_plot_file_name_constant(self, tmp_path: Path) -> None:
        """Saved file is named exactly 'metrics_plot.png'."""
        cb = MetricsPlotCallback(output_dir=str(tmp_path))
        trainer = _make_trainer({"val/mAP_50": 0.5}, epoch=0)
        cb.on_validation_end(trainer, MagicMock())
        cb.on_fit_end(trainer, MagicMock())
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "metrics_plot.png"
