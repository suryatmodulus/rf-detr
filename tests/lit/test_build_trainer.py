# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for build_trainer() stub — PTL Ch3/T5."""

import pytest

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.lit import build_trainer
from rfdetr.lit.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr.lit.callbacks.coco_eval import COCOEvalCallback
from rfdetr.lit.callbacks.drop_schedule import DropPathCallback
from rfdetr.lit.callbacks.ema import RFDETREMACallback
from rfdetr.lit.callbacks.metrics import MetricsPlotCallback


def _mc(**kwargs):
    """Minimal RFDETRBaseConfig for tests."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(kwargs)
    return RFDETRBaseConfig(**defaults)


def _tc(tmp_path, **kwargs):
    """Minimal TrainConfig for tests."""
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


class TestBuildTrainerReturnType:
    """build_trainer() must return a PTL Trainer."""

    def test_returns_trainer_instance(self, tmp_path):
        """Return value must be a pytorch_lightning.Trainer."""
        from pytorch_lightning import Trainer

        trainer = build_trainer(_tc(tmp_path), _mc())
        assert isinstance(trainer, Trainer)


class TestBuildTrainerCallbacks:
    """build_trainer() must wire the correct callback set."""

    def test_coco_eval_always_present(self, tmp_path):
        """COCOEvalCallback is always included regardless of config flags."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False, early_stopping=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert COCOEvalCallback in types

    def test_best_model_always_present(self, tmp_path):
        """BestModelCallback is always included."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert BestModelCallback in types

    def test_metrics_plot_always_present(self, tmp_path):
        """MetricsPlotCallback is always included."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert MetricsPlotCallback in types

    def test_ema_callback_when_use_ema_true(self, tmp_path):
        """RFDETREMACallback is added when use_ema=True."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREMACallback in types

    def test_no_ema_callback_when_use_ema_false(self, tmp_path):
        """RFDETREMACallback is absent when use_ema=False."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREMACallback not in types

    def test_drop_path_callback_when_drop_path_nonzero(self, tmp_path):
        """DropPathCallback is added when drop_path > 0."""
        trainer = build_trainer(_tc(tmp_path, drop_path=0.1), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert DropPathCallback in types

    def test_no_drop_path_callback_when_drop_path_zero(self, tmp_path):
        """DropPathCallback is absent when drop_path == 0."""
        trainer = build_trainer(_tc(tmp_path, drop_path=0.0), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert DropPathCallback not in types

    def test_early_stopping_when_enabled(self, tmp_path):
        """RFDETREarlyStopping is added when early_stopping=True."""
        trainer = build_trainer(_tc(tmp_path, early_stopping=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREarlyStopping in types

    def test_no_early_stopping_when_disabled(self, tmp_path):
        """RFDETREarlyStopping is absent when early_stopping=False."""
        trainer = build_trainer(_tc(tmp_path, early_stopping=False), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREarlyStopping not in types

    def test_segmentation_config_accepted(self, tmp_path):
        """SegmentationTrainConfig is accepted without error."""
        seg_tc = SegmentationTrainConfig(
            dataset_dir=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            num_workers=0,
        )
        trainer = build_trainer(seg_tc, _mc(segmentation_head=True))
        assert isinstance(trainer, __import__("pytorch_lightning").Trainer)
