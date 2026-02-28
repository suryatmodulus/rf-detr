# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for build_trainer() — PTL Ch3/T5 (callbacks) and Ch4/T1 (precision, loggers, trainer kwargs)."""

import warnings

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
    """Minimal TrainConfig for tests.

    Loggers are disabled by default to avoid requiring optional deps (tensorboard,
    wandb, mlflow) in the CPU test environment.  Logger-specific tests override these
    explicitly via kwargs or mocking.
    """
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
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
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
        )
        trainer = build_trainer(seg_tc, _mc(segmentation_head=True))
        assert isinstance(trainer, __import__("pytorch_lightning").Trainer)


class TestBuildTrainerPrecision:
    """build_trainer() must resolve training precision from model_config.amp + device caps."""

    def test_amp_false_gives_32_true(self, tmp_path):
        """amp=False always produces '32-true' regardless of device."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False))
        assert trainer.precision == "32-true"

    def test_amp_true_cpu_gives_32_true(self, tmp_path):
        """amp=True on CPU (no CUDA) must fall back to '32-true'."""
        import unittest.mock as mock

        with mock.patch("torch.cuda.is_available", return_value=False):
            trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert trainer.precision == "32-true"

    def test_amp_true_cuda_no_bf16_gives_16_mixed(self, tmp_path):
        """amp=True with CUDA but no bf16 support must produce '16-mixed'."""
        import unittest.mock as mock

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert trainer.precision == "16-mixed"

    def test_amp_true_cuda_bf16_gives_bf16_mixed(self, tmp_path):
        """amp=True with CUDA + bf16 support must produce 'bf16-mixed'."""
        import unittest.mock as mock

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))
        assert trainer.precision == "bf16-mixed"


class TestBuildTrainerEMAShardingGuard:
    """EMA must be disabled and a UserWarning emitted for sharded strategies.

    PTL validates strategy+accelerator compatibility at Trainer construction time,
    so tests that exercise sharded strategies mock Trainer to capture the callback
    list without triggering platform-specific validation.
    """

    @pytest.mark.parametrize(
        "strategy",
        [
            pytest.param("fsdp", id="fsdp"),
            pytest.param("deepspeed", id="deepspeed"),
            pytest.param("deepspeed_stage_2", id="deepspeed_stage_2"),
        ],
    )
    def test_ema_disabled_for_sharded_strategy(self, tmp_path, strategy):
        """EMA callback must be absent when a sharded strategy is requested."""
        import unittest.mock as mock

        tc = _tc(tmp_path, use_ema=True)
        # Inject strategy via monkey-patch (field not yet in TrainConfig until T4-2).
        tc.__dict__["strategy"] = strategy

        captured_callbacks = []

        def _fake_trainer(**kwargs):
            captured_callbacks.extend(kwargs.get("callbacks", []))
            return mock.MagicMock()

        with (
            mock.patch("rfdetr.lit.Trainer", side_effect=_fake_trainer),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            build_trainer(tc, _mc())

        types = [type(cb) for cb in captured_callbacks]
        assert RFDETREMACallback not in types

    def test_ema_sharding_emits_user_warning(self, tmp_path):
        """A UserWarning is emitted when EMA is requested with a sharded strategy."""
        import unittest.mock as mock

        tc = _tc(tmp_path, use_ema=True)
        tc.__dict__["strategy"] = "fsdp"

        with (
            mock.patch("rfdetr.lit.Trainer", return_value=mock.MagicMock()),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            build_trainer(tc, _mc())

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert any("EMA disabled" in str(w.message) for w in user_warns)

    def test_ema_enabled_for_non_sharded_strategy(self, tmp_path):
        """EMA callback must be present for non-sharded strategies."""
        trainer = build_trainer(_tc(tmp_path, use_ema=True), _mc())
        types = [type(cb) for cb in trainer.callbacks]
        assert RFDETREMACallback in types


class TestBuildTrainerLoggers:
    """build_trainer() must wire loggers from TrainConfig flags."""

    def test_no_loggers_produces_no_logger(self, tmp_path):
        """When all logger flags are off, Trainer must have no active logger."""
        trainer = build_trainer(
            _tc(tmp_path, use_ema=False),  # _tc already sets all loggers to False
            _mc(),
        )
        # PTL 2.6 returns None (not False) when logger=False is passed.
        assert not trainer.logger

    def test_tensorboard_logger_wired(self, tmp_path):
        """TensorBoardLogger is added when tensorboard=True (dep mocked)."""
        import unittest.mock as mock

        from pytorch_lightning.loggers import TensorBoardLogger

        fake_logger = mock.MagicMock(spec=TensorBoardLogger)
        with mock.patch("rfdetr.lit.TensorBoardLogger", return_value=fake_logger):
            trainer = build_trainer(
                _tc(tmp_path, tensorboard=True, use_ema=False),
                _mc(),
            )
        assert fake_logger in trainer.loggers

    def test_mlflow_logger_wired(self, tmp_path):
        """MLFlowLogger is added when mlflow=True (dep mocked)."""
        import unittest.mock as mock

        from pytorch_lightning.loggers import MLFlowLogger

        fake_logger = mock.MagicMock(spec=MLFlowLogger)
        with mock.patch("rfdetr.lit.MLFlowLogger", return_value=fake_logger):
            trainer = build_trainer(
                _tc(tmp_path, mlflow=True, use_ema=False),
                _mc(),
            )
        assert fake_logger in trainer.loggers

    def test_missing_tensorboard_dep_warns_not_crashes(self, tmp_path):
        """If tensorboard package is absent, a UserWarning is emitted and training continues."""
        import unittest.mock as mock

        with mock.patch("rfdetr.lit.TensorBoardLogger", side_effect=ModuleNotFoundError("no tensorboard")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trainer = build_trainer(
                    _tc(tmp_path, tensorboard=True, use_ema=False),
                    _mc(),
                )
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert any("TensorBoard" in str(w.message) for w in user_warns)
        assert not trainer.logger  # no logger wired despite flag=True

    def test_clearml_flag_emits_warning(self, tmp_path):
        """clearml=True must emit a UserWarning (no native PTL logger)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_trainer(
                _tc(tmp_path, clearml=True, use_ema=False),
                _mc(),
            )
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert any("ClearML" in str(w.message) for w in user_warns)

    def test_multiple_loggers_combined(self, tmp_path):
        """Multiple loggers can be wired simultaneously."""
        import unittest.mock as mock

        from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

        fake_tb = mock.MagicMock(spec=TensorBoardLogger)
        fake_mlflow = mock.MagicMock(spec=MLFlowLogger)
        with (
            mock.patch("rfdetr.lit.TensorBoardLogger", return_value=fake_tb),
            mock.patch("rfdetr.lit.MLFlowLogger", return_value=fake_mlflow),
        ):
            trainer = build_trainer(
                _tc(tmp_path, tensorboard=True, mlflow=True, use_ema=False),
                _mc(),
            )
        assert fake_tb in trainer.loggers
        assert fake_mlflow in trainer.loggers


class TestBuildTrainerKwargs:
    """build_trainer() must pass the correct kwargs to Trainer."""

    def test_gradient_clip_val_default(self, tmp_path):
        """gradient_clip_val defaults to 0.1 when clip_max_norm is not yet in TrainConfig."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        assert trainer.gradient_clip_val == pytest.approx(0.1)

    def test_accumulate_grad_batches(self, tmp_path):
        """accumulate_grad_batches maps from grad_accum_steps."""
        trainer = build_trainer(_tc(tmp_path, grad_accum_steps=8, use_ema=False), _mc())
        assert trainer.accumulate_grad_batches == 8

    def test_max_epochs(self, tmp_path):
        """max_epochs maps from config.epochs."""
        trainer = build_trainer(_tc(tmp_path, epochs=42, use_ema=False), _mc())
        assert trainer.max_epochs == 42

    def test_log_every_n_steps(self, tmp_path):
        """log_every_n_steps is fixed at 50."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc())
        assert trainer.log_every_n_steps == 50

    def test_default_root_dir(self, tmp_path):
        """default_root_dir maps from config.output_dir."""
        out = str(tmp_path / "my_output")
        trainer = build_trainer(_tc(tmp_path, output_dir=out, use_ema=False), _mc())
        assert str(trainer.default_root_dir) == out


class TestBuildTrainerSeed:
    """build_trainer() must call seed_everything when seed is set."""

    def test_seed_is_applied(self, tmp_path):
        """seed_everything is called when a seed is present on the config."""
        import unittest.mock as mock

        tc = _tc(tmp_path, use_ema=False)
        tc.__dict__["seed"] = 42  # injected until T4-2 promotes the field

        with mock.patch("rfdetr.lit.seed_everything") as mock_seed:
            build_trainer(tc, _mc())
        mock_seed.assert_called_once_with(42, workers=True)

    def test_no_seed_skips_seed_everything(self, tmp_path):
        """seed_everything is not called when seed is None (default)."""
        import unittest.mock as mock

        with mock.patch("rfdetr.lit.seed_everything") as mock_seed:
            build_trainer(_tc(tmp_path, use_ema=False), _mc())
        mock_seed.assert_not_called()
