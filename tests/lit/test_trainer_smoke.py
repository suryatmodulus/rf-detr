# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests: Trainer(fast_dev_run=2).fit(module, datamodule) — T7.

Verifies that the PTL training loop runs end-to-end without error for both
detection and segmentation configurations.  All heavy operations (build_model,
build_criterion_and_postprocessors, build_dataset, get_param_dict) are patched
so no real dataset or GPU is required.

Chapter 1 gate: these must pass before Chapter 2 begins.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.utils.data
from pytorch_lightning import Trainer

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.lit import build_trainer
from rfdetr.lit.datamodule import RFDETRDataModule
from rfdetr.lit.module import RFDETRModule

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _base_mc(**overrides):
    """Return a minimal RFDETRBaseConfig with pretrain_weights disabled."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _base_tc(tmp_path, **overrides):
    """Return a minimal detection TrainConfig suitable for smoke tests."""
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        drop_path=0.0,
        num_workers=0,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _base_seg_tc(tmp_path, **overrides):
    """Return a minimal SegmentationTrainConfig suitable for smoke tests."""
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        drop_path=0.0,
        num_workers=0,
        tensorboard=False,
    )
    defaults.update(overrides)
    return SegmentationTrainConfig(**defaults)


class _TinyModel(nn.Module):
    """Minimal real nn.Module that satisfies the RFDETRModule model contract.

    Has a single trainable parameter so the optimizer has something to update
    and the loss has a gradient path back through the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, samples, targets=None):
        return {"dummy": self.dummy}

    def update_drop_path(self, *args, **kwargs) -> None:
        pass

    def update_dropout(self, *args, **kwargs) -> None:
        pass

    def reinitialize_detection_head(self, *args, **kwargs) -> None:
        pass


class _FakeCriterion:
    """Callable criterion that returns a loss connected to the model output.

    Keeps a gradient path from the loss back to _TinyModel.dummy so that
    loss.backward() does not error when the Trainer calls it.
    """

    weight_dict = {"loss_ce": 1.0}

    def __call__(self, outputs, targets):
        dummy = outputs.get("dummy", torch.zeros(1))
        return {"loss_ce": dummy.mean()}


class _FakeDataset(torch.utils.data.Dataset):
    """Dataset with a working __getitem__ that returns (image, target) pairs.

    The image is a (3, 32, 32) float tensor; the target dict includes the
    fields expected by RFDETRModule (boxes, labels, image_id, orig_size).
    """

    def __init__(self, length: int = 20) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([32, 32]),
            "size": torch.tensor([32, 32]),
        }
        return image, target


class _FakeDatasetWithMasks(_FakeDataset):
    """Like _FakeDataset but includes binary instance masks (for segmentation)."""

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        target["masks"] = torch.zeros(1, 32, 32, dtype=torch.bool)
        return image, target


def _fake_postprocess(outputs, orig_sizes):
    """Postprocessor returning empty detection lists — one dict per image."""
    n = orig_sizes.shape[0]
    return [
        {
            "boxes": torch.zeros(0, 4),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        for _ in range(n)
    ]


def _make_param_dicts(model: nn.Module):
    """Build a minimal param-dict list for AdamW from all trainable parameters."""
    return [{"params": p, "lr": 1e-4} for p in model.parameters() if p.requires_grad]


def _make_trainer() -> Trainer:
    """Create a Trainer configured for minimal smoke-test runs."""
    return Trainer(
        fast_dev_run=2,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )


# ---------------------------------------------------------------------------
# Smoke test classes
# ---------------------------------------------------------------------------


class TestDetectionSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for detection."""

    def test_fit_runs_without_error(self, tmp_path):
        """Full PTL fit loop runs 2 train + 2 val batches without raising."""
        mc = _base_mc()
        tc = _base_tc(tmp_path)

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.lit.module.build_model", return_value=tiny_model),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_training_step_called_expected_times(self, tmp_path):
        """fast_dev_run=2 must run exactly 2 training steps."""
        mc = _base_mc()
        tc = _base_tc(tmp_path)

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.lit.module.build_model", return_value=tiny_model),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            original_training_step = module.training_step
            call_count = []

            def _counting_training_step(batch, batch_idx):
                call_count.append(1)
                return original_training_step(batch, batch_idx)

            module.training_step = _counting_training_step
            _make_trainer().fit(module, datamodule)

        assert sum(call_count) == 2

    def test_val_step_called_expected_times(self, tmp_path):
        """fast_dev_run=2 must run exactly 2 validation steps."""
        mc = _base_mc()
        tc = _base_tc(tmp_path)

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.lit.module.build_model", return_value=tiny_model),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            original_validation_step = module.validation_step
            call_count = []

            def _counting_val_step(batch, batch_idx):
                call_count.append(1)
                return original_validation_step(batch, batch_idx)

            module.validation_step = _counting_val_step
            _make_trainer().fit(module, datamodule)

        assert sum(call_count) == 2

    def test_loss_decreases_or_is_finite(self, tmp_path):
        """Training loss must be finite (not NaN/inf) for the run to be valid."""
        mc = _base_mc()
        tc = _base_tc(tmp_path)

        tiny_model = _TinyModel()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        losses = []

        def _recording_criterion(outputs, targets):
            dummy = outputs.get("dummy", torch.zeros(1))
            loss = dummy.mean()
            losses.append(loss.detach().item())
            return {"loss_ce": loss}

        fake_criterion = MagicMock(side_effect=_recording_criterion)
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        with (
            patch("rfdetr.lit.module.build_model", return_value=tiny_model),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

        assert len(losses) > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in losses)


class TestSegmentationSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for segmentation."""

    def test_fit_runs_without_error(self, tmp_path):
        """Full PTL fit loop runs 2 train + 2 val batches without raising."""
        mc = _base_mc(segmentation_head=True)
        tc = _base_seg_tc(tmp_path)

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDatasetWithMasks(length=20)

        with (
            patch("rfdetr.lit.module.build_model", return_value=tiny_model),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_segmentation_config_accepted(self, tmp_path):
        """SegmentationTrainConfig must be accepted by both module and datamodule."""
        mc = _base_mc(segmentation_head=True)
        tc = _base_seg_tc(tmp_path)

        with (
            patch("rfdetr.lit.module.build_model", return_value=_TinyModel()),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=_FakeDatasetWithMasks()),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            assert isinstance(module.train_config, SegmentationTrainConfig)
            assert isinstance(datamodule.train_config, SegmentationTrainConfig)


class TestBuildTrainerSmoke:
    """Smoke tests for the ``build_trainer()`` public factory.

    Verifies that the full callback stack wired by ``build_trainer`` runs
    end-to-end with ``fast_dev_run``, using mocked internals so no real
    dataset or GPU is required.
    """

    def test_fit_via_build_trainer(self, tmp_path):
        """build_trainer() + trainer.fit(module, datamodule=datamodule) must not raise."""
        mc = _base_mc()
        tc = _base_tc(tmp_path, use_ema=False, run_test=False)

        with (
            patch("rfdetr.lit.module.build_model", return_value=_TinyModel()),
            patch(
                "rfdetr.lit.module.build_criterion_and_postprocessors",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.lit.datamodule.build_dataset", return_value=_FakeDataset(length=20)),
            patch(
                "rfdetr.lit.module.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
            trainer.fit(module, datamodule=datamodule)
