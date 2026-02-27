# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for _build_args_from_configs — the canonical config-to-Namespace mapping."""

import pytest

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.lit._args import _build_args_from_configs


def _base_model_config(**overrides):
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _base_train_config(tmp_path, **overrides):
    defaults = dict(
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        lr_drop=8,
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


@pytest.fixture
def base_model_config():
    return _base_model_config


@pytest.fixture
def base_train_config(tmp_path):
    return lambda **overrides: _base_train_config(tmp_path, **overrides)


@pytest.fixture
def seg_train_config(tmp_path):
    def _make(**overrides):
        defaults = dict(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=10,
            tensorboard=False,
        )
        defaults.update(overrides)
        return SegmentationTrainConfig(**defaults)

    return _make


class TestBuildArgsFromConfigs:
    """_build_args_from_configs maps ModelConfig + TrainConfig to a Namespace."""

    def test_returns_namespace(self, base_model_config, base_train_config):
        """The return value has attribute access (argparse.Namespace or similar)."""
        args = _build_args_from_configs(base_model_config(), base_train_config())
        assert hasattr(args, "encoder")

    def test_forwards_model_config_fields(self, base_model_config, base_train_config):
        """All key ModelConfig fields are faithfully mapped."""
        mc = base_model_config(num_classes=7)
        args = _build_args_from_configs(mc, base_train_config())

        assert args.encoder == mc.encoder
        assert args.num_classes == 7
        assert args.hidden_dim == mc.hidden_dim
        assert args.resolution == mc.resolution
        assert args.patch_size == mc.patch_size
        assert args.num_windows == mc.num_windows
        assert args.segmentation_head == mc.segmentation_head
        assert args.positional_encoding_size == mc.positional_encoding_size

    def test_forwards_train_config_fields(self, base_model_config, base_train_config):
        """All key TrainConfig fields are faithfully mapped."""
        tc = base_train_config(lr=3e-4, epochs=20, weight_decay=5e-5, batch_size=4, num_workers=0)
        args = _build_args_from_configs(base_model_config(), tc)

        assert args.lr == pytest.approx(3e-4)
        assert args.epochs == 20
        assert args.weight_decay == pytest.approx(5e-5)
        assert args.batch_size == 4
        assert args.num_workers == 0

    def test_forwards_dataset_fields(self, base_model_config, base_train_config):
        """Dataset-routing fields are forwarded to the Namespace."""
        tc = base_train_config(multi_scale=True, expanded_scales=True, dataset_file="coco")
        args = _build_args_from_configs(base_model_config(), tc)

        assert args.multi_scale is True
        assert args.expanded_scales is True
        assert args.dataset_file == "coco"

    def test_num_queries_from_subclass_config(self, base_train_config):
        """num_queries is read from subclass config attributes."""
        mc = _base_model_config()  # RFDETRBaseConfig has num_queries=300
        args = _build_args_from_configs(mc, base_train_config())
        assert args.num_queries == 300

    def test_resume_none_becomes_empty_string(self, base_model_config, base_train_config):
        """resume=None (the default) is converted to '' for the legacy Namespace."""
        tc = base_train_config()
        assert tc.resume is None
        args = _build_args_from_configs(base_model_config(), tc)
        assert args.resume == ""

    def test_segmentation_extras_forwarded_from_seg_config(self, base_model_config, seg_train_config):
        """SegmentationTrainConfig mask loss coefficients are forwarded."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        args = _build_args_from_configs(mc, tc)

        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)

    def test_segmentation_extras_default_for_plain_config(self, base_model_config, base_train_config):
        """mask_* attributes default to 5.0 for a plain TrainConfig (not segmentation)."""
        args = _build_args_from_configs(base_model_config(), base_train_config())
        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)

    def test_segmentation_head_flag_forwarded(self, base_train_config):
        """segmentation_head=True from ModelConfig reaches the Namespace."""
        mc = _base_model_config(segmentation_head=True)
        args = _build_args_from_configs(mc, base_train_config())
        assert args.segmentation_head is True
