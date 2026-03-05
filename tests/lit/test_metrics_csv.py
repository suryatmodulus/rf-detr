# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests: metrics.csv contains all columns used by plot_metrics().

Runs a minimal PTL training loop (1 epoch, 2 batches each) using mocked model
internals so no real dataset or GPU is required.  After training, reads the
CSVLogger output and asserts that every metric column that ``plot_metrics()``
needs is present and has at least one non-NaN value.

Also verifies that ``train/loss`` is logged at the same scale as ``val/loss``
(i.e. NOT divided by ``grad_accum_steps`` before logging).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.utils.data

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.lit import build_trainer
from rfdetr.lit.datamodule import RFDETRDataModule
from rfdetr.lit.module import RFDETRModule

# ---------------------------------------------------------------------------
# Fakes (minimal, mirrors test_trainer_smoke.py)
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Single-parameter model so the optimizer and loss have a gradient path."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, samples, targets=None):
        return {"dummy": self.dummy}

    def update_drop_path(self, *a, **kw) -> None:
        pass

    def update_dropout(self, *a, **kw) -> None:
        pass

    def reinitialize_detection_head(self, *a, **kw) -> None:
        pass


class _FakeCriterion:
    """Criterion that returns a loss connected to the model output."""

    weight_dict = {"loss_ce": 1.0}

    def __call__(self, outputs, targets):
        dummy = outputs.get("dummy", torch.zeros(1))
        return {"loss_ce": dummy.mean()}


class _FakeDataset(torch.utils.data.Dataset):
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


def _fake_postprocess(outputs, orig_sizes):
    """Return a non-empty prediction so COCOEvalCallback has something to score."""
    n = orig_sizes.shape[0]
    return [
        {
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
        for _ in range(n)
    ]


def _make_param_dicts(model: nn.Module) -> list[dict]:
    return [{"params": p, "lr": 1e-4} for p in model.parameters() if p.requires_grad]


def _base_mc(**kw) -> RFDETRBaseConfig:
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(kw)
    return RFDETRBaseConfig(**defaults)


def _base_tc(tmp_path, **kw) -> TrainConfig:
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
    defaults.update(kw)
    return TrainConfig(**defaults)


def _fit_and_read_csv(mc: RFDETRBaseConfig, tc: TrainConfig, criterion=None) -> pd.DataFrame:
    """Run 1 epoch (2 train + 2 val batches) and return the resulting metrics.csv."""
    fake_criterion = criterion or _FakeCriterion()
    with (
        patch("rfdetr.lit.module.build_model", return_value=_TinyModel()),
        patch(
            "rfdetr.lit.module.build_criterion_and_postprocessors",
            return_value=(fake_criterion, MagicMock(side_effect=_fake_postprocess)),
        ),
        patch("rfdetr.lit.datamodule.build_dataset", return_value=_FakeDataset(length=20)),
        patch(
            "rfdetr.lit.module.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        ),
    ):
        module = RFDETRModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            log_every_n_steps=1,
        )
        trainer.fit(module, datamodule=datamodule)

    csv_path = Path(tc.output_dir) / "metrics.csv"
    assert csv_path.exists(), "CSVLogger must write metrics.csv to output_dir"
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Expected columns (must exist and have ≥1 non-NaN row after one epoch)
# ---------------------------------------------------------------------------

_REQUIRED_DETECTION = frozenset(
    {
        "train/loss",
        "val/loss",
        "val/mAP_50",
        "val/mAP_50_95",
        "val/mAR",
    }
)

_REQUIRED_DETECTION_EMA = _REQUIRED_DETECTION | frozenset(
    {
        "val/ema_mAP_50",
        "val/ema_mAP_50_95",
        "val/ema_mAR",
    }
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectionMetricsCSV:
    """metrics.csv contains all columns that plot_metrics() needs for detection."""

    def test_base_metrics_present_without_ema(self, tmp_path):
        """Without EMA all core val/* columns must appear in metrics.csv with non-NaN data."""
        mc = _base_mc()
        tc = _base_tc(tmp_path, use_ema=False, run_test=False)
        df = _fit_and_read_csv(mc, tc)

        missing = _REQUIRED_DETECTION - set(df.columns)
        assert not missing, f"Missing columns in metrics.csv: {sorted(missing)}"

        all_nan = {c for c in _REQUIRED_DETECTION if df[c].isna().all()}
        assert not all_nan, f"Columns with all-NaN values: {sorted(all_nan)}"

    def test_ema_metrics_present_with_ema_enabled(self, tmp_path):
        """With use_ema=True the ema_* aliases must also appear in metrics.csv."""
        mc = _base_mc()
        tc = _base_tc(tmp_path, use_ema=True, run_test=False)
        df = _fit_and_read_csv(mc, tc)

        missing = _REQUIRED_DETECTION_EMA - set(df.columns)
        assert not missing, f"Missing EMA columns in metrics.csv: {sorted(missing)}"

        all_nan = {c for c in _REQUIRED_DETECTION_EMA if df[c].isna().all()}
        assert not all_nan, f"EMA columns with all-NaN values: {sorted(all_nan)}"

    def test_train_loss_is_unscaled(self, tmp_path):
        """train/loss must be logged at the raw criterion scale, not divided by grad_accum_steps.

        With grad_accum_steps=4 the old code divided the logged value by 4,
        making train/loss ~4× smaller than val/loss.  After the fix the logged
        value equals the raw weighted criterion output so both losses are on the
        same scale.
        """
        FIXED_LOSS = 5.0
        GRAD_ACCUM = 4

        class _FixedCriterion:
            weight_dict = {"loss_ce": 1.0}

            def __call__(self, outputs, targets):
                # Loss is always FIXED_LOSS, connected to model params for gradient.
                dummy = outputs.get("dummy", torch.zeros(1))
                return {"loss_ce": dummy.mean() * 0 + FIXED_LOSS}

        mc = _base_mc()
        tc = _base_tc(tmp_path, use_ema=False, run_test=False, grad_accum_steps=GRAD_ACCUM)
        df = _fit_and_read_csv(mc, tc, criterion=_FixedCriterion())

        logged = df["train/loss"].dropna().mean()
        expected_unscaled = FIXED_LOSS
        expected_if_divided = FIXED_LOSS / GRAD_ACCUM

        assert abs(logged - expected_unscaled) < abs(logged - expected_if_divided), (
            f"train/loss={logged:.4f} is closer to the grad-accum-divided value "
            f"({expected_if_divided:.4f}) than the raw criterion output "
            f"({expected_unscaled:.4f}). The division must have been removed."
        )
