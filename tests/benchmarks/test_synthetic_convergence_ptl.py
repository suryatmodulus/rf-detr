# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PTL-native convergence test: train_ptl() + compat.evaluate throughout.

Unlike :mod:`test_synthetic_convergence`, which evaluates via the legacy
``engine.evaluate`` path, this test uses only PTL-native APIs:

* :meth:`~rfdetr.detr.RFDETR.train_ptl` for training.
* :func:`~rfdetr.lit.compat.evaluate` (wrapping a temporary
  :class:`~rfdetr.lit.module.RFDETRModule`) for evaluation.

This avoids manually wiring ``criterion``, ``postprocess``, and ``args``,
reducing boilerplate while keeping the same correctness guarantees.
"""
import json
import math
import os
from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningModule

from rfdetr import RFDETRNano
from rfdetr.config import TrainConfig
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.lit.compat import evaluate as compat_evaluate
from rfdetr.lit.module import RFDETRModule
from rfdetr.main import populate_args
from rfdetr.util import misc as utils


def _make_ptl_module_from(rfdetr_obj, dataset_dir: Path, output_dir: Path) -> RFDETRModule:
    """Build a temporary RFDETRModule from a trained/untrained RFDETR instance.

    Creates an :class:`~rfdetr.lit.module.RFDETRModule` with the same
    architecture as *rfdetr_obj*, then copies its current weights (from
    ``rfdetr_obj.model.model``).  No pretrain weights are downloaded; the
    module's weights are entirely derived from *rfdetr_obj*.

    Args:
        rfdetr_obj: A (possibly trained) :class:`~rfdetr.detr.RFDETR` instance.
        dataset_dir: Dataset directory forwarded to :class:`~rfdetr.config.TrainConfig`.
        output_dir: Output directory forwarded to :class:`~rfdetr.config.TrainConfig`.

    Returns:
        A weight-synced :class:`~rfdetr.lit.module.RFDETRModule` ready for
        ``compat.evaluate``.
    """
    train_config = TrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
    )
    module = RFDETRModule(rfdetr_obj.model_config, train_config)
    module.model.load_state_dict(rfdetr_obj.model.model.state_dict())
    module.model.eval()

    # Guarantee the returned object is a genuine PTL LightningModule so that
    # callers can trust they are using the PTL evaluation path.
    assert isinstance(module, RFDETRModule), (
        f"Expected RFDETRModule, got {type(module).__name__}"
    )
    assert isinstance(module, LightningModule), (
        "Module must be a pytorch_lightning.LightningModule"
    )
    return module


@pytest.mark.gpu
@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_ptl_training_improves_performance(
    tmp_path: Path,
    synthetic_shape_dataset_dir: Path,
) -> None:
    """Verify that train_ptl() improves model performance on synthetic data.

    Mirrors :func:`test_synthetic_training_improves_performance` but:

    * Drives training via :meth:`~rfdetr.detr.RFDETR.train_ptl`.
    * Evaluates via :func:`~rfdetr.lit.compat.evaluate` (PTL path) rather
      than the legacy ``engine.evaluate``.

    Same thresholds (mAP >= 35 %, F1 >= 35 %, loss drop to 70 %) apply.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = synthetic_shape_dataset_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFDETRNano(pretrain_weights=None, num_classes=4, device=str(device))

    # Build args only for dataloader construction — shared between pre and
    # post evaluation so both use the identical data split and resolution.
    args = populate_args(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        class_names=["square", "triangle", "circle"],
        batch_size=4,
        grad_accum_steps=1,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
        device=str(device),
        amp=False,
        use_ema=True,
        square_resize_div_64=True,
        epochs=10,
    )

    train_dataset = build_dataset(image_set="train", args=args, resolution=args.resolution)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        sampler=torch.utils.data.SequentialSampler(train_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    train_ds = get_coco_api_from_dataset(train_dataset)

    val_dataset = build_dataset(image_set="val", args=args, resolution=args.resolution)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        args.batch_size,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)

    # ------------------------------------------------------------------
    # Pre-training baseline via PTL compat path.
    # ------------------------------------------------------------------
    ptl_pre = _make_ptl_module_from(model, dataset_dir, output_dir)

    # Confirm evaluation uses the PTL module, not the raw nn.Module directly.
    _spot_key = next(iter(model.model.model.state_dict()))
    assert torch.equal(
        model.model.model.state_dict()[_spot_key],
        ptl_pre.model.state_dict()[_spot_key],
    ), f"Pre-training weight copy failed: '{_spot_key}' differs"

    with torch.no_grad():
        base_stats_val, _ = compat_evaluate(ptl_pre, val_loader, base_ds, device)
        base_stats_train, _ = compat_evaluate(ptl_pre, train_loader, train_ds, device)
    del ptl_pre  # free memory before training

    Path(output_dir / "base_stats_val.json").write_text(json.dumps(base_stats_val, indent=2))
    Path(output_dir / "base_stats_train.json").write_text(json.dumps(base_stats_train, indent=2))
    base_map = base_stats_val["results_json"]["map"]
    base_f1_score = base_stats_val["results_json"]["f1_score"]
    base_loss_bbox = base_stats_train["loss_bbox"]
    base_loss_giou = base_stats_train["loss_giou"]

    assert math.isfinite(base_loss_bbox), f"Base loss {base_loss_bbox:.3f} must be finite."
    assert math.isfinite(base_loss_giou), f"Base loss {base_loss_giou:.3f} must be finite."
    assert math.isfinite(base_map), f"Base mAP50 {base_map:.3f} must be finite."
    assert math.isfinite(base_f1_score), f"Base F1 {base_f1_score:.3f} must be finite."
    assert base_map <= 0.05, f"Base mAP50 {base_map:.3f} should be low before training."
    assert base_f1_score <= 0.05, f"Base F1 {base_f1_score:.3f} should be low before training."

    # ------------------------------------------------------------------
    # Train via the PyTorch Lightning stack.
    # ------------------------------------------------------------------
    model.train_ptl(
        dataset_file="roboflow",
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=10,
        batch_size=4,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
        lr=1e-3,
        warmup_epochs=1.0,
        use_ema=True,
        multi_scale=False,
        run_test=False,
        device=str(device),
    )

    # ------------------------------------------------------------------
    # Post-training evaluation via PTL compat path.
    # model.model.model has been updated by train_ptl() via the sync-back.
    # ------------------------------------------------------------------
    ptl_post = _make_ptl_module_from(model, dataset_dir, output_dir)

    # Confirm the post-training module carries the trained weights (not the
    # original random init), by checking the spot-key changed after training.
    assert not torch.equal(
        ptl_pre.model.state_dict()[_spot_key],
        ptl_post.model.state_dict()[_spot_key],
    ), (
        f"Post-training weights at '{_spot_key}' are identical to pre-training — "
        "train_ptl() may not have updated the model."
    )
    # And that ptl_post matches the trained model exactly.
    assert torch.equal(
        model.model.model.state_dict()[_spot_key],
        ptl_post.model.state_dict()[_spot_key],
    ), f"Post-training weight sync failed: '{_spot_key}' differs between RFDETR and PTL module"

    with torch.no_grad():
        final_stats_val, _ = compat_evaluate(ptl_post, val_loader, base_ds, device)
        final_stats_train, _ = compat_evaluate(ptl_post, train_loader, train_ds, device)

    Path(output_dir / "final_stats_val.json").write_text(json.dumps(final_stats_val, indent=2))
    Path(output_dir / "final_stats_train.json").write_text(json.dumps(final_stats_train, indent=2))
    final_map = final_stats_val["results_json"]["map"]
    final_f1_score = final_stats_val["results_json"]["f1_score"]
    final_loss_bbox = final_stats_train["loss_bbox"]
    final_loss_giou = final_stats_train["loss_giou"]

    threshold_map = 0.35
    threshold_f1_score = 0.35
    threshold_loss = 0.7
    assert math.isfinite(final_loss_bbox), f"Final loss {final_loss_bbox:.3f} must be finite."
    assert math.isfinite(final_loss_giou), f"Final loss {final_loss_giou:.3f} must be finite."
    assert math.isfinite(final_map), f"Final mAP {final_map:.3f} must be finite."
    assert math.isfinite(final_f1_score), f"Final F1 {final_f1_score:.3f} must be finite."
    assert final_map >= threshold_map, (
        f"Final mAP {final_map:.3f} should reach at least {threshold_map} after PTL training."
    )
    assert final_f1_score >= threshold_f1_score, (
        f"Final F1 {final_f1_score:.3f} should reach at least {threshold_f1_score} after PTL training."
    )
    assert final_loss_bbox <= base_loss_bbox * threshold_loss, (
        f"Loss {base_loss_bbox:.3f} -> {final_loss_bbox:.3f} should drop to at least {threshold_loss * 100}%."
    )
    assert final_loss_giou <= base_loss_giou * threshold_loss, (
        f"Loss {base_loss_giou:.3f} -> {final_loss_giou:.3f} should drop to at least {threshold_loss * 100}%."
    )
