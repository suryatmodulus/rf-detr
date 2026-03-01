# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PTL parity tests: RFDETRModule evaluation must match legacy engine.evaluate.

For every detection and segmentation model variant, this module:

1. Loads pretrained weights via the legacy ``RFDETR`` wrapper.
2. Evaluates via the legacy :func:`rfdetr.engine.evaluate` path (baseline).
3. Copies the same weights into a fresh :class:`~rfdetr.lit.module.RFDETRModule`.
4. Asserts the module is a genuine ``pytorch_lightning.LightningModule``.
5. Evaluates via :func:`rfdetr.lit.compat.evaluate` (PTL path).
6. Asserts numerical parity between the two paths (< 1e-4).
7. Asserts the PTL path meets the same COCO baseline thresholds as the
   original :mod:`test_coco_inference` benchmark.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch
from pytorch_lightning import LightningModule

from rfdetr import (
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)
from rfdetr.config import TrainConfig
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.detr import RFDETR
from rfdetr.engine import evaluate as engine_evaluate
from rfdetr.lit.compat import evaluate as compat_evaluate
from rfdetr.lit.module import RFDETRModule
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.misc import collate_fn


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _build_ptl_module(rfdetr_obj: RFDETR, tmp_path: Path) -> RFDETRModule:
    """Copy pretrained weights from *rfdetr_obj* into a fresh RFDETRModule.

    Builds an :class:`~rfdetr.lit.module.RFDETRModule` with the same
    architecture as *rfdetr_obj* (no pretrain download), loads the weights
    from ``rfdetr_obj.model.model``, and runs PTL-identity assertions before
    returning.

    Args:
        rfdetr_obj: A pretrained :class:`~rfdetr.detr.RFDETR` instance.
        tmp_path: Temporary directory used as ``output_dir`` in TrainConfig.

    Returns:
        Weight-synced :class:`~rfdetr.lit.module.RFDETRModule` ready for
        :func:`rfdetr.lit.compat.evaluate`.
    """
    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path),
    )
    ptl_module = RFDETRModule(rfdetr_obj.model_config, train_config)
    ptl_module.model.load_state_dict(rfdetr_obj.model.model.state_dict())
    ptl_module.model.eval()

    # Assert PTL lineage — the evaluated object must be a genuine
    # LightningModule, not a raw nn.Module or a duck-typed stand-in.
    assert isinstance(ptl_module, RFDETRModule), (
        f"Expected RFDETRModule, got {type(ptl_module).__name__}"
    )
    assert isinstance(ptl_module, LightningModule), (
        "ptl_module must be a pytorch_lightning.LightningModule — "
        "this confirms evaluation runs through the PTL stack"
    )

    # Spot-check that the weight copy succeeded.
    _first_key = next(iter(rfdetr_obj.model.model.state_dict()))
    assert torch.equal(
        rfdetr_obj.model.model.state_dict()[_first_key],
        ptl_module.model.state_dict()[_first_key],
    ), f"Weight copy failed: '{_first_key}' differs between legacy model and PTL module"

    return ptl_module


# ---------------------------------------------------------------------------
# Detection benchmark
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRNano, 0.67, 0.66, None, 6, id="nano"),
        pytest.param(RFDETRSmall, 0.72, 0.70, 500, 6, id="small"),
        pytest.param(RFDETRMedium, 0.73, 0.71, 500, 4, id="medium"),
        pytest.param(RFDETRLarge, 0.74, 0.72, 500, 2, id="large"),
    ],
)
def test_ptl_detection_module_matches_legacy(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: Optional[int],
    batch_size: int,
) -> None:
    """PTL-loaded detection model must match legacy engine.evaluate on COCO val2017.

    Same thresholds and parametrization as
    :func:`test_coco_detection_inference_benchmark`.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    images_root, annotations_path = download_coco_val

    # 1. Load pretrained model via legacy API.
    rfdetr = model_cls(device=device_str)
    config = rfdetr.model_config
    args = rfdetr.model.args
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500
    if not hasattr(args, "fp16_eval"):
        args.fp16_eval = False
    if not hasattr(args, "segmentation_head"):
        args.segmentation_head = False

    criterion, postprocess = build_criterion_and_postprocessors(args)

    # 2. Build shared COCO val dataloader.
    transforms = make_coco_transforms_square_div_64(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    val_dataset = CocoDetection(images_root, annotations_path, transforms=transforms)
    if num_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(num_samples, len(val_dataset)))))
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 1,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)

    # 3. Evaluate via legacy engine.evaluate (baseline).
    rfdetr.model.model.eval()
    with torch.no_grad():
        legacy_stats, _ = engine_evaluate(
            rfdetr.model.model, criterion, postprocess, data_loader, base_ds, device, args=args
        )

    legacy_map = legacy_stats["results_json"]["map"]
    legacy_f1 = legacy_stats["results_json"]["f1_score"]

    # 4. Build PTL module from same weights.
    ptl_module = _build_ptl_module(rfdetr, tmp_path)

    # 5. Evaluate via compat.evaluate (PTL path).
    with torch.no_grad():
        ptl_stats, _ = compat_evaluate(ptl_module, data_loader, base_ds, device)

    ptl_map = ptl_stats["results_json"]["map"]
    ptl_f1 = ptl_stats["results_json"]["f1_score"]

    # Debug dump.
    test_id = request.node.callspec.id
    debug_dir = os.environ.get("COCO_BENCHMARK_DEBUG_DIR", tempfile.gettempdir())
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    for name, stats in [("legacy", legacy_stats), ("ptl", ptl_stats)]:
        path = Path(debug_dir) / f"ptl_detection_{test_id}_{name}.json"
        path.write_text(json.dumps(stats, indent=2))

    print(f"[{test_id}] legacy: mAP@50={legacy_map:.4f} F1={legacy_f1:.4f}")
    print(f"[{test_id}] ptl:    mAP@50={ptl_map:.4f} F1={ptl_f1:.4f}")

    # 6. Parity: same weights must produce identical metrics.
    assert abs(ptl_map - legacy_map) < 1e-4, (
        f"PTL mAP@50 {ptl_map:.6f} differs from legacy {legacy_map:.6f} by > 1e-4"
    )

    # 7. Baseline thresholds (identical to test_coco_detection_inference_benchmark).
    assert ptl_map >= threshold_map, f"PTL mAP@50 {ptl_map:.4f} < {threshold_map}"
    assert ptl_f1 >= threshold_f1, f"PTL F1 {ptl_f1:.4f} < {threshold_f1}"


# ---------------------------------------------------------------------------
# Segmentation benchmark
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_segm_map", "threshold_segm_f1", "num_samples", "batch_size"),
    [
        pytest.param(RFDETRSegNano, 0.63, 0.64, 500, 6, id="nano"),
        pytest.param(RFDETRSegSmall, 0.66, 0.67, 100, 6, id="small"),
        pytest.param(RFDETRSegMedium, 0.68, 0.68, 100, 4, id="medium"),
        pytest.param(RFDETRSegLarge, 0.70, 0.69, 100, 2, id="large"),
        pytest.param(RFDETRSegXLarge, 0.72, 0.70, 100, 2, id="xlarge"),
        pytest.param(RFDETRSeg2XLarge, 0.73, 0.71, 100, 2, id="2xlarge"),
    ],
)
def test_ptl_segmentation_module_matches_legacy(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_segm_map: float,
    threshold_segm_f1: float,
    num_samples: Optional[int],
    batch_size: int,
) -> None:
    """PTL-loaded segmentation model must match legacy engine.evaluate on COCO val2017.

    Same thresholds and parametrization as
    :func:`test_coco_segmentation_inference_benchmark`.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    images_root, annotations_path = download_coco_val

    # 1. Load pretrained model via legacy API.
    rfdetr = model_cls(device=device_str)
    config = rfdetr.model_config
    args = rfdetr.model.args
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500
    if not hasattr(args, "fp16_eval"):
        args.fp16_eval = False
    if not hasattr(args, "mask_ce_loss_coef"):
        args.mask_ce_loss_coef = 5.0
    if not hasattr(args, "mask_dice_loss_coef"):
        args.mask_dice_loss_coef = 5.0
    if not hasattr(args, "mask_point_sample_ratio"):
        args.mask_point_sample_ratio = 16

    criterion, postprocess = build_criterion_and_postprocessors(args)

    # 2. Build shared COCO val dataloader with mask loading enabled.
    transforms = make_coco_transforms_square_div_64(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    val_dataset = CocoDetection(images_root, annotations_path, transforms=transforms, include_masks=True)
    if num_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(num_samples, len(val_dataset)))))
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 1,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)

    # 3. Evaluate via legacy engine.evaluate (baseline).
    rfdetr.model.model.eval()
    with torch.no_grad():
        legacy_stats, _ = engine_evaluate(
            rfdetr.model.model, criterion, postprocess, data_loader, base_ds, device, args=args
        )

    legacy_segm_map = legacy_stats["results_json_masks"]["map"]
    legacy_segm_f1 = legacy_stats["results_json_masks"]["f1_score"]

    # 4. Build PTL module from same weights.
    ptl_module = _build_ptl_module(rfdetr, tmp_path)

    # 5. Evaluate via compat.evaluate (PTL path).
    with torch.no_grad():
        ptl_stats, _ = compat_evaluate(ptl_module, data_loader, base_ds, device)

    ptl_segm_map = ptl_stats["results_json_masks"]["map"]
    ptl_segm_f1 = ptl_stats["results_json_masks"]["f1_score"]

    # Debug dump.
    test_id = request.node.callspec.id
    debug_dir = os.environ.get("COCO_BENCHMARK_DEBUG_DIR", tempfile.gettempdir())
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    for name, stats in [("legacy", legacy_stats), ("ptl", ptl_stats)]:
        path = Path(debug_dir) / f"ptl_segmentation_{test_id}_{name}.json"
        path.write_text(json.dumps(stats, indent=2))

    print(f"[{test_id}] legacy: segm_mAP@50={legacy_segm_map:.4f} segm_F1={legacy_segm_f1:.4f}")
    print(f"[{test_id}] ptl:    segm_mAP@50={ptl_segm_map:.4f} segm_F1={ptl_segm_f1:.4f}")

    # 6. Parity: same weights must produce identical segmentation metrics.
    assert abs(ptl_segm_map - legacy_segm_map) < 1e-4, (
        f"PTL segm_mAP@50 {ptl_segm_map:.6f} differs from legacy {legacy_segm_map:.6f} by > 1e-4"
    )

    # 7. Baseline thresholds (identical to test_coco_segmentation_inference_benchmark).
    assert ptl_segm_map >= threshold_segm_map, (
        f"PTL segm_mAP@50 {ptl_segm_map:.4f} < {threshold_segm_map}"
    )
    assert ptl_segm_f1 >= threshold_segm_f1, (
        f"PTL segm_F1 {ptl_segm_f1:.4f} < {threshold_segm_f1}"
    )
