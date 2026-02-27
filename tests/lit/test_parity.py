# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Parity tests: COCOEvalCallback vs legacy coco_extended_metrics (PTL Ch2/T5).

Asserts that the new ``COCOEvalCallback`` produces mAP50 and F1 values within
the tolerances specified in MIGRATION_PT_LIGHTNING.md Chapter 2:

    |Δ mAP50| ≤ 0.005
    |Δ F1|    ≤ 0.01

The intermediate scenario (2 classes, mixed TP/FP, varying confidence)
mirrors the ``intermediate_scenario_cocoeval`` fixture in
``tests/util/test_metrics.py`` so the same data drives both paths.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr.engine import coco_extended_metrics
from rfdetr.lit.callbacks.coco_eval import COCOEvalCallback

# ---------------------------------------------------------------------------
# Tolerances (per MIGRATION_PT_LIGHTNING.md Chapter 2 milestone)
# ---------------------------------------------------------------------------

_MAP50_TOL = 0.005
_F1_TOL = 0.01

# ---------------------------------------------------------------------------
# Scenario constants — match tests/util/test_metrics.py exactly
# ---------------------------------------------------------------------------

_BOX_SIZE = 200
_BOX_SPACING = 250
_ROW_SPACING = 260


# ---------------------------------------------------------------------------
# Shared scenario builder
# ---------------------------------------------------------------------------


def _make_contained_pred_box(gt_box: list[float], target_iou: float) -> list[float]:
    """Return a prediction box centred inside *gt_box* with the given IoU.

    Uses the contained-box formula IoU = pred_area / gt_area, valid when the
    prediction is a square fully inside a square GT.

    Args:
        gt_box: COCO-format ``[x, y, w, h]`` for the ground-truth box.
        target_iou: Desired IoU between prediction and GT.

    Returns:
        COCO-format ``[x, y, w, h]`` for the prediction box.
    """
    x, y, gt_size, _ = gt_box
    if target_iou >= 1.0:
        return list(gt_box)
    pred_size = gt_size * math.sqrt(target_iou)
    offset = (gt_size - pred_size) / 2
    return [x + offset, y + offset, pred_size, pred_size]


def _build_intermediate_scenario() -> dict:
    """Build intermediate scenario raw data: 2 classes, mixed TP/FP.

    Exactly mirrors the ``intermediate_scenario_cocoeval`` fixture in
    ``tests/util/test_metrics.py``:

    - Class 1: 10 GTs, 10 TP predictions (IoU ≥ 0.525, varying confidence).
    - Class 2: 10 GTs, 10 TP predictions (IoU ≥ 0.525) + 10 FP predictions
      (IoU = 0, lower confidence).

    Returns:
        Dict with keys:

        - ``image_id`` (int)
        - ``image_width``, ``image_height`` (int)
        - ``categories`` (list of ``{"id": int, "name": str}``)
        - ``gt_abs_xywh``: list of ``(cat_id, [x, y, w, h])``
        - ``pred_abs_xywh``: list of ``(cat_id, [x, y, w, h], score)``
    """
    image_id = 1
    n_boxes = 10

    class1_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class1_confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    class2_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class2_confs = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    class2_fp_confs = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    image_width = n_boxes * _BOX_SPACING
    image_height = 3 * _ROW_SPACING

    gt_abs_xywh: list[tuple[int, list[float]]] = []
    pred_abs_xywh: list[tuple[int, list[float], float]] = []

    # Row 0: Class 1 — 10 GTs each paired with one TP prediction
    for i, (iou, conf) in enumerate(zip(class1_ious, class1_confs)):
        gt_box = [float(i * _BOX_SPACING), 0.0, float(_BOX_SIZE), float(_BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        gt_abs_xywh.append((1, gt_box))
        pred_abs_xywh.append((1, pred_box, conf))

    # Row 1: Class 2 — 10 GTs each paired with one TP prediction
    for i, (iou, conf) in enumerate(zip(class2_ious, class2_confs)):
        gt_box = [float(i * _BOX_SPACING), float(_ROW_SPACING), float(_BOX_SIZE), float(_BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        gt_abs_xywh.append((2, gt_box))
        pred_abs_xywh.append((2, pred_box, conf))

    # Row 2: Class 2 — 10 FP predictions (no GTs on this row, IoU = 0)
    for i, conf in enumerate(class2_fp_confs):
        fp_box = [float(i * _BOX_SPACING), float(2 * _ROW_SPACING), float(_BOX_SIZE), float(_BOX_SIZE)]
        pred_abs_xywh.append((2, fp_box, conf))

    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
        "gt_abs_xywh": gt_abs_xywh,
        "pred_abs_xywh": pred_abs_xywh,
    }


# ---------------------------------------------------------------------------
# Legacy path runner
# ---------------------------------------------------------------------------


def _run_legacy(scenario: dict) -> dict:
    """Run legacy ``COCOeval`` + ``coco_extended_metrics`` on *scenario* data.

    Args:
        scenario: Raw scenario dict from :func:`_build_intermediate_scenario`.

    Returns:
        Dict with keys ``"map50"``, ``"f1"``, ``"precision"``, ``"recall"``
        (all floats).
    """
    images = [
        {
            "id": scenario["image_id"],
            "width": scenario["image_width"],
            "height": scenario["image_height"],
        }
    ]
    annotations = []
    ann_id = 1
    for cat_id, box in scenario["gt_abs_xywh"]:
        annotations.append(
            {
                "id": ann_id,
                "image_id": scenario["image_id"],
                "category_id": cat_id,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
        )
        ann_id += 1

    predictions = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        predictions.append(
            {
                "image_id": scenario["image_id"],
                "category_id": cat_id,
                "bbox": box,
                "score": score,
            }
        )

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": scenario["categories"],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = coco_extended_metrics(coco_eval)
    return {
        "map50": float(coco_eval.stats[1]),
        "f1": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


# ---------------------------------------------------------------------------
# New callback path runner
# ---------------------------------------------------------------------------


def _run_callback(scenario: dict) -> dict:
    """Drive :class:`COCOEvalCallback` with *scenario* data and return logged metrics.

    Converts COCO-format boxes to the formats expected by the callback:

    - Predictions: absolute xyxy (already in absolute coords, just reformat).
    - Targets: normalised CxCyWH + ``orig_size``.

    Args:
        scenario: Raw scenario dict from :func:`_build_intermediate_scenario`.

    Returns:
        Dict mapping metric key (e.g. ``"val/mAP_50"``) to float value.
    """
    W = scenario["image_width"]
    H = scenario["image_height"]

    # Predictions: COCO [x, y, w, h] absolute → absolute xyxy tensors
    pred_boxes_xyxy: list[list[float]] = []
    pred_scores_list: list[float] = []
    pred_labels_list: list[int] = []
    for cat_id, box, score in scenario["pred_abs_xywh"]:
        x, y, w, h = box
        pred_boxes_xyxy.append([x, y, x + w, y + h])
        pred_scores_list.append(score)
        pred_labels_list.append(cat_id)

    preds = [
        {
            "boxes": torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
            "scores": torch.tensor(pred_scores_list, dtype=torch.float32),
            "labels": torch.tensor(pred_labels_list, dtype=torch.long),
        }
    ]

    # Targets: COCO [x, y, w, h] absolute → normalised CxCyWH for the callback
    gt_boxes_norm: list[list[float]] = []
    gt_labels_list: list[int] = []
    for cat_id, box in scenario["gt_abs_xywh"]:
        x, y, w, h = box
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        wn = w / W
        hn = h / H
        gt_boxes_norm.append([cx, cy, wn, hn])
        gt_labels_list.append(cat_id)

    targets = [
        {
            "boxes": torch.tensor(gt_boxes_norm, dtype=torch.float32),
            "labels": torch.tensor(gt_labels_list, dtype=torch.long),
            "orig_size": torch.tensor([H, W]),  # [H, W] as expected by _convert_targets
        }
    ]

    cb = COCOEvalCallback(max_dets=500)
    trainer = MagicMock()
    module = MagicMock()
    logged: dict[str, float] = {}
    module.log.side_effect = lambda key, val: logged.__setitem__(key, float(val))

    cb.setup(trainer, module, stage="validate")
    cb.on_validation_batch_end(trainer, module, {"results": preds, "targets": targets}, None, 0)
    cb.on_validation_epoch_end(trainer, module)

    return logged


# ---------------------------------------------------------------------------
# Parity tests — mAP50
# ---------------------------------------------------------------------------


class TestMAPParityDetection:
    """val/mAP_50 from COCOEvalCallback agrees with legacy COCOeval.

    Tolerance: ``|Δ mAP50| ≤ 0.005``.
    """

    def test_map50_within_tolerance(self) -> None:
        """|Δ mAP50| ≤ 0.005 on the intermediate scenario."""
        scenario = _build_intermediate_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        delta = abs(legacy["map50"] - new["val/mAP_50"])
        assert delta <= _MAP50_TOL, (
            f"mAP50 parity failed: legacy={legacy['map50']:.4f}, "
            f"new={new['val/mAP_50']:.4f}, delta={delta:.4f} > {_MAP50_TOL}"
        )

    def test_map50_95_is_logged(self) -> None:
        """val/mAP_50_95 is always logged.

        Note: torchmetrics returns -1.0 as a sentinel when ``map`` cannot be
        computed with non-default ``max_detection_thresholds`` (e.g. 500).
        The assertion only verifies the key is present; use ``val/mAP_50`` for
        numeric comparisons.
        """
        new = _run_callback(_build_intermediate_scenario())
        assert "val/mAP_50_95" in new

    def test_mar_logged_and_nonnegative(self) -> None:
        """val/mAR is logged and non-negative."""
        new = _run_callback(_build_intermediate_scenario())
        assert "val/mAR" in new
        assert new["val/mAR"] >= 0.0


# ---------------------------------------------------------------------------
# Parity tests — F1 / precision / recall
# ---------------------------------------------------------------------------


class TestF1ParityDetection:
    """F1, precision, recall from COCOEvalCallback agree with legacy path.

    Tolerance: ``|Δ| ≤ 0.01`` for all three metrics.
    """

    def test_f1_within_tolerance(self) -> None:
        """|Δ F1| ≤ 0.01 on the intermediate scenario."""
        scenario = _build_intermediate_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        delta = abs(legacy["f1"] - new["val/F1"])
        assert delta <= _F1_TOL, (
            f"F1 parity failed: legacy={legacy['f1']:.4f}, "
            f"new={new['val/F1']:.4f}, delta={delta:.4f} > {_F1_TOL}"
        )

    def test_precision_within_tolerance(self) -> None:
        """|Δ precision| ≤ 0.01 on the intermediate scenario."""
        scenario = _build_intermediate_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        delta = abs(legacy["precision"] - new["val/precision"])
        assert delta <= _F1_TOL, (
            f"Precision parity failed: legacy={legacy['precision']:.4f}, "
            f"new={new['val/precision']:.4f}, delta={delta:.4f} > {_F1_TOL}"
        )

    def test_recall_within_tolerance(self) -> None:
        """|Δ recall| ≤ 0.01 on the intermediate scenario."""
        scenario = _build_intermediate_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        delta = abs(legacy["recall"] - new["val/recall"])
        assert delta <= _F1_TOL, (
            f"Recall parity failed: legacy={legacy['recall']:.4f}, "
            f"new={new['val/recall']:.4f}, delta={delta:.4f} > {_F1_TOL}"
        )


# ---------------------------------------------------------------------------
# Sanity-check parity at the boundary scenarios
# ---------------------------------------------------------------------------


class TestBoundaryScenarioParity:
    """Both paths report consistent values on boundary (all-TP, all-FP) scenarios."""

    def _build_perfect_scenario(self) -> dict:
        """Two classes, 5 GTs each with IoU=0.96 TP predictions, score=1.0."""
        image_id = 1
        image_width = 5 * _BOX_SPACING
        image_height = 2 * _ROW_SPACING

        gt_abs_xywh: list[tuple[int, list[float]]] = []
        pred_abs_xywh: list[tuple[int, list[float], float]] = []

        for cat_id, row_y in [(1, 0), (2, _ROW_SPACING)]:
            for i in range(5):
                gt_box = [float(i * _BOX_SPACING), float(row_y), float(_BOX_SIZE), float(_BOX_SIZE)]
                pred_box = _make_contained_pred_box(gt_box, target_iou=0.96)
                gt_abs_xywh.append((cat_id, gt_box))
                pred_abs_xywh.append((cat_id, pred_box, 1.0))

        return {
            "image_id": image_id,
            "image_width": image_width,
            "image_height": image_height,
            "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
            "gt_abs_xywh": gt_abs_xywh,
            "pred_abs_xywh": pred_abs_xywh,
        }

    def _build_degenerate_scenario(self) -> dict:
        """Two classes: GTs on left, FP predictions on right (IoU=0), score=1.0."""
        image_id = 1
        pred_x_offset = 5 * _BOX_SPACING + 100  # gap ensures zero IoU
        image_width = pred_x_offset + 5 * _BOX_SPACING
        image_height = 2 * _ROW_SPACING

        gt_abs_xywh: list[tuple[int, list[float]]] = []
        pred_abs_xywh: list[tuple[int, list[float], float]] = []

        for cat_id, row_y in [(1, 0), (2, _ROW_SPACING)]:
            for i in range(5):
                gt_box = [float(i * _BOX_SPACING), float(row_y), float(_BOX_SIZE), float(_BOX_SIZE)]
                fp_box = [float(pred_x_offset + i * _BOX_SPACING), float(row_y), float(_BOX_SIZE), float(_BOX_SIZE)]
                gt_abs_xywh.append((cat_id, gt_box))
                pred_abs_xywh.append((cat_id, fp_box, 1.0))

        return {
            "image_id": image_id,
            "image_width": image_width,
            "image_height": image_height,
            "categories": [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}],
            "gt_abs_xywh": gt_abs_xywh,
            "pred_abs_xywh": pred_abs_xywh,
        }

    def test_perfect_scenario_map50_near_one(self) -> None:
        """Both paths report mAP50 ≥ 0.99 on the perfect scenario."""
        scenario = self._build_perfect_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        assert legacy["map50"] >= 0.99, f"Legacy mAP50 unexpectedly low: {legacy['map50']:.4f}"
        assert new["val/mAP_50"] >= 0.99, f"Callback mAP50 unexpectedly low: {new['val/mAP_50']:.4f}"

    def test_perfect_scenario_f1_near_one(self) -> None:
        """Both paths report F1 ≥ 0.99 on the perfect scenario."""
        scenario = self._build_perfect_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        assert legacy["f1"] >= 0.99, f"Legacy F1 unexpectedly low: {legacy['f1']:.4f}"
        assert new["val/F1"] >= 0.99, f"Callback F1 unexpectedly low: {new['val/F1']:.4f}"

    def test_degenerate_scenario_f1_is_zero(self) -> None:
        """Both paths report F1 = 0.0 on the degenerate (all-FP) scenario."""
        scenario = self._build_degenerate_scenario()
        legacy = _run_legacy(scenario)
        new = _run_callback(scenario)

        assert legacy["f1"] == pytest.approx(0.0), f"Legacy F1 should be 0, got {legacy['f1']:.4f}"
        assert new["val/F1"] == pytest.approx(0.0), f"Callback F1 should be 0, got {new['val/F1']:.4f}"
