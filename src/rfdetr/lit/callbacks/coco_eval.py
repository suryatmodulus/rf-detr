# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""COCOEvalCallback — torchmetrics-based mAP and F1 evaluation (Phase 3)."""

from typing import Any

import numpy as np
import torch
from pytorch_lightning import Callback
from torchmetrics.detection import MeanAveragePrecision

from rfdetr.engine import (
    build_matching_data,
    distributed_merge_matching_data,
    init_matching_accumulator,
    merge_matching_data,
    sweep_confidence_thresholds,
)
from rfdetr.util.box_ops import box_cxcywh_to_xyxy


class COCOEvalCallback(Callback):
    """Validation callback that computes mAP (via torchmetrics) and macro-F1.

    Accumulates predictions and targets across validation batches, then at
    epoch end computes:

    - ``val/mAP_50_95``, ``val/mAP_50``, ``val/mAP_75``, ``val/mAR`` using
      ``torchmetrics.detection.MeanAveragePrecision``.
    - Per-class ``val/AP/<name>`` when class names are available.
    - ``val/F1``, ``val/precision``, ``val/recall`` from a confidence-threshold
      sweep over compact per-class matching data (DDP-safe).

    For segmentation models (``segmentation=True``) additional metrics
    ``val/segm_mAP_50_95`` and ``val/segm_mAP_50`` are logged.

    Args:
        max_dets: Maximum detections per image passed to
            ``MeanAveragePrecision``. Defaults to 500.
        segmentation: When ``True``, evaluate both bbox and segm IoU using
            ``backend="faster_coco_eval"``. Defaults to ``False``.
    """

    def __init__(self, max_dets: int = 500, segmentation: bool = False) -> None:
        super().__init__()
        self._max_dets = max_dets
        self._segmentation = segmentation
        self._class_names: list[str] = []
        self._f1_local: dict[int, dict[str, Any]] = init_matching_accumulator()

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Instantiate ``MeanAveragePrecision`` after DDP device placement.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``.
        """
        iou_type: Any = ["bbox", "segm"] if self._segmentation else "bbox"
        kwargs: dict[str, Any] = dict(
            class_metrics=True,
            max_detection_thresholds=[1, 10, self._max_dets],
        )
        if self._segmentation:
            kwargs["backend"] = "faster_coco_eval"
        self.map_metric = MeanAveragePrecision(iou_type=iou_type, **kwargs)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Pull class names from the DataModule once the datasets are set up.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        dm = trainer.datamodule
        if dm is not None and hasattr(dm, "class_names"):
            self._class_names = dm.class_names

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate predictions and matching data for one validation batch.

        Expects ``outputs`` to be the dict returned by
        ``RFDETRModule.validation_step``:
        ``{"results": list[dict], "targets": list[dict]}``.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``validation_step``.
            batch: Raw batch (unused here).
            batch_idx: Batch index within the validation epoch.
        """
        preds: list[dict[str, torch.Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"])

        self.map_metric.update(preds, targets)

        iou_type = "segm" if self._segmentation else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_local, batch_matching)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log mAP and F1 metrics at the end of the validation epoch.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        metrics = self.map_metric.compute()

        # torchmetrics prefixes all keys when iou_type is a list (e.g. "bbox_map")
        pfx = "bbox_" if self._segmentation else ""
        mar_key = f"{pfx}mar_{self._max_dets}"

        pl_module.log("val/mAP_50_95", metrics[f"{pfx}map"])
        pl_module.log("val/mAP_50", metrics[f"{pfx}map_50"])
        pl_module.log("val/mAP_75", metrics[f"{pfx}map_75"])
        pl_module.log("val/mAR", metrics[mar_key])

        if self._segmentation:
            pl_module.log("val/segm_mAP_50_95", metrics["segm_map"])
            pl_module.log("val/segm_mAP_50", metrics["segm_map_50"])

        # Per-class AP (safe: class_id maps to class_names by value)
        pc_key = f"{pfx}map_per_class"
        if pc_key in metrics and "classes" in metrics:
            for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
                idx = int(class_id)
                name = self._class_names[idx] if idx < len(self._class_names) else str(idx)
                pl_module.log(f"val/AP/{name}", ap)

        # F1 sweep — gather compact matching state across all DDP ranks
        merged = distributed_merge_matching_data(self._f1_local)
        if merged:
            sorted_ids = sorted(merged.keys())
            per_class_list = [merged[cid] for cid in sorted_ids]
            classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged[cid]["total_gt"] > 0]
            f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
            best = max(f1_results, key=lambda x: x["macro_f1"])
            pl_module.log("val/F1", float(best["macro_f1"]))
            pl_module.log("val/precision", float(best["macro_precision"]))
            pl_module.log("val/recall", float(best["macro_recall"]))
        else:
            pl_module.log("val/F1", 0.0)
            pl_module.log("val/precision", 0.0)
            pl_module.log("val/recall", 0.0)

        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions and matching data for one test batch.

        Mirrors :meth:`on_validation_batch_end` for the test evaluation loop
        triggered by ``trainer.test()`` at the end of training.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``test_step``.
            batch: Raw batch (unused here).
            batch_idx: Batch index within the test epoch.
            dataloader_idx: Index of the test dataloader (unused here).
        """
        preds: list[dict[str, torch.Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"])

        self.map_metric.update(preds, targets)

        iou_type = "segm" if self._segmentation else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_local, batch_matching)

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log mAP and F1 under ``test/`` prefix at end of test epoch.

        Mirrors :meth:`on_validation_epoch_end` for the test evaluation loop.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        metrics = self.map_metric.compute()

        pfx = "bbox_" if self._segmentation else ""
        mar_key = f"{pfx}mar_{self._max_dets}"

        pl_module.log("test/mAP_50_95", metrics[f"{pfx}map"])
        pl_module.log("test/mAP_50", metrics[f"{pfx}map_50"])
        pl_module.log("test/mAP_75", metrics[f"{pfx}map_75"])
        pl_module.log("test/mAR", metrics[mar_key])

        if self._segmentation:
            pl_module.log("test/segm_mAP_50_95", metrics["segm_map"])
            pl_module.log("test/segm_mAP_50", metrics["segm_map_50"])

        pc_key = f"{pfx}map_per_class"
        if pc_key in metrics and "classes" in metrics:
            for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
                idx = int(class_id)
                name = self._class_names[idx] if idx < len(self._class_names) else str(idx)
                pl_module.log(f"test/AP/{name}", ap)

        merged = distributed_merge_matching_data(self._f1_local)
        if merged:
            sorted_ids = sorted(merged.keys())
            per_class_list = [merged[cid] for cid in sorted_ids]
            classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged[cid]["total_gt"] > 0]
            f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
            best = max(f1_results, key=lambda x: x["macro_f1"])
            pl_module.log("test/F1", float(best["macro_f1"]))
            pl_module.log("test/precision", float(best["macro_precision"]))
            pl_module.log("test/recall", float(best["macro_recall"]))
        else:
            pl_module.log("test/F1", 0.0)
            pl_module.log("test/precision", 0.0)
            pl_module.log("test/recall", 0.0)

        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_preds(self, preds: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Normalise prediction dicts from ``PostProcess`` for torchmetrics.

        ``PostProcess.forward`` returns masks with shape ``[K, 1, H, W]``
        (the extra channel is introduced by ``F.interpolate`` which requires
        4-D input).  Both ``torchmetrics.MeanAveragePrecision`` and
        ``engine.build_matching_data`` expect ``[K, H, W]``, so squeeze the
        channel dim when present.

        TODO(post-migration): audit whether ``PostProcess.forward`` should
        drop the channel dim itself (returning ``[K, H, W]`` directly), or
        whether other callers (e.g. ``RFDETR.predict``) rely on the 4-D shape
        and handle ``.squeeze(1)`` themselves.  See regression fix — Bug 4.

        Args:
            preds: Raw per-image prediction dicts from ``PostProcess``.

        Returns:
            Per-image dicts with ``masks`` squeezed to ``[K, H, W]`` when
            applicable; all other keys are passed through unchanged.
        """
        out = []
        for p in preds:
            entry = dict(p)
            if "masks" in entry and entry["masks"].ndim == 4 and entry["masks"].shape[1] == 1:
                entry["masks"] = entry["masks"].squeeze(1)
            out.append(entry)
        return out

    def _convert_targets(self, targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Convert targets from normalised CxCyWH to absolute xyxy boxes.

        Also passes ``iscrowd`` and ``masks`` through unchanged.

        Args:
            targets: Per-image target dicts with ``boxes`` in normalised
                CxCyWH format and ``orig_size`` as ``[H, W]``.

        Returns:
            Per-image dicts with ``boxes`` in absolute xyxy, ``labels``,
            and optionally ``masks`` and ``iscrowd``.
        """
        out = []
        for t in targets:
            h, w = t["orig_size"].tolist()
            scale = t["boxes"].new_tensor([w, h, w, h])
            boxes = box_cxcywh_to_xyxy(t["boxes"]) * scale
            entry: dict[str, torch.Tensor] = {"boxes": boxes, "labels": t["labels"]}
            if "masks" in t:
                entry["masks"] = t["masks"].bool()
            if "iscrowd" in t:
                entry["iscrowd"] = t["iscrowd"]
            out.append(entry)
        return out
