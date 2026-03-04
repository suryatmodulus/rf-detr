# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""COCOEvalCallback — torchmetrics-based mAP and F1 evaluation (Phase 3)."""

import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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

    def __init__(
        self,
        max_dets: int = 500,
        segmentation: bool = False,
        in_notebook: bool | None = None,
    ) -> None:
        super().__init__()
        self._max_dets = max_dets
        self._segmentation = segmentation
        self._class_names: list[str] = []
        self._cat_id_to_name: dict[int, str] = {}
        self._f1_local: dict[int, dict[str, Any]] = init_matching_accumulator()
        self._output_widget: Any = None  # ipywidgets.Output, created lazily
        self._in_notebook: bool = False
        if in_notebook is None:
            with contextlib.suppress(ImportError):
                from IPython import get_ipython

                self._in_notebook = get_ipython() is not None
        else:
            self._in_notebook = in_notebook

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

        Builds a ``category_id → name`` mapping from the COCO annotation
        metadata so that per-class AP is logged under the class name regardless
        of whether the dataset uses sequential or non-sequential category IDs.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        dm = trainer.datamodule
        if dm is None:
            return
        if hasattr(dm, "class_names"):
            self._class_names = dm.class_names or []
        # Build cat_id → name from the COCO annotation object when available.
        for attr in ("_dataset_train", "_dataset_val"):
            dataset = getattr(dm, attr, None)
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                self._cat_id_to_name = {k: v["name"] for k, v in coco.cats.items()}
                return
        # Fallback: treat class_names as 0-based sequential labels.
        self._cat_id_to_name = {i: name for i, name in enumerate(self._class_names)}

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
        self._compute_and_log(trainer, pl_module, "val")

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
        self._compute_and_log(trainer, pl_module, "test")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_and_log(self, trainer: Any, pl_module: Any, split: str) -> None:
        """Shared epoch-end logic for validation and test evaluation loops.

        Computes mAP (via ``self.map_metric``), runs the F1 confidence-threshold
        sweep, logs all scalar metrics via ``pl_module.log``, prints two summary
        tables to the terminal, and resets internal accumulators.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            split: Metric namespace — ``"val"`` or ``"test"``.
        """
        metrics = self.map_metric.compute()

        # torchmetrics prefixes all keys when iou_type is a list (e.g. "bbox_map")
        pfx = "bbox_" if self._segmentation else ""
        mar_key = f"{pfx}mar_{self._max_dets}"

        overall: dict[str, float] = {
            "mAP 50:95": float(metrics[f"{pfx}map"]),
            "mAP 50": float(metrics[f"{pfx}map_50"]),
            "mAP 75": float(metrics[f"{pfx}map_75"]),
            f"mAR @{self._max_dets}": float(metrics[mar_key]),
        }

        pl_module.log(f"{split}/mAP_50_95", metrics[f"{pfx}map"])
        pl_module.log(f"{split}/mAP_50", metrics[f"{pfx}map_50"])
        pl_module.log(f"{split}/mAP_75", metrics[f"{pfx}map_75"])
        pl_module.log(f"{split}/mAR", metrics[mar_key])

        if self._segmentation:
            overall["segm mAP 50:95"] = float(metrics["segm_map"])
            overall["segm mAP 50"] = float(metrics["segm_map_50"])
            pl_module.log(f"{split}/segm_mAP_50_95", metrics["segm_map"])
            pl_module.log(f"{split}/segm_mAP_50", metrics["segm_map_50"])

        # F1 sweep — run first so per-class F1/prec/rec are available when
        # building the unified per-class table rows below.
        merged = distributed_merge_matching_data(self._f1_local)
        # category_id → {f1, precision, recall} at the best macro-F1 threshold
        f1_by_cid: dict[int, dict[str, float]] = {}
        if merged:
            sorted_ids = sorted(merged.keys())
            per_class_list = [merged[cid] for cid in sorted_ids]
            classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged[cid]["total_gt"] > 0]
            f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
            best = max(f1_results, key=lambda x: x["macro_f1"])
            overall["F1"] = float(best["macro_f1"])
            overall["Precision"] = float(best["macro_precision"])
            overall["Recall"] = float(best["macro_recall"])
            pl_module.log(f"{split}/F1", float(best["macro_f1"]))
            pl_module.log(f"{split}/precision", float(best["macro_precision"]))
            pl_module.log(f"{split}/recall", float(best["macro_recall"]))
            for k, cid in enumerate(sorted_ids):
                f1_by_cid[cid] = {
                    "f1": float(best["per_class_f1"][k]),
                    "precision": float(best["per_class_prec"][k]),
                    "recall": float(best["per_class_rec"][k]),
                }
        else:
            overall["F1"] = 0.0
            overall["Precision"] = 0.0
            overall["Recall"] = 0.0
            pl_module.log(f"{split}/F1", 0.0)
            pl_module.log(f"{split}/precision", 0.0)
            pl_module.log(f"{split}/recall", 0.0)

        # Per-class AR from torchmetrics (keyed by category_id)
        ar_pc_key = f"{pfx}mar_{self._max_dets}_per_class"
        ar_by_cid: dict[int, float] = {}
        if ar_pc_key in metrics and "classes" in metrics:
            for class_id, ar in zip(metrics["classes"], metrics[ar_pc_key]):
                ar_by_cid[int(class_id)] = float(ar)

        # Unified per-class rows: AP 50:95 | AR | F1 | Precision | Recall
        # Classes with no ground-truth annotations are skipped (pycocotools
        # returns -1 for AP and torchmetrics returns NaN for AR on such classes,
        # so they would show as all dashes in the table).
        per_class: list[dict[str, Any]] = []
        pc_key = f"{pfx}map_per_class"
        if pc_key in metrics and "classes" in metrics:
            for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
                ap_f = float(ap)
                ar_f = ar_by_cid.get(int(class_id), float("nan"))
                if ap_f < 0 and ar_f != ar_f:  # no ground-truth: skip ghost class
                    continue
                idx = int(class_id)
                name = self._cat_id_to_name.get(idx, str(idx))
                pl_module.log(f"{split}/AP/{name}", ap)
                row: dict[str, Any] = {"name": name, "ap": ap_f, "ar": ar_f}
                row.update(f1_by_cid.get(idx, {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}))
                per_class.append(row)

        self._print_metrics_tables(trainer, split, overall, per_class)
        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()

    def _print_metrics_tables(
        self,
        trainer: Any,
        split: str,
        overall: dict[str, float],
        per_class: list[dict[str, Any]],
    ) -> None:
        """Print two tables to the terminal: overall metrics and per-class metrics.

        The overall table is transposed (metrics as columns, one value row) with
        true merged group-header cells rendered via box-drawing characters:
        ``mAP`` spans sub-columns 50:95 / 50 / 75, ``mAR`` spans ``@N``, and
        ``F1 sweep`` spans F1 / Prec / Recall.  The per-class table uses a
        standard Rich ``Table`` with columns for AP 50:95, AR, F1, Prec, Recall.

        Only runs on the global-zero rank to avoid duplicate output in DDP.

        Args:
            trainer: The PTL Trainer (used to check ``is_global_zero``).
            split: ``"val"`` or ``"test"``.
            overall: Ordered mapping of metric label → scalar value.
            per_class: Per-class dicts with keys ``name``, ``ap``, ``ar``,
                ``f1``, ``precision``, ``recall``; skipped when empty.
        """
        if not getattr(trainer, "is_global_zero", True):
            return
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            return

        def _fmt(v: float) -> str:
            if v != v or v < 0:  # NaN or pycocotools sentinel -1 → em-dash
                return "—"
            return f"{v:.4f}"

        console = Console(force_terminal=True)
        title_pfx = split.capitalize()

        def _render_all() -> None:
            # Table 1: Overall metrics — colour-free merged-header table.
            console.print(self._render_overall_merged(title_pfx, overall))

            # Table 2: Per-class metrics (Rich Table)
            if per_class:
                t2 = Table(
                    title=f"{title_pfx} — Per-class Metrics",
                    title_style="bold cyan",
                    show_header=True,
                    header_style="bold cyan",
                )
                t2.add_column("Class", style="dim", no_wrap=True)
                t2.add_column("AP 50:95", justify="right")
                t2.add_column("AR", justify="right")
                t2.add_column("F1", justify="right")
                t2.add_column("Precision", justify="right")
                t2.add_column("Recall", justify="right")
                for row in per_class:
                    t2.add_row(
                        row["name"],
                        _fmt(row["ap"]),
                        _fmt(row["ar"]),
                        _fmt(row["f1"]),
                        _fmt(row["precision"]),
                        _fmt(row["recall"]),
                    )
                console.print(t2)

        if self._in_notebook:
            # Lazily create an ipywidgets.Output on the first table print so it
            # anchors below the progress bar that is already visible.  Subsequent
            # epochs clear only the widget's isolated slot — the main cell output
            # (and PTL's progress bar) is never touched, so there is no flicker.
            if self._output_widget is None:
                with contextlib.suppress(ImportError):
                    import ipywidgets as widgets
                    from IPython.display import display

                    self._output_widget = widgets.Output()
                    display(self._output_widget)

            if self._output_widget is not None:
                self._output_widget.clear_output(wait=True)
                with self._output_widget:
                    _render_all()
                return

        _render_all()

    def _render_overall_merged(self, title_pfx: str, overall: dict[str, float]) -> str:
        """Render the overall metrics table with merged group-header cells.

        Uses only plain Unicode box-drawing characters (no ANSI colour codes)
        so the output renders correctly in both terminals and Jupyter/Colab
        notebook widgets.

        .. code-block:: text

                        Val — Overall Metrics
            ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃          mAP          ┃   mAR   ┃        F1 sweep       ┃
            ┡━━━━━━━━━┳━━━━━━┳━━━━━━╇━━━━━━━━━╇━━━━━━┳━━━━━━┳━━━━━━━━━┩
            │  50:95  │  50  │  75  │  @500   │  F1  │ Prec │ Recall  │
            ├─────────┼──────┼──────┼─────────┼──────┼──────┼─────────┤
            │    —    │0.1510│0.1228│  0.4017 │0.1573│0.2607│  0.1562 │
            └─────────┴──────┴──────┴─────────┴──────┴──────┴─────────┘

        Args:
            title_pfx: Capitalised split name used in the title (e.g. ``"Val"``).
            overall: Ordered mapping of metric label → scalar value.

        Returns:
            Multi-line plain-text string ready to pass to ``console.print()``.
        """

        def _fmt(v: float) -> str:
            if v != v or v < 0:  # NaN or pycocotools sentinel -1 → em-dash
                return "—"
            return f"{v:.4f}"

        mar_lbl = f"@{self._max_dets}"
        mar_key = f"mAR @{self._max_dets}"

        # Groups: (group_name, [(sub_label, formatted_value), ...])
        groups: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "mAP",
                [
                    ("50:95", _fmt(overall["mAP 50:95"])),
                    ("50", _fmt(overall["mAP 50"])),
                    ("75", _fmt(overall["mAP 75"])),
                ],
            ),
            ("mAR", [(mar_lbl, _fmt(overall[mar_key]))]),
            (
                "F1 sweep",
                [
                    ("F1", _fmt(overall["F1"])),
                    ("Prec", _fmt(overall["Precision"])),
                    ("Recall", _fmt(overall["Recall"])),
                ],
            ),
        ]
        if "segm mAP 50:95" in overall:
            groups.append(
                (
                    "segm mAP",
                    [
                        ("50:95", _fmt(overall["segm mAP 50:95"])),
                        ("50", _fmt(overall["segm mAP 50"])),
                    ],
                )
            )

        # Flatten sub-columns and compute widths (+2 for single-space padding each side)
        flat: list[tuple[str, str]] = [(s, v) for _, cols in groups for s, v in cols]
        widths: list[int] = [max(len(s), len(v)) + 2 for s, v in flat]

        # Expand widths so each group label fits in its merged cell
        col = 0
        for grp, cols in groups:
            nc = len(cols)
            cell_w = sum(widths[col : col + nc]) + (nc - 1)  # nc-1 internal separators
            needed = len(grp) + 2
            if needed > cell_w:
                for k in range(needed - cell_w):
                    widths[col + k % nc] += 1
            col += nc

        # Compute group spans: (start_col, end_col_inclusive, name)
        spans: list[tuple[int, int, str]] = []
        col = 0
        for grp, cols in groups:
            nc = len(cols)
            spans.append((col, col + nc - 1, grp))
            col += nc

        grp_ends = {end for start, end, _ in spans[:-1]}
        n = len(flat)

        def grp_w(start: int, end: int) -> int:
            """Merged cell width for columns start..end inclusive."""
            return sum(widths[start : end + 1]) + (end - start)

        # Box-drawing character sets
        BH, BL = "━", "─"
        VH, VL = "┃", "│"
        TL, TR = "┏", "┓"
        T_DN = "┳"  # heavy T-down: top-border internal group separator
        TR_L, TR_R = "┡", "┩"  # transition-row left/right edges
        GRP_J = "╇"  # transition-row at group boundary: heavy-up, heavy-horiz, light-down
        SUB_J = "┯"  # transition-row within group: no-up, heavy-horiz, light-down
        ML, MR, MX = "├", "┤", "┼"
        BL_C, BR_C, BT = "└", "┘", "┴"

        # Title (centred over the full table width)
        inner_w = sum(widths) + n - 1
        title = f"{title_pfx} — Overall Metrics"
        title_line = title.center(inner_w + 2)

        # Row 1: top border — group-level separators only
        r1 = TL
        for i, (s, e, _) in enumerate(spans):
            r1 += BH * grp_w(s, e)
            r1 += T_DN if i < len(spans) - 1 else TR

        # Row 2: group labels centred in merged cells
        r2 = VH
        for s, e, grp in spans:
            r2 += grp.center(grp_w(s, e)) + VH

        # Row 3: transition row — heavy horizontal; ╇ at group ends, ┯ within groups
        r3 = TR_L
        for i, w in enumerate(widths):
            r3 += BH * w
            if i < n - 1:
                r3 += GRP_J if i in grp_ends else SUB_J
        r3 += TR_R

        # Row 4: sub-labels with light borders
        r4 = VL
        for i, (sub, _) in enumerate(flat):
            r4 += sub.center(widths[i]) + VL

        # Row 5: light separator between sub-labels and values
        r5 = ML
        for i, w in enumerate(widths):
            r5 += BL * w
            r5 += MX if i < n - 1 else MR

        # Row 6: values
        r6 = VL
        for i, (_, val) in enumerate(flat):
            r6 += val.center(widths[i]) + VL

        # Row 7: bottom border
        r7 = BL_C
        for i, w in enumerate(widths):
            r7 += BL * w
            r7 += BT if i < n - 1 else BR_C

        return "\n".join([title_line, r1, r2, r3, r4, r5, r6, r7])

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
                masks = t["masks"].bool()
                # PostProcess resizes predicted masks to orig_size; resize GT
                # masks to match so that mask-IoU comparisons are size-consistent.
                if masks.shape[-2:] != (int(h), int(w)):
                    masks = (
                        F.interpolate(
                            masks.float().unsqueeze(1),
                            size=(int(h), int(w)),
                            mode="nearest",
                        )
                        .squeeze(1)
                        .bool()
                    )
                entry["masks"] = masks
            if "iscrowd" in t:
                entry["iscrowd"] = t["iscrowd"]
            out.append(entry)
        return out
