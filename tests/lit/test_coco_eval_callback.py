# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for COCOEvalCallback (PTL Ch3/T3)."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from rfdetr.lit.callbacks.coco_eval import COCOEvalCallback

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pl_module() -> MagicMock:
    """Return a minimal mock LightningModule."""
    return MagicMock(name="pl_module")


def _make_trainer(datamodule=None) -> MagicMock:
    """Return a minimal mock Trainer with an optional DataModule."""
    trainer = MagicMock(name="trainer")
    trainer.datamodule = datamodule
    return trainer


def _detection_preds(n: int = 0) -> list[dict]:
    """Return a list with one per-image prediction dict."""
    return [
        {
            "boxes": torch.zeros(n, 4),
            "scores": torch.zeros(n),
            "labels": torch.zeros(n, dtype=torch.long),
        }
    ]


def _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1, label=1) -> list[dict]:
    """Return a single-image target dict with one box in normalised CxCyWH."""
    return [
        {
            "boxes": torch.tensor([[cx, cy, w, h]]),
            "labels": torch.tensor([label]),
            "orig_size": torch.tensor([100, 200]),  # H=100, W=200
        }
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetup:
    """setup() creates map_metric with correct configuration."""

    def test_detection_iou_type_is_bbox(self) -> None:
        """Detection mode uses iou_type='bbox'."""
        cb = COCOEvalCallback(max_dets=300, segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert "bbox" in cb.map_metric.iou_type
        assert "segm" not in cb.map_metric.iou_type

    def test_detection_max_detection_thresholds(self) -> None:
        """max_dets is forwarded to max_detection_thresholds."""
        cb = COCOEvalCallback(max_dets=300, segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert 300 in cb.map_metric.max_detection_thresholds

    def test_segmentation_iou_type_includes_segm(self) -> None:
        """Segmentation mode uses iou_type=['bbox','segm']."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert "segm" in cb.map_metric.iou_type

    def test_map_metric_created_on_every_setup_call(self) -> None:
        """Repeated setup() calls replace map_metric (idempotent)."""
        cb = COCOEvalCallback()
        trainer, module = _make_trainer(), _make_pl_module()
        cb.setup(trainer, module, stage="fit")
        first = cb.map_metric
        cb.setup(trainer, module, stage="validate")
        assert cb.map_metric is not first


class TestOnFitStart:
    """on_fit_start() populates class names from the datamodule."""

    def test_class_names_loaded_from_datamodule(self) -> None:
        """Class names are taken from trainer.datamodule.class_names."""
        dm = MagicMock()
        dm.class_names = ["cat", "dog"]
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._class_names == ["cat", "dog"]

    def test_no_datamodule_leaves_class_names_empty(self) -> None:
        """Absent datamodule keeps class_names as empty list."""
        trainer = _make_trainer(datamodule=None)
        cb = COCOEvalCallback()
        cb.on_fit_start(trainer, _make_pl_module())
        assert cb._class_names == []

    def test_datamodule_without_class_names_attr_leaves_empty(self) -> None:
        """DataModule without class_names attr keeps class_names empty."""
        dm = MagicMock(spec=[])  # no attributes
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._class_names == []


class TestOnValidationBatchEnd:
    """on_validation_batch_end() updates map_metric and accumulates F1 data."""

    def test_map_metric_update_called_once_per_batch(self) -> None:
        """map_metric.update is called exactly once per batch."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        cb.on_validation_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0)

        assert cb.map_metric.update.call_count == 1

    def test_f1_accumulator_grows_across_batches(self) -> None:
        """Calling on_validation_batch_end twice accumulates more GT in F1 state."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")

        # target with 1 non-crowd GT for class 1
        outputs = {"results": _detection_preds(0), "targets": _detection_targets(label=1)}
        cb.on_validation_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0)
        total_after_1 = sum(v["total_gt"] for v in cb._f1_local.values())

        cb.on_validation_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 1)
        total_after_2 = sum(v["total_gt"] for v in cb._f1_local.values())

        assert total_after_2 == total_after_1 * 2

    def test_targets_converted_before_update(self) -> None:
        """map_metric.update receives targets with absolute xyxy boxes."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        captured = {}

        def _capture_update(preds, targets):
            captured["targets"] = targets

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.update.side_effect = _capture_update

        outputs = {
            "results": _detection_preds(0),
            "targets": _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1),
        }
        cb.on_validation_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0)

        # Expected: CxCyWH(0.5,0.5,0.1,0.1) × scale(W=200,H=100) → xyxy(90,45,110,55)
        boxes = captured["targets"][0]["boxes"]
        assert boxes.shape == (1, 4)
        assert boxes[0, 0].item() == pytest.approx(90.0)
        assert boxes[0, 1].item() == pytest.approx(45.0)
        assert boxes[0, 2].item() == pytest.approx(110.0)
        assert boxes[0, 3].item() == pytest.approx(55.0)


class TestOnValidationEpochEnd:
    """on_validation_epoch_end() logs metrics and resets state."""

    @staticmethod
    def _minimal_metrics(pfx: str = "", max_dets: int = 500) -> dict:
        """Return a minimal torchmetrics-style metrics dict."""
        return {
            f"{pfx}map": torch.tensor(0.4),
            f"{pfx}map_50": torch.tensor(0.6),
            f"{pfx}map_75": torch.tensor(0.3),
            f"{pfx}mar_{max_dets}": torch.tensor(0.5),
        }

    def test_detection_core_metrics_are_logged(self) -> None:
        """val/mAP_50_95, val/mAP_50, val/mAP_75, val/mAR are always logged."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/mAP_50_95" in logged_keys
        assert "val/mAP_50" in logged_keys
        assert "val/mAP_75" in logged_keys
        assert "val/mAR" in logged_keys

    def test_f1_metrics_logged_when_gt_present(self) -> None:
        """val/F1, val/precision, val/recall are logged when GT exists."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        # Inject non-empty F1 accumulator
        cb._f1_local = {
            0: {
                "scores": np.array([0.9], dtype=np.float32),
                "matches": np.array([1], dtype=np.int64),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        }
        module = _make_pl_module()
        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/F1" in logged_keys
        assert "val/precision" in logged_keys
        assert "val/recall" in logged_keys

    def test_f1_metrics_zero_when_no_gt(self) -> None:
        """val/F1 == 0.0 when no predictions were accumulated (empty epoch)."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        f1_call = next(c for c in module.log.call_args_list if c.args[0] == "val/F1")
        assert f1_call.args[1] == pytest.approx(0.0)

    def test_state_reset_after_epoch(self) -> None:
        """map_metric.reset() is called and _f1_local is cleared after epoch end."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        cb._f1_local = {
            0: {
                "scores": np.array([0.9], dtype=np.float32),
                "matches": np.array([1], dtype=np.int64),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        }

        cb.on_validation_epoch_end(_make_trainer(), _make_pl_module())

        cb.map_metric.reset.assert_called_once()
        assert cb._f1_local == {}

    def test_segmentation_extra_metrics_logged(self) -> None:
        """val/segm_mAP_50_95 and val/segm_mAP_50 are logged in segm mode."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        segm_metrics = self._minimal_metrics(pfx="bbox_")
        segm_metrics["segm_map"] = torch.tensor(0.35)
        segm_metrics["segm_map_50"] = torch.tensor(0.55)
        cb.map_metric.compute.return_value = segm_metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/segm_mAP_50_95" in logged_keys
        assert "val/segm_mAP_50" in logged_keys

    def test_per_class_ap_logged_when_classes_present(self) -> None:
        """val/AP/<name> is logged for each class when class metrics are present."""
        cb = COCOEvalCallback()
        cb._class_names = ["cat", "dog"]
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = self._minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/AP/cat" in logged_keys
        assert "val/AP/dog" in logged_keys

    def test_per_class_ap_falls_back_to_str_id_when_no_class_names(self) -> None:
        """val/AP/<id> is logged when class_names is empty."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = self._minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5])
        metrics["classes"] = torch.tensor([3])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/AP/3" in logged_keys


class TestConvertTargets:
    """_convert_targets() converts normalised CxCyWH to absolute xyxy."""

    def test_box_conversion_known_values(self) -> None:
        """CxCyWH(0.5,0.5,0.4,0.6) × (W=100,H=200) → xyxy(30,40,70,160)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([200, 100]),  # H=200, W=100
            }
        ]
        out = cb._convert_targets(targets)
        boxes = out[0]["boxes"]
        # cx=0.5*100=50, cy=0.5*200=100, w=0.4*100=40, h=0.6*200=120
        # → x1=50-20=30, y1=100-60=40, x2=50+20=70, y2=100+60=160
        assert boxes[0, 0].item() == pytest.approx(30.0)
        assert boxes[0, 1].item() == pytest.approx(40.0)
        assert boxes[0, 2].item() == pytest.approx(70.0)
        assert boxes[0, 3].item() == pytest.approx(160.0)

    def test_labels_passed_through(self) -> None:
        """labels tensor is preserved unchanged."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert out[0]["labels"][0].item() == 7

    def test_masks_passed_through_as_bool(self) -> None:
        """masks tensor is cast to bool and included in output."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([8, 8]),
                "masks": torch.ones(1, 8, 8, dtype=torch.uint8),
            }
        ]
        out = cb._convert_targets(targets)
        assert "masks" in out[0]
        assert out[0]["masks"].dtype == torch.bool

    def test_iscrowd_passed_through(self) -> None:
        """iscrowd tensor is included when present."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
                "iscrowd": torch.tensor([1]),
            }
        ]
        out = cb._convert_targets(targets)
        assert "iscrowd" in out[0]
        assert out[0]["iscrowd"][0].item() == 1

    def test_no_masks_no_iscrowd_keys_absent(self) -> None:
        """Output dict contains exactly boxes and labels when extras are absent."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert set(out[0].keys()) == {"boxes", "labels"}


class TestOnTestBatchEnd:
    """on_test_batch_end() mirrors on_validation_batch_end() for the test loop."""

    def test_map_metric_update_called_once_per_batch(self) -> None:
        """map_metric.update is called exactly once per test batch."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0)

        assert cb.map_metric.update.call_count == 1

    def test_f1_accumulator_grows_across_batches(self) -> None:
        """Calling on_test_batch_end twice accumulates more GT in F1 state."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets(label=1)}
        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0)
        total_after_1 = sum(v["total_gt"] for v in cb._f1_local.values())

        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 1)
        total_after_2 = sum(v["total_gt"] for v in cb._f1_local.values())

        assert total_after_2 == total_after_1 * 2

    def test_targets_converted_before_update(self) -> None:
        """map_metric.update receives targets with absolute xyxy boxes."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        captured = {}

        def _capture_update(preds, targets):
            captured["targets"] = targets

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.update.side_effect = _capture_update

        outputs = {
            "results": _detection_preds(0),
            "targets": _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1),
        }
        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0)

        # Expected: CxCyWH(0.5,0.5,0.1,0.1) × scale(W=200,H=100) → xyxy(90,45,110,55)
        boxes = captured["targets"][0]["boxes"]
        assert boxes.shape == (1, 4)
        assert boxes[0, 0].item() == pytest.approx(90.0)
        assert boxes[0, 1].item() == pytest.approx(45.0)
        assert boxes[0, 2].item() == pytest.approx(110.0)
        assert boxes[0, 3].item() == pytest.approx(55.0)

    def test_dataloader_idx_param_has_default(self) -> None:
        """on_test_batch_end must accept calls with an explicit dataloader_idx."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        # Must not raise with explicit dataloader_idx=0
        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0, dataloader_idx=0)


class TestOnTestEpochEnd:
    """on_test_epoch_end() logs metrics under test/ prefix and resets state."""

    @staticmethod
    def _minimal_metrics(pfx: str = "", max_dets: int = 500) -> dict:
        """Return a minimal torchmetrics-style metrics dict."""
        return {
            f"{pfx}map": torch.tensor(0.4),
            f"{pfx}map_50": torch.tensor(0.6),
            f"{pfx}map_75": torch.tensor(0.3),
            f"{pfx}mar_{max_dets}": torch.tensor(0.5),
        }

    def test_detection_core_metrics_are_logged(self) -> None:
        """test/mAP_50_95, test/mAP_50, test/mAP_75, test/mAR are always logged."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "test/mAP_50_95" in logged_keys
        assert "test/mAP_50" in logged_keys
        assert "test/mAP_75" in logged_keys
        assert "test/mAR" in logged_keys

    def test_val_prefix_not_logged(self) -> None:
        """test_epoch_end must not emit val/ keys — prefixes must not bleed across loops."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("val/") for k in logged_keys)

    def test_f1_metrics_logged_when_gt_present(self) -> None:
        """test/F1, test/precision, test/recall are logged when GT exists."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        cb._f1_local = {
            0: {
                "scores": np.array([0.9], dtype=np.float32),
                "matches": np.array([1], dtype=np.int64),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        }
        module = _make_pl_module()
        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "test/F1" in logged_keys
        assert "test/precision" in logged_keys
        assert "test/recall" in logged_keys

    def test_f1_metrics_zero_when_no_gt(self) -> None:
        """test/F1 == 0.0 when no predictions were accumulated (empty epoch)."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        f1_call = next(c for c in module.log.call_args_list if c.args[0] == "test/F1")
        assert f1_call.args[1] == pytest.approx(0.0)

    def test_state_reset_after_epoch(self) -> None:
        """map_metric.reset() is called and _f1_local is cleared after test epoch end."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = self._minimal_metrics()
        cb._f1_local = {
            0: {
                "scores": np.array([0.9], dtype=np.float32),
                "matches": np.array([1], dtype=np.int64),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        }

        cb.on_test_epoch_end(_make_trainer(), _make_pl_module())

        cb.map_metric.reset.assert_called_once()
        assert cb._f1_local == {}

    def test_segmentation_extra_metrics_logged(self) -> None:
        """test/segm_mAP_50_95 and test/segm_mAP_50 are logged in segm mode."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        segm_metrics = self._minimal_metrics(pfx="bbox_")
        segm_metrics["segm_map"] = torch.tensor(0.35)
        segm_metrics["segm_map_50"] = torch.tensor(0.55)
        cb.map_metric.compute.return_value = segm_metrics
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "test/segm_mAP_50_95" in logged_keys
        assert "test/segm_mAP_50" in logged_keys

    def test_per_class_ap_logged_when_classes_present(self) -> None:
        """test/AP/<name> is logged for each class when class metrics are present."""
        cb = COCOEvalCallback()
        cb._class_names = ["cat", "dog"]
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = self._minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "test/AP/cat" in logged_keys
        assert "test/AP/dog" in logged_keys

    def test_per_class_ap_falls_back_to_str_id_when_no_class_names(self) -> None:
        """test/AP/<id> is logged when class_names is empty."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = self._minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5])
        metrics["classes"] = torch.tensor([3])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "test/AP/3" in logged_keys
