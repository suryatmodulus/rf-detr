# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for resuming training from checkpoint."""

import warnings
from pathlib import Path

from rfdetr import RFDETRNano


def test_resume_with_completed_epochs_returns_early(synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
    """Passing start_epoch emits DeprecationWarning; training completes without error.

    In the legacy engine.py path, ``start_epoch=epochs`` caused the training loop
    to be skipped (``range(start_epoch, epochs)`` was empty), which triggered an
    ``UnboundLocalError`` when accessing ``test_stats``.

    In the PTL path, ``start_epoch`` is a deprecated kwarg that is absorbed and
    ignored (PTL resumes automatically via ``ckpt_path``).  Training runs normally
    for the requested number of epochs and must not raise any exception.

    Args:
        synthetic_shape_dataset_dir: Path to a synthetic COCO-style dataset.
        tmp_path: Pytest temporary directory.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train_ptl(
            dataset_dir=str(synthetic_shape_dataset_dir),
            epochs=1,
            start_epoch=1,
            batch_size=1,
            grad_accum_steps=1,
            output_dir=str(output_dir),
            device="cpu",
        )

    depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("start_epoch" in str(w.message) for w in depr), "Expected a DeprecationWarning mentioning start_epoch"


def test_resume_with_completed_epochs_calls_on_train_end_callback(
    synthetic_shape_dataset_dir: Path, tmp_path: Path
) -> None:
    """Old-style on_train_end callbacks are not forwarded to PTL.

    In the legacy engine.py path, callbacks added to ``model.callbacks["on_train_end"]``
    were invoked at the end of training (including when the loop was skipped).
    In the PTL path the old-style callback dict on the model instance is not consulted;
    use PTL ``Callback`` objects via ``build_trainer()`` instead.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.callbacks["on_train_end"].append(_callback)

    model.train_ptl(
        dataset_dir=str(synthetic_shape_dataset_dir),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        device="cpu",
    )

    # Old-style callbacks on model.callbacks are no longer invoked in the PTL path.
    assert callback_calls == 0
