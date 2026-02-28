# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""PyTorch Lightning integration layer for RF-DETR.

This package provides the transitional Lightning module, data module, callbacks,
and CLI required for the PTL migration. It coexists with the existing engine.py /
main.py until those are removed in a later chapter.

Exports:
    RFDETRModule: LightningModule wrapping the RF-DETR model and training loop.
    RFDETRDataModule: LightningDataModule wrapping dataset construction and loaders.
    build_trainer: Factory that assembles a PTL Trainer from RF-DETR configs.
"""

from pytorch_lightning import Trainer

from rfdetr.lit.callbacks import (
    BestModelCallback,
    DropPathCallback,
    MetricsPlotCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from rfdetr.lit.callbacks.coco_eval import COCOEvalCallback
from rfdetr.lit.datamodule import RFDETRDataModule
from rfdetr.lit.module import RFDETRModule


def build_trainer(train_config: "TrainConfig", model_config: "ModelConfig") -> Trainer:  # type: ignore[name-defined]
    """Assemble a PTL ``Trainer`` with the Chapter 3 callback stack.

    This is a Chapter 3 stub.  Chapter 4 will extend it with logger wiring,
    gradient clipping, ``sync_bn``, precision resolution, and ``LightningCLI``
    integration.

    Args:
        train_config: Training hyperparameter configuration.
        model_config: Architecture configuration (used for segmentation flag).

    Returns:
        A configured ``pytorch_lightning.Trainer`` instance.
    """
    # TODO(Chapter 4): resolve precision from model_config.amp + device capability.
    # TODO(Chapter 4): configure TensorBoard / WandB / MLflow / ClearML loggers.
    # TODO(Chapter 4): add gradient_clip_val, sync_batchnorm, strategy from config.

    tc = train_config
    callbacks = []

    # EMA — disabled automatically for sharded strategies (Chapter 4 handles warning).
    if tc.use_ema:
        callbacks.append(RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau))

    # Drop-path / dropout scheduling.
    # vit_encoder_num_layers lives in the legacy argparse Namespace (default 12);
    # Chapter 4 will promote it to ModelConfig.
    if tc.drop_path > 0.0:
        callbacks.append(DropPathCallback(drop_path=tc.drop_path))

    # COCO mAP + F1 evaluation.
    callbacks.append(
        COCOEvalCallback(
            max_dets=tc.eval_max_dets,
            segmentation=model_config.segmentation_head,
        )
    )

    # Best-model checkpointing.
    callbacks.append(
        BestModelCallback(
            output_dir=tc.output_dir,
            monitor_ema="val/ema_mAP_50_95" if tc.use_ema else None,
            run_test=tc.run_test,
        )
    )

    # Optional early stopping.
    if tc.early_stopping:
        callbacks.append(
            RFDETREarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                use_ema=tc.early_stopping_use_ema,
            )
        )

    # Metrics plot saved at end of training.
    callbacks.append(MetricsPlotCallback(output_dir=tc.output_dir))

    return Trainer(
        max_epochs=tc.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        enable_progress_bar=tc.progress_bar,
        default_root_dir=tc.output_dir,
        logger=False,  # TODO(Chapter 4): wire loggers from tc.tensorboard / tc.wandb etc.
    )


__all__ = [
    "BestModelCallback",
    "DropPathCallback",
    "MetricsPlotCallback",
    "RFDETRDataModule",
    "RFDETREMACallback",
    "RFDETREarlyStopping",
    "RFDETRModule",
    "build_trainer",
]
