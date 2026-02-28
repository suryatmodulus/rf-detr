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

import warnings

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger

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
    """Assemble a PTL ``Trainer`` with the full RF-DETR callback and logger stack.

    Resolves training precision from ``model_config.amp`` and device capability,
    guards EMA against sharded strategies, wires conditional loggers, and applies
    promoted training knobs (gradient clipping, sync_batchnorm, strategy).

    Fields ``clip_max_norm``, ``seed``, and ``sync_bn`` are read via ``getattr``
    with safe defaults so this function works before those fields are promoted to
    ``TrainConfig`` in T4-2.

    Args:
        train_config: Training hyperparameter configuration.
        model_config: Architecture configuration (used for precision and segmentation).

    Returns:
        A configured ``pytorch_lightning.Trainer`` instance.
    """
    tc = train_config

    # --- Seed ---
    seed = getattr(tc, "seed", None)
    if seed is not None:
        seed_everything(seed, workers=True)

    # --- Precision resolution ---
    def _resolve_precision() -> str:
        if not model_config.amp:
            return "32-true"
        if torch.cuda.is_available():
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            if is_bf16_supported():
                return "bf16-mixed"
            return "16-mixed"
        return "32-true"

    # --- Strategy + EMA sharding guard ---
    strategy = getattr(tc, "strategy", "auto")
    sharded = any(s in str(strategy).lower() for s in ("fsdp", "deepspeed"))
    enable_ema = bool(tc.use_ema) and not sharded
    if tc.use_ema and sharded:
        warnings.warn(
            f"EMA disabled: RFDETREMACallback is not compatible with sharded strategies "
            f"(strategy={strategy!r}). Set use_ema=False to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    # --- Build callbacks ---
    callbacks = []

    if enable_ema:
        callbacks.append(RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau))

    # Drop-path / dropout scheduling (vit_encoder_num_layers defaults to 12).
    if tc.drop_path > 0.0:
        callbacks.append(DropPathCallback(drop_path=tc.drop_path))

    # COCO mAP + F1 evaluation.
    callbacks.append(
        COCOEvalCallback(
            max_dets=tc.eval_max_dets,
            segmentation=model_config.segmentation_head,
        )
    )

    # Best-model checkpointing — monitor EMA metric only when EMA is active.
    callbacks.append(
        BestModelCallback(
            output_dir=tc.output_dir,
            monitor_ema="val/ema_mAP_50_95" if enable_ema else None,
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

    # --- Build loggers ---
    # Each logger is guarded by a try/except because tensorboard, wandb, and mlflow
    # are optional dependencies (installed via the [metrics] extra).  A missing dep
    # emits a UserWarning instead of crashing.
    loggers = []

    if tc.tensorboard:
        try:
            loggers.append(
                TensorBoardLogger(
                    save_dir=tc.output_dir,
                    name=tc.run or "rfdetr",
                    version="",
                )
            )
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"TensorBoard logging disabled: {exc}. Install with `pip install tensorboard`.",
                UserWarning,
                stacklevel=2,
            )

    if tc.wandb:
        try:
            loggers.append(
                WandbLogger(
                    name=tc.run,
                    project=tc.project,
                    save_dir=tc.output_dir,
                )
            )
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"WandB logging disabled: {exc}. Install with `pip install wandb`.",
                UserWarning,
                stacklevel=2,
            )

    if tc.mlflow:
        try:
            loggers.append(
                MLFlowLogger(
                    experiment_name=tc.project or "rfdetr",
                    run_name=tc.run,
                    save_dir=tc.output_dir,
                )
            )
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"MLflow logging disabled: {exc}. Install with `pip install mlflow`.",
                UserWarning,
                stacklevel=2,
            )

    if tc.clearml:
        warnings.warn(
            "ClearML logging is not supported via a native PTL logger in this version. "
            "Metrics will not be logged to ClearML. Use the ClearML SDK callback directly "
            "or wait for a dedicated ClearML PTL logger integration.",
            UserWarning,
            stacklevel=2,
        )

    # --- Promoted config fields (getattr fallbacks until T4-2 promotes them) ---
    clip_max_norm: float = getattr(tc, "clip_max_norm", 0.1)
    sync_bn: bool = getattr(tc, "sync_bn", False)

    return Trainer(
        max_epochs=tc.epochs,
        accelerator="auto",
        devices="auto",
        strategy=strategy,
        precision=_resolve_precision(),
        accumulate_grad_batches=tc.grad_accum_steps,
        gradient_clip_val=clip_max_norm,
        sync_batchnorm=sync_bn,
        callbacks=callbacks,
        logger=loggers if loggers else False,
        enable_progress_bar=tc.progress_bar,
        default_root_dir=tc.output_dir,
        log_every_n_steps=50,
        deterministic=False,
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
