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
"""

from rfdetr.lit.datamodule import RFDETRDataModule
from rfdetr.lit.module import RFDETRModule

__all__ = ["RFDETRModule", "RFDETRDataModule"]
