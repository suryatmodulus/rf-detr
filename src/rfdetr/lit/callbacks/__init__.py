# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightning callbacks for RF-DETR training (Phase 3+)."""

from rfdetr.lit.callbacks.ema import RFDETREMACallback

__all__ = ["RFDETREMACallback"]
