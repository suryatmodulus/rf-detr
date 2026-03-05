# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exponential Moving Average callback compatible with ``ModelEma``."""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
from pytorch_lightning.callbacks import WeightAveraging


class RFDETREMACallback(WeightAveraging):
    """Exponential Moving Average with optional tau-based warm-up.

    Drop-in replacement for ``rfdetr.util.utils.ModelEma`` that delegates
    weight averaging to PyTorch Lightning's ``WeightAveraging`` callback.
    The ``_avg_fn`` reproduces the exact same formula as ``ModelEma``
    (1-indexed ``updates`` counter, optional ``tau`` warm-up).

    .. important::

        This is a **custom EMA implementation** ported from the legacy
        ``ModelEma`` for behavioural parity.  It has **not** been
        experimentally compared against
        :class:`pytorch_lightning.callbacks.EMAWeightAveraging` (available
        since PTL 2.6.0), which uses a simpler fixed-decay formula without
        the tau warm-up.  Before stabilising this API, run ablations
        comparing training mAP curves with both implementations.  If the PTL
        baseline proves equivalent, replace this class with::

            EMAWeightAveraging(decay=tc.ema_decay, update_starting_at_epoch=0)

        to eliminate the custom subclass entirely.

    Args:
        decay: Base EMA decay factor. Corresponds to ``TrainConfig.ema_decay``.
        tau: Warm-up time constant (in optimizer steps). When > 0 the
            effective decay ramps from 0 towards *decay* following
            ``decay * (1 - exp(-updates / tau))``. Corresponds to
            ``TrainConfig.ema_tau``.
    """

    def __init__(self, decay: float = 0.993, tau: int = 100) -> None:
        warnings.warn(
            "RFDETREMACallback uses a custom EMA implementation (tau warm-up) "
            "ported from the legacy ModelEma.  pytorch_lightning.callbacks.EMAWeightAveraging "
            "(PTL 2.6+) provides a simpler fixed-decay alternative that may be equivalent in "
            "practice.  See the class docstring for migration guidance.",
            UserWarning,
            stacklevel=2,
        )
        # Must be set before super().__init__ because avg_fn=self._avg_fn
        # creates a reference to the bound method stored in _kwargs.
        self._decay = decay
        self._tau = tau
        super().__init__(use_buffers=True, avg_fn=self._avg_fn)

    def _avg_fn(
        self,
        averaged_param: torch.Tensor,
        model_param: torch.Tensor,
        num_averaged: int,
    ) -> torch.Tensor:
        """Compute the EMA update for a single parameter tensor.

        Matches the ``ModelEma`` formula where ``updates`` is 1-indexed:
        PTL's ``num_averaged`` starts at 0 (incremented *after* calling
        ``avg_fn``), so ``updates = num_averaged + 1`` reproduces the
        same sequence of effective decay values.

        Args:
            averaged_param: Current EMA parameter value.
            model_param: Corresponding live model parameter value.
            num_averaged: Number of models averaged so far (0-indexed).

        Returns:
            Updated EMA parameter tensor.
        """
        updates = num_averaged + 1  # match ModelEma 1-indexed counter
        if self._tau > 0:
            effective_decay = self._decay * (1 - math.exp(-updates / self._tau))
        else:
            effective_decay = self._decay
        return averaged_param * effective_decay + model_param * (1.0 - effective_decay)

    def should_update(
        self,
        step_idx: Optional[int] = None,
        epoch_idx: Optional[int] = None,
    ) -> bool:
        """Return ``True`` after every optimizer step and every epoch end.

        The base ``WeightAveraging`` only updates on steps. This override
        also triggers an update at epoch boundaries, matching RF-DETR's
        existing EMA behaviour.

        Args:
            step_idx: Index of the last optimizer step, or ``None``.
            epoch_idx: Index of the last epoch, or ``None``.

        Returns:
            Whether the averaged model should be updated.
        """
        return step_idx is not None or epoch_idx is not None
