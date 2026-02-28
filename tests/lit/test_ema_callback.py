# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit and parity tests for RFDETREMACallback."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from rfdetr.lit.callbacks.ema import RFDETREMACallback
from rfdetr.util.utils import ModelEma


class TestAvgFnDecayFormula:
    """Verify the tau / no-tau decay formula matches ModelEma."""

    @pytest.mark.parametrize(
        "num_averaged",
        [
            pytest.param(0, id="step-0"),
            pytest.param(5, id="step-5"),
            pytest.param(99, id="step-99"),
        ],
    )
    def test_tau_zero_uses_fixed_decay(self, num_averaged: int) -> None:
        """With tau=0 the effective decay equals the base decay at every step."""
        decay = 0.99
        cb = RFDETREMACallback(decay=decay, tau=0)
        ema_val = torch.tensor(1.0)
        model_val = torch.tensor(2.0)

        result = cb._avg_fn(ema_val, model_val, num_averaged)

        expected = ema_val * decay + model_val * (1.0 - decay)
        assert torch.allclose(result, expected, atol=1e-7)

    def test_tau_warmup_at_step_1(self) -> None:
        """At the first call (num_averaged=0) with tau>0 the effective decay
        uses updates=1 matching ModelEma's 1-indexed counter."""
        decay = 0.993
        tau = 100
        cb = RFDETREMACallback(decay=decay, tau=tau)
        ema_val = torch.tensor(1.0)
        model_val = torch.tensor(2.0)

        result = cb._avg_fn(ema_val, model_val, num_averaged=0)

        updates = 1  # num_averaged + 1
        effective_decay = decay * (1 - math.exp(-updates / tau))
        expected = ema_val * effective_decay + model_val * (1.0 - effective_decay)
        assert torch.allclose(result, expected, atol=1e-7)


class TestModelEmaParity:
    """Ensure N-step EMA weights match ModelEma exactly."""

    def test_avg_fn_matches_modelema_weight_parity(self) -> None:
        """Simulate 500 update steps and compare final EMA weights with
        ModelEma.module to confirm numerical parity."""
        torch.manual_seed(42)
        n_steps = 500
        decay = 0.993
        tau = 100

        model = nn.Linear(4, 4)
        model_ema = ModelEma(model, decay=decay, tau=tau)
        cb = RFDETREMACallback(decay=decay, tau=tau)

        # Initialise manual EMA state from model (same as ModelEma deepcopy)
        ema_weights: dict[str, torch.Tensor] = {name: p.clone() for name, p in model.named_parameters()}

        for step in range(n_steps):
            # Perturb model parameters
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)

            # Update legacy ModelEma
            model_ema.update(model)

            # Replicate update via callback avg_fn
            model_weights = {name: p.clone() for name, p in model.named_parameters()}
            for name in ema_weights:
                ema_weights[name] = cb._avg_fn(ema_weights[name], model_weights[name], step)

        # Compare
        legacy_state = dict(model_ema.module.named_parameters())
        for name, cb_val in ema_weights.items():
            assert torch.allclose(cb_val, legacy_state[name], atol=1e-5), (
                f"Parity failed for {name}: max diff = {(cb_val - legacy_state[name]).abs().max().item()}"
            )


class TestShouldUpdate:
    """Verify should_update triggers on steps and epochs."""

    def test_should_update_on_step(self) -> None:
        cb = RFDETREMACallback()
        assert cb.should_update(step_idx=42) is True

    def test_should_update_on_epoch(self) -> None:
        cb = RFDETREMACallback()
        assert cb.should_update(epoch_idx=3) is True

    def test_should_update_neither(self) -> None:
        cb = RFDETREMACallback()
        assert cb.should_update() is False
