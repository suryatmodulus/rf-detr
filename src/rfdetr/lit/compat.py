# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Legacy compatibility adapters for the migration period (Phase 0 / T6).

Provides thin wrappers that convert PTL validation_step outputs back to the
stats schema expected by benchmark tests and the existing engine.evaluate() API.
"""

# TODO(Chapter 1 / T6): implement engine.evaluate() compat wrapper
