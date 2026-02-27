# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Package-level pytest fixtures for tests/lit/.

Provides cross-test cleanup that prevents class-level state from leaking
between individual tests in the lit/ test package.
"""

import pytest


@pytest.fixture(autouse=True)
def _restore_rfdetr_module_trainer_property():
    """Restore RFDETRModule.trainer to the LightningModule parent property after each test.

    Several unit tests in test_module.py patch the ``trainer`` property directly
    on the ``RFDETRModule`` class (``type(module).trainer = property(...)``).
    Without cleanup this mutates the class for the remainder of the session and
    breaks ``Trainer.fit()`` calls in smoke tests (PTL cannot set ``.trainer``
    on the module because the patched property has no setter).

    This fixture deletes any class-level override from ``RFDETRModule.__dict__``
    after every test, so the next test starts with a clean class that inherits
    PTL's read/write ``trainer`` descriptor from ``LightningModule``.
    """
    yield
    # Lazy import so the fixture does not force module import at collection time.
    from rfdetr.lit.module import RFDETRModule

    if "trainer" in RFDETRModule.__dict__:
        delattr(RFDETRModule, "trainer")
