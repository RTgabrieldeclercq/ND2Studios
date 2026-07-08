"""Image registration engine (V1.56).

Pure, Qt-free routines for estimating a spatial transform that aligns a
*moving* image/series onto a *reference* and resampling through it. Used by the
registration :class:`~nd2studios.core.plugin_registry.EnhancementPlugin`
(per-channel drift/rigid stabilization) and available for a future cross-channel
``RegistrationMethod`` registry (review Part 3.4). No PySide6 import.
"""
from __future__ import annotations
