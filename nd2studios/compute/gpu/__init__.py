"""
V1.39 Phase 7 — optional GPU acceleration for hot analysis ops.

This package is intentionally *optional*. Every public function works
unchanged when CuPy / cucim are absent: detection returns
``available=False``, ``is_gpu_enabled()`` returns False, and the ops
in :mod:`nd2studios.compute.gpu.ops` pass straight through to their
scipy / scikit-image originals.

Callers import named functions from this package rather than swapping
between numpy and cupy in their own code — the host↔device boundary
lives entirely inside :mod:`nd2studios.compute.gpu.ops`, so consumers
always receive a numpy array and never have to think about which
backend ran underneath.

Public API
----------

- :func:`gpu_status` — one-shot dict describing GPU availability.
- :func:`log_gpu_status` — emit a single startup info line via the
  stdlib :mod:`logging` module.
- :func:`configure` — flip the runtime GPU/CPU dispatch flag.
- :func:`is_gpu_enabled` — read the flag.
- :func:`to_xp` / :func:`to_numpy` — boundary helpers for advanced
  callers (most code never needs them).
- :func:`gaussian` / :func:`gaussian_filter` / :func:`gaussian_laplace`
  / :func:`threshold_otsu` — the four ops the V1.33 baseline flagged
  as analysis hotspots, wrapped so they accept and return numpy
  arrays regardless of the active backend.
"""
from __future__ import annotations

from nd2studios.compute.gpu.array import (
    configure,
    is_gpu_enabled,
    to_numpy,
    to_xp,
)
from nd2studios.compute.gpu.detect import gpu_status, log_gpu_status
from nd2studios.compute.gpu.ops import (
    gaussian,
    gaussian_filter,
    gaussian_laplace,
    threshold_otsu,
)

__all__ = [
    "configure",
    "gaussian",
    "gaussian_filter",
    "gaussian_laplace",
    "gpu_status",
    "is_gpu_enabled",
    "log_gpu_status",
    "threshold_otsu",
    "to_numpy",
    "to_xp",
]
