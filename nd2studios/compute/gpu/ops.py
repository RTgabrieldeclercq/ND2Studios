"""
GPU-aware reimplementations of the four hot analysis ops (V1.39 Phase 7).

Each function in this module mirrors the signature of the scipy /
scikit-image original so call sites swap their import and nothing
else. On a machine without CuPy / cucim, dispatch falls through to
the CPU implementation transparently.

Ops covered (chosen from the V1.33 profiling baseline):

- :func:`gaussian` — :func:`skimage.filters.gaussian`
  (``preserve_range=True`` is honoured). Used by tear detection.
- :func:`gaussian_filter` — :func:`scipy.ndimage.gaussian_filter`.
  Used by the Bright/Dark Spots DoG response.
- :func:`gaussian_laplace` — :func:`scipy.ndimage.gaussian_laplace`.
  Used by the Bright/Dark Spots LoG response.
- :func:`threshold_otsu` — :func:`skimage.filters.threshold_otsu`.
  Used by tear detection.

Boundary policy
---------------

Every function:

1. Returns numpy. The caller does not need to know which backend ran.
2. Catches GPU failures (``cupy.cuda.memory.OutOfMemoryError``,
   missing ``cucim``, NVML / driver errors) and falls back to CPU
   silently. A single warning per op name is logged the first time
   it fires — subsequent failures of the same op are silent so a
   busy Cellpose run does not flood the log.
3. Skips GPU dispatch for arrays smaller than ~256² (see
   :func:`~nd2studios.compute.gpu.array.should_dispatch_to_gpu`).
   Transfer cost dominates below that size; CPU wins by avoiding the
   round-trip.

Imports of scipy / scikit-image are top-level — they are required
dependencies of ND2Studios and always available. Imports of CuPy and
cucim are deferred to first use so this module is importable on
CPU-only machines without paying the cost of touching the GPU
namespace.
"""
from __future__ import annotations

import logging
from typing import Set

import numpy as np
from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
from scipy.ndimage import gaussian_laplace as _scipy_gaussian_laplace
from skimage.filters import gaussian as _sk_gaussian
from skimage.filters import threshold_otsu as _sk_threshold_otsu

from nd2studios.compute.gpu.array import (
    is_gpu_enabled,
    should_dispatch_to_gpu,
    to_numpy,
    to_xp,
)

log = logging.getLogger(__name__)

# Names of ops whose GPU branch has already logged a fallback warning.
# Lives at module scope so it persists across calls within one process.
_WARNED: Set[str] = set()


def _warn_once(op: str, exc: BaseException) -> None:
    """Log a single fallback warning per op name."""
    if op in _WARNED:
        return
    _WARNED.add(op)
    log.warning(
        "GPU op %s fell back to CPU (%s: %s) — future failures silent.",
        op, type(exc).__name__, exc,
    )


def _try_cucim_filters():
    """Return ``cucim.skimage.filters`` or ``None`` if cucim absent."""
    try:
        from cucim.skimage import filters as _cucim_filters
        return _cucim_filters
    except Exception:  # noqa: BLE001
        return None


def _try_cupyx_ndimage():
    """Return ``cupyx.scipy.ndimage`` or ``None`` if absent.

    Used for the scipy.ndimage-style ops (``gaussian_filter`` and
    ``gaussian_laplace``) where cucim does not provide a drop-in
    replacement. ``cupyx`` ships with CuPy.
    """
    try:
        import cupyx.scipy.ndimage as _cupyx_ndi
        return _cupyx_ndi
    except Exception:  # noqa: BLE001
        return None


def gaussian(image, sigma: float, *, preserve_range: bool = True):
    """Gaussian blur — :func:`skimage.filters.gaussian` shape-compatible.

    Only ``image``, ``sigma``, and ``preserve_range`` are exposed
    here because those are the only kwargs the V1.38 callers pass.
    Add more keyword passthroughs if a new caller needs them.
    """
    if not should_dispatch_to_gpu(image, op="gaussian"):
        return _sk_gaussian(image, sigma=sigma, preserve_range=preserve_range)

    cucim = _try_cucim_filters()
    if cucim is None:
        return _sk_gaussian(image, sigma=sigma, preserve_range=preserve_range)
    try:
        arr = to_xp(image)
        out = cucim.gaussian(arr, sigma=sigma, preserve_range=preserve_range)
        return to_numpy(out)
    except Exception as exc:  # noqa: BLE001
        _warn_once("gaussian", exc)
        return _sk_gaussian(image, sigma=sigma, preserve_range=preserve_range)


def gaussian_filter(image, sigma):
    """:func:`scipy.ndimage.gaussian_filter` shape-compatible."""
    if not should_dispatch_to_gpu(image, op="gaussian_filter"):
        return _scipy_gaussian_filter(image, sigma=sigma)

    cupyx_ndi = _try_cupyx_ndimage()
    if cupyx_ndi is None:
        return _scipy_gaussian_filter(image, sigma=sigma)
    try:
        arr = to_xp(image)
        out = cupyx_ndi.gaussian_filter(arr, sigma=sigma)
        return to_numpy(out)
    except Exception as exc:  # noqa: BLE001
        _warn_once("gaussian_filter", exc)
        return _scipy_gaussian_filter(image, sigma=sigma)


def gaussian_laplace(image, sigma):
    """:func:`scipy.ndimage.gaussian_laplace` shape-compatible."""
    if not should_dispatch_to_gpu(image, op="gaussian_laplace"):
        return _scipy_gaussian_laplace(image, sigma=sigma)

    cupyx_ndi = _try_cupyx_ndimage()
    if cupyx_ndi is None:
        return _scipy_gaussian_laplace(image, sigma=sigma)
    try:
        arr = to_xp(image)
        out = cupyx_ndi.gaussian_laplace(arr, sigma=sigma)
        return to_numpy(out)
    except Exception as exc:  # noqa: BLE001
        _warn_once("gaussian_laplace", exc)
        return _scipy_gaussian_laplace(image, sigma=sigma)


def threshold_otsu(image) -> float:
    """:func:`skimage.filters.threshold_otsu` shape-compatible.

    Returns a Python float regardless of backend so callers can
    compare against literal numeric thresholds without an explicit
    cast.
    """
    if not should_dispatch_to_gpu(image, op="threshold_otsu"):
        return float(_sk_threshold_otsu(image))

    cucim = _try_cucim_filters()
    if cucim is None:
        return float(_sk_threshold_otsu(image))
    try:
        arr = to_xp(image)
        return float(cucim.threshold_otsu(arr))
    except Exception as exc:  # noqa: BLE001
        _warn_once("threshold_otsu", exc)
        return float(_sk_threshold_otsu(image))


# Re-export the GPU-aware flag so callers can branch on it for
# things like ``models.Cellpose(gpu=is_gpu_enabled())`` without
# having to import from two places.
__all__ = [
    "gaussian",
    "gaussian_filter",
    "gaussian_laplace",
    "is_gpu_enabled",
    "threshold_otsu",
]
