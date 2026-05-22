"""
Numpy ↔ CuPy dispatch flag (V1.39 Phase 7).

A single process-wide ``_USE_GPU`` boolean drives whether
:mod:`nd2studios.compute.gpu.ops` routes through CuPy / cucim or
through scipy / scikit-image. The flag is read inside each op call
(not captured at import time) so the Performance dialog can flip it
mid-session without restarting the app.

Three rules govern dispatch:

1. **Hard env override beats GUI choice.** Setting
   ``ND2_DISABLE_GPU_ANALYSIS=1`` pins the flag to False; the
   Performance dialog's checkbox becomes inert.
2. **GPU is opt-in, even when available.** The default at startup is
   False — the user must explicitly turn it on. This keeps "what just
   changed?" questions tractable when comparing analysis results.
3. **Configure validates.** Calling :func:`configure(True)` on a
   machine without CuPy is a no-op that returns False; callers can
   rely on the return value to update GUI state.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


# Hard env override. Same convention as
# ``ND2_DISABLE_GPU_DISPLAY`` from V1.36 Phase 4.
DISABLE_ENV = "ND2_DISABLE_GPU_ANALYSIS"

# Minimum array dimension below which we never dispatch to GPU.
# Host↔device transfer dominates for tiny arrays; the threshold is
# deliberately conservative so a 256² (or larger) frame goes to GPU
# while a 32×32 manual-mask patch stays on CPU.
_MIN_SIZE_FOR_GPU = 256 * 256


_USE_GPU: bool = False


def _hard_disabled() -> bool:
    """True iff the env override is set; checked on every flip."""
    return os.environ.get(DISABLE_ENV, "") == "1"


def _has_cupy() -> bool:
    """Cheap CuPy importability probe (also implicitly checks device)."""
    try:
        import cupy as cp  # noqa: F401
        return bool(cp.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def configure(use_gpu: bool) -> bool:
    """Flip the dispatch flag; return the effective value.

    Always returns False if the env override is set or CuPy is not
    importable, regardless of the requested value. Callers (e.g. the
    Performance dialog) should use the return value, not their own
    input, to update displayed state.
    """
    global _USE_GPU
    if not use_gpu:
        _USE_GPU = False
        return False
    if _hard_disabled():
        _USE_GPU = False
        return False
    if not _has_cupy():
        _USE_GPU = False
        return False
    _USE_GPU = True
    return True


def is_gpu_enabled() -> bool:
    """Read the dispatch flag. Lockless; flips are atomic on CPython."""
    return _USE_GPU


def should_dispatch_to_gpu(array: Any) -> bool:
    """Return True iff this array is big enough to benefit from GPU.

    Used by :mod:`compute.gpu.ops` to decide between transfer-and-run
    on GPU vs. running CPU in place. The threshold lives here so it
    can be tuned without touching every op.
    """
    if not _USE_GPU:
        return False
    if array is None:
        return False
    try:
        return int(np.asarray(array).size) >= _MIN_SIZE_FOR_GPU
    except Exception:  # noqa: BLE001 — defensive; never fail the caller
        return False


def to_xp(arr: Any) -> Any:
    """Move ``arr`` onto the active backend.

    Numpy in/out when GPU is off; numpy → cupy when on. Already-cupy
    arrays pass through unchanged. Used by :mod:`compute.gpu.ops` and
    by callers that batch multiple ops on the same array (so they
    can transfer once and keep the device tensor across calls).
    """
    if not _USE_GPU:
        return np.asarray(arr)
    try:
        import cupy as cp
        return cp.asarray(arr)
    except Exception:  # noqa: BLE001 — fall back rather than break
        return np.asarray(arr)


def to_numpy(arr: Any) -> np.ndarray:
    """Move ``arr`` back to host numpy regardless of source backend.

    Idempotent on numpy input. A CuPy ndarray copies to host; any
    other type round-trips through ``np.asarray`` so downstream code
    always gets a real numpy array.
    """
    if arr is None:
        return arr  # type: ignore[return-value]
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(arr)
