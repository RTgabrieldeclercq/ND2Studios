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

import logging
import os
import time
from typing import Any, Optional, Set

import numpy as np


log = logging.getLogger(__name__)


# Hard env override. Same convention as
# ``ND2_DISABLE_GPU_DISPLAY`` from V1.36 Phase 4.
DISABLE_ENV = "ND2_DISABLE_GPU_ANALYSIS"

# Minimum array dimension below which we never dispatch to GPU.
# Host↔device transfer dominates for tiny arrays; the threshold is
# deliberately conservative so a 256² (or larger) frame goes to GPU
# while a 32×32 manual-mask patch stays on CPU.
_MIN_SIZE_FOR_GPU = 256 * 256


_USE_GPU: bool = False

# ── V1.41 proactive VRAM guard ─────────────────────────────────────
# ``memGetInfo`` is cheap individually but a recipe running 30 ops/sec
# burns ~6,000 driver calls per minute. Cache the free-bytes value for
# a short TTL so the guard imposes near-zero amortized cost.
_VRAM_CACHE_TTL_S = 0.20
_vram_cache_ts: float = 0.0
_vram_cache_free_bytes: int = 0

# Names of ops whose VRAM guard has already logged a fallback warning,
# so a tight loop doesn't spam the log on every frame.
_VRAM_WARNED: Set[str] = set()

# Default safety multiplier: cuCIM / cupyx kernels allocate intermediate
# buffers of similar size to the input. 2.5× covers a 2D blur (input +
# kernel buffer + output ≈ 3× nbytes) with a margin.
_VRAM_SAFETY_FACTOR_DEFAULT = 2.5


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


def _cached_free_vram_bytes() -> int:
    """Return free VRAM in bytes, cached for :data:`_VRAM_CACHE_TTL_S`.

    Returns 0 on any failure (driver not loaded, CuPy not importable,
    device gone). Zero is treated as "not enough" by the guard so we
    fall back to CPU rather than dispatching against an unknown state.
    """
    global _vram_cache_ts, _vram_cache_free_bytes
    now = time.monotonic()
    if (now - _vram_cache_ts) < _VRAM_CACHE_TTL_S:
        return _vram_cache_free_bytes
    try:
        import cupy as cp
        free, _total = cp.cuda.runtime.memGetInfo()
        _vram_cache_free_bytes = int(free)
    except Exception:  # noqa: BLE001 — never fail the caller
        _vram_cache_free_bytes = 0
    _vram_cache_ts = now
    return _vram_cache_free_bytes


def _vram_warn_once(op: str, needed: int, free: int) -> None:
    """Log a single VRAM-shortage warning per op name.

    Mirrors the pattern in :mod:`compute.gpu.ops` so the operator
    sees one informative line per op rather than per-frame spam.
    """
    if op in _VRAM_WARNED:
        return
    _VRAM_WARNED.add(op)
    log.warning(
        "GPU op %s skipped to CPU: needs %.2f GB VRAM, %.2f GB free "
        "— future occurrences silent.",
        op or "<unspecified>",
        needed / 1024**3,
        free / 1024**3,
    )


def should_dispatch_to_gpu(
    array: Any,
    *,
    op: str = "",
    safety_factor: float = _VRAM_SAFETY_FACTOR_DEFAULT,
) -> bool:
    """Return True iff this array is big enough to benefit from GPU
    *and* fits in current free VRAM.

    Used by :mod:`compute.gpu.ops` to decide between transfer-and-run
    on GPU vs. running CPU in place. The size threshold filters out
    transfer-bound tiny arrays; the proactive VRAM check (V1.41)
    catches the OOM before dispatch instead of catching the
    :class:`cupy.cuda.memory.OutOfMemoryError` after the kernel has
    already torched the device pool.

    ``op`` is an optional event name forwarded to
    :func:`_vram_warn_once` so the log records *which* op fell back.
    Callers (e.g. ``nd2studios.compute.gpu.ops.gaussian``) already
    pass their own name.

    ``safety_factor`` accounts for kernel-internal scratch buffers
    (cuCIM / cupyx temporaries are typically 1–2× the input). The
    default of 2.5× is conservative; a known-light op can lower it.
    """
    if not _USE_GPU:
        return False
    if array is None:
        return False
    try:
        np_view = np.asarray(array)
        size = int(np_view.size)
    except Exception:  # noqa: BLE001
        return False
    if size < _MIN_SIZE_FOR_GPU:
        return False

    try:
        needed = int(np_view.nbytes * float(safety_factor))
    except Exception:  # noqa: BLE001
        return True  # If nbytes is unreadable, fall back to size-only check.

    free = _cached_free_vram_bytes()
    if free <= 0:
        # VRAM probe failed — be safe and let the op handle its own
        # fallback (the ops module catches CuPy errors anyway).
        return True
    if needed > free:
        _vram_warn_once(op, needed, free)
        return False
    return True


def reset_vram_cache() -> None:
    """Invalidate the cached free-VRAM reading.

    Tests and the diagnostics panel call this so they observe a
    fresh ``memGetInfo`` value rather than the previous 200 ms-stale
    one. Not needed during normal operation.
    """
    global _vram_cache_ts, _vram_cache_free_bytes
    _vram_cache_ts = 0.0
    _vram_cache_free_bytes = 0


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
