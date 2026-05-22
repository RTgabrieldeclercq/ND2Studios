"""
GPU detection and one-time startup logging (V1.39 Phase 7).

Two public helpers:

- :func:`gpu_status` — returns a dict describing whether CuPy is
  importable, whether at least one CUDA device is visible, and
  (when available) the device name and total memory in GiB. Always
  returns; never raises — the CUDA driver layer can throw runtime
  errors that need to be swallowed so the rest of the app keeps
  going.
- :func:`log_gpu_status` — emits a single info line summarizing the
  result. Called once at app startup from ``__main__``.

The detection cost is paid only on the first call; subsequent calls
re-evaluate (cheap) so a user plugging in an eGPU mid-session sees
the new state on the next Performance dialog open.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def gpu_status() -> Dict[str, Any]:
    """Return a dict describing GPU availability.

    Schema::

        {
            "available": bool,
            "reason": str,            # human-readable diagnostic
            "device_name": str | None,
            "memory_gb": float,        # 0.0 when unavailable
            "cucim": bool,             # cucim importable
        }

    The ``cucim`` flag matters because CuPy alone is not enough to
    accelerate the scikit-image-style ops in
    :mod:`nd2studios.compute.gpu.ops`. Without ``cucim`` we can still
    move arrays to the GPU but every op would fall back to CPU; the
    flag lets the Performance dialog explain that to the user.
    """
    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"cupy not importable: {type(exc).__name__}",
            "device_name": None,
            "memory_gb": 0.0,
            "cucim": False,
        }

    # Cupy is importable; probe the device. Failures here are
    # typically driver / WSL mismatch / no CUDA card.
    import cupy as cp  # second import is cheap, satisfies type checkers

    try:
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"no CUDA devices: {type(exc).__name__}: {exc}",
            "device_name": None,
            "memory_gb": 0.0,
            "cucim": _has_cucim(),
        }

    if n <= 0:
        return {
            "available": False,
            "reason": "no CUDA devices",
            "device_name": None,
            "memory_gb": 0.0,
            "cucim": _has_cucim(),
        }

    try:
        cp.cuda.Device(0).use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": (
                "cupy import succeeded but device query failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            "device_name": None,
            "memory_gb": 0.0,
            "cucim": _has_cucim(),
        }

    raw_name = props.get("name", b"")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8", errors="replace")
    return {
        "available": True,
        "reason": "ok",
        "device_name": str(raw_name).strip(),
        "memory_gb": float(total_bytes) / (1024 ** 3),
        "free_memory_gb": float(free_bytes) / (1024 ** 3),
        "cucim": _has_cucim(),
    }


def _has_cucim() -> bool:
    """Cheap one-shot cucim availability probe."""
    try:
        import cucim  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def log_gpu_status() -> None:
    """Emit one info line summarizing GPU availability.

    Called from :mod:`nd2studios.__main__` after Qt is up so the
    message lands in the same sink as the rest of ND2Studios's
    startup output. Safe to call multiple times — each call refreshes
    the status — but the convention is "once per app launch".
    """
    s = gpu_status()
    if s["available"]:
        cucim_note = "" if s["cucim"] else " (cucim missing — analysis ops will run on CPU)"
        log.info(
            "GPU available: %s, %.1f GB%s",
            s["device_name"], s["memory_gb"], cucim_note,
        )
    else:
        log.info("GPU not available: %s", s["reason"])
