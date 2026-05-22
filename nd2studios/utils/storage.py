"""Storage-class detection — addendum Phase 2 Addition A.

The addendum (``08_addendum_parallelism.md``) calls out that NVMe-class
drives sustain useful work from 2–3 concurrent reader threads, whereas
spinning disks and SATA SSDs are saturated by one. ND2Studios's V1.34
:class:`~nd2studios.utils.threading.IOWorker` defaults to a single
reader; this module is the runtime hint that lets the viewer (and any
other multi-worker caller) ask for a sensible thread count without
hard-coding it.

Detection is best-effort: we don't claim to identify every NVMe device
on every OS, only to err on the side of *fewer* workers when in doubt.
A wrong "fast" verdict wastes a couple of MB of file-handle state; a
wrong "slow" verdict just falls back to ND2Studios's V1.34 behavior.

Example
-------
>>> from pathlib import Path
>>> from nd2studios.utils.storage import (
...     is_fast_storage, recommended_io_thread_count,
... )
>>> is_fast_storage(Path("/data/run42.nd2"))
True
>>> recommended_io_thread_count(Path("/data/run42.nd2"))
2
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import psutil  # already a project dependency via utils/resources.py
except Exception:  # noqa: BLE001 — psutil is optional defense-in-depth
    psutil = None  # type: ignore[assignment]


def _linux_is_nvme(path: Path) -> Optional[bool]:
    """Linux: read ``/sys/block/<dev>/queue/rotational`` for the backing device.

    Returns ``True`` if the device is non-rotational (NVMe / SSD),
    ``False`` if rotational (HDD), and ``None`` when the lookup is
    inconclusive (psutil missing, mount-point mismatch, sysfs absent).
    """
    if psutil is None:
        return None
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:  # noqa: BLE001
        return None
    path_str = str(path.resolve())
    dev = None
    best_len = -1
    for p in partitions:
        if path_str.startswith(p.mountpoint) and len(p.mountpoint) > best_len:
            dev = p.device
            best_len = len(p.mountpoint)
    if not dev:
        return None
    base = Path(dev).name.rstrip("0123456789")
    rot_path = Path(f"/sys/block/{base}/queue/rotational")
    try:
        if rot_path.exists():
            return rot_path.read_text().strip() == "0"
    except OSError:
        return None
    return None


def is_fast_storage(path: Path) -> bool:
    """Return ``True`` when *path* is on NVMe-class storage.

    Heuristic and platform-specific:

    - **Linux** — consults ``/sys/block/.../queue/rotational``.
    - **Windows / macOS** — returns ``False`` (conservative). Detecting
      NVMe on these reliably requires WMI / IOKit calls we don't want
      to introduce as new dependencies. The fallback matches the V1.34
      single-reader behavior, so the conservative answer is also the
      safe answer.

    Returns ``False`` on any error.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if os.name != "posix":
        return False
    verdict = _linux_is_nvme(path)
    return bool(verdict) if verdict is not None else False


def recommended_io_thread_count(path: Path) -> int:
    """Return the number of IOWorker threads to spawn for *path*.

    - 1 reader for spinning disks, SATA SSDs, and any storage class we
      can't positively identify.
    - 2 readers for NVMe-class storage (sustained 4 GB/s+ throughput
      with high queue depth — one decoder thread leaves the device
      under-utilised).

    Capped at 2: ND2 / TIFF file handles aren't free (per-file lock
    state inside the ``nd2`` library), and beyond 2 the queue-depth
    benefit plateaus while the prefetcher cache hit rate climbs and
    starves the readers anyway.
    """
    return 2 if is_fast_storage(path) else 1
