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

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

try:
    import psutil  # already a project dependency via utils/resources.py
except Exception:  # noqa: BLE001 — psutil is optional defense-in-depth
    psutil = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


# Threshold (MB/s) above which a read probe classifies the drive as
# fast.  Modern SATA SSDs sustain ~500 MB/s sequential, NVMe ~3-7 GB/s,
# 7200 RPM spinning rust ~150 MB/s.  400 MB/s is comfortably above
# spinning + over-the-network mounts while below most SSDs.
_FAST_STORAGE_THRESHOLD_MBPS = 400.0
_READ_PROBE_BYTES = 16 * 1024 * 1024  # 16 MB chunk for the probe


def _probe_cache_path() -> Path:
    """Per-user JSON cache for storage-class verdicts.

    Keyed by drive letter (Windows) or mount point (Linux/macOS) so
    repeated imports from the same device skip the probe.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home()))
        return Path(base) / "nd2studios" / "storage_probe.json"
    return Path.home() / ".config" / "nd2studios" / "storage_probe.json"


def _load_probe_cache() -> Dict[str, bool]:
    p = _probe_cache_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.debug("storage probe cache read failed: %s", exc)
    return {}


def _save_probe_cache(cache: Dict[str, bool]) -> None:
    p = _probe_cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.debug("storage probe cache write failed: %s", exc)


def _windows_drive_key(path: Path) -> Optional[str]:
    """Return the drive letter (uppercase, with trailing colon) for *path*.

    Used as the cache key on Windows.  Returns ``None`` if the path
    isn't on a lettered drive (network UNC, virtual filesystem).
    """
    try:
        drive = os.path.splitdrive(str(path.resolve()))[0]
    except Exception:  # noqa: BLE001
        return None
    if not drive:
        return None
    return drive.upper()


def _read_probe_mbps(path: Path) -> Optional[float]:
    """Measure read throughput by reading up to 16 MB from *path*.

    Returns MB/s, or ``None`` if the path can't be opened or the
    measurement is unreliable (tiny file, EOF before timing
    stabilises).  Adds OS read-ahead cache effects but in practice
    those still leave NVMe / spinning rust well apart from each other.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size <= 0:
        return None
    to_read = min(_READ_PROBE_BYTES, st.st_size)
    try:
        start = time.perf_counter()
        with open(path, "rb") as f:
            buf = f.read(to_read)
        elapsed = time.perf_counter() - start
    except OSError:
        return None
    if elapsed <= 0.0 or not buf:
        return None
    return (len(buf) / (1024 * 1024)) / elapsed


def windows_is_fast_storage(path: Path) -> Optional[bool]:
    """Windows storage-class probe.  Returns ``True``/``False``/``None``.

    Strategy:

    1. Check the per-user JSON cache for the drive letter.
    2. If not cached, read up to 16 MB from *path*, time it, and
       classify against :data:`_FAST_STORAGE_THRESHOLD_MBPS`.
    3. Cache the verdict by drive letter so subsequent imports from
       the same drive skip the probe.

    Returns ``None`` only when the path isn't on a lettered drive
    (UNC paths, virtual filesystems) or no readable file is at the
    location — callers should treat ``None`` as "unknown, be
    conservative".
    """
    key = _windows_drive_key(path)
    if key is None:
        return None

    cache = _load_probe_cache()
    if key in cache:
        return bool(cache[key])

    mbps = _read_probe_mbps(path)
    if mbps is None:
        return None
    verdict = mbps >= _FAST_STORAGE_THRESHOLD_MBPS
    cache[key] = verdict
    _save_probe_cache(cache)
    log.info(
        "Storage probe for %s: %.0f MB/s -> %s",
        key, mbps, "fast" if verdict else "slow",
    )
    return verdict


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
    - **Windows** (V1.41) — runs a cached read probe via
      :func:`windows_is_fast_storage`. The first import from each
      drive letter pays a ~30 ms read; subsequent imports hit the
      JSON cache instantly.
    - **macOS / unknown** — returns ``False`` (conservative). The
      fallback matches the V1.34 single-reader behavior, so the
      conservative answer is also the safe answer.

    Returns ``False`` on any error.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if os.name == "posix":
        verdict = _linux_is_nvme(path)
        return bool(verdict) if verdict is not None else False
    if os.name == "nt":
        verdict = windows_is_fast_storage(path)
        return bool(verdict) if verdict is not None else False
    return False


def log_default_storage_class(default_dir: Path) -> None:
    """Emit a one-line info log describing the storage class of *default_dir*.

    Called from ``__main__`` after the GPU + RAM detection lines so
    the user sees consistent startup diagnostics. When the path is a
    directory we look for a real file inside to probe — the probe
    needs a readable file to time, and directories typically report
    ``st_size == 0`` on Windows which would short-circuit the probe.
    If no probeable file is found we say so explicitly rather than
    misclassify the drive.
    """
    try:
        p = default_dir if isinstance(default_dir, Path) else Path(default_dir)
    except Exception:  # noqa: BLE001
        return
    probe_target = p
    if p.is_dir():
        # First regular file under the directory.  Cap the walk
        # depth so we don't spelunk a recursive symlink at startup.
        probe_target = None
        for entry in p.iterdir():
            try:
                if entry.is_file() and entry.stat().st_size > 1_000_000:
                    probe_target = entry
                    break
            except OSError:
                continue
        if probe_target is None:
            log.info("Storage class for %s: deferred until first file open", p)
            return
    fast = is_fast_storage(probe_target)
    log.info(
        "Storage class for %s: %s",
        p,
        "fast (NVMe/SSD)" if fast else "slow / unknown",
    )


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
