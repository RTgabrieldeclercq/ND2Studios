"""Profiling helpers — backend-pure (no Qt imports).

Context managers and small dataclasses used by the ``profiling/``
harness to record wall time, CPU time, RSS, and Python-side allocations
around blocks of code. Kept in :mod:`nd2studios.utils` so future
optimization phases can reuse the same `Measurement` shape from worker
tests and ad-hoc scripts.

Example
-------
>>> from nd2studios.utils.profiling import measure
>>> with measure("warmup") as m:
...     do_stuff()
>>> m.wall_ms, m.rss_after_mb
"""
from __future__ import annotations

import gc
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import psutil


@dataclass
class Measurement:
    """One timing/memory record produced by :func:`measure`.

    All durations are in milliseconds; all memory figures in megabytes.
    ``extra`` carries scenario-specific extras (e.g. per-frame FPS
    statistics from :func:`fps_from_durations`).
    """

    name: str
    wall_ms: float = 0.0
    cpu_ms: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    py_alloc_peak_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rss_mb() -> float:
    """Resident-set-size of the current process in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@contextmanager
def measure(name: str, *, track_pyalloc: bool = True):
    """Time a block of code and capture RSS and Python allocations.

    Parameters
    ----------
    name
        Label that ends up in :attr:`Measurement.name`.
    track_pyalloc
        Whether to enable :mod:`tracemalloc` for the block. Adds a
        small amount of overhead — turn it off for very hot, very
        short measurements (e.g. inner scrub loops measured one
        frame at a time).
    """
    gc.collect()
    if track_pyalloc:
        tracemalloc.start()
    rss_before = _rss_mb()
    rss_peak = rss_before
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    m = Measurement(name=name, rss_before_mb=rss_before)
    try:
        yield m
    finally:
        m.wall_ms = (time.perf_counter() - wall_start) * 1000.0
        m.cpu_ms = (time.process_time() - cpu_start) * 1000.0
        rss_after = _rss_mb()
        m.rss_after_mb = rss_after
        m.rss_peak_mb = max(rss_peak, rss_after)
        if track_pyalloc:
            try:
                _, peak = tracemalloc.get_traced_memory()
                m.py_alloc_peak_mb = peak / (1024 * 1024)
            finally:
                tracemalloc.stop()


def fps_from_durations(durations_ms: List[float]) -> Dict[str, float]:
    """Summarise per-frame durations (ms) into FPS / latency stats.

    Returns a dict with ``mean_fps``, ``p50_fps``, ``p99_fps`` and the
    same three durations expressed in milliseconds, plus ``frames``.
    Safe to call with an empty list — returns zeros.
    """
    if not durations_ms:
        return {
            "mean_fps": 0.0, "p50_fps": 0.0, "p99_fps": 0.0,
            "mean_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0,
            "frames": 0,
        }
    ordered = sorted(durations_ms)
    n = len(ordered)
    p50 = ordered[n // 2]
    p99 = ordered[min(n - 1, int(n * 0.99))]
    mean = sum(ordered) / n
    return {
        "mean_fps": 1000.0 / mean if mean > 0 else 0.0,
        "p50_fps": 1000.0 / p50 if p50 > 0 else 0.0,
        "p99_fps": 1000.0 / p99 if p99 > 0 else 0.0,
        "mean_ms": mean,
        "p50_ms": p50,
        "p99_ms": p99,
        "frames": n,
    }
