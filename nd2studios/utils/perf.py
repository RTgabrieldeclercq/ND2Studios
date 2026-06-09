"""Lightweight performance instrumentation for ND2Studios (V1.41).

Gated entirely by the ``ND2_PERF_LOG=1`` environment variable. When unset
(the common case), every public helper in this module short-circuits to a
no-op — the decorator returns the original function unchanged, the context
manager skips the clock reads, and ``log_event`` returns immediately. The
goal is zero measurable overhead at call sites in normal operation.

When enabled, rows are appended to ``~/.nd2studios/perf.csv`` with the
schema::

    timestamp_ns, event, duration_us, frame_idx, cache_hit, extra

Three usage patterns are provided:

* :func:`perf_log` — function decorator. Reads ``self._last_cache_hit`` /
  ``self._last_frame_idx`` from the first positional arg if present, so a
  method can stash context immediately before returning.
* :class:`perf_block` — context manager. Use when the timed region is not
  a whole function or when frame/cache context is computed inside.
* :func:`log_event` — raw row emitter. Use when timing comes from an
  external clock (e.g., a worker that reports its own duration).
"""
from __future__ import annotations

import atexit
import csv
import logging
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_ENABLED = os.environ.get("ND2_PERF_LOG") == "1"
_LOCK = threading.Lock()
_FH: Optional[Any] = None
_WRITER: Optional[Any] = None
_PATH: Optional[Path] = None
_HEADER = ("timestamp_ns", "event", "duration_us", "frame_idx", "cache_hit", "extra")


def is_enabled() -> bool:
    """Return True if perf logging is active for this process."""
    return _ENABLED


def perf_csv_path() -> Optional[Path]:
    """Return the active perf log path, or ``None`` if not yet opened."""
    return _PATH


def _open_writer() -> None:
    global _FH, _WRITER, _PATH
    if _WRITER is not None:
        return
    perf_dir = Path.home() / ".nd2studios"
    perf_dir.mkdir(parents=True, exist_ok=True)
    _PATH = perf_dir / "perf.csv"
    write_header = (not _PATH.exists()) or _PATH.stat().st_size == 0
    _FH = _PATH.open("a", newline="", encoding="utf-8")
    _WRITER = csv.writer(_FH)
    if write_header:
        _WRITER.writerow(_HEADER)
        _FH.flush()


def log_event(
    event: str,
    duration_us: float,
    *,
    frame_idx: Optional[int] = None,
    cache_hit: Optional[bool] = None,
    extra: str = "",
) -> None:
    """Append a single perf row. No-op when disabled.

    Safe to call from any thread — guarded by a module-level lock so writes
    do not interleave.
    """
    if not _ENABLED:
        return
    with _LOCK:
        try:
            _open_writer()
            _WRITER.writerow((
                time.monotonic_ns(),
                event,
                f"{duration_us:.1f}",
                "" if frame_idx is None else int(frame_idx),
                "" if cache_hit is None else ("1" if cache_hit else "0"),
                extra,
            ))
            _FH.flush()
        except Exception as exc:  # noqa: BLE001
            log.debug("perf.log_event failed: %s", exc)


def perf_log(event: str) -> Callable[[Callable], Callable]:
    """Time a function and append a perf row on exit.

    When ND2_PERF_LOG is unset the decorator returns the original function
    unchanged, so there is no call-site overhead in the common case.

    If the wrapped callable is a bound method, the decorator reads
    ``self._last_cache_hit`` and ``self._last_frame_idx`` (if set) and
    includes them in the row. Methods that want this context should set
    those attributes immediately before returning.
    """
    if not _ENABLED:
        def passthrough(fn: Callable) -> Callable:
            return fn
        return passthrough

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                dur_us = (time.perf_counter_ns() - start) / 1000.0
                hit: Optional[bool] = None
                idx: Optional[int] = None
                if args:
                    candidate = args[0]
                    hit = getattr(candidate, "_last_cache_hit", None)
                    idx = getattr(candidate, "_last_frame_idx", None)
                log_event(event, dur_us, frame_idx=idx, cache_hit=hit)
        return wrapper
    return decorator


class perf_block:
    """Context manager that times an arbitrary code block.

    Example::

        with perf_block("composite", frame_idx=t, cache_hit=False):
            rgb = composite_channels(...)

    No-op when ND2_PERF_LOG is unset — the ``__enter__`` / ``__exit__``
    calls are still issued but skip the clock reads, so cost is roughly
    one attribute access per use.
    """

    __slots__ = ("event", "frame_idx", "cache_hit", "extra", "_start")

    def __init__(
        self,
        event: str,
        *,
        frame_idx: Optional[int] = None,
        cache_hit: Optional[bool] = None,
        extra: str = "",
    ) -> None:
        self.event = event
        self.frame_idx = frame_idx
        self.cache_hit = cache_hit
        self.extra = extra
        self._start = 0

    def __enter__(self) -> "perf_block":
        if _ENABLED:
            self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *_exc) -> None:
        if not _ENABLED:
            return
        dur_us = (time.perf_counter_ns() - self._start) / 1000.0
        log_event(
            self.event,
            dur_us,
            frame_idx=self.frame_idx,
            cache_hit=self.cache_hit,
            extra=self.extra,
        )

    def set_cache_hit(self, hit: bool) -> None:
        """Update the cache_hit flag after the block has entered."""
        self.cache_hit = hit

    def set_frame_idx(self, idx: int) -> None:
        """Update the frame_idx after the block has entered."""
        self.frame_idx = idx


def _close() -> None:
    """Flush and close the perf CSV. Registered with atexit."""
    global _FH, _WRITER
    with _LOCK:
        if _FH is not None:
            try:
                _FH.flush()
                _FH.close()
            except Exception:  # noqa: BLE001
                pass
        _FH = None
        _WRITER = None


atexit.register(_close)
