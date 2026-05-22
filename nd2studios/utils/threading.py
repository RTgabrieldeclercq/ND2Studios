"""V1.34 Phase 2 — foreground I/O worker for ``MultiAxisViewer``.

Until V1.33, a slider movement that landed on coords whose planes
were not yet in :class:`~nd2studios.backend.frame_cache.FrameCache`
caused :meth:`MultiAxisViewer._compose_current_frame` to read each
enabled channel synchronously via ``volume.get_frame(...)`` on the GUI
thread. The :class:`~nd2studios.workers.prefetch_worker.PrefetchManager`
only fills the cache for *neighbors* of the previous coords, so every
fresh user gesture incurred ``n_channels`` synchronous reads before
the next repaint could happen.

:class:`IOWorker` closes that gap. It owns its own ``volume.reopen()``
handle (the ``nd2`` library's per-file state is not thread-safe), runs
a priority queue on a dedicated :class:`~PySide6.QtCore.QThread`, and
emits a :class:`numpy.ndarray` back to the GUI through a queued
signal connection.

Coexists with the existing prefetcher: the prefetcher does the
*speculative* ±5 T neighbor reads at low priority; the IOWorker does
the *foreground* reads the viewer is actually about to display. Two
threads on one file is the deliberate trade made in the V1.34 plan
(``CodeLog/ClaudesPlan/V1.34_phase2_lazy_loading_threading.md``).

Example
-------
>>> from nd2studios.utils.threading import (
...     PlaneRequest, IOWorker, start_io_worker,
... )
>>> worker, thread = start_io_worker(volume)
>>> worker.plane_ready.connect(self._on_io_plane_ready)
>>> worker.submit(PlaneRequest(
...     request_id=42, c=0, m=0, t=10, z=0, z_mode="max",
... ))
>>> # later, on app exit:
>>> worker.stop(); thread.quit(); thread.wait(2000)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal


@dataclass(order=True)
class PlaneRequest:
    """One request for a single ``(H, W)`` plane.

    Ordered by ``(priority, request_id)`` so the worker's
    :class:`~queue.PriorityQueue` services lower-priority numeric
    values first (Phase 2 doc convention: 0 = foreground). The
    ``c, m, t, z, z_mode`` block is the same key
    :class:`~nd2studios.backend.frame_cache.FrameCache` uses, so
    callers can read straight from the cache on a hit and only fall
    through to the worker on a miss.
    """

    priority: int
    request_id: int
    c: int = field(compare=False, default=0)
    m: int = field(compare=False, default=0)
    t: int = field(compare=False, default=0)
    z: int = field(compare=False, default=0)
    z_mode: str = field(compare=False, default="none")
    z_start: Optional[int] = field(compare=False, default=None)
    z_end: Optional[int] = field(compare=False, default=None)

    @property
    def cache_key(self) -> Tuple[int, int, int, int, str]:
        """The :class:`FrameCache` key for this request."""
        return (self.c, self.m, self.t, self.z, self.z_mode)


def _normalize_to_2d(frame: np.ndarray) -> Optional[np.ndarray]:
    """Coerce any reasonable frame shape into 2D ``(H, W)``.

    Duplicated intentionally from
    :meth:`MultiAxisViewer._normalize_to_2d` — running normalization
    on the worker thread keeps the GUI slot trivial and means the
    plane handed to the cache is already display-ready. Both the
    viewer's and this module's normalizers must stay in sync; the
    viewer's version remains for the in-memory ``channels`` path that
    does not go through the worker.
    """
    if frame is None:
        return None
    a = np.squeeze(np.asarray(frame))
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            lum = (
                0.299 * r.astype(np.float32)
                + 0.587 * g.astype(np.float32)
                + 0.114 * b.astype(np.float32)
            )
            return lum.astype(
                a.dtype if np.issubdtype(a.dtype, np.integer) else np.float32
            )
        return a[0]
    while a.ndim > 2:
        a = a[0]
    return a if a.ndim == 2 else None


class IOWorker(QObject):
    """Runs on a dedicated :class:`QThread`. Reads planes off the GUI thread.

    The worker is a plain :class:`QObject` that gets moved to a thread
    via :meth:`QObject.moveToThread`; the :func:`start_io_worker`
    helper wires that up. Its only public surface is:

    - :meth:`submit` — enqueue a :class:`PlaneRequest`. Safe to call
      from the GUI thread.
    - :meth:`stop` — signal the run loop to exit. Safe to call from
      the GUI thread.
    - The :attr:`plane_ready` and :attr:`error` signals — both emit
      with auto-connection semantics so any GUI slot connected with
      the default connection type runs on the main thread.

    The worker opens its file handle lazily inside :meth:`run_loop`
    (not the constructor) so the handle lives entirely on the worker
    thread. This matters because the ``nd2`` library's per-file state
    is not thread-safe; reusing a main-thread handle from a worker
    could segfault.
    """

    # request_id, cache_key tuple, normalized (H, W) ndarray
    plane_ready = Signal(int, tuple, object)
    # request_id, error message
    error = Signal(int, str)

    def __init__(self, reader_factory: Callable[[], Any]) -> None:
        super().__init__()
        self._reader_factory = reader_factory
        self._queue: "PriorityQueue[PlaneRequest]" = PriorityQueue()
        self._stop_event = threading.Event()
        self._seq = 0  # tie-breaker for equal-priority requests

    def submit(self, req: PlaneRequest) -> None:
        """Enqueue a :class:`PlaneRequest`. Safe from any thread."""
        self._queue.put(req)

    def cancel_all(self) -> None:
        """Drop all pending requests without stopping the worker."""
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass

    def stop(self) -> None:
        """Signal the run loop to exit and unblock the queue."""
        self._stop_event.set()
        # Push a sentinel so ``queue.get(timeout=...)`` returns promptly.
        self._queue.put(PlaneRequest(priority=-(2 ** 31), request_id=-1))

    def run_loop(self) -> None:
        """Worker loop — connected to :meth:`QThread.started`."""
        try:
            reader = self._reader_factory()
        except Exception as exc:
            # No handle, no useful work. Report once and exit.
            self.error.emit(-1, f"{type(exc).__name__}: {exc}")
            return

        try:
            while not self._stop_event.is_set():
                try:
                    req = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                if self._stop_event.is_set():
                    break
                if req.request_id < 0:
                    # Stop sentinel.
                    continue
                try:
                    raw = reader.get_frame(
                        c=req.c, m=req.m, t=req.t, z=req.z,
                        z_mode=req.z_mode,
                        z_start=req.z_start, z_end=req.z_end,
                    )
                    plane = _normalize_to_2d(raw)
                    if plane is None:
                        # Unreadable shape — surface as an error so the
                        # viewer can decide what to do; do not silently
                        # drop the request.
                        self.error.emit(
                            req.request_id,
                            f"frame at {req.cache_key!r} normalized to None",
                        )
                        continue
                    self.plane_ready.emit(req.request_id, req.cache_key, plane)
                except Exception as exc:
                    self.error.emit(
                        req.request_id, f"{type(exc).__name__}: {exc}",
                    )
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def start_io_worker(volume: Any) -> Tuple[IOWorker, QThread]:
    """Create an :class:`IOWorker` on a dedicated :class:`QThread` and start it.

    Returns the ``(worker, thread)`` pair. The caller is responsible
    for keeping references alive and for clean shutdown on app exit
    or when opening a new file::

        worker.stop()
        thread.quit()
        thread.wait(2000)

    The ``volume`` must expose ``reopen() -> <same type>`` —
    :class:`~nd2studios.backend.nd2_volume.LazyND2Volume`,
    :class:`~nd2studios.backend.nd2_volume.LazyMultiFileND2Volume`,
    and :class:`~nd2studios.backend.tiff_loader.LazyMultiFileTIFFVolume`
    all do. The reopened handle lives entirely on the worker thread.
    """
    if not hasattr(volume, "reopen"):
        raise TypeError(
            "IOWorker volume must expose reopen() — "
            f"{type(volume).__name__} does not"
        )

    factory = volume.reopen  # bound method; no shared state across threads

    thread = QThread()
    thread.setObjectName("ND2StudiosIOWorker")
    worker = IOWorker(factory)
    worker.moveToThread(thread)
    thread.started.connect(worker.run_loop)
    thread.start()
    return worker, thread
