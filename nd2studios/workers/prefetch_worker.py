"""V1.35 Phase 3 — velocity-aware background frame prefetcher.

``PrefetchManager`` is a single ``QThread`` that reads neighboring
frames into a :class:`~nd2studios.backend.frame_cache.FrameCache` ahead
of slider movement. It opens its **own** volume handle (via a factory
callable) so it never touches the main-thread volume's
``nd2.ND2File`` — that library's per-file state is not thread-safe.

V1.35 changes vs V1.17:

- The T window is **velocity-biased**. The manager remembers the
  previous ``(m, t, z, z_mode)`` plus the wall-clock time the request
  arrived; from successive calls it estimates planes-per-second along
  T. Forward scrubbing (≥ +1 planes/sec) biases the window forward
  (``[t - 2, t + radius_t]``); backward scrubbing biases it backward;
  a stop falls back to a symmetric ``radius_t // 2`` window. Roughly
  doubles the effective lookahead during fast scrubs without enlarging
  the prefetch queue.
- ``radius_t`` and ``radius_z`` are now constructor parameters so
  :class:`~nd2studios.widgets.multi_axis_viewer.MultiAxisViewer` can
  size the lookahead adaptively from
  :func:`nd2studios.utils.resources.detect`. Defaults stay at 5 for
  back-compat with any scripts that build the manager directly.

The Z-axis fallback (used only when T is degenerate) stays symmetric
— Z scrubbing in ND2Studios is rare enough that hand-tuning it is
gratuitous churn.

Usage::

    mgr = PrefetchManager(_factory, cache, channels=list(range(n_c)),
                          radius_t=8)
    mgr.frame_ready.connect(self._on_prefetch_ready)
    mgr.start()
    mgr.request_neighbors(m, t, z, t_range=(0, n_t-1),
                          z_range=(0, n_z-1), z_mode="max")
    mgr.stop()
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal


# Velocity threshold (planes/sec along T) above which the window is
# biased forward / backward. Below |v| ≤ this value the user is
# effectively stopped and we fall back to a symmetric window. 1.0
# corresponds to "one slider step per second" — anything slower is
# treated as scrutiny rather than scrubbing.
_VELOCITY_THRESHOLD = 1.0

# Maximum age of the previous-focus timestamp before we discard the
# velocity estimate. If the user pauses for longer than this the next
# move is treated as a fresh gesture, not a continuation. Keeps the
# bias from sticking in the forward direction after a long idle.
_VELOCITY_HISTORY_TTL_SEC = 1.5


class PrefetchManager(QThread):
    """Background thread that fills a :class:`FrameCache` with neighbors.

    Signals
    -------
    frame_ready(c, m, t, z, z_mode):
        Emitted after a frame is stored in the cache. Delivered to the
        main thread via Qt's automatic queued-connection mechanism.
    """

    frame_ready = Signal(int, int, int, int, str)

    def __init__(
        self,
        reader_factory: Callable,
        cache,
        channels: List[int],
        parent=None,
        *,
        radius_t: int = 5,
        radius_z: int = 5,
    ) -> None:
        super().__init__(parent)
        self._reader_factory = reader_factory
        self._cache = cache
        self._channels = list(channels)
        self._radius_t = max(1, int(radius_t))
        self._radius_z = max(1, int(radius_z))

        self._queue: deque = deque()
        self._queue_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_flag = threading.Event()

        # Velocity tracking — updated on every ``request_neighbors`` call.
        # ``_focus_prev`` is the (m, t, z, z_mode) we were last asked
        # about; ``_focus_time`` is its perf-counter timestamp.
        self._focus_prev: Optional[Tuple[int, int, int, str]] = None
        self._focus_time: float = 0.0

        self.setObjectName("PrefetchManager")

    # ── public API ──────────────────────────────────────────────────────────

    def request_neighbors(
        self,
        m: int,
        t: int,
        z: int,
        t_range: tuple,
        z_range: tuple,
        z_mode: str = "max",
        n: Optional[int] = None,
    ) -> None:
        """Replace the prefetch queue with a velocity-biased window.

        On a multi-T volume the queue is built from a T-axis window
        whose direction is set by the user's recent scrub velocity.
        On a single-T volume the window falls back to symmetric ±radius_z
        Z neighbors. Nearest neighbors are queued first.

        Parameters
        ----------
        m, t, z, z_mode:
            Current focus position. Channel is implicit — every
            channel in ``self._channels`` is queued for each target
            ``(m, t', z')``.
        t_range, z_range:
            Inclusive bounds on the corresponding axis; the window is
            clamped into these.
        n:
            Legacy override for the radius. ``None`` (the default) uses
            the constructor's ``radius_t`` / ``radius_z``. V1.0 callers
            that passed ``n=5`` keep working unchanged.
        """
        t_min, t_max = t_range
        z_min, z_max = z_range

        # Velocity estimate along T (planes / sec). Positive = forward.
        v_t = self._update_velocity_t((m, t, z, z_mode))
        radius_t = int(n) if n is not None else self._radius_t
        radius_z = int(n) if n is not None else self._radius_z

        candidates: list = []
        if t_max > t_min:
            t_lo, t_hi = self._biased_t_window(t, v_t, radius_t)
            # Nearest-first: walk outward in expanding deltas around t,
            # constrained to [t_lo, t_hi] ∩ [t_min, t_max].
            for delta in range(1, max(radius_t, abs(t_hi - t), abs(t - t_lo)) + 1):
                for sign in (1, -1):
                    nt = t + sign * delta
                    if nt < t_min or nt > t_max:
                        continue
                    if nt < t_lo or nt > t_hi:
                        continue
                    candidates.append((m, nt, z, z_mode))
        else:
            for delta in range(1, radius_z + 1):
                for sign in (1, -1):
                    nz = z + sign * delta
                    if z_min <= nz <= z_max:
                        candidates.append((m, t, nz, z_mode))

        tasks: list = []
        for pm, pt, pz, pzm in candidates:
            for c in self._channels:
                key = (c, pm, pt, pz, pzm)
                if not self._cache.contains(key):
                    tasks.append(key)

        with self._queue_lock:
            self._queue.clear()
            self._queue.extend(tasks)
        self._wakeup.set()

    def cancel_all(self) -> None:
        """Discard all pending prefetch work; preserve velocity history."""
        with self._queue_lock:
            self._queue.clear()

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 2 s for it to exit."""
        self._stop_flag.set()
        self._wakeup.set()
        self.wait(2000)

    # ── internal ────────────────────────────────────────────────────────────

    def _update_velocity_t(
        self, focus: Tuple[int, int, int, str]
    ) -> float:
        """Estimate T-axis velocity in planes/sec; update history.

        Resets to 0 when ``m`` or ``z_mode`` changed (different scrub
        gesture entirely) or when the previous focus is stale beyond
        :data:`_VELOCITY_HISTORY_TTL_SEC`. Otherwise returns
        ``(t_new - t_prev) / dt``.
        """
        now = time.perf_counter()
        m, t, _z, z_mode = focus
        prev = self._focus_prev
        prev_time = self._focus_time

        velocity = 0.0
        if prev is not None:
            p_m, p_t, _p_z, p_zmode = prev
            dt = now - prev_time
            if (
                dt > 0.0
                and dt <= _VELOCITY_HISTORY_TTL_SEC
                and p_m == m
                and p_zmode == z_mode
            ):
                velocity = (t - p_t) / dt

        self._focus_prev = focus
        self._focus_time = now
        return velocity

    def _biased_t_window(
        self, t: int, velocity: float, radius: int
    ) -> Tuple[int, int]:
        """Return ``(t_lo, t_hi)`` biased by scrub direction.

        Forward scrubbing widens the upper bound; backward scrubbing
        widens the lower bound. A 2-plane cushion in the off-direction
        catches the user reversing course mid-scrub.
        """
        if velocity >= _VELOCITY_THRESHOLD:
            return (t - 2, t + radius)
        if velocity <= -_VELOCITY_THRESHOLD:
            return (t - radius, t + 2)
        half = max(1, radius // 2)
        return (t - half, t + half)

    # ── QThread.run ─────────────────────────────────────────────────────────

    def run(self) -> None:
        reader = self._reader_factory()
        try:
            while not self._stop_flag.is_set():
                self._wakeup.wait()
                self._wakeup.clear()

                while not self._stop_flag.is_set():
                    with self._queue_lock:
                        if not self._queue:
                            break
                        key = self._queue.popleft()

                    if self._cache.contains(key):
                        continue

                    c, m, t, z, z_mode = key
                    try:
                        frame = reader.get_frame(
                            c=c, m=m, t=t, z=z, z_mode=z_mode,
                        )
                        if frame is None:
                            continue
                        arr = np.asarray(frame)
                        if arr.ndim > 2:
                            arr = arr.squeeze()
                        if arr.ndim == 2:
                            self._cache.put(key, arr)
                            self.frame_ready.emit(c, m, t, z, z_mode)
                    except Exception:
                        pass
        finally:
            if hasattr(reader, "close"):
                reader.close()
