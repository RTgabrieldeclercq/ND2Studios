"""
``PreRenderWorker`` — background QThread that pre-composes all (M, T) frames
into ``(H, W, 3) uint8`` numpy arrays and deposits them in a shared dict.

After :class:`LoadWorker` materializes the ND2 file into RAM, this worker
runs the same LUT + compositing pipeline that
:meth:`MultiAxisViewer._compose_current_frame` uses, but entirely off the
GUI thread.  The shared ``cache`` dict is written by the worker and read by
the main thread; CPython's GIL makes individual ``dict[key] = value``
assignments atomic, so no additional locking is needed.

Priority order: all T frames for ``priority_m`` first, then all other M
positions.  The worker emits :attr:`finished` after the priority-M series
is complete so the viewer can switch to the fast QPixmap path sooner; it
continues filling remaining M positions until cancelled or done.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from nd2studios.backend.materialized_dataset import MaterializedDataset
from nd2studios.widgets.lut_histogram import apply_lut

# Upper bound on cached frames before the worker restricts itself to
# priority_m only.  At 2048×2048 × 3 channels uint8 each frame is ~12 MB;
# 500 frames ≈ 6 GB which is a reasonable ceiling on a workstation.
_MAX_FRAMES = 500


class PreRenderWorker(QThread):
    """Off-GUI-thread frame compositor for smooth slider and FPS playback."""

    progress = Signal(int)           # 0–100 percent
    frame_cached = Signal(int, int)  # (m, t) — individual frame ready
    finished = Signal()              # priority-M series done (or cancelled)
    error = Signal(str)

    # Class-level list keeps alive workers that have no other Python referent,
    # preventing GC while the C++ thread is still running.
    _running: List["PreRenderWorker"] = []

    def __init__(
        self,
        volume: MaterializedDataset,
        lut_snapshot: Dict[str, Tuple[float, float, float]],
        chip_snapshot: Dict[str, Tuple[bool, Tuple[int, int, int]]],
        priority_m: int,
        cache: Dict[Tuple[int, int], np.ndarray],
        z_mode: str = "max",
        z_index: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._volume = volume
        self._lut_snapshot = lut_snapshot
        self._chip_snapshot = chip_snapshot
        self._priority_m = priority_m
        self._cache = cache
        self._z_mode = z_mode
        self._z_index = int(z_index)
        self._cancelled = False
        self._finished_emitted = False
        PreRenderWorker._running.append(self)
        self.finished.connect(lambda: self._remove_self())

    def cancel(self) -> None:
        self._cancelled = True

    def _remove_self(self) -> None:
        try:
            PreRenderWorker._running.remove(self)
        except ValueError:
            pass

    def _emit_finished_once(self) -> None:
        if not self._finished_emitted:
            self._finished_emitted = True
            self.finished.emit()

    def run(self) -> None:
        try:
            self._render()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            # Guarantee exactly one finished emission regardless of whether
            # _render() already emitted it (after priority_m) or not (error).
            self._emit_finished_once()

    def _render(self) -> None:
        volume = self._volume
        n_m = volume.n_multipoints
        n_t = volume.n_timepoints

        restrict_to_priority = (n_m * n_t) > _MAX_FRAMES

        if restrict_to_priority:
            m_order = [self._priority_m]
        else:
            other_m = [m for m in range(n_m) if m != self._priority_m]
            m_order = [self._priority_m] + other_m

        # Count only the work we'll actually do for progress reporting.
        total = len(m_order) * n_t
        done = 0

        for m in m_order:
            if self._cancelled:
                break
            for t in range(n_t):
                if self._cancelled:
                    break
                self._cache[(m, t)] = self._compose_frame(m, t)
                self.frame_cached.emit(m, t)
                done += 1
                self.progress.emit(int(100 * done / total))

            # Emit finished exactly once — right after the priority-M series
            # completes.  The viewer uses this to activate the fast QPixmap
            # path without waiting for every other M position to render.
            if m == self._priority_m:
                self._emit_finished_once()
                # Continue rendering remaining M positions silently.

    def _compose_frame(self, m: int, t: int) -> np.ndarray:
        """Pure numpy composite — no Qt, safe on a background thread."""
        volume = self._volume
        h, w = volume.height, volume.width
        if h == 0 or w == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        composite = np.zeros((h, w, 3), dtype=np.float32)

        for name in volume.channel_names:
            enabled, rgb = self._chip_snapshot.get(name, (False, (255, 255, 255)))
            if not enabled:
                continue
            lo, hi, gamma = self._lut_snapshot.get(name, (0.0, 65535.0, 1.0))

            arr = volume.channels.get(name)
            if arr is None:
                continue
            try:
                zstack = arr[m, t]  # (Z, H, W)
            except IndexError:
                continue
            n_z = zstack.shape[0]
            if n_z <= 1:
                plane = zstack[0]
            elif self._z_mode == "none":
                plane = zstack[max(0, min(self._z_index, n_z - 1))]
            elif self._z_mode == "max":
                plane = zstack.max(axis=0)
            elif self._z_mode == "min":
                plane = zstack.min(axis=0)
            else:
                plane = zstack.mean(axis=0).astype(zstack.dtype)
            if plane.ndim != 2:
                continue

            mapped = apply_lut(plane, lo, hi, gamma).astype(np.float32)
            r, g, b = rgb
            composite[..., 0] += mapped * (r / 255.0)
            composite[..., 1] += mapped * (g / 255.0)
            composite[..., 2] += mapped * (b / 255.0)

        return np.clip(composite, 0, 255).astype(np.uint8)
