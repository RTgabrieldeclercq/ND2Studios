"""Frame-accurate progress accounting (V1.41).

The historical progress story in ND2Studios was inconsistent: most exporters
reported ``(t + 1) / n_t * 100`` (ignoring multipoint **M** and **Z**, so a
4-position, 40-slice file jumped to 100 % after the first position), the recipe
worker reported coarse *phases*, and only ``image_sequence_exporter`` counted
``M * T * Z`` correctly.

:class:`FrameProgress` makes one contract for every long-running operation:
*count real frames*. The caller computes the true total up front (usually
``T * M * Z * C`` via :func:`total_frames_for`) and calls :meth:`advance` once
per frame processed. The accountant maps the running count onto an integer
percentage and only fires the emit callback **when that integer changes** — so
the bar steps at 1 % thresholds and the GUI is never flooded with signals, no
matter how many millions of frames are processed.

The emit callback is the existing GUI-facing contract — typically
``BaseWorker.set_progress`` (``Callable[[int], None]`` accepting 0–100). An
optional output band ``(lo, hi)`` lets a worker reserve part of the bar for a
phase (e.g. load reserves 25–95 % for materialization) while still feeding a
single accountant.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence


def total_frames_for(
    meta,
    *,
    channels: Optional[int] = None,
    z_collapsed: bool = False,
    include_channels: bool = True,
) -> int:
    """Derive a real frame total from an :class:`ND2Metadata`-like object.

    Parameters
    ----------
    meta:
        Any object exposing ``n_timepoints``, ``n_multipoints``, ``n_zslices``
        and ``n_channels`` (i.e. :class:`~nd2studios.backend.nd2_loader.ND2Metadata`).
    channels:
        Override the channel count (e.g. only *enabled* channels for an export).
        Defaults to ``meta.n_channels``.
    z_collapsed:
        When ``True`` the Z axis is treated as already projected (one plane per
        ``(M, T)``), so Z does not multiply into the total.
    include_channels:
        When ``False`` the channel axis is excluded (e.g. compositing produces
        one RGB frame per ``(M, T, Z)`` regardless of channel count).
    """
    n_t = max(1, int(getattr(meta, "n_timepoints", 1)))
    n_m = max(1, int(getattr(meta, "n_multipoints", 1)))
    n_z = 1 if z_collapsed else max(1, int(getattr(meta, "n_zslices", 1)))
    if include_channels:
        n_c = max(1, int(channels if channels is not None else getattr(meta, "n_channels", 1)))
    else:
        n_c = 1
    return n_t * n_m * n_z * n_c


class FrameProgress:
    """Count processed frames and emit integer percent at 1 % thresholds.

    Parameters
    ----------
    total_frames:
        Total number of frames the operation will process. Clamped to ``>= 1``
        so an empty job still reports 0 → 100 cleanly.
    emit_cb:
        Called with an int in ``[lo, hi]`` whenever the mapped percentage
        changes. ``None`` makes the accountant a no-op sink (useful for headless
        callers that don't care about progress).
    lo, hi:
        Output band. The internal 0–100 fraction is linearly mapped onto
        ``[lo, hi]`` so a worker can reserve a slice of the bar for one phase.
    """

    def __init__(
        self,
        total_frames: int,
        emit_cb: Optional[Callable[[int], None]],
        *,
        lo: int = 0,
        hi: int = 100,
    ) -> None:
        self._total = max(1, int(total_frames))
        self._emit = emit_cb
        self._lo = int(lo)
        self._hi = int(hi)
        self._done = 0
        self._last_emitted: Optional[int] = None

    @property
    def total(self) -> int:
        return self._total

    @property
    def done(self) -> int:
        return self._done

    def _mapped(self) -> int:
        frac = self._done / self._total
        if frac > 1.0:
            frac = 1.0
        return self._lo + int(round(frac * (self._hi - self._lo)))

    def _maybe_emit(self) -> None:
        if self._emit is None:
            return
        value = self._mapped()
        if value != self._last_emitted:
            self._last_emitted = value
            self._emit(value)

    def advance(self, n: int = 1) -> None:
        """Mark ``n`` more frames done and emit if the percent changed."""
        if n <= 0:
            return
        self._done += n
        if self._done > self._total:
            self._done = self._total
        self._maybe_emit()

    def set_done(self, done: int) -> None:
        """Set the absolute number of completed frames (clamped)."""
        self._done = max(0, min(int(done), self._total))
        self._maybe_emit()

    def finish(self) -> None:
        """Force the bar to the top of the band."""
        self._done = self._total
        self._maybe_emit()

    def child(self, lo: int, hi: int) -> "FrameProgress":
        """A sub-accountant feeding the same callback over a narrower band.

        Convenience for multi-phase jobs that still want frame counting within
        each phase (the parent's band is ignored — pass absolute ``lo``/``hi``).
        """
        return FrameProgress(self._total, self._emit, lo=lo, hi=hi)

    def iter_count(self, items: Sequence) -> Iterable:
        """Yield from ``items`` calling :meth:`advance` after each — sugar for
        ``for x in fp.iter_count(frames): process(x)``."""
        for item in items:
            yield item
            self.advance()
