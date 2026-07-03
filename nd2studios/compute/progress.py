"""Progress reporter for the V1.37 Phase 5 job runner.

:class:`ProgressReporter` is a tiny :class:`QObject` carrying a
single Qt signal that emits ``(job_key, fraction, message)`` on each
update. The runner instantiates one per job, hands it to the job's
:meth:`run` method, and routes its signal back to GUI-thread slots
via the runner's own ``job_progress`` signal (queued connection by
default, so the emission marshals correctly).

To keep existing :class:`AnalysisPipeline` implementations working
without edits, the reporter also exposes a
:meth:`as_pipeline_progress_cb` adapter that returns the
``Callable[[int], None]`` shape pipelines expect for their
``progress_cb=...`` argument.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Signal


class ProgressReporter(QObject):
    """Emit fine-grained job progress on a Qt signal.

    Parameters
    ----------
    key:
        The job's coalescing key — emitted on every update so the
        GUI receiver can filter by job (multiple jobs may report
        concurrently).
    """

    # job_key, fraction in [0.0, 1.0], optional status message
    progress = Signal(str, float, str)
    # job_key, m_index, t (frame), labels (2-D int array) — per-frame streaming so
    # the GUI can paint the overlay live as each frame finishes.
    frame = Signal(str, int, int, object)

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    @property
    def key(self) -> str:
        return self._key

    def report_frame(self, m: int, t: int, labels) -> None:
        """Emit one finished frame's labels (``(m, t, labels)``) for live overlay."""
        self.frame.emit(self._key, int(m), int(t), labels)

    def update(self, fraction: float, message: str = "") -> None:
        """Report progress in ``[0.0, 1.0]`` plus an optional message.

        ``fraction`` is clamped to ``[0.0, 1.0]`` so a buggy job can't
        push a progress bar past 100 %.
        """
        f = max(0.0, min(1.0, float(fraction)))
        self.progress.emit(self._key, f, message)

    def as_pipeline_progress_cb(self) -> Callable[[int], None]:
        """Return a ``progress_cb`` shim accepting an int 0–100.

        ``AnalysisPipeline.run`` was designed against
        :class:`BaseWorker`, which emits integer percent. Wrap the
        reporter so existing pipelines can stay verbatim.
        """
        def _cb(percent: int) -> None:
            self.update(float(percent) / 100.0)
        return _cb
