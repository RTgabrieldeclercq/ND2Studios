"""V1.37 Phase 5 — :class:`JobRunner` (QThreadPool-backed dispatcher).

The runner is the single coordination point for the new
short-and-cancellable jobs ND2Studios spawns on the Analysis page.
It complements (does not replace) the long-running
:class:`~nd2studios.workers.base_worker.BaseWorker` subclasses
(``RecipeWorker``, ``BatchWorker``, ``LoadWorker``, ``IOWorker``,
``PrefetchWorker``, ``StitchWorker``) — each of those owns its
file-handle or recipe-pipeline lifecycle and stays on a dedicated
``QThread``. The runner is for sub-second to few-second jobs whose
defining property is "should be cancellable and coalesce with the
next submission".

Coalescing: the moment :meth:`submit` is called with a key already
in use, the existing job's token is cancelled before the new job
is enqueued. The old job typically unwinds within one
``token.check()`` interval and its outcome is silently dropped
(``job_cancelled`` is emitted; no ``job_done``).

Thread-safety: the active-jobs dict is guarded by an
:class:`RLock`. Signal emissions cross thread boundaries via
Qt's default auto-connection, which queues over the GUI thread's
event loop — i.e. GUI slots run on the main thread, exactly as
``BaseWorker`` callers already expect.
"""
from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from nd2studios.compute.cancellation import CancelledError
from nd2studios.compute.jobs import AnalysisJob, JobResult
from nd2studios.compute.progress import ProgressReporter
from nd2studios.utils.resources import recommended_worker_count


class _RunnerSignals(QObject):
    """Signal sink shared by every :class:`_Runnable`.

    Living on the runner (not on individual jobs) so the GUI can
    connect once at startup and receive results for every job.
    """

    done = Signal(object)            # JobResult
    cancelled = Signal(str)          # key
    progress = Signal(str, float, str)


class _Runnable(QRunnable):
    """Wrap a single :class:`AnalysisJob` for :class:`QThreadPool`.

    Created once per submission. Forwards ``CancelledError`` to the
    ``cancelled`` signal and any other exception to ``done`` with
    ``ok=False`` — the runner converts both into the appropriate
    public-facing signal. After signalling, calls ``finalize_cb``
    to let the runner forget the job from its active-map.
    """

    def __init__(
        self,
        job: AnalysisJob,
        signals: _RunnerSignals,
        progress: ProgressReporter,
        finalize_cb: Callable[[AnalysisJob], None],
    ) -> None:
        super().__init__()
        self._job = job
        self._signals = signals
        self._progress = progress
        self._finalize_cb = finalize_cb

    @Slot()
    def run(self) -> None:  # noqa: D401 — Qt-mandated method name
        try:
            if self._job.token.is_cancelled():
                self._signals.cancelled.emit(self._job.key)
                return
            value = self._job.run(self._progress)
            self._signals.done.emit(
                JobResult(self._job.key, ok=True, value=value)
            )
        except CancelledError:
            self._signals.cancelled.emit(self._job.key)
        except Exception as exc:  # noqa: BLE001 — report all failures
            self._signals.done.emit(
                JobResult(
                    self._job.key,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            # Drop the runner's reference to this job so the dict
            # doesn't grow indefinitely; and disconnect the per-job
            # ProgressReporter so it can be garbage-collected with
            # the runnable. Forgetting either leaks one QObject per
            # submission.
            try:
                self._finalize_cb(self._job)
            except Exception:  # noqa: BLE001 — never fail finalization
                pass
            try:
                self._progress.progress.disconnect()
            except (RuntimeError, TypeError):
                pass


class JobRunner(QObject):
    """Submit, cancel, and listen to :class:`AnalysisJob` instances.

    A single instance lives on :class:`MainWindow.job_runner`. Pages
    that opt in to background analysis use it; pages still on the
    legacy ``QThread`` workers are unaffected.

    Signals
    -------
    job_done(JobResult)
        Emitted once for every successfully-completed *or* failed
        job. Use ``result.ok`` to discriminate.
    job_cancelled(str)
        Emitted when a job exits via :class:`CancelledError`
        (either because the user clicked Cancel or because a
        newer submission with the same key superseded it).
    job_progress(str, float, str)
        Forwarded from each job's :class:`ProgressReporter`.
    """

    job_done = Signal(object)             # JobResult
    job_cancelled = Signal(str)           # key
    job_progress = Signal(str, float, str)  # key, fraction, message

    def __init__(
        self,
        max_workers: Optional[int] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(
            max_workers if max_workers is not None
            else recommended_worker_count()
        )
        self._signals = _RunnerSignals()
        # Re-emit on the public signals so connectors only see one
        # source of truth.
        self._signals.done.connect(self.job_done)
        self._signals.cancelled.connect(self.job_cancelled)
        self._signals.progress.connect(self.job_progress)
        self._active: Dict[str, AnalysisJob] = {}
        self._lock = RLock()

    # ── public API ────────────────────────────────────────────────────────
    def submit(self, job: AnalysisJob) -> None:
        """Schedule ``job``.

        If a job with the same :attr:`AnalysisJob.key` is already
        active, it is cancelled before the new one runs. The
        previous job's outcome (after it unwinds) will be a
        ``job_cancelled`` emission; receivers should ignore it
        for their UI state and react to the new job's
        ``job_done`` instead.
        """
        with self._lock:
            existing = self._active.get(job.key)
            if existing is not None:
                existing.cancel()
            self._active[job.key] = job

        progress = ProgressReporter(job.key)
        progress.progress.connect(self._signals.progress)
        runnable = _Runnable(job, self._signals, progress, self._finalize)
        self._pool.start(runnable)

    def cancel(self, key: str) -> None:
        """Cancel the active job under ``key`` (if any)."""
        with self._lock:
            job = self._active.get(key)
        if job is not None:
            job.cancel()

    def cancel_all(self) -> None:
        """Cancel every active job."""
        with self._lock:
            for job in self._active.values():
                job.cancel()

    def shutdown(self, wait_ms: int = 5000) -> None:
        """Cancel all jobs and wait for the pool to drain.

        Call from :meth:`MainWindow.closeEvent` so the app does
        not exit while a worker thread is mid-frame. ``wait_ms``
        is the upper bound; if it expires, the pool is forcibly
        released by Qt (the OS thread continues but won't keep
        the process alive once the main loop has returned).
        """
        self.cancel_all()
        self._pool.waitForDone(wait_ms)

    def has_active(self, key: str) -> bool:
        """Return ``True`` if a job is currently active under ``key``."""
        with self._lock:
            return key in self._active

    # ── internals ─────────────────────────────────────────────────────────
    def _finalize(self, job: AnalysisJob) -> None:
        """Drop ``job`` from the active map, if it's still the entry there.

        A newer submission under the same key may have already
        replaced this job by the time finalize runs (the old job's
        ``CancelledError`` arrived just after the new ``submit``).
        In that case we leave the new job in place — checking
        identity, not just key.
        """
        with self._lock:
            existing = self._active.get(job.key)
            if existing is job:
                self._active.pop(job.key, None)
