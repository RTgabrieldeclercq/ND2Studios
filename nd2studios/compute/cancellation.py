"""Cancellation primitives for the V1.37 Phase 5 job runner.

:class:`CancellationToken` is the thread-safe replacement for the
``BaseWorker._cancelled`` bool flag. It wraps a
:class:`threading.Event` so callers can either poll it via
:meth:`is_cancelled` (the shape the existing
:meth:`AnalysisPipeline.run` expects for its ``cancelled_cb``
argument) or have it raise :class:`CancelledError` via :meth:`check`
for cooperative unwinding inside the new
:class:`~nd2studios.compute.jobs.AnalysisJob` subclasses.

The bool-flag pattern in ``BaseWorker`` is not memory-safe across
threads — Python guarantees attribute writes are atomic but not
visible to other threads without a memory barrier. ``Event``
internally uses a lock, which gives us the visibility guarantee
plus an interface that already matches what we want.

Example
-------
>>> tok = CancellationToken()
>>> tok.is_cancelled()
False
>>> tok.cancel()
>>> tok.is_cancelled()
True
>>> tok.check()
Traceback (most recent call last):
    ...
nd2studios.compute.cancellation.CancelledError
"""
from __future__ import annotations

import threading


class CancelledError(Exception):
    """Raised by :meth:`CancellationToken.check` when cancellation
    has been requested.

    The job runner catches this as a normal control-flow signal —
    the offending job is not treated as an error and the page
    receives a ``job_cancelled`` rather than ``job_done(ok=False)``.
    """


class CancellationToken:
    """Thread-safe cancellation flag built on :class:`threading.Event`.

    Two interfaces are exposed because callers fall into two camps:

    - Existing :class:`AnalysisPipeline` implementations take a
      ``cancelled_cb`` callable that returns ``bool``. Pass
      :attr:`is_cancelled` (bound method) and they keep working
      verbatim.
    - New code that wants tighter loops can call :meth:`check`
      inside its hot path; cancellation unwinds via
      :class:`CancelledError`.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Safe to call from any thread."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested.

        Non-raising. Matches the ``Callable[[], bool]`` shape that
        :meth:`AnalysisPipeline.run` accepts for ``cancelled_cb``.
        """
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`CancelledError` if cancellation was requested.

        Use inside a job's inner loop to unwind cleanly. The runner
        catches the exception and reports a cancellation, not an
        error.
        """
        if self._event.is_set():
            raise CancelledError()
