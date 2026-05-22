"""Base classes for jobs run on the V1.37 Phase 5 job runner.

An :class:`AnalysisJob` is a unit of background work the runner can
schedule, cancel, and coalesce against other submissions with the
same key. Implementers override :meth:`run`, sprinkle
``self.token.check()`` inside long loops for cooperative
cancellation, and call ``progress.update(fraction, message)`` as
work advances.

:class:`JobResult` is the payload the runner emits via its
``job_done`` signal — a tagged-union-shaped dataclass so GUI slots
can fan out on ``result.key`` and react to ``result.ok``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from nd2studios.compute.cancellation import CancellationToken
from nd2studios.compute.progress import ProgressReporter


@dataclass(frozen=True)
class JobResult:
    """Outcome of an :class:`AnalysisJob` run.

    Emitted on the runner's ``job_done`` signal once the worker
    thread finishes. ``ok=True`` and ``value`` set on success;
    ``ok=False`` and ``error`` set on exception. A cancelled job
    does *not* produce a :class:`JobResult` — it emits the
    separate ``job_cancelled`` signal so receivers can distinguish
    "user pressed Cancel" from "pipeline raised RuntimeError".
    """

    key: str
    ok: bool
    value: Any = None
    error: Optional[str] = None


class AnalysisJob(ABC):
    """Abstract base for jobs scheduled by :class:`JobRunner`.

    Subclasses fill in :meth:`run` and optionally override
    :attr:`coalesce_key` if they want to use a key different from
    the one supplied at construction (rare). The default behavior
    is: the runner cancels any in-flight job sharing the same
    :attr:`key` before scheduling the new one.

    Parameters
    ----------
    key:
        Coalescing key. Two jobs with the same key are mutually
        exclusive on the runner — submitting the second cancels
        the first.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self.token = CancellationToken()

    def cancel(self) -> None:
        """Request cancellation of this job."""
        self.token.cancel()

    @abstractmethod
    def run(self, progress: ProgressReporter) -> Any:
        """Perform the work and return the result.

        Implementers should:

        - call ``progress.update(fraction, message)`` at sensible
          checkpoints (every iteration for slow loops, every N
          iterations for tight ones);
        - call ``self.token.check()`` inside loops so cancellation
          unwinds promptly via :class:`CancelledError`;
        - never touch Qt widgets directly — the worker thread is
          not the GUI thread. Snapshot any widget state at
          construction time and let signals carry results back.
        """
        ...
