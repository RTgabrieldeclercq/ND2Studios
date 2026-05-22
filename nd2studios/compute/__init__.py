"""V1.37 Phase 5 — Background analysis runner with cancellation.

A thin coordination layer on top of ``BaseWorker`` / ``AnalysisWorker``
that gives the rest of the app:

- :class:`CancellationToken` — :class:`threading.Event`-backed,
  exposes both ``is_cancelled()`` (non-raising, compatible with the
  existing ``AnalysisPipeline.run(cancelled_cb=…)`` contract) and
  ``check()`` (raises :class:`CancelledError`, for cooperative
  cancellation inside the new job classes).
- :class:`ProgressReporter` — Qt-aware progress signaling; emits on
  the GUI thread via the runner's queued connections.
- :class:`AnalysisJob` — ABC that pairs a unique key with a
  ``run(progress) -> Any`` implementation.
- :class:`JobRunner` — :class:`QThreadPool`-backed dispatcher that
  coalesces submissions by key (latest wins) and emits
  ``job_done`` / ``job_cancelled`` / ``job_progress`` on the GUI
  thread.
- :class:`ResultStore` — small thread-safe dict for caching the
  most recent result of each job key.
- :func:`PipelinePreviewJob`, :func:`PipelineCommitJob` — adapters
  that bridge the existing ``AnalysisPipeline`` registry to the
  job framework without touching the pipelines themselves.

See ``CodeLog/ClaudesPlan/V1.37_phase5_background_analysis.md`` for
the design rationale and the scope-out list.
"""
from __future__ import annotations

from nd2studios.compute.cancellation import CancellationToken, CancelledError
from nd2studios.compute.jobs import AnalysisJob, JobResult
from nd2studios.compute.pipeline_jobs import (
    BuildPyramidJob,
    PipelineCommitJob,
    PipelinePreviewJob,
)
from nd2studios.compute.progress import ProgressReporter
from nd2studios.compute.result_store import ResultStore
from nd2studios.compute.runner import JobRunner

__all__ = [
    "AnalysisJob",
    "BuildPyramidJob",
    "CancellationToken",
    "CancelledError",
    "JobResult",
    "JobRunner",
    "PipelineCommitJob",
    "PipelinePreviewJob",
    "ProgressReporter",
    "ResultStore",
]
