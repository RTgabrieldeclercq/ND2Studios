"""Bridge :class:`AnalysisPipeline` instances to the V1.37 job runner.

The existing pipeline registry (see
:mod:`nd2studios.core.analysis_registry`) defines
``AnalysisPipeline.run(channels, metadata, params, progress_cb,
cancelled_cb) -> AnalysisResult``. These two adapter jobs let the
:class:`~nd2studios.compute.runner.JobRunner` schedule that
``run()`` method without touching any concrete pipeline:

- :class:`PipelinePreviewJob` runs the pipeline on a single
  ``(M, T)`` frame. It's the fast tier the Analysis page reaches
  for whenever a parameter slider or viewer-coords slider
  changes. Output is an :class:`AnalysisResult` whose
  ``label_masks`` dict has one entry of shape ``(1, H, W)``.
- :class:`PipelineCommitJob` runs the pipeline on a full
  ``(T, H, W)`` channel stack — exactly what the legacy
  :class:`~nd2studios.workers.analysis_worker.AnalysisWorker`
  used to do. The page's existing multi-M state machine
  (``_run_queue`` / ``_start_next_m_run``) is unchanged; it just
  submits a commit job per M instead of starting an
  ``AnalysisWorker``.

Both jobs hand the pipeline ``self.token.is_cancelled`` as
``cancelled_cb`` (the bool-returning shape the existing
``AnalysisPipeline`` contract expects) and a 0–100 percent
``progress_cb`` derived from
:meth:`ProgressReporter.as_pipeline_progress_cb`. They do not
require pipeline implementations to change.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np

from nd2studios.compute.jobs import AnalysisJob
from nd2studios.compute.progress import ProgressReporter
from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult


class PipelinePreviewJob(AnalysisJob):
    """Run an :class:`AnalysisPipeline` on a single ``(1, H, W)`` frame.

    Parameters
    ----------
    key:
        Coalescing key — pages typically use ``"analysis_preview"``
        so rapid slider movement collapses to one in-flight job.
    pipeline_cls:
        The pipeline class to instantiate. Construction is cheap;
        we don't reuse instances across submissions so the
        pipeline's internal state is always fresh.
    channel_name:
        Name of the channel to feed the pipeline. The job builds
        a one-entry ``channels`` dict — pipelines that consume a
        single channel work as-is; pipelines that need a
        counter-stain are out of scope for preview (those run
        only on commit).
    frame:
        The ``(1, H, W)`` plane to analyze. Caller is responsible
        for extracting this from the active volume on the GUI
        thread (fast on a cache hit).
    metadata:
        ND2 metadata dict, augmented with ``pixel_size_um``.
        Carried verbatim into ``pipeline.run``.
    params:
        Param dict from the page's :class:`ParamEditor`.
    tag:
        Free-form payload returned alongside the result so the
        page's slot can verify the result still matches the
        currently-displayed ``(M, T, Z)``. Stale results are
        silently dropped.
    """

    def __init__(
        self,
        key: str,
        pipeline_cls: Type[AnalysisPipeline],
        channel_name: str,
        frame: np.ndarray,
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        tag: Any = None,
    ) -> None:
        super().__init__(key)
        self._pipeline_cls = pipeline_cls
        self._channel_name = channel_name
        self._frame = frame
        self._metadata = metadata
        self._params = params
        self.tag = tag

    def run(self, progress: ProgressReporter) -> AnalysisResult:
        # Pipelines expect channels keyed by name; pass the single
        # frame as a (1, H, W) stack so existing per-frame loops
        # iterate exactly once.
        channels = {self._channel_name: self._frame}
        pipeline = self._pipeline_cls()
        progress.update(0.0, f"Screening {pipeline.name}")
        result = pipeline.run(
            channels,
            self._metadata,
            self._params,
            progress_cb=progress.as_pipeline_progress_cb(),
            cancelled_cb=self.token.is_cancelled,
        )
        progress.update(1.0, "Done")
        return result


class PipelineCommitJob(AnalysisJob):
    """Run an :class:`AnalysisPipeline` on a full multi-channel stack.

    Drop-in replacement for one ``AnalysisWorker`` invocation. The
    Analysis page's multi-M state machine submits one commit job
    per M position, sequentially, exactly as before — only the
    worker class changes.

    Parameters
    ----------
    key:
        Typically ``"analysis_commit"``. Submitting a new commit
        cancels the in-flight one (same coalescing rule as
        previews).
    pipeline_cls:
        The pipeline class.
    channels:
        ``{channel_name: (T, H, W) ndarray}`` for this M position.
    metadata:
        ND2 metadata + ``pixel_size_um``.
    params:
        Param dict, with any per-M shape injection
        (Manual Mask) already done by the caller.
    m_index:
        The M position this job corresponds to. Carried in the
        :attr:`tag` so the page can route the result back into
        ``_results_per_m`` without inventing a new signal.
    """

    def __init__(
        self,
        key: str,
        pipeline_cls: Type[AnalysisPipeline],
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        m_index: int,
    ) -> None:
        super().__init__(key)
        self._pipeline_cls = pipeline_cls
        self._channels = channels
        self._metadata = metadata
        self._params = params
        self.m_index = m_index
        self.tag = m_index

    def run(self, progress: ProgressReporter) -> AnalysisResult:
        pipeline = self._pipeline_cls()
        progress.update(0.0, f"Running {pipeline.name}")
        result = pipeline.run(
            self._channels,
            self._metadata,
            self._params,
            progress_cb=progress.as_pipeline_progress_cb(),
            cancelled_cb=self.token.is_cancelled,
        )
        progress.update(1.0, "Done")
        return result


class BuildPyramidJob(AnalysisJob):
    """Build a per-source multi-resolution pyramid in the background.

    V1.39 Phase 7. Wraps :meth:`PyramidStage.build` into a
    :class:`JobRunner`-friendly unit so the build is cancellable,
    progress-reported, and coalesced (submitting a new build cancels
    any in-flight one). Coalesce key convention: ``"pyramid_build"``.

    The job runs synchronously inside a worker thread and emits the
    resulting :class:`StageRecord` on success. Failures (missing
    zarr, source too small) propagate as ``JobResult.ok=False`` with
    the exception's message — the Performance dialog surfaces this
    as a status note. Cancellation leaves a partial Zarr store on
    disk; the next build wipes it cleanly.
    """

    def __init__(
        self,
        key: str,
        stage,
        volume,
    ) -> None:
        super().__init__(key)
        self._stage = stage
        self._volume = volume

    def run(self, progress: ProgressReporter):
        progress.update(0.0, "Building pyramid")

        def _percent_cb(pct: int) -> None:
            try:
                frac = max(0.0, min(1.0, float(pct) / 100.0))
            except (TypeError, ValueError):
                return
            progress.update(frac, f"Building pyramid ({int(pct)}%)")

        record = self._stage.build(
            self._volume,
            progress_cb=_percent_cb,
            cancel_cb=self.token.is_cancelled,
        )
        progress.update(1.0, "Pyramid built")
        return record
