"""Pipelines page — GA3-style node-graph editor (V1.45).

Fuses the Recipe / Analysis / Results flow into one node board. There are two
sub-tabs: **Processing** (commit a recipe via Apply) and a merged **Analysis**
tab that holds analysis pipelines (pink), results ops (green), if-else branch
nodes (purple triangles) and special action nodes (orange hexagons; Dismiss is
a red rectangle) in one scene, executed by **Run**.

Layout (spec §4.2): a horizontal ``QSplitter`` with the node board on the left
and the viewer on the right. The board has a sub-tab selector + a control bar
(Add, Preview ON/OFF, Apply / Run, Undo, preview progress) above a single
``QGraphicsView`` that swaps one ``NodeScene`` per sub-tab. Run walks the merged
graph painting each node shaded (pending) → gold (executing) → normal (done),
branching at if-else nodes; Pause drops back to editor mode (all un-shaded).

Interaction model (carried from the Processing rounds):
- Vertical node flow — inputs anchor on top, outputs on bottom; the stage
  accent is a left-edge lip. Wires are rounded-elbow Béziers; double-click a
  wire to disconnect.
- **Sticky preview** — a single click opens a node's parameter pop-up; a
  *double-click* promotes a node to the *previewed* node (golden outline on it
  and its upstream lineage) whose result drives the viewer.

Preview / Apply run through ``MainWindow.job_runner`` (coalesced, debounced
~300 ms, honoring ``heavy_ops_blocked()``):
- **Processing** preview applies the linearized recipe to the channels and
  shows the processed image (``set_channels``).
- **Analysis** preview runs the previewed pipeline on the current frame and
  overlays its label masks via ``MultiAxisViewer.set_frame_post_process`` — the
  same overlay path the Analysis page uses. A processed *base image* is shown
  underneath. Apply runs the pipeline on the current M's full stack
  (``PipelineCommitJob``) into ``record.analysis_results``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabBar,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nd2studios.backend.celltracker_bridge import (
    augment_rows_with_metrics, build_tracked_df,
)
from nd2studios.backend.object_tracker import (
    link_objects, link_objects_with_params,
)
from nd2studios.backend.results_engine import (
    EXPORT_ORG_MAP,
    compute_measurements,
    export_organized,
)
from nd2studios.compute import JobResult, PipelineCommitJob
from nd2studios.compute.jobs import AnalysisJob
from nd2studios.compute.progress import ProgressReporter
from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph import (
    IF_ELSE_OP_KEY,
    PIPELINE_EXTENSION,
    SPECIAL_CT_FIELDS_OP_KEY,
    SPECIAL_CT_METRICS_OP_KEY,
    SPECIAL_DISMISS_OP_KEY,
    SPECIAL_DVC_OP_KEY,
    SPECIAL_EXPORT_OP_KEY,
    SPECIAL_INTERP_MAP_OP_KEY,
    SPECIAL_CHECKPOINT_OP_KEY,
    SPECIAL_PAUSE_OP_KEY,
    SPECIAL_REGISTER_OP_KEY,
    SPECIAL_REVIEW_OP_KEY,
    SPECIAL_SEND_RESULTS_OP_KEY,
    SPECIAL_TRACK_OP_KEY,
    SPECIAL_VALIDATE_OP_KEY,
    Condition,
    GROUP_ROW,
    LENS_OBJECT,
    CHANNEL_ALL_OP_KEY,
    CHANNEL_PREFIX,
    CroppedVolume,
    GraphRunner,
    PinnedProcessedVolume,
    PipelineDoc,
    ProcessedFrameVolume,
    RegisteredFrameVolume,
    Stage,
    analysis_input_spec,
    analysis_output_spec,
    analysis_pipeline_name_for_op_key,
    apply_recipe,
    channel_all_spec,
    channel_name_for_op_key,
    channel_recipes,
    channel_sets,
    channel_source_spec,
    describe_condition,
    enhancement_specs,
    evaluate_condition,
    has_channel_wiring,
    input_node,
    is_channel_source_op,
    load_pipeline,
    merged_action_specs,
    output_nodes,
    param_specs_for,
    partition_rows,
    predecessor,
    processing_input_spec,
    processing_output_spec,
    recipe_for_node,
    save_pipeline,
    topological_order,
)
from nd2studios.pipeline_graph.model import LOOP_KIND, NodeCategory, NodeRole
from nd2studios.widgets.icon_button import bind_toggle_icon, icon_button, make_icon, scaled
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
from nd2studios.widgets.popout_window import PopOutWindow
from nd2studios.widgets.node_board import (
    AddNodeDialog, ConditionBuilderDialog, MeasurementSelectDialog, NodeScene,
    ParamPopup,
)
from nd2studios.pipeline_graph.registry_adapter import (
    RESULTS_PREFIX, results_op_name_for_op_key,
)

_log = logging.getLogger(__name__)

# Coalescing keys for the shared JobRunner (a fresh submit cancels the previous).
_PREVIEW_KEY = "pipeline_preview"            # processing image / analysis+results base
_ANALYSIS_PREVIEW_KEY = "pipeline_analysis_preview"   # single-frame overlay
_ANALYSIS_COMMIT_KEY = "pipeline_analysis_commit"     # full-M commit
_RESULTS_PREVIEW_KEY = "pipeline_results_preview"     # measurements for the table
_RUN_ANALYSIS_KEY = "pipeline_run_analysis"           # Run: analysis node compute (per M)
_RUN_MEASURE_KEY = "pipeline_run_measure"             # Run: per-M measurement in the analysis loop
_RUN_TRACK_KEY = "pipeline_run_track"                 # Run: per-M default tracking (off the GUI thread)
_RUN_TRACKOBJ_KEY = "pipeline_run_trackobj"           # Run: Track Objects node (off the GUI thread)
_RUN_DVC_KEY = "pipeline_run_dvc"                     # Run: DVC (ALDVC) node (off the GUI thread)
_RUN_REGISTER_KEY = "pipeline_run_register"           # Run: Registration node (off the GUI thread)
_RUN_RESULTS_KEY = "pipeline_run_results"             # Run: explicit measurement node (legacy)
_RUN_LOOPCOMBINE_KEY = "pipeline_run_loopcombine"     # Run: loop union-dedup combine (off the GUI thread)
_PV_SCREEN_KEY = "pipeline_preview_walk"              # Preview walk: screen+measure selected planes

_STAGE_ACCENT = {
    Stage.PROCESSING: Settings.ACCENT_CYAN,
    Stage.ANALYSIS: Settings.ACCENT_PINK,
    Stage.RESULTS: Settings.ACCENT_GREEN,
}
_STAGE_LABEL = {
    Stage.PROCESSING: "Processing",
    Stage.ANALYSIS: "Analysis",
    Stage.RESULTS: "Results",
}


class _BoardView(QGraphicsView):
    """Graphics view with wheel-zoom and rubber-band selection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)


class _ProcessingPreviewJob(AnalysisJob):
    """Run a Processing recipe on the selected preview plane(s) (Qt-free executor).

    ``planes`` maps each ``(m, t)`` the user is previewing — the current frame, or
    every frame of a multi-frame selection — to its raw channels
    (``{name: (1, H, W)}``). Each plane is processed in turn; the result is
    ``{(m, t): {name: (1, H, W)}}``. Runs on a worker thread (progress per plane),
    polling the token so a superseded job reports ``job_cancelled``.
    """

    def __init__(
        self,
        key: str,
        planes: Dict[Any, Dict[str, Any]],
        recipe: List,
        normalized: bool,
        recipe_by_channel: Optional[Dict[str, List]] = None,
    ) -> None:
        super().__init__(key)
        self._planes = planes
        self._recipe = recipe
        self._normalized = normalized
        self._recipe_by_channel = recipe_by_channel

    def run(self, progress: ProgressReporter) -> Dict[Any, Dict[str, Any]]:
        progress.update(0.0, "Processing preview")
        out: Dict[Any, Dict[str, Any]] = {}
        total = max(1, len(self._planes))
        for i, (mt, channels) in enumerate(self._planes.items()):
            self.token.check()
            out[mt] = apply_recipe(
                channels, self._recipe, self._normalized,
                cancelled_cb=self.token.is_cancelled,
                recipe_by_channel=self._recipe_by_channel,
            )
            progress.update((i + 1) / total, f"Processing {i + 1}/{total}")
        self.token.check()  # superseded? -> CancelledError -> job_cancelled
        progress.update(1.0, "Done")
        return out


def _merge_analysis_results(dst: AnalysisResult, src: AnalysisResult,
                            first: bool) -> None:
    """Merge one channel's ``src`` result into the multi-channel ``dst`` (V1.48).

    ``label_masks`` are keyed by channel name, so merging preserves one entry per
    channel (and the row-level ``segmentation_channel`` downstream). Overlay style
    is taken from the first channel."""
    dst.label_masks.update(src.label_masks or {})
    if getattr(src, "secondary_label_masks", None):
        dst.secondary_label_masks.update(src.secondary_label_masks)
    if getattr(src, "volumetric_voxel_counts", None):
        if dst.volumetric_voxel_counts is None:
            dst.volumetric_voxel_counts = {}
        dst.volumetric_voxel_counts.update(src.volumetric_voxel_counts)
    if first:
        dst.overlay_color = src.overlay_color
        dst.overlay_alpha = src.overlay_alpha
        dst.overlay_outline = getattr(src, "overlay_outline", False)
        dst.secondary_overlay_color = src.secondary_overlay_color
        dst.secondary_overlay_alpha = src.secondary_overlay_alpha


class _AnalysisPreviewJob(AnalysisJob):
    """Run an analysis pipeline on the selected preview plane(s), per channel.

    ``frames_by_mt`` maps each previewed ``(m, t)`` to ``{channel: (1, H, W)}``
    for every wired channel (V1.48). Each plane is screened once per channel and
    the per-channel label masks are merged into one :class:`AnalysisResult`; the
    result is ``{(m, t): AnalysisResult}``. Runs on a worker thread (progress per
    plane), polling the token so a superseded job reports ``job_cancelled``.
    """

    def __init__(self, key: str, pipeline_cls, channel_names,
                 frames_by_mt: Dict[tuple, Dict[str, Any]], metadata: Dict[str, Any],
                 params: Dict[str, Any]) -> None:
        super().__init__(key)
        self._pipeline_cls = pipeline_cls
        self._channel_names = list(channel_names)
        self._frames_by_mt = frames_by_mt
        self._metadata = metadata
        self._params = params

    def run(self, progress: ProgressReporter) -> Dict[tuple, AnalysisResult]:
        out: Dict[tuple, AnalysisResult] = {}
        total = max(1, len(self._frames_by_mt))
        for i, (mt, chan_frames) in enumerate(self._frames_by_mt.items()):
            self.token.check()
            merged = AnalysisResult()
            for j, ch in enumerate(self._channel_names):
                frame = chan_frames.get(ch)
                if frame is None:
                    continue
                self.token.check()
                params = dict(self._params)
                params["channel_name"] = ch
                res = self._pipeline_cls().run(
                    {ch: frame}, self._metadata, params,
                    progress_cb=None, cancelled_cb=self.token.is_cancelled,
                )
                _merge_analysis_results(merged, res, first=(j == 0))
            out[mt] = merged
            progress.update((i + 1) / total, f"Screening {i + 1}/{total}")
        self.token.check()
        return out


class _MultiChannelCommitJob(AnalysisJob):
    """Run an analysis pipeline once per wired channel on a full stack and merge
    the per-channel label masks (V1.48).

    Replaces :class:`PipelineCommitJob` on the channel-wire Run: each channel's
    run keys its ``label_masks`` by that channel, so the merged result carries one
    entry per channel and downstream measurement / tracking (grouped by the
    row-level ``segmentation_channel``) stay per-channel automatically. Streams
    each channel's frames to the live overlay via ``_frame_cb``.
    """

    def __init__(self, key: str, pipeline_cls, channels: Dict[str, Any],
                 metadata: Dict[str, Any], params: Dict[str, Any], m_index: int,
                 seg_channels) -> None:
        super().__init__(key)
        self._pipeline_cls = pipeline_cls
        self._channels = channels
        self._metadata = metadata
        self._params = params
        self.m_index = m_index
        self.tag = m_index
        self._seg_channels = list(seg_channels)

    def run(self, progress: ProgressReporter) -> AnalysisResult:
        merged = AnalysisResult()
        n = max(1, len(self._seg_channels))
        for i, ch in enumerate(self._seg_channels):
            self.token.check()
            params = dict(self._params)
            params["channel_name"] = ch
            params["_frame_cb"] = (
                lambda t, lbl, _m=self.m_index: progress.report_frame(_m, t, lbl))
            res = self._pipeline_cls().run(
                self._channels, self._metadata, params,
                progress_cb=progress.as_pipeline_progress_cb(),
                cancelled_cb=self.token.is_cancelled,
            )
            _merge_analysis_results(merged, res, first=(i == 0))
            progress.update((i + 1) / n, f"Segmenting {ch}")
        return merged


class _ResultsMeasureJob(AnalysisJob):
    """Compute per-object measurements for a committed analysis result.

    Wraps ``results_engine.compute_measurements`` off the GUI thread. The
    processed channels are materialized inside the worker (they may be a lazy
    ``EnhancedDataset``) so the GUI never blocks. Returns the measurement rows
    for the Results table.
    """

    def __init__(
        self,
        key: str,
        label_masks: Dict[str, Any],
        channel_source: Any,
        metadata: Dict[str, Any],
        m_index: int,
        voxel_counts: Any = None,
        metrics: Any = None,
    ) -> None:
        super().__init__(key)
        self._label_masks = label_masks
        self._channel_source = channel_source
        self._metadata = metadata
        self._m_index = m_index
        self._voxel_counts = voxel_counts
        self._metrics = metrics

    def run(self, progress: ProgressReporter) -> List[Dict[str, Any]]:
        # ``compute_measurements`` is imported at module load (main thread) so
        # skimage/scipy are never first-imported here on a worker thread.
        progress.update(0.0, "Computing measurements")
        channels: Dict[str, Any] = {}
        src = self._channel_source
        try:
            keys = list(src.keys())
        except Exception:  # noqa: BLE001
            keys = []
        for name in keys:
            self.token.check()
            try:
                channels[name] = np.asarray(src[name])
            except Exception:  # noqa: BLE001
                continue
        rows = compute_measurements(
            self._label_masks, channels, self._metadata,
            m_index=self._m_index, volumetric_voxel_counts=self._voxel_counts,
            metrics=self._metrics,
        )
        self.token.check()
        progress.update(1.0, "Done")
        return rows


class _TrackJob(AnalysisJob):
    """Link objects into tracks off the GUI thread.

    Wraps ``object_tracker.link_objects`` / ``link_objects_with_params`` so the
    Hungarian (or SerialTrack / Cell-Tracker) linker — whose cost grows with the
    object count and can take many seconds on a dense field of thousands of nuclei
    (and minutes for SerialTrack) — never blocks the Qt event loop (the symptom is
    the window going "not responding" right at the segmentation→tracking
    transition). The linker mutates ``rows`` in place and the result is returned;
    the run walk waits for the job before touching those rows again, so the
    in-place mutation is safe across the thread hand-off.

    ``params is None`` → the default per-M ``link_objects`` (centroid, default
    thresholds); otherwise a Track Objects node's ``link_objects_with_params``.
    """

    def __init__(self, key: str, rows: List[Dict[str, Any]],
                 params: Optional[Dict[str, Any]] = None,
                 pixel_size_um: Optional[float] = None,
                 label_masks: Optional[Dict[Tuple[str, int], Any]] = None) -> None:
        super().__init__(key)
        self._rows = rows
        self._params = params
        self._pixel_size_um = pixel_size_um
        # {(segmentation_channel, m_position): (T,H,W) label image} for the
        # mask-overlap (IoU) method; None for the other linkers.
        self._label_masks = label_masks

    def run(self, progress: ProgressReporter) -> List[Dict[str, Any]]:
        progress.update(0.0, "Tracking objects")

        # Per-frame progress from the linker so a long track (dense field →
        # O(n³) Hungarian) shows movement instead of a frozen bar, and can be
        # cancelled mid-run: token.check() raises on Stop between frames.
        def _cb(fraction: float, message: str = "") -> None:
            self.token.check()
            progress.update(fraction, message or "Tracking objects")

        if self._params is None:
            link_objects(self._rows, progress_cb=_cb)
        else:
            link_objects_with_params(
                self._rows, self._params, pixel_size_um=self._pixel_size_um,
                label_masks=self._label_masks, progress_cb=_cb)
        self.token.check()
        progress.update(1.0, "Done")
        return self._rows


class _LoopCombineJob(AnalysisJob):
    """Union + overlap-dedup a loop's iterations off the GUI thread (V1.49).

    ``deduplicate_objects`` compares detections across every iteration by mask
    IoU and repaints a merged label volume — on a dense field (hundreds of nuclei
    × many frames × several iterations) that is seconds-to-minutes of NumPy work.
    Running it here keeps the window responsive instead of freezing right at the
    end of the loop. Returns ``(rows, merged_masks)`` where ``merged_masks`` is
    keyed by ``(m_position, channel)``; the GUI thread rebuilds the per-M results
    from those (a light, reference-only step)."""

    def __init__(self, key: str, iterations: List[Any],
                 dedup: Dict[str, Any]) -> None:
        super().__init__(key)
        self._iterations = iterations
        self._dedup = dict(dedup or {})

    def run(self, progress: ProgressReporter):
        from nd2studios.pipeline_graph.loop import (
            IterationResult, deduplicate_objects,
        )
        progress.update(0.1, "Combining iterations (overlap dedup)")
        its = [IterationResult(i, {}, it["rows"], it["masks"])
               for i, it in enumerate(self._iterations)]
        self.token.check()
        rows, merged = deduplicate_objects(
            its,
            metric=str(self._dedup.get("metric", "iou")),
            iou_threshold=float(self._dedup.get("iou_threshold", 0.3)),
            centroid_distance=float(self._dedup.get("centroid_distance", 10.0)))
        self.token.check()
        progress.update(1.0, "Done")
        return rows, merged


def _bin_xy(vol: "np.ndarray", d: int) -> "np.ndarray":
    """Block-mean downsample a ``(H,W)`` or ``(Z,H,W)`` array by ``d`` in X and Y
    (crops to a multiple of ``d`` first). ``d<=1`` is a no-op. Keeps huge 4k² DVC
    tractable — displacements come out in downsampled voxels, so the caller scales
    the voxel size by ``d`` and the field/backdrop stay in the same pixel space."""
    import numpy as _np
    d = int(d)
    if d <= 1:
        return vol
    a = _np.asarray(vol)
    hy, hx = a.shape[-2] - a.shape[-2] % d, a.shape[-1] - a.shape[-1] % d
    if hy < d or hx < d:
        return a
    a = a[..., :hy, :hx]
    new = a.shape[:-2] + (hy // d, d, hx // d, d)
    return a.reshape(new).mean(axis=(-3, -1)).astype(_np.float32)


class _DVCJob(AnalysisJob):
    """Run the ALDVC field **series** over a timelapse, off the GUI thread.

    For each multipoint it iterates the deformed frames and correlates, per the
    tracking mode, either the fixed reference→frame (``cumulative``) or the
    previous→frame (``incremental``) volume pair — exactly how FranckLab's
    ``main_ALDVC.m`` walks ``ImgSeqNum = 2..N``. Volumes are read here (the app
    reads ``_raw_volume`` off-thread in other workers too), one Z-stack at a time
    via ``get_volume`` (so a 121 GB file never lands in RAM whole), optionally
    Z-ranged and XY-downsampled for tractability. DVC itself (IC-GN + ADMM) fans
    across processes inside the engine.

    Faithful to ALDVC's series handling: each frame **warm-starts** its seed from
    the previous frame's field (FFT only on the first frame, or every frame when
    ``newFFTSearch``); in **incremental** mode the per-step increments are then
    **accumulated** into a cumulative field (:mod:`nd2studios.backend.dvc.tracking`).
    Returns ``{m: {"primary": {t: (DVCResult, backdrop)}, "increment": {t: DVCResult}}}``
    — ``primary`` is the cumulative field (accumulated in incremental mode);
    ``increment`` carries the raw per-step increments for the viewer toggle.
    """

    def __init__(self, key: str, vol, c_idx, m_list, frames, ref_frame, mode,
                 z_start, z_end, downsample, voxel_size_um, params) -> None:
        super().__init__(key)
        self._vol = vol
        self._c = int(c_idx)
        self._m_list = list(m_list)
        self._frames = list(frames)                # deformed frames to compute
        self._ref_frame = int(ref_frame)
        self._mode = str(mode)
        self._z0 = int(z_start)
        self._z1 = z_end                            # None = all Z
        self._down = max(1, int(downsample))
        self._voxel = voxel_size_um
        self._params = params

    def _read_volume(self, m: int, t: int):
        import numpy as _np
        v = _np.asarray(self._vol.get_volume(
            self._c, m=int(m), t=int(t), z_start=self._z0, z_end=self._z1))
        if v.ndim == 3 and v.shape[0] == 1:         # singleton Z → 2D DIC
            v = v[0]
        return _bin_xy(v, self._down)

    def run(self, progress: ProgressReporter):
        import numpy as _np
        from nd2studios.backend.dvc.method import ALDVCMethod
        from nd2studios.backend.dvc.mesh import build_grid
        from nd2studios.backend.dvc.tracking import build_accumulated_results
        method = ALDVCMethod()
        newfft = bool(self._params.get("newFFTSearch", False))
        strain_type = str(self._params.get("strain_type", "infinitesimal"))
        subset = int(self._params.get("subset_size", 16))
        spacing = int(self._params.get("subset_spacing", 10))
        out: Dict[int, Dict[str, Any]] = {}
        total = max(1, len(self._m_list) * max(1, len(self._frames)))
        done = 0
        for m in self._m_list:
            ref_vol = (self._read_volume(m, self._ref_frame)
                       if self._mode == "cumulative" else None)
            prev_vol, prev_t, prev_u = None, None, None
            primary: Dict[int, Any] = {}      # t -> (result, backdrop)
            incr: List[Any] = []              # [(t, result)] for incremental
            vshape = None
            for t in self._frames:
                self.token.check()
                if self._mode == "cumulative":
                    rvol = ref_vol
                else:                               # incremental: previous frame
                    rvol = (prev_vol if (prev_vol is not None and prev_t == t - 1)
                            else self._read_volume(m, t - 1))
                dvol = self._read_volume(m, t)
                vshape = dvol.shape
                base = done / total
                progress.update(base, f"Correlating M{int(m) + 1} T{int(t) + 1}…")

                def _cb(pct: int, _b=base, _tot=total, _m=int(m), _t=int(t)) -> None:
                    self.token.check()
                    progress.update(min(1.0, _b + (float(pct) / 100.0) / _tot),
                                    f"DVC M{_m + 1} T{_t + 1}")

                # Warm-start each frame from the previous frame's field (ALDVC U0);
                # FFT-seed only the first frame, or every frame when newFFTSearch.
                warm = (prev_u is not None and not newfft)
                res = method.run(rvol, dvol, self._voxel, self._params,
                                 progress_cb=_cb, cancelled_cb=self.token.is_cancelled,
                                 u0_seed=prev_u, use_fft_seed=not warm)
                bg = dvol.max(axis=0) if dvol.ndim == 3 else dvol
                prev_u = _np.moveaxis(_np.asarray(res.displacement_field), -1, 0)
                prev_vol, prev_t = dvol, t
                primary[int(t)] = (res, _np.asarray(bg))
                if self._mode == "incremental":
                    incr.append((int(t), res))
                done += 1
            if self._mode == "incremental" and incr and vshape is not None:
                progress.update(min(1.0, done / total),
                                f"Accumulating M{int(m) + 1}…")
                grid = build_grid(vshape, subset, spacing)
                accum = build_accumulated_results(
                    grid, incr, self._voxel, strain_type=strain_type)
                primary = {t: (accum[t], primary[t][1]) for t in accum}
                out[int(m)] = {"primary": primary,
                               "increment": {t: r for t, r in incr}}
            else:
                out[int(m)] = {"primary": primary, "increment": {}}
        return out


class _RegisterJob(AnalysisJob):
    """Estimate an image-registration transform on the reference channel and apply
    it to all requested channels, off the GUI thread.

    Per multipoint it reads each channel's projected ``(T,H,W)`` series (one frame
    at a time via ``get_frame``, so a huge file never lands in RAM whole),
    registers the **reference** channel with :class:`RigidRegistration`, then
    applies the SAME per-frame transform to every requested channel (register once,
    apply to all — preserving colocalization). Returns
    ``{m: {"channel", "raw_ref" (T,H,W), "aligned" {name:(T,H,W)}, "shifts" (T,2),
    "confidence" (T,), "pixel_size_um", "model", "reference_mode"}}``.
    """

    def __init__(self, key: str, vol, ref_c_idx, apply_c_indices, channel_names,
                 m_list, n_t, z_index, z_mode, pixel_size_um, params) -> None:
        super().__init__(key)
        self._vol = vol
        self._ref_c = int(ref_c_idx)
        self._apply_c = list(apply_c_indices)
        self._names = list(channel_names)
        self._m_list = list(m_list)
        self._n_t = int(n_t)
        self._zi = int(z_index)
        self._zmode = str(z_mode)
        self._px = pixel_size_um
        self._params = params

    def _read_series(self, c_idx: int, m: int):
        import numpy as _np
        frames = []
        for t in range(self._n_t):
            self.token.check()
            frames.append(_np.asarray(self._vol.get_frame(
                int(c_idx), m=int(m), t=int(t), z=self._zi, z_mode=self._zmode)))
        return _np.stack(frames)

    def run(self, progress: ProgressReporter):
        import numpy as _np
        from nd2studios.backend.registration import estimate as _est
        from nd2studios.backend.registration.method import RigidRegistration
        method = RigidRegistration()
        order = int(self._params.get("interp_order", 1))
        out: Dict[int, Dict[str, Any]] = {}
        total = max(1, len(self._m_list))
        for mi, m in enumerate(self._m_list):
            self.token.check()
            base = mi / total
            progress.update(base, f"Registering M{int(m) + 1}…")
            ref_series = self._read_series(self._ref_c, m)

            def _cb(pct: int, _b=base, _tot=total, _m=int(m)) -> None:
                self.token.check()
                progress.update(min(1.0, _b + (float(pct) / 100.0) * 0.5 / _tot),
                                f"Register M{_m + 1}")

            res = method.run(ref_series, ref_series, self._px, self._params,
                             progress_cb=_cb, cancelled_cb=self.token.is_cancelled)
            tf = res.diagnostics.get("transforms")
            aligned: Dict[str, Any] = {self._names[self._ref_c]: _np.asarray(res.aligned)}
            for ci in self._apply_c:
                if int(ci) == self._ref_c:
                    continue
                self.token.check()
                series = self._read_series(int(ci), m)
                aligned[self._names[int(ci)]] = _est.apply_series(
                    series, tf, interp_order=order)
            out[int(m)] = {
                "channel": self._names[self._ref_c],
                "raw_ref": ref_series,
                "aligned": aligned,
                "transforms": tf,   # per-frame effective transforms (apply downstream)
                "shifts": _np.asarray(res.shifts_px),
                "confidence": _np.asarray(res.confidence),
                "gated": (None if tf.get("gated") is None
                          else _np.asarray(tf.get("gated"))),
                "min_confidence": float(self._params.get("min_confidence", 0.2)),
                "pixel_size_um": self._px,
                "model": res.model,
                "reference_mode": res.reference_mode,
            }
        return out


class _ResultsScreenMeasureJob(AnalysisJob):
    """Screen + measure the selected preview plane(s) for the results table.

    ``planes_frames`` maps each previewed ``(m, t)`` to its processed channels
    ``{name: (1, H, W)}``. For each plane the analysis pipeline is run on the
    analyzed channel and the masks are measured (selected ``metrics`` only), with
    every row tagged with its real ``frame`` (t) and ``m_position`` (m). Returns
    ``{"rows": [...all planes...], "results": {(m, t): AnalysisResult}}`` so the
    page can fill the table (all selected frames) *and* paint the per-plane
    overlay. Worker thread; token-polled for prompt supersession.
    """

    def __init__(self, key: str, pipeline_cls, channel_names,
                 planes_frames: Dict[tuple, Dict[str, Any]],
                 metadata: Dict[str, Any], params: Dict[str, Any],
                 metrics: Any = None) -> None:
        super().__init__(key)
        self._pipeline_cls = pipeline_cls
        # V1.48: list of wired channels to segment (was a single channel_name).
        self._channel_names = ([channel_names] if isinstance(channel_names, str)
                               else list(channel_names))
        self._planes_frames = planes_frames
        self._metadata = metadata
        self._params = params
        self._metrics = metrics

    def run(self, progress: ProgressReporter) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        results: Dict[tuple, Any] = {}
        items = list(self._planes_frames.items())
        total = max(1, len(items))
        for i, ((m, t), chans) in enumerate(items):
            self.token.check()
            seg = [c for c in self._channel_names if chans.get(c) is not None]
            if not seg and chans:
                seg = [next(iter(chans.keys()))]
            res = AnalysisResult()
            for j, ch in enumerate(seg):
                self.token.check()
                params = dict(self._params)
                params["channel_name"] = ch
                one = self._pipeline_cls().run(
                    {ch: chans.get(ch)}, self._metadata, params,
                    progress_cb=None, cancelled_cb=self.token.is_cancelled,
                )
                _merge_analysis_results(res, one, first=(j == 0))
            results[(m, t)] = res
            plane_rows = compute_measurements(
                getattr(res, "label_masks", {}) or {}, chans, self._metadata,
                m_index=m, metrics=self._metrics,
            )
            for r in plane_rows:
                r["frame"] = int(t)
                r["m_position"] = int(m)
            rows.extend(plane_rows)
            progress.update((i + 1) / total, f"Measuring {i + 1}/{total}")
        self.token.check()
        progress.update(1.0, "Done")
        return {"rows": rows, "results": results}


class PipelinesPage(QWidget):
    """Page 5 (new): node-graph editor across Processing / Analysis / Results."""

    def __init__(self, main_window=None) -> None:
        super().__init__()
        self.main_window = main_window
        self._runner = getattr(main_window, "job_runner", None)
        if self._runner is not None:
            self._runner.job_done.connect(self._on_runner_done)
            self._runner.job_cancelled.connect(self._on_runner_cancelled)
            self._runner.job_progress.connect(self._on_runner_progress)
            self._runner.job_frame.connect(self._on_run_frame)

        self._doc = PipelineDoc.empty()
        self._stage = Stage.PROCESSING
        # V1.45 merge: Analysis + Results share one sub-tab/scene (Stage.ANALYSIS
        # holds analysis + results + logic + special nodes). Two sub-tabs total.
        self._enabled_stages = {Stage.PROCESSING, Stage.ANALYSIS}
        self._stages = (Stage.PROCESSING, Stage.ANALYSIS)
        self._scenes: Dict[Stage, NodeScene] = {}
        self._selected_node_id: str = ""
        # The "previewed" node per stage — golden outline + drives the viewer.
        # Sticky: only a double-click changes it; selecting / clicking off does
        # not.
        self._preview_node_ids: Dict[Stage, str] = {
            Stage.PROCESSING: "", Stage.ANALYSIS: "", Stage.RESULTS: "",
        }
        self._normalized = False
        self._bridge_counter = 0           # Processing #N
        self._analysis_bridge_counter = 0  # Analysis #N
        self._results_bridge_counter = 0   # Results #N

        # Analysis preview / commit state. Screen results are per previewed
        # plane: {(m, t): AnalysisResult} — the current frame, or every frame of
        # a multi-frame selection.
        self._analysis_screen_results: Dict[tuple, AnalysisResult] = {}
        self._analysis_results_per_m: Dict[int, AnalysisResult] = {}
        self._analysis_commit_pipeline: str = ""
        self._analysis_commit_m: int = 0
        # V1.46 label-streaming scratch dir (lazily created on analysis Apply).
        self._analysis_scratch: Optional[str] = None

        # Results preview state.
        self._results_overlay_result: Optional[AnalysisResult] = None
        self._results_rows: List[Dict[str, Any]] = []
        self._results_view_mode: str = "split"
        # Overlay tabs on the viewer: which layer paints over the processed base.
        # The base image is cached; switching just recomputes the overlay layer.
        self._overlay_tab_keys = [
            "image", "segmentation", "tracks", "vectors_cells", "vectors_field",
            "spatial", "serialtrack", "dvc", "registration",
        ]
        self._overlay_mode: str = "segmentation"
        # Track-colour overlay cache (built once per tracked result; Phase 5).
        self._track_colormap: Optional[Dict[int, tuple]] = None
        self._track_long_ids: set = set()
        # Tracked rows captured at Track-Objects time for the tracks / vectors
        # overlays — a *stable* source so a downstream Dismiss / if-else filter
        # (which mutates ``_results_rows``) never blanks the visualization.
        self._track_overlay_rows: List[Dict[str, Any]] = []
        # Live per-frame streaming state (during a Run): the latest labels per
        # (m, t) for the live overlay, and the running cells-per-frame counts.
        self._live_seg: Dict[tuple, np.ndarray] = {}
        self._live_counts: Dict[int, int] = {}
        # M position the current preview was computed for (so a viewer M change
        # triggers a recompute for the newly-displayed position).
        self._preview_m: int = 0
        # The (M, T) plane(s) the Processing preview is processing (isolated
        # processed planes laid over the raw volume) — the current frame, or every
        # frame of a multi-frame selection. FOLLOWS the user: navigating /
        # selecting re-processes in place.
        self._processing_planes: List[tuple] = []
        self._proc_volume = None  # live PinnedProcessedVolume (updated in place)

        # Preview crop (V1.46) — a preview-only XY sub-region. When set, preview
        # (Processing / Analysis / Results, plus overlays + plots) runs on
        # ``(x, y, w, h)`` in raw-image pixels instead of the whole frame; a full
        # Run always processes the whole frame. Not persisted to .nd2s; reset when
        # the active file changes.
        self._preview_crop: Optional[Tuple[int, int, int, int]] = None
        self._crop_selecting: bool = False  # crop tool armed (drawing a rect)
        # Undo stack of prior crop states (each None or an (x,y,w,h) rect) so the
        # ↩ button can step back to the previous crop (or to no-crop).
        self._crop_history: List[Optional[Tuple[int, int, int, int]]] = []
        # Whether the active Run is scoped to the preview crop (chosen via the
        # Run button's dropdown when a crop is set). Only meaningful while
        # ``_run_active``. ``_run_results_crop`` records the crop geometry the
        # committed Run masks were computed at (None = full frame) so overlays
        # only paint them when the current display geometry matches.
        self._run_cropped: bool = False
        self._run_results_crop: Optional[Tuple[int, int, int, int]] = None

        # Run state machine (merged Analysis tab): a GraphRunner walks the graph
        # while the page paints shaded/current/done and parks on async compute.
        self._run_active = False
        self._runner_obj: Optional[GraphRunner] = None
        self._run_states: Dict[str, str] = {}     # node_id -> shaded/current/done
        self._run_context: Dict[str, Any] = {}    # {'rows': [...], 'result': ...}
        self._run_pending: str = ""               # node id of an in-flight async step
        self._run_paused = False                  # paused mid-graph (resume on Run)
        self._run_paused_nodes: set = set()       # node-id snapshot at pause time
        # V1.53 checkpoint store: node_id -> {"hash": <upstream hash>, "data":
        # <frozen run snapshot>}. When a Run reaches a checkpoint it freezes the
        # accumulated masks / rows / tracks here; a later Run whose upstream hash
        # still matches resumes from the checkpoint with this data restored (the
        # upstream nodes never re-run). Session-scoped RAM (cleared on file change).
        self._checkpoint_store: Dict[str, Dict[str, Any]] = {}
        # Multi-M analysis loop: Run processes the WHOLE file (every multipoint),
        # not just the current frame. State for the per-M analysis→measure walk.
        self._run_m_queue: List[int] = []
        self._run_m_total = 0
        self._run_current_m = 0
        self._run_results_by_m: Dict[int, Any] = {}
        self._run_all_rows: List[Dict[str, Any]] = []
        self._run_analysis_ctx: Optional[Dict[str, Any]] = None
        # The current multipoint's processed channels ({name: (T, H, W)}), built
        # per-M by ``_advance_run_m`` and reused for that M's measurement — so a
        # Run analyzes the true per-M repertoire, not a single cached M (V1.48).
        self._run_channels_m: Optional[Dict[str, Any]] = None
        # Off-thread tracking hand-off (so the linker never blocks the GUI): the
        # rows awaiting the per-M default track job, and the Track Objects node
        # whose linker job is in flight.
        self._run_pending_track_rows: Optional[List[Dict[str, Any]]] = None
        self._run_trackobj_node = None
        # Preview walk: a scoped Run on the selected planes that shades the graph,
        # branches at if-else, and (on a deliberate trigger) pops interactive
        # nodes. ``_pv_interactive_next`` is armed by a double-click and consumed
        # by the next preview; plain navigation re-runs non-interactively.
        self._pv_interactive_next = False
        self._pv_target = ""
        self._pv_interactive = False
        self._pv_screen_set: set = set()
        self._pv_states: Dict[str, str] = {}
        self._pv_walking = False
        # V1.49: the Analysis preview is a scoped Run that ONLY regenerates on a
        # node double-click — never on navigation / param edits / selection / crop.
        # A double-click arms this; ``_do_preview`` consumes it (and skips the
        # analysis preview when it is not armed).
        self._analysis_preview_armed = False

        # Preview debounce (mirrors analysis_page's 300 ms _screen_debounce).
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(300)
        self._preview_debounce.timeout.connect(self._do_preview)

        self._popup = ParamPopup(self)
        self._popup.params_changed.connect(self._on_popup_params_changed)
        self._popup.edit_requested.connect(self._on_popup_edit)

        self._build_ui()
        self._select_stage(Stage.PROCESSING)

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Left: node board ────────────────────────────────────────────
        board = QWidget()
        board.setObjectName("nodeBoard")
        board_layout = QVBoxLayout(board)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(6)
        board_layout.addWidget(self._build_subtab_selector())
        board_layout.addWidget(self._build_control_bar())

        self._view = _BoardView()
        self._view.setObjectName("nodeBoardView")
        for stage in self._stages:
            self._scenes[stage] = self._build_scene(stage)
        board_layout.addWidget(self._view, stretch=1)

        # "Coming soon" overlay for deferred sub-tabs.
        self._coming_soon = QLabel("Coming soon", self._view)
        self._coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._coming_soon.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 14pt; background: transparent;"
        )
        self._coming_soon.setVisible(False)

        # ── Right: viewer (+ Results CSV table) ──────────────────────────
        right_pane = self._build_right_pane()

        splitter.addWidget(board)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 700])
        outer.addWidget(splitter)

    def _build_right_pane(self) -> QWidget:
        """Viewer + (Results-only) measurements table, with an image/table/split
        mode bar. The table panel and mode bar are hidden outside the Results
        sub-tab (where the right pane is just the image viewer)."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        # Mode segmented control (Results only).
        self._results_mode_bar = QWidget()
        mb = QHBoxLayout(self._results_mode_bar)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.setSpacing(4)
        mb.addStretch(1)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns: Dict[str, QPushButton] = {}
        for mode, label, icon in (("image", "Image", "fa5s.image"),
                                  ("table", "Table", "fa5s.table"),
                                  ("split", "Split", "fa5s.columns")):
            btn = icon_button(icon, f"{label} view", text=f" {label}",
                              checkable=True, object_name="pipelineToolBtn")
            btn.clicked.connect(lambda _c=False, m=mode: self._set_results_view_mode(m))
            self._mode_group.addButton(btn)
            self._mode_btns[mode] = btn
            mb.addWidget(btn)
        self._results_mode_bar.setVisible(False)
        col.addWidget(self._results_mode_bar)

        # Viewer wrapped with an overlay-tab bar: each tab switches which layer
        # paints over the same cached processed base (cheap — only the overlay
        # recomputes, never the base image).
        self.viewer = MultiAxisViewer(self, show_tile_preview=False)
        self.viewer.coords_changed.connect(self._on_viewer_coords)
        self.viewer.selection_changed.connect(self._on_viewer_selection)
        # Preview crop (V1.46): a drag emits the rect; a single click enters the
        # corner into the dialog. Both only act while the crop tool is armed.
        self.viewer.crop_rect_selected.connect(self._on_preview_crop_drag)
        self.viewer.canvas.clicked.connect(self._on_preview_crop_click)

        self._viewer_container = QWidget()
        vc = QVBoxLayout(self._viewer_container)
        vc.setContentsMargins(0, 0, 0, 0)
        vc.setSpacing(2)
        self._overlay_tabbar = QTabBar()
        self._overlay_tabbar.setObjectName("overlayTabBar")
        self._overlay_tabbar.setExpanding(False)
        self._overlay_tabbar.setDrawBase(False)
        for label in ("Image", "Segmentation", "Tracks",
                      "Vectors: cells", "Vectors: field", "Spatial Maps",
                      "SerialTrack", "DVC", "Registration"):
            self._overlay_tabbar.addTab(label)
        self._overlay_tabbar.setCurrentIndex(
            self._overlay_tab_keys.index(self._overlay_mode))
        self._overlay_tabbar.currentChanged.connect(self._on_overlay_tab_changed)

        # Overlay tab row + a maximize button that pops the whole viewer
        # (overlay tabs + image + every control) out into a full window.
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.setSpacing(4)
        tab_row.addWidget(self._overlay_tabbar)
        # V1.49.x: loop iteration selector — step the overlay/table/plots through a
        # saved loop's iterations ("Combined" + each "#k …"). Hidden until a loop
        # ran with "Save all iterations".
        self._iter_label = QLabel("Iteration:")
        self._iter_label.setStyleSheet(f"color:{Settings.FG_SECONDARY};font:8pt;")
        self._iter_label.setVisible(False)
        self._iter_combo = QComboBox()
        self._iter_combo.setObjectName("loopIterCombo")
        self._iter_combo.setToolTip(
            "Show a saved loop iteration's result (or the Combined result) in the "
            "viewer, table and plots.")
        self._iter_combo.setVisible(False)
        self._iter_combo.currentIndexChanged.connect(self._on_iteration_selected)
        tab_row.addSpacing(8)
        tab_row.addWidget(self._iter_label)
        tab_row.addWidget(self._iter_combo)
        tab_row.addStretch(1)
        # Undo arrow (left of Preview Crop) — step back to the previous crop.
        self._btn_crop_undo = icon_button(
            "fa5s.undo", "Undo the last preview-crop change (restore the "
            "previous crop region).",
            object_name="pipelineToolBtn", icon_px=12,
        )
        self._btn_crop_undo.setEnabled(False)
        self._btn_crop_undo.clicked.connect(self._on_crop_undo)
        tab_row.addWidget(self._btn_crop_undo)
        # Preview Crop — restrict preview compute + display to an XY sub-region.
        self._btn_preview_crop = icon_button(
            "fa5s.crop-alt",
            "Preview Crop — restrict the preview to a sub-region.\n"
            "Toggle on, then drag a rectangle on the image, or click a point to "
            "enter the top-left corner + width/height manually.\n"
            "Preview (segmentation, tracking, measurements, overlays and plots) "
            "then runs on the crop only. Toggle off to clear. A full Run always "
            "uses the whole frame.",
            text="Preview Crop", checkable=True,
            object_name="pipelineToolBtn", icon_px=14,
        )
        self._btn_preview_crop.toggled.connect(self._on_preview_crop_toggled)
        tab_row.addWidget(self._btn_preview_crop)
        self._lbl_preview_crop = QLabel("")
        self._lbl_preview_crop.setObjectName("previewCropLabel")
        self._lbl_preview_crop.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        tab_row.addWidget(self._lbl_preview_crop)
        self._btn_viewer_popout = icon_button(
            "fa5s.expand", "Maximize the image viewer in a separate window",
            object_name="pipelineToolBtn", icon_px=14,
        )
        self._btn_viewer_popout.clicked.connect(lambda: self._toggle_popout("viewer"))
        tab_row.addWidget(self._btn_viewer_popout)
        vc.addLayout(tab_row)
        # The viewer area is a stack: the image viewer (overlay tabs paint over it)
        # and the Spatial Maps panel (its own canvas + sidebar). The "Spatial Maps"
        # overlay tab swaps the stack to the panel; the others swap back.
        self._viewer_stack = QStackedWidget()
        self._viewer_stack.addWidget(self.viewer)            # index 0
        self._spatial_panel: Optional[SpatialMapsPanel] = None
        self._serialtrack_panel = None  # type: Optional[Any]
        self._dvc_panel = None  # type: Optional[Any]
        # DVC field series keyed by multipoint → {frame(t): DVCResult} (+ per-frame
        # display backdrops), so the DVC tab plays through frames like the viewer.
        self._dvc_series_by_m: Dict[int, Dict[int, Any]] = {}     # primary (cumulative)
        self._dvc_incr_by_m: Dict[int, Dict[int, Any]] = {}       # raw increments
        self._dvc_bg_by_mt: Dict[int, Dict[int, Any]] = {}
        # Registration (V1.56): per-multipoint result bundle keyed by M →
        # {channel, raw_ref, aligned {ch:(T,H,W)}, shifts, confidence, ...}.
        self._registration_panel = None  # type: Optional[Any]
        self._reg_by_m: Dict[int, Dict[str, Any]] = {}
        vc.addWidget(self._viewer_stack, stretch=1)

        self._results_table_panel = QWidget()
        tp = QVBoxLayout(self._results_table_panel)
        tp.setContentsMargins(0, 0, 0, 0)
        tp.setSpacing(4)
        self._results_summary = QLabel("No measurements yet — preview a Results node.")
        self._results_summary.setObjectName("resultsSummary")
        self._results_summary.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        tp.addWidget(self._results_summary)
        self._results_table = QTableView()
        self._results_table.setObjectName("resultsTable")
        self._results_table.setSortingEnabled(True)
        self._results_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._results_table.setAlternatingRowColors(False)
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.verticalHeader().setVisible(False)
        tp.addWidget(self._results_table, stretch=1)

        # Bottom data panel: a tab widget holding the Measurements table plus the
        # plot tabs generated by a Run (cells/frame, histograms, track stats).
        self._data_tabs = QTabWidget()
        self._data_tabs.setObjectName("pipelineDataTabs")
        self._data_tabs.addTab(self._results_table_panel, "Measurements")
        # Maximize button (top-right corner) pops the whole data panel — every
        # plot/measurements tab — out into a full window, preserving tab state.
        self._btn_plots_popout = icon_button(
            "fa5s.expand", "Maximize the plots / data panel in a separate window",
            object_name="pipelineToolBtn", icon_px=14,
        )
        self._btn_plots_popout.clicked.connect(lambda: self._toggle_popout("plots"))
        self._data_tabs.setCornerWidget(
            self._btn_plots_popout, Qt.Corner.TopRightCorner)
        # Plot canvases keyed by tab name, (re)built lazily by the Run handlers.
        self._plot_canvases: Dict[str, Any] = {}
        # Live pop-out windows, keyed "viewer" / "plots" (None when docked).
        self._popout_windows: Dict[str, Optional[PopOutWindow]] = {
            "viewer": None, "plots": None,
        }
        self._data_tabs.setVisible(False)

        self._right_split = QSplitter(Qt.Orientation.Vertical)
        self._right_split.setChildrenCollapsible(False)
        self._right_split.addWidget(self._viewer_container)
        self._right_split.addWidget(self._data_tabs)
        self._right_split.setStretchFactor(0, 1)
        self._right_split.setStretchFactor(1, 1)
        col.addWidget(self._right_split, stretch=1)
        return container

    # ── Maximize / pop-out (V1.46.3) ─────────────────────────────────────────
    def _toggle_popout(self, kind: str) -> None:
        """Pop the viewer or plots panel out into a full window, or dock it back.

        The live widget is reparented (not rebuilt), so its full state — overlay
        tabs, sliders, channel LUTs, zoom/pan, plot tabs — is preserved on both
        the way out and the way back.
        """
        if kind == "viewer":
            widget, index, title, btn = (
                self._viewer_container, 0, "Image Viewer",
                self._btn_viewer_popout,
            )
        else:
            widget, index, title, btn = (
                self._data_tabs, 1, "Plots & Data",
                self._btn_plots_popout,
            )

        existing = self._popout_windows.get(kind)
        if existing is not None:
            # Already popped out — dock it back (also triggers _restore below).
            existing.restore()
            return

        def _restore(w: QWidget, _idx: int = index, _kind: str = kind,
                     _btn: QPushButton = btn) -> None:
            self._popout_windows[_kind] = None
            self._right_split.insertWidget(_idx, w)
            w.setVisible(True)
            self._right_split.setStretchFactor(0, 1)
            self._right_split.setStretchFactor(1, 1)
            # Give both panes real estate so the restored panel is never
            # collapsed to 0 px (which would look like it vanished).
            try:
                h = max(self._right_split.height(), 2)
                self._right_split.setSizes([h // 2, h // 2])
            except Exception:  # noqa: BLE001
                pass
            _btn.setIcon(make_icon("fa5s.expand"))
            _btn.setToolTip(
                "Maximize the image viewer in a separate window" if _kind == "viewer"
                else "Maximize the plots / data panel in a separate window"
            )
            # Docked again → the Spatial Maps panel uses its compact top-bar layout.
            if _kind == "viewer" and self._spatial_panel is not None:
                self._spatial_panel.set_compact(True)
            # Re-apply the docked visibility for the current sub-tab / mode.
            self._update_merged_view_mode()

        widget.setParent(None)
        win = PopOutWindow(
            widget, title=title, on_restore=_restore, parent=self.window(),
        )
        self._popout_windows[kind] = win
        btn.setIcon(make_icon("fa5s.compress"))
        btn.setToolTip("Restore this panel back into the main window")
        # Maximized → the Spatial Maps panel shows its full Cell-Tracker sidebar.
        if kind == "viewer" and self._spatial_panel is not None:
            self._spatial_panel.set_compact(False)
        win.showMaximized()

    def _set_panel_visible(self, kind: str, visible: bool) -> None:
        """Show/hide a docked panel. A panel that is currently popped out owns
        its own window, so leave its visibility alone."""
        if self._popout_windows.get(kind) is not None:
            return
        widget = self._viewer_container if kind == "viewer" else self._data_tabs
        widget.setVisible(visible)

    def _build_subtab_selector(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("subTabBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._subtab_group = QButtonGroup(self)
        self._subtab_group.setExclusive(True)
        self._subtab_btns: Dict[Stage, QPushButton] = {}
        for stage in self._stages:
            btn = QPushButton(_STAGE_LABEL[stage])
            btn.setCheckable(True)
            btn.setObjectName("subTabBtn")
            accent = _STAGE_ACCENT[stage]
            btn.setStyleSheet(
                "QPushButton#subTabBtn{padding:6px 14px;border:none;"
                f"border-bottom:2px solid transparent;color:{Settings.FG_SECONDARY};"
                "background:transparent;font:bold 10pt;}"
                "QPushButton#subTabBtn:checked{"
                f"color:{accent};border-bottom:2px solid {accent};}}"
                f"QPushButton#subTabBtn:hover{{color:{accent};}}"
            )
            btn.clicked.connect(lambda _c=False, s=stage: self._select_stage(s))
            self._subtab_group.addButton(btn)
            self._subtab_btns[stage] = btn
            layout.addWidget(btn)
        layout.addStretch(1)
        return bar

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("pipelineControlBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._btn_add = icon_button(
            "fa5s.plus", "Add a node (pick from the catalog)", text=" Add",
            object_name="pipelineToolBtn",
        )
        self._btn_add.clicked.connect(self._on_add_node)
        layout.addWidget(self._btn_add)

        self._btn_preview = icon_button(
            "fa5s.eye", "Toggle live preview", text=" Preview",
            checkable=True, object_name="pipelineToolBtn",
        )
        self._btn_preview.setChecked(True)
        bind_toggle_icon(self._btn_preview, "fa5s.eye-slash", "fa5s.eye",
                         color=Settings.FG_SECONDARY, color_checked=Settings.ACCENT_CYAN)
        self._btn_preview.toggled.connect(self._on_preview_toggled)
        layout.addWidget(self._btn_preview)

        # Scissors (V1.48): arm to cut wires — drag across any wire (channel or
        # standard bridging) to sever it, or click a single wire.
        self._btn_scissors = icon_button(
            "fa5s.cut",
            "Cut wires — drag across any wire (channel or standard) to cut it, "
            "or click a wire. Toggle off to resume editing.",
            text=" Cut", checkable=True, object_name="pipelineToolBtn",
        )
        self._btn_scissors.toggled.connect(self._on_scissors_toggled)
        layout.addWidget(self._btn_scissors)

        # Loop connector (V1.49): arm, then drag from a node's bottom output to
        # the top input of the same or an upstream node to add an iteration loop.
        self._btn_loop = icon_button(
            "fa5s.redo",
            "Loop connector — drag from a node's bottom back to the top of the "
            "same or an upstream node to iterate that region (parameter sweep / "
            "count / until a condition). Toggle off to resume editing.",
            text=" Loop", checkable=True, object_name="pipelineToolBtn",
        )
        self._btn_loop.toggled.connect(self._on_loop_mode_toggled)
        self._btn_loop.setVisible(False)  # Analysis tab only (shown in _select_stage)
        layout.addWidget(self._btn_loop)

        self._btn_apply = icon_button(
            "fa5s.check", "Commit this stage's result", text=" Apply",
            object_name="pipelineToolBtn",
        )
        self._btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self._btn_apply)

        # Run (merged Analysis tab): execute the graph — shaded → gold → normal,
        # branching at if-else nodes. Replaces Apply on that tab.
        self._btn_run = icon_button(
            "fa5s.play", "Run the pipeline (executes the graph)", text=" Run",
            object_name="pipelineToolBtn",
        )
        self._btn_run.clicked.connect(self._on_run_button)
        self._btn_run.setVisible(False)
        layout.addWidget(self._btn_run)

        self._btn_undo = icon_button(
            "fa5s.undo", "Undo (coming soon)", text=" Undo",
            object_name="pipelineToolBtn",
        )
        self._btn_undo.setEnabled(False)
        layout.addWidget(self._btn_undo)

        # Preview progress — sits next to the action buttons and animates while
        # the pipeline is applied to the image in the preview viewer.
        self._preview_progress = QProgressBar()
        self._preview_progress.setObjectName("pipelinePreviewProgress")
        self._preview_progress.setRange(0, 100)
        self._preview_progress.setValue(0)
        self._preview_progress.setTextVisible(False)
        self._preview_progress.setFixedWidth(scaled(110))
        self._preview_progress.setFixedHeight(scaled(10))
        self._preview_progress.setToolTip("Applying pipeline to the preview…")
        self._preview_progress.setVisible(False)
        layout.addWidget(self._preview_progress)

        self._chk_normalize = QCheckBox("Normalize")
        self._chk_normalize.setToolTip(
            "Frame-mean normalize channels before the recipe."
        )
        self._chk_normalize.toggled.connect(self._on_normalize_toggled)
        layout.addWidget(self._chk_normalize)

        layout.addStretch(1)

        self._btn_save = icon_button("fa5s.save", "Save pipeline (.nd2s_pipeline.json)",
                                     object_name="pipelineToolBtn")
        self._btn_save.clicked.connect(self._on_save)
        layout.addWidget(self._btn_save)
        self._btn_load = icon_button("fa5s.folder-open", "Load pipeline",
                                     object_name="pipelineToolBtn")
        self._btn_load.clicked.connect(self._on_load)
        layout.addWidget(self._btn_load)

        self._hint = QLabel("Add nodes via the Add button or right-click • "
                            "double-click a node to preview it • double-click a "
                            "wire to disconnect")
        self._hint.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        layout.addWidget(self._hint)
        return bar

    def _build_scene(self, stage: Stage) -> NodeScene:
        scene = NodeScene(self._doc.slice_for(stage), _STAGE_ACCENT[stage])
        if stage is Stage.PROCESSING:
            scene.action_specs = enhancement_specs()
            scene.output_spec = processing_output_spec()
            scene.on_output_created = self._on_output_node_created
        elif stage is Stage.ANALYSIS:
            # Merged tab: analysis pipelines + results ops + if-else + specials,
            # each colored by category. Output node still bridges to Export.
            scene.action_specs = merged_action_specs()
            scene.output_spec = analysis_output_spec()
            scene.on_output_created = self._on_analysis_output_created
        else:
            scene.allow_add = False
        scene.selection_changed.connect(self._on_scene_selection)
        scene.node_double_clicked.connect(self._on_node_double_clicked)
        scene.graph_changed.connect(self._on_graph_changed)
        scene.node_renamed.connect(self._on_node_renamed)
        scene.loop_edge_edit_requested.connect(self._on_loop_edge_edit)  # V1.49
        return scene

    # ── sub-tab switching ──────────────────────────────────────────────────
    def _select_stage(self, stage: Stage) -> None:
        # Graph is authoritative (V1.46): leaving Processing re-derives the
        # committed recipe from the node graph, so an empty / disconnected graph
        # clears any stale recipe before the Analysis base image reads it.
        if self._stage is Stage.PROCESSING and stage is not Stage.PROCESSING:
            self._sync_committed_recipe_from_graph()
            # Stop a queued / in-flight processing preview so its (async) result
            # can't overwrite the Analysis base image after the switch.
            self._preview_debounce.stop()
            if self._runner is not None:
                self._runner.cancel(_PREVIEW_KEY)
        self._stage = stage
        # The processing preview volume is per-visit; rebuild it (via set_volume)
        # on the next preview rather than updating a stale instance in place.
        self._proc_volume = None
        self._view.setScene(self._scenes[stage])
        self._subtab_btns[stage].setChecked(True)
        accent = _STAGE_ACCENT[stage]
        self._view.setStyleSheet(
            f"#nodeBoardView{{border:1px solid {accent};border-radius:6px;"
            f"background:{Settings.BG_PRIMARY};}}"
        )
        deferred = stage not in self._enabled_stages
        self._coming_soon.setVisible(deferred)
        if deferred:
            self._coming_soon.setGeometry(self._view.rect())
        self._popup.hide()
        # Apply commits a recipe on Processing; the merged Analysis tab uses Run.
        is_proc = stage is Stage.PROCESSING
        self._btn_apply.setVisible(is_proc)
        self._btn_apply.setEnabled(is_proc and not deferred)
        self._btn_run.setVisible(not is_proc and not deferred)
        self._btn_run.setEnabled(not is_proc and not deferred and not self._run_active)
        # Loop connector lives on the merged Analysis tab (where the graph runs).
        self._btn_loop.setVisible(stage is Stage.ANALYSIS and not deferred)
        self._btn_add.setEnabled(not deferred and self._scenes[stage].allow_add)
        self._set_preview_progress(visible=False)

        # Overlay hook + Results table: on the merged tab these depend on the
        # previewed node's category (analysis overlay vs results overlay+table);
        # Processing shows a plain processed image.
        self._update_merged_view_mode()

        if not deferred:
            self._ensure_input_node(stage)
            self._update_preview_highlight()
            if stage is Stage.ANALYSIS:
                self._show_base_image()
            self._request_preview()
        # Center on the whole graph for this sub-tab (deferred so the view has
        # its final size after the scene swap / layout).
        QTimer.singleShot(0, self._frame_all_nodes)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        if self._coming_soon.isVisible():
            self._coming_soon.setGeometry(self._view.rect())

    # ── merged-tab mode (analysis overlay vs results overlay + table) ────────
    def _merged_mode(self) -> str:
        """On the merged Analysis tab, the active sub-mode is driven by the
        previewed node's category: results / logic (if-else) / special nodes show
        the measurements table + overlay (``"results"``); an analysis node shows
        the mask overlay only (``"analysis"``). Processing → ``"processing"``."""
        if self._stage is not Stage.ANALYSIS:
            return "processing"
        node = self._doc.analysis.nodes.get(self._preview_target(Stage.ANALYSIS))
        cat = getattr(node, "category", None) if node is not None else None
        if cat in (NodeCategory.RESULTS, NodeCategory.LOGIC, NodeCategory.SPECIAL):
            return "results"
        return "analysis"

    def _update_merged_view_mode(self) -> None:
        """Set the viewer overlay hook + Results table visibility for the current
        sub-tab / previewed-node category."""
        if self._stage is Stage.PROCESSING:
            self.viewer.set_frame_post_process(None)
            self.viewer.invalidate_post_process_cache()
            self._results_mode_bar.setVisible(False)
            self._overlay_tabbar.setVisible(False)
            self._set_panel_visible("viewer", True)
            self._set_panel_visible("plots", False)
            return
        # Analysis (merged) stage — the overlay tabs + data/plots tabs apply here.
        self.viewer.set_frame_post_process(self._composite_pipeline_overlay)
        self._overlay_tabbar.setVisible(True)
        self._update_overlay_tabs_available()
        if self._merged_mode() == "results":
            self._results_mode_bar.setVisible(True)
            self._set_results_view_mode(self._results_view_mode)
        else:
            self._results_mode_bar.setVisible(False)
            self._set_panel_visible("viewer", True)
            self._set_panel_visible("plots", True)
        self.viewer.invalidate_post_process_cache()

    # ── selection / popup ───────────────────────────────────────────────────
    def _on_scene_selection(self, node_id: str) -> None:
        # A single click selects a node and opens its parameter pop-up (ACTION
        # nodes only). The previewed node is *sticky* — it changes only on a
        # double-click — so selecting another node, or clicking off the canvas,
        # never disturbs which result the viewer shows or which node is golden.
        self._selected_node_id = node_id
        if node_id:
            node = self._current_slice().nodes.get(node_id)
            if node is not None and node.role is NodeRole.ACTION:
                self._open_popup_for(node_id)
            else:
                self._popup.hide()
        else:
            self._popup.hide()

    def _open_popup_for(self, node_id: str) -> None:
        node = self._current_slice().nodes.get(node_id)
        if node is None:
            return
        specs = param_specs_for(node.op_key)
        # Inject the live channel names into 'choice' params (channel_name /
        # counterstain_channel / intensity_channel), exactly as the Analysis page
        # does on activation, so the user can pick a channel. Harmless for nodes
        # without such params, so it runs regardless of stage (the CellTracker
        # Metrics / Field Maps special nodes are RESULTS-stage but need it).
        self._inject_channel_choices(specs, self._current_channel_names())
        if node.stage is Stage.RESULTS:
            self._inject_result_choices(specs, self._get_analysis_result_names())
        item = self._scenes[self._stage].node_item(node_id)
        if item is not None:
            scene_pt = item.mapToScene(item.boundingRect().topRight())
            view_pt = self._view.mapFromScene(scene_pt)
            global_pt = self._view.viewport().mapToGlobal(view_pt)
        else:
            global_pt = self.mapToGlobal(self.rect().center())
        # Nodes with a richer editor get an "Edit…" button in the popup.
        edit_label = None
        if node.op_key == IF_ELSE_OP_KEY:
            edit_label = "Edit branch condition…"
        elif (node.op_key.startswith(RESULTS_PREFIX)
              and results_op_name_for_op_key(node.op_key) == "Compute Measurements"):
            edit_label = "Select measurements…"
        elif node.op_key in (SPECIAL_VALIDATE_OP_KEY, SPECIAL_REVIEW_OP_KEY):
            edit_label = "Review objects…"
        elif node.op_key in (SPECIAL_CT_FIELDS_OP_KEY, SPECIAL_INTERP_MAP_OP_KEY):
            edit_label = "Spatial map templates…"
        elif node.op_key == SPECIAL_REGISTER_OP_KEY:
            edit_label = "Pick ROI…"
        self._popup.show_for(node.title, specs, node.params, global_pt,
                             edit_label=edit_label)

    def _on_popup_params_changed(self, values: Dict[str, Any]) -> None:
        if not self._selected_node_id:
            return
        node = self._current_slice().nodes.get(self._selected_node_id)
        if node is None:
            return
        node.params = dict(values)
        # Refresh only when the edited node feeds the previewed (golden) node —
        # i.e. it is the golden node itself or one of the nodes before it.
        # Editing a downstream / unrelated node can't change the preview.
        target = self._preview_target(self._stage)
        if not target:
            self._request_preview()
            return
        up_nodes, _ = self._upstream_subgraph(self._stage, target)
        if self._selected_node_id in up_nodes:
            self._request_preview()

    def _on_node_double_clicked(self, node_id: str) -> None:
        # Double-click promotes a node to the previewed node: it (and its
        # upstream lineage) take the golden outline and its result drives the
        # viewer. Works for every node — results / logic / special nodes preview
        # the upstream analysis result (table + overlay); editing a node's
        # condition / metrics is via the param popup's "Edit…" button instead.
        if self._stage not in self._enabled_stages:
            return
        node = self._current_slice().nodes.get(node_id)
        if node is None:
            return
        # In the Analysis tab a double-click is a fresh start: cancel any preview
        # analysis still computing and reset its scratch before previewing the
        # newly-chosen node, so results never mix or land stale.
        if self._stage is Stage.ANALYSIS:
            self._reset_analysis_preview()
        self._preview_node_ids[self._stage] = node_id
        self._update_preview_highlight()
        # Promoting a results/logic/special node switches the merged tab into its
        # results overlay + measurements-table mode (and back for analysis).
        self._update_merged_view_mode()
        if self._stage is Stage.ANALYSIS:
            # Re-assert the (cropped) base image so the freshly-previewed node's
            # overlay lands on a current base — the reset cleared the old overlay.
            self._show_base_image()
            # Only a double-click regenerates the analysis preview (V1.49).
            self._analysis_preview_armed = True
        # A double-click is the *deliberate* trigger: the next preview walk runs
        # interactively (shaded steps + pop-ups for Validate/Review).
        self._pv_interactive_next = True
        self._request_preview()

    def _on_popup_edit(self) -> None:
        """The param popup's "Edit…" button — open the node's editor dialog."""
        node = self._current_slice().nodes.get(self._selected_node_id)
        if node is None:
            return
        if node.op_key == IF_ELSE_OP_KEY:
            self._edit_if_else_condition(node)
        elif node.op_key in (SPECIAL_VALIDATE_OP_KEY, SPECIAL_REVIEW_OP_KEY):
            self._preview_validate(node)
        elif node.op_key in (SPECIAL_CT_FIELDS_OP_KEY, SPECIAL_INTERP_MAP_OP_KEY):
            self._edit_spatial_templates(node)
        elif node.op_key == SPECIAL_REGISTER_OP_KEY:
            self._edit_registration_roi(node)
        elif node.op_key.startswith(RESULTS_PREFIX):
            self._edit_measurement_metrics(node)

    def _edit_spatial_templates(self, node) -> None:
        """Pick which saved templates a Spatial Maps node carries (and import
        external JSON template files into the library)."""
        from nd2studios.widgets.spatial_maps_panel import SpatialTemplatePicker
        names = list(node.params.get("templates") or [])
        dlg = SpatialTemplatePicker(names, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        node.params["templates"] = dlg.get_names()
        # Keep the popup editor's hidden value in sync (it round-trips params).
        self._popup.editor.set_values({"templates": node.params["templates"]})
        item = self._scenes[self._stage].node_item(node.id)
        if item is not None:
            item.setToolTip(
                f"Templates: {', '.join(node.params['templates']) or '(none)'}")
        # If the panel is live, re-apply so the change is reflected immediately.
        if (self._spatial_panel is not None
                and self._viewer_stack.currentWidget() is self._spatial_panel):
            self._spatial_panel.set_node_templates(node.params["templates"])

    # ── Registration ROI picker (V1.60) ────────────────────────────────────
    def _edit_registration_roi(self, node) -> None:
        """Pick the region the Registration transform is *estimated* on (whole
        frame, a rectangle, or a freeform shape drawn on the viewer); it is always
        *applied* full-frame. Restricting the estimate to a static landmark stops
        moving cells / artifacts from corrupting whole-frame correlation. Stored as
        a serializable spec in ``node.params['roi']`` (a hidden param)."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QComboBox, QSpinBox, QLabel,
            QDialogButtonBox, QWidget,
        )
        record = self._active_record()
        vol = getattr(record, "_raw_volume", None) if record is not None else None
        H = int(getattr(vol, "height", 0) or getattr(record, "height", 0) or 0)
        W = int(getattr(vol, "width", 0) or getattr(record, "width", 0) or 0)
        if H <= 0 or W <= 0:
            H = W = 100000  # engine clamps the box to the real frame anyway
        cur = node.params.get("roi") or None
        cur_kind = cur.get("kind") if isinstance(cur, dict) else None

        dlg = QDialog(self)
        dlg.setWindowTitle("Registration ROI")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "Estimate the alignment on this region; the transform is applied to the\n"
            "whole frame. Pick a static landmark (bead / substrate feature) so moving\n"
            "cells or artifacts don't corrupt the estimate."))
        cmb = QComboBox()
        cmb.addItems(["Whole frame", "Rectangle", "Freeform (draw on image)"])
        lay.addWidget(cmb)
        form = QFormLayout()
        sx, sy, sw, sh = (QSpinBox() for _ in range(4))
        for s, hi in ((sx, W), (sy, H), (sw, W), (sh, H)):
            s.setRange(0, max(1, int(hi)))
        if cur_kind == "rect":
            sx.setValue(int(cur.get("x", 0))); sy.setValue(int(cur.get("y", 0)))
            sw.setValue(int(cur.get("w", 0))); sh.setValue(int(cur.get("h", 0)))
            cmb.setCurrentIndex(1)
        else:
            sx.setValue(W // 4); sy.setValue(H // 4)
            sw.setValue(max(1, W // 2)); sh.setValue(max(1, H // 2))
            cmb.setCurrentIndex(0 if cur_kind is None else 2)
        for lbl, s in (("x", sx), ("y", sy), ("width", sw), ("height", sh)):
            form.addRow(lbl, s)
        rect_box = QWidget(); rect_box.setLayout(form)
        lay.addWidget(rect_box)
        rect_box.setVisible(cmb.currentIndex() == 1)
        cmb.currentIndexChanged.connect(
            lambda _i: rect_box.setVisible(cmb.currentIndex() == 1))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mode = cmb.currentIndex()
        if mode == 1:
            w, h = int(sw.value()), int(sh.value())
            roi = ({"kind": "rect", "x": int(sx.value()), "y": int(sy.value()),
                    "w": w, "h": h} if w > 0 and h > 0 else None)
            self._set_registration_roi(node, roi)
        elif mode == 2:
            # Arm the viewer to draw a shape; captured in _on_registration_shape.
            self._roi_capture_node = node.id
            if not getattr(self, "_roi_shape_connected", False):
                try:
                    self.viewer.shape_drawn.connect(self._on_registration_shape)
                    self._roi_shape_connected = True
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.viewer.set_draw_mode("polygon")
            except Exception:  # noqa: BLE001
                pass
            self._set_status("Registration ROI: draw a shape on the image "
                             "(close the polygon to finish).")
        else:
            self._set_registration_roi(node, None)

    def _on_registration_shape(self, mode: str, verts) -> None:
        """Capture a freeform ROI shape drawn on the viewer for the armed node."""
        node_id = getattr(self, "_roi_capture_node", None)
        if not node_id:
            return  # a shape drawn for some other purpose — ignore
        self._roi_capture_node = None
        try:
            self.viewer.set_draw_mode(None)
        except Exception:  # noqa: BLE001
            pass
        node = self._current_slice().nodes.get(node_id)
        if node is None:
            return
        pts = [[float(v[0]), float(v[1])] for v in (verts or [])]  # (row, col)
        if len(pts) < 3:
            self._set_status("Registration ROI: shape needs ≥3 points — unchanged.")
            return
        self._set_registration_roi(
            node, {"kind": "shapes",
                   "shapes": [{"type": str(mode) or "polygon", "vertices": pts}]})

    def _set_registration_roi(self, node, roi) -> None:
        node.params["roi"] = roi
        try:  # keep the popup's hidden slot in sync (it round-trips params)
            self._popup.editor.set_values({"roi": roi})
        except Exception:  # noqa: BLE001
            pass
        summary = self._registration_roi_summary(roi)
        item = self._scenes[self._stage].node_item(node.id)
        if item is not None:
            item.setToolTip(f"Registration ROI: {summary}")
        self._set_status(f"Registration ROI set: {summary}")

    @staticmethod
    def _registration_roi_summary(roi) -> str:
        if not roi:
            return "whole frame"
        if roi.get("kind") == "rect":
            return (f"rectangle x{roi['x']} y{roi['y']} "
                    f"{roi['w']}×{roi['h']} px")
        if roi.get("kind") == "shapes":
            return f"{len(roi.get('shapes', []))} freeform shape(s)"
        return "whole frame"

    def _preview_validate(self, node) -> None:
        """Review objects in preview mode, on the current M's screened T planes.
        Honors the node's ``mode`` (single-object panels vs whole-frame viewer)
        and the upstream tracking — tracked rows review as tracks, untracked rows
        as single objects (no Track Objects node ⇒ per-object). Applies
        accept/reject back onto the preview's measurement rows."""
        record = self._active_record()
        m, _, _ = self.viewer.coords()
        planes = sorted(
            (mt for mt in self._analysis_screen_results if mt[0] == m),
            key=lambda mt: mt[1])
        if record is None or not planes:
            self._set_status(
                "Preview the analysis first — no screened objects on this M.")
            return
        ts = [t for (_m, t) in planes]
        first = self._analysis_screen_results[planes[0]]
        seg_ch = next(iter(getattr(first, "label_masks", {}) or {}), "")
        if not seg_ch:
            self._set_status("Nothing to review.")
            return
        try:
            mask_stack = np.stack([
                np.asarray(self._analysis_screen_results[(m, t)].label_masks[seg_ch])[0]
                for t in ts])
            names = self._current_channel_names()
            channels = {}
            for ch in names:
                frames = [self._extract_processed_frame(record, ch, m, t) for t in ts]
                if all(f is not None for f in frames):
                    channels[ch] = np.stack([f[0] for f in frames])
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Review failed to assemble frames: {exc}")
            return
        # Rows for this M, copied + re-framed to the stack index; '_orig' links
        # each copy back to the real row (track_id is preserved from the walk).
        rows = []
        for r in self._results_rows:
            if int(r.get("m_position", -1)) == m and int(r.get("frame", -1)) in ts:
                rc = dict(r)
                rc["_orig"] = r
                rc["frame"] = ts.index(int(r.get("frame")))
                rows.append(rc)
        if not rows:
            self._set_status("No measured objects on this M to review.")
            return
        mode = (node.params or {}).get("mode", "Single objects")
        cd = getattr(record, "channel_display", {}) or {}
        try:
            if mode == "Whole frame":
                from nd2studios.widgets.whole_frame_review_dialog import (
                    WholeFrameReviewDialog,
                )
                dlg = WholeFrameReviewDialog(
                    {m: rows}, {m: {seg_ch: mask_stack}},
                    lambda mm: channels, cd, parent=self)
            else:
                from nd2studios.widgets.track_validation_dialog import (
                    TrackValidationDialog,
                )
                dlg = TrackValidationDialog(
                    rows, {seg_ch: mask_stack}, channels, cd, parent=self,
                    include_untracked=True,
                    overlay_style=self._overlay_style())
            dlg.exec()
            drop = {id(rc["_orig"]) for rc in dlg.rejected_rows}
            accept = {id(rc["_orig"]) for rc in dlg.accepted_rows}
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Review failed: {exc}")
            return
        kept = []
        for r in self._results_rows:
            if id(r) in drop:
                continue
            if id(r) in accept:
                r["track_validation"] = "accepted"
            kept.append(r)
        self._results_rows = kept
        self._populate_results_table(kept)
        self._set_status(
            f"Reviewed M{m + 1} over {len(ts)} frame(s): "
            f"{len(accept)} accepted, {len(drop)} rejected.")

    def _edit_measurement_metrics(self, node) -> None:
        """Open the metric picker for a Compute Measurements node."""
        current = node.params.get("metrics")
        sel = set(current) if current is not None else None
        dlg = MeasurementSelectDialog(sel, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        node.params["metrics"] = sorted(dlg.selected())
        self._set_status(f"Measurements: {len(node.params['metrics'])} metric(s) selected.")
        # If this node is in the previewed chain, recompute the table.
        target = self._preview_target(self._stage)
        up_nodes, _ = self._upstream_subgraph(self._stage, target) if target else (set(), set())
        if node.id == target or node.id in up_nodes:
            self._request_preview()

    def _edit_if_else_condition(self, node) -> None:
        """Open the condition builder for an if-else node and store the result."""
        dlg = ConditionBuilderDialog(
            node.params.get("condition"), self._current_channel_names(),
            metrics=self._available_metric_columns(node),
            lens=node.params.get("lens"), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cond = dlg.result_condition()
        node.params["condition"] = cond.to_dict()
        item = self._scenes[self._stage].node_item(node.id)
        if item is not None:
            item.setToolTip(f"If TRUE: {describe_condition(cond)}")
        self._set_status(f"Branch condition: {describe_condition(cond)}")

    # Row keys that identify / index an object rather than measure it — excluded
    # from the metric-comparison dropdown.
    _NON_METRIC_KEYS = frozenset({
        "frame", "m_position", "label_id", "track_id", "segmentation_channel",
    })

    def _available_metric_columns(self, node) -> List[str]:
        """Metric columns the if-else can branch on: every measurement column
        already present in the computed rows, plus the tracking / Cell-Tracker
        columns contributed by any upstream Track Objects / Cell-Tracker Metrics
        node (offered even before a Run has produced rows)."""
        cols: List[str] = []
        seen: set = set()

        def _add(c: str) -> None:
            if c and c not in seen and c not in self._NON_METRIC_KEYS:
                cols.append(c)
                seen.add(c)

        for r in (self._results_rows or []):
            for k in r.keys():
                _add(str(k))
        if self._has_upstream_celltracker(node):
            from nd2studios.backend.celltracker_bridge import METRIC_COLUMNS
            _add("track_length")
            for c in METRIC_COLUMNS:
                _add(c)
        return cols

    def _has_upstream_celltracker(self, node) -> bool:
        """True when a Track Objects or Cell-Tracker Metrics node lies anywhere
        upstream of ``node`` in the analysis slice (walks incoming edges)."""
        from nd2studios.pipeline_graph.registry_adapter import (
            SPECIAL_TRACK_OP_KEY, SPECIAL_CT_METRICS_OP_KEY,
        )
        targets = {SPECIAL_TRACK_OP_KEY, SPECIAL_CT_METRICS_OP_KEY}
        sl = self._doc.analysis
        seen: set = set()
        stack = [node.id]
        while stack:
            nid = stack.pop()
            for e in sl.edges.values():
                if e.dst_node == nid and e.src_node not in seen:
                    seen.add(e.src_node)
                    up = sl.nodes.get(e.src_node)
                    if up is not None and up.op_key in targets:
                        return True
                    stack.append(e.src_node)
        return False

    # ── add node (catalog dialog) ────────────────────────────────────────────
    def _on_add_node(self) -> None:
        scene = self._scenes[self._stage]
        if not scene.allow_add:
            return
        specs = list(scene.action_specs)
        if scene.output_spec is not None:
            specs.append(scene.output_spec)
        if not specs:
            return
        dlg = AddNodeDialog(specs, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        spec = dlg.selected_spec()
        if spec is None:
            return
        scene.add_node_from_spec(spec, self._new_node_pos())

    def _new_node_pos(self) -> tuple:
        """Drop point for a newly added node: the center of the visible canvas,
        nudged by a small per-add cascade so successive adds don't stack."""
        center = self._view.mapToScene(self._view.viewport().rect().center())
        step = (self._bridge_counter + len(self._current_slice().nodes)) % 6
        offset = step * 26
        return (center.x() - 86 + offset, center.y() - 22 + offset)

    # ── graph change / output bridge ─────────────────────────────────────────
    def _on_graph_changed(self) -> None:
        self._clear_preview_shading()
        self._update_preview_highlight()
        # V1.53: drop frozen checkpoint data for nodes that no longer exist (a
        # deleted checkpoint), so its masks/rows are released. Surviving
        # checkpoints stay valid unless their upstream hash changes (checked at Run).
        if self._checkpoint_store:
            live = set(self._doc.analysis.nodes)
            for nid in [k for k in self._checkpoint_store if k not in live]:
                self._checkpoint_store.pop(nid, None)
        # V1.48: recolor channel wires / propagation strands after any wiring
        # change (channel connect/disconnect, node add/delete).
        scene = self._scenes.get(self._stage)
        if scene is not None:
            scene.refresh_channel_visuals()
        self._request_preview()

    def _clear_preview_shading(self) -> None:
        """Drop any lingering preview-walk shading (not during an active Run)."""
        if self._run_active:
            return
        self._pv_states = {}
        scene = self._scenes.get(Stage.ANALYSIS)
        if scene is not None:
            scene.clear_run_states()

    def _reset_analysis_preview(self) -> None:
        """Cancel any in-flight preview analysis and drop its scratch state so a
        freshly-previewed node starts from a clean slate.

        Called on a double-click in the Analysis tab: a new preview should not
        land on top of, or mix with, an analysis that was already computing. A
        full Run owns the runner + viewer, so this is a no-op while running."""
        if self._run_active:
            return
        # Cancel the preview-analysis jobs this page owns (leave other pages'
        # jobs and a full Run untouched).
        if self._runner is not None:
            for key in (_ANALYSIS_PREVIEW_KEY, _RESULTS_PREVIEW_KEY,
                        _PV_SCREEN_KEY, _ANALYSIS_COMMIT_KEY):
                self._runner.cancel(key)
        # Stop a queued (debounced) preview for the previous target from firing.
        self._preview_debounce.stop()
        self._pv_walking = False
        # Drop preview scratch: per-plane overlay results, measurement rows,
        # walk shading, plots, progress. (Committed Apply / Run results survive.)
        self._analysis_screen_results = {}
        self._results_rows = []
        # Drop track-overlay scratch too, so the Tracks / Vectors tabs don't
        # linger with stale data from the previously-previewed node.
        self._track_colormap = None
        self._track_overlay_rows = []
        self._populate_results_table([])
        self._clear_preview_shading()
        self._update_preview_plots()
        self._set_preview_progress(visible=False)
        self.viewer.invalidate_post_process_cache()

    def _on_output_node_created(self, node) -> None:
        self._bridge_counter += 1
        node.title = f"Processing #{self._bridge_counter}"
        from nd2studios.pipeline_graph.model import Bridge, PortType, new_id
        bridge = Bridge(
            id=new_id("bridge"),
            producer_stage=Stage.PROCESSING,
            consumer_stage=Stage.ANALYSIS,
            name=node.title,
            payload_type=PortType.IMAGE,
        )
        self._doc.bridges[bridge.id] = bridge
        node.bridge_id = bridge.id

    def _on_analysis_output_created(self, node) -> None:
        self._analysis_bridge_counter += 1
        node.title = f"Analysis #{self._analysis_bridge_counter}"
        from nd2studios.pipeline_graph.model import Bridge, PortType, new_id
        bridge = Bridge(
            id=new_id("bridge"),
            producer_stage=Stage.ANALYSIS,
            consumer_stage=Stage.RESULTS,
            name=node.title,
            payload_type=PortType.BINARY,
        )
        self._doc.bridges[bridge.id] = bridge
        node.bridge_id = bridge.id

    def _on_node_renamed(self, node_id: str, title: str) -> None:
        """Keep an output node's bridge name in sync with its (edited) title, so
        the rename is reflected wherever the bridge is referenced (Analysis /
        Export source labels). The node may live in any slice, so look it up by
        id (ids are unique across the document)."""
        node = None
        for sl in (self._doc.processing, self._doc.analysis, self._doc.results):
            if node_id in sl.nodes:
                node = sl.nodes[node_id]
                break
        if node is None:
            return
        if node.bridge_id and node.bridge_id in self._doc.bridges:
            self._doc.bridges[node.bridge_id].name = title

    # ── golden preview highlight ──────────────────────────────────────────────
    def _update_preview_highlight(self) -> None:
        """Gold the previewed node and the pipeline *upstream* of it.

        Only the nodes/bridges that feed into the previewed node — the chain that
        actually produces what the viewer shows — are highlighted. Nodes and
        wires downstream of it stay un-gilded.
        """
        scene = self._scenes.get(self._stage)
        if scene is None:
            return
        if self._stage not in self._enabled_stages:
            scene.set_highlight(set(), set(), "")
            return
        target = self._preview_target(self._stage)
        if target:
            node_ids, edge_ids = self._upstream_subgraph(self._stage, target)
        else:
            node_ids, edge_ids = set(), set()
        scene.set_highlight(node_ids, edge_ids, target)

    def _upstream_subgraph(self, stage: Stage, target_id: str):
        """All nodes + edges upstream of ``target_id`` (walking incoming edges
        back toward the input), plus ``target_id`` itself. Downstream nodes/wires
        are excluded."""
        sl = self._doc.slice_for(stage)
        node_ids: set = set()
        edge_ids: set = set()
        if target_id not in sl.nodes:
            return node_ids, edge_ids
        stack = [target_id]
        while stack:
            cur = stack.pop()
            if cur in node_ids:
                continue
            node_ids.add(cur)
            for e in sl.incoming(cur):  # edges feeding cur — go to predecessors
                edge_ids.add(e.id)
                if e.src_node not in node_ids:
                    stack.append(e.src_node)
        return node_ids, edge_ids

    # ── preview ──────────────────────────────────────────────────────────────
    def _on_preview_toggled(self, on: bool) -> None:
        if on:
            if self._stage is Stage.ANALYSIS:
                self._show_base_image()
            self._request_preview()
        else:
            self._set_preview_progress(visible=False)

    def _on_normalize_toggled(self, on: bool) -> None:
        self._normalized = bool(on)
        self._request_preview()

    def _on_scissors_toggled(self, on: bool) -> None:
        """Arm the wire cutter across all scenes; swap the view to a cut cursor
        (and off rubber-band select) so a drag cuts instead of selecting."""
        for scene in self._scenes.values():
            scene.set_cut_mode(on)
        if on:
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self._view.viewport().unsetCursor()

    def _on_loop_mode_toggled(self, on: bool) -> None:
        """Arm loop-wire mode on the Analysis scene; a drag then makes a loop
        back-edge instead of a structural wire. Mutually exclusive with cut."""
        if on and self._btn_scissors.isChecked():
            self._btn_scissors.setChecked(False)
        for scene in self._scenes.values():
            scene.set_loop_mode(on)
        if on:
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self._view.viewport().unsetCursor()

    def _loop_body_param_options(self, region) -> list:
        """Numeric ParamSpec options for the nodes in a loop region, for the Loop
        Settings dialog's sweep-axis pickers (V1.49)."""
        sl = self._doc.analysis
        options = []
        for nid in region.body:
            node = sl.nodes.get(nid)
            if node is None:
                continue
            params = []
            for spec in param_specs_for(node.op_key):
                if getattr(spec, "param_type", "") in ("float", "int"):
                    params.append({
                        "name": spec.name,
                        "label": spec.label or spec.name,
                        "min": getattr(spec, "min_val", None),
                        "max": getattr(spec, "max_val", None),
                        "step": getattr(spec, "step", None),
                        "default": spec.default,
                    })
            if params:
                options.append({"node_id": nid, "title": node.title,
                                "params": params})
        return options

    def _on_loop_edge_edit(self, edge_id: str) -> None:
        """Open the Loop Settings dialog for a loop edge (V1.49)."""
        from nd2studios.pipeline_graph.loop import loop_region
        from nd2studios.widgets.node_board.loop_dialog import LoopSettingsDialog
        sl = self._doc.analysis
        edge = sl.edges.get(edge_id)
        if edge is None or edge.kind != LOOP_KIND:
            return
        region = loop_region(sl, edge)
        entry = sl.nodes.get(region.entry)
        exit_ = sl.nodes.get(region.exit)
        region_label = (f"{exit_.title if exit_ else '?'} → "
                        f"{entry.title if entry else '?'} "
                        f"({len(region.body)} node(s))")
        options = self._loop_body_param_options(region)
        channels = list(self._channel_display_names()) \
            if hasattr(self, "_channel_display_names") else []
        metrics = self._upstream_metric_columns(region.exit) \
            if hasattr(self, "_upstream_metric_columns") else []
        dlg = LoopSettingsDialog(
            edge.params, options, region_label=region_label,
            channels=channels, metrics=metrics, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            edge.params = dlg.result_config()
            self._on_graph_changed()

    def _on_viewer_coords(self, m: int, t: int, z: int) -> None:
        # The base is a lazy on-demand volume, so the viewer navigates M/T/Z by
        # itself. We only re-run the pipeline when the *set of planes we should be
        # previewing* actually changes — so browsing frames that are already
        # processed (e.g. arrowing through a selection) does NOT recompute.
        if self._stage not in self._enabled_stages:
            return
        if not self._btn_preview.isChecked():
            return
        self._preview_m = m
        desired = set(self._selected_planes(m, t))
        if self._stage is Stage.PROCESSING:
            if desired != set(self._processing_planes):
                self._request_preview()
        elif self._stage is Stage.ANALYSIS:
            # Both analysis- and results-mode previews screen the selected planes
            # into ``_analysis_screen_results``; re-run whenever that set changes
            # (so navigating / selecting frames updates the overlay AND the table).
            if desired != set(self._analysis_screen_results.keys()):
                self._request_preview()

    def _on_viewer_selection(self, axis: str, sel) -> None:
        # A multi-frame selection on the M or T strip re-runs the preview so every
        # selected (M, T) plane goes through the pipeline at once (Processing) /
        # is screened (Analysis). Z is projected, so it doesn't change the set.
        if axis not in ("m", "t"):
            return
        if self._stage not in self._enabled_stages:
            return
        if not self._btn_preview.isChecked():
            return
        self._request_preview()

    def _request_preview(self) -> None:
        if self._run_active:
            return  # a Run owns the viewer/runner — don't fight it with previews
        if self._stage not in self._enabled_stages:
            return
        if not self._btn_preview.isChecked():
            return
        self._preview_debounce.start()

    def _do_preview(self) -> None:
        if self._runner is None or self._pv_walking:
            return
        if self._stage is Stage.PROCESSING:
            self._do_processing_preview()
        elif self._stage is Stage.ANALYSIS:
            # V1.49: the Analysis preview only (re)generates on a node double-click.
            # Navigation / param edits / selection / crop still call _request_preview
            # (to refresh the base image etc.) but must NOT recompute the scoped Run.
            if not self._analysis_preview_armed:
                return
            self._analysis_preview_armed = False
            interactive = self._pv_interactive_next
            self._pv_interactive_next = False
            # A double-click preview is a mini "Run": segment → measure → track →
            # spatial → plots on the selected frames, up to (and including) the
            # double-clicked node — driven by the same walk the Run button uses.
            target = self._resolve_preview_run_target()
            if target:
                self._start_preview_walk(target, interactive)

    def _preview_target(self, stage: Stage) -> str:
        """Node to preview: the sticky previewed node, else the primary output,
        else the first action node, else the input."""
        sl = self._doc.slice_for(stage)
        pid = self._preview_node_ids.get(stage, "")
        if pid and pid in sl.nodes:
            return pid
        outs = output_nodes(sl)
        if outs:
            return outs[0].id
        for node in sl.nodes.values():
            if node.role is NodeRole.ACTION:
                return node.id
        inp = input_node(sl)
        return inp.id if inp is not None else ""

    def _resolve_preview_run_target(self) -> str:
        """The node a double-click preview runs *up to* (a scoped Run, V1.49).

        Normally the double-clicked (sticky) node, so the scoped Run covers only
        the chain feeding it — clicking StarDist previews segmentation; clicking
        Track Objects adds tracks/vectors; clicking the final node runs it all.

        If the clicked node has no analysis feeding it (e.g. the raw input, or a
        disconnected node), there is nothing to run "up to" — fall back to the
        whole downstream chain (the last output, else the last analysis action) so
        double-clicking the input still runs the pipeline."""
        sl = self._doc.analysis
        target = self._preview_target(Stage.ANALYSIS)
        if target and self._upstream_analysis_node(target) is not None:
            return target
        outs = output_nodes(sl)
        if outs:
            return outs[-1].id
        for nid in reversed(topological_order(sl)):
            n = sl.nodes.get(nid)
            if (n is not None and n.role is NodeRole.ACTION
                    and n.op_key.startswith("analysis:")):
                return nid
        return target

    # ── single-frame on-demand preview (shared) ────────────────────────────────
    def _show_preview_volume(self, volume) -> None:
        """Display a lazy, on-demand ``volume`` at the current coords.

        The viewer reads only the *displayed* plane via ``volume.get_frame``, so
        the recipe is applied to **one frame at a time** (no full-stack
        materialization) while M/T/Z navigation stays fully live. Re-asserts the
        active stage's overlay hook on top.
        """
        record = self._active_record()
        cd = record.channel_display if record is not None else {}
        m, t, z = self.viewer.coords()
        self._preview_m = m
        # set_volume reconfigures the strips and CLEARS the tile selection; save
        # the user's M/T/Z selection and restore it after, so a multi-frame
        # selection survives the volume swap (and keeps driving the preview).
        saved_sel = {ax: set(self.viewer.axis_selection(ax))
                     for ax in ("m", "t", "z")}
        self.viewer.set_volume(
            volume,
            channel_display=cd,
            z_mode=getattr(record, "z_view_mode", None) or "max",
            z_index=int(getattr(record, "z_view_index", 0) or 0),
            m=m, t=t, z=z,
            stage_xy_um=(getattr(record, "nd2_metadata", {}) or {}).get("stage_xy_um"),
        )
        for ax, sel in saved_sel.items():
            if sel:
                self.viewer.set_axis_selection(ax, sel)
        if self._stage is Stage.ANALYSIS:
            self.viewer.set_frame_post_process(self._composite_analysis_overlay)
        elif self._stage is Stage.RESULTS:
            self.viewer.set_frame_post_process(self._composite_results_overlay)
        else:
            self.viewer.set_frame_post_process(None)
        self.viewer.invalidate_post_process_cache()
        self._set_preview_progress(visible=False)

    # ── Processing preview ────────────────────────────────────────────────────
    def _do_processing_preview(self) -> None:
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        if getattr(self.main_window, "heavy_ops_blocked", lambda: False)():
            self._set_status("Preview paused — low memory")
            return
        target = self._preview_target(Stage.PROCESSING)
        if not target:
            return
        try:
            recipe = recipe_for_node(self._doc.processing, target)
        except ValueError:
            # Node not connected back to the input — nothing to show yet.
            return
        # V1.48: with channel wiring, preview per-channel (unwired channels stay
        # raw) up to the previewed node's chain.
        rbc = None
        if has_channel_wiring(self._doc.processing):
            names = list((record._raw_channels or {}).keys())
            try:
                rbc = channel_recipes(self._doc.processing, target, names)
            except Exception:  # noqa: BLE001
                rbc = None
        # Preview the plane(s) the user is on: the current frame, or — when a
        # multi-frame range is selected on the frame strip — every selected
        # frame. Read just those planes' raw channels (cheap), process them in a
        # background job (progress bar + recipe off the render thread). The result
        # is shown only on those planes (see _on_runner_done); every other frame
        # stays raw. The set follows the user — re-runs on navigation / selection.
        m, t, _ = self.viewer.coords()
        self._preview_m = m
        planes = self._selected_planes(m, t)
        self._processing_planes = planes
        frames = self._read_planes_frames(record, planes)
        if not frames:
            return
        job = _ProcessingPreviewJob(
            _PREVIEW_KEY, frames, recipe, self._normalized, recipe_by_channel=rbc,
        )
        self._set_preview_progress(visible=True, value=0)
        self._runner.submit(job)

    def _selected_planes(self, m: int, t: int) -> List[tuple]:
        """``(M, T)`` planes to preview: the cartesian product of the frame-strip
        **M** and **T** selections (each defaulting to the current position when
        empty), so selecting M positions *and* T frames previews every
        combination. With no selection at all → just the current ``(m, t)``."""
        try:
            msel = sorted(int(x) for x in self.viewer.axis_selection("m"))
        except Exception:  # noqa: BLE001
            msel = []
        try:
            tsel = sorted(int(x) for x in self.viewer.axis_selection("t"))
        except Exception:  # noqa: BLE001
            tsel = []
        ms = msel or [int(m)]
        ts = tsel or [int(t)]
        return [(mm, tt) for mm in ms for tt in ts]

    # ── Preview crop (V1.46) ────────────────────────────────────────────────
    def _crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """The active crop ``(x, y, w, h)`` in raw-image pixels, or None.

        Composes two rectangles (both in original-frame coords): the **preview
        crop** (only on cropped Runs / outside a Run) and the **registration
        common-region crop** (V1.60 — always active once a Registration node with
        "Crop to common region" has published it, even on a full Run, so every
        downstream consumer sees the border-free registered image). When both are
        set the effective crop is their intersection."""
        preview = (None if (self._run_active and not self._run_cropped)
                   else self._preview_crop)
        return self._intersect_crop_rects(preview, self._registration_crop_rect())

    def _registration_crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """The published registration common-region crop as ``(x, y, w, h)``, or
        None. Stored on the record as ``(y0, y1, x0, x1)`` by ``_finish_register``."""
        rec = self._active_record()
        c = getattr(rec, "_registration_crop", None) if rec is not None else None
        if not c:
            return None
        y0, y1, x0, x1 = (int(v) for v in c)
        if y1 <= y0 or x1 <= x0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    @staticmethod
    def _intersect_crop_rects(a, b):
        """Intersect two ``(x, y, w, h)`` rects (either may be None). If they don't
        overlap, fall back to ``b`` (the registration crop) — the meaningful region."""
        if a is None:
            return b
        if b is None:
            return a
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x1 <= x0 or y1 <= y0:
            return b
        return (x0, y0, x1 - x0, y1 - y0)

    def _crop_frame(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Slice a frame's XY to the preview crop if one is active.

        Accepts ``(H, W)``, ``(1, H, W)`` or ``(T, H, W)`` and crops the last two
        axes, returning the same rank. Pass-through when no crop is active."""
        rect = self._crop_rect()
        if rect is None or arr is None:
            return arr
        x, y, w, h = rect
        return np.asarray(arr)[..., y:y + h, x:x + w]

    def _maybe_crop_volume(self, vol):
        """Wrap ``vol`` in a :class:`CroppedVolume` when a preview crop is active,
        so the displayed base image is the cropped region."""
        rect = self._crop_rect()
        if rect is None or vol is None:
            return vol
        return CroppedVolume(vol, rect)

    def _maybe_register_volume(self, record, vol):
        """Wrap ``vol`` in a :class:`RegisteredFrameVolume` when a Registration node
        has published per-M transforms for ``record`` — so the displayed base image
        (and any crop taken from it) is drift-corrected. Pass-through otherwise, so
        graphs without a Registration node are unaffected. Wrap *before* the crop so
        the crop is a region of the registered image (V1.59)."""
        by_m = getattr(record, "_registration_by_m", None)
        if not isinstance(by_m, dict) or not by_m or vol is None:
            return vol
        order = int(getattr(record, "_registration_interp_order", 1) or 1)
        return RegisteredFrameVolume(vol, by_m, interp_order=order)

    def _register_frame(self, record, frame, m: int, t: int):
        """Apply the stored ``(m, t)`` registration transform to a single
        ``(1,H,W)`` / ``(H,W)`` frame (no-op when no Registration node ran for this
        M, or the transform is identity). Used so a paused, cropped *preview* runs
        on the drift-corrected image — registration applied to the full frame here,
        the crop taken by the caller afterward (register → crop) (V1.59)."""
        if frame is None:
            return frame
        by_m = getattr(record, "_registration_by_m", None)
        tf = by_m.get(int(m)) if isinstance(by_m, dict) else None
        if not tf:
            return frame
        order = int(getattr(record, "_registration_interp_order", 1) or 1)
        from nd2studios.backend.registration import estimate as _est
        arr = np.asarray(frame)
        try:
            if arr.ndim == 3:  # (1, H, W)
                return _est.apply_frame(arr[0], tf, int(t), interp_order=order)[None, ...]
            return _est.apply_frame(arr, tf, int(t), interp_order=order)
        except Exception:  # noqa: BLE001 — never break the preview read path
            return frame

    def _crop_channel_for_run(self, arr) -> np.ndarray:
        """Crop a whole-stack channel to the active crop for a cropped Run.

        Slices the last two axes. Tries the source's own slicing first (a numpy
        view, or a lazy proxy that materializes only the crop) and falls back to
        materialize-then-crop, so a huge lazy source doesn't have to fully
        materialize when it supports slicing."""
        rect = self._crop_rect()
        if rect is None:
            return np.asarray(arr)
        x, y, w, h = rect
        try:
            return np.asarray(arr[..., y:y + h, x:x + w])
        except Exception:  # noqa: BLE001 — source doesn't support fancy slicing
            return np.asarray(arr)[..., y:y + h, x:x + w]

    def _preview_metadata(self, record) -> Dict[str, Any]:
        """Metadata for a preview job — pixel size plus, when a crop is active,
        ``height`` / ``width`` overridden to the crop dims so field-based analysis
        (Vectors: field, Spatial Maps) sizes to the cropped frame."""
        md = dict(record.nd2_metadata)
        md["pixel_size_um"] = record.pixel_size_um
        rect = self._crop_rect()
        if rect is not None:
            _, _, w, h = rect
            md["height"] = int(h)
            md["width"] = int(w)
        return md

    def _on_preview_crop_toggled(self, on: bool) -> None:
        """Toggle the crop tool. On → arm the rubber-band selection; off → clear
        any active crop and restore the full-frame preview."""
        if on:
            self._crop_selecting = True
            # With Preview off the viewer may have no image to draw on — show the
            # navigable base so the user can rubber-band a crop regardless.
            if (not self._btn_preview.isChecked()
                    and getattr(self.viewer, "_volume", None) is None):
                self._show_base_image(force=True)
            self.viewer.set_crop_mode(True)
            self._set_status(
                "Preview crop: drag a rectangle on the image, or click a point to "
                "enter the region manually.")
        else:
            self._crop_selecting = False
            self.viewer.set_crop_mode(False)
            had_crop = self._preview_crop is not None
            if had_crop:
                self._push_crop_history(self._preview_crop)
            self._preview_crop = None
            self._lbl_preview_crop.setText("")
            if had_crop:
                self._on_crop_changed()

    def _push_crop_history(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Record a prior crop state so ↩ can restore it."""
        self._crop_history.append(rect)
        if len(self._crop_history) > 50:
            self._crop_history.pop(0)
        self._update_crop_undo_enabled()

    def _update_crop_undo_enabled(self) -> None:
        btn = getattr(self, "_btn_crop_undo", None)
        if btn is not None:
            btn.setEnabled(bool(self._crop_history))

    def _on_crop_undo(self) -> None:
        """Restore the previous crop state (or no-crop) from the undo stack."""
        if not self._crop_history:
            return
        prev = self._crop_history.pop()
        self._update_crop_undo_enabled()
        self._apply_crop_state(prev)

    def _apply_crop_state(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Set the preview crop to ``rect`` (or None) and sync the UI + preview.

        Used by undo — does not itself push history (the caller manages the
        stack)."""
        self._crop_selecting = False
        self.viewer.set_crop_mode(False)
        self._preview_crop = tuple(rect) if rect is not None else None
        if self._preview_crop is not None:
            cx, cy, cw, ch = self._preview_crop
            self._lbl_preview_crop.setText(f"Crop {cw}×{ch} @({cx},{cy})")
        else:
            self._lbl_preview_crop.setText("")
        # Reflect the active-crop state in the toggle button without recursing
        # into its handler.
        self._btn_preview_crop.blockSignals(True)
        self._btn_preview_crop.setChecked(self._preview_crop is not None)
        self._btn_preview_crop.blockSignals(False)
        self._on_crop_changed()

    def _on_preview_crop_drag(self, x: int, y: int, w: int, h: int) -> None:
        if not self._crop_selecting:
            return
        self._prompt_and_apply_crop(int(x), int(y), int(w), int(h))

    def _on_preview_crop_click(self, iy: float, ix: float) -> None:
        # Only a click made while the crop tool is armed enters a crop corner;
        # after a crop is applied the tool disarms so clicks navigate normally.
        if not self._crop_selecting:
            return
        self._prompt_and_apply_crop(int(ix), int(iy), 0, 0)

    def _prompt_and_apply_crop(self, x: int, y: int, w: int, h: int) -> None:
        """Confirm/edit the rectangle in a dialog and apply it as the preview
        crop. Disarms the tool afterwards so the image stays navigable."""
        rect = self._show_preview_crop_dialog(x, y, w, h)
        self._crop_selecting = False
        self.viewer.set_crop_mode(False)
        if rect is None:
            # Cancelled — keep any existing crop; if there is none, disarm the
            # button so its checked state reflects reality.
            if self._preview_crop is None:
                self._btn_preview_crop.blockSignals(True)
                self._btn_preview_crop.setChecked(False)
                self._btn_preview_crop.blockSignals(False)
            return
        # Record the prior state (crop or none) so ↩ can restore it.
        if rect != self._preview_crop:
            self._push_crop_history(self._preview_crop)
        self._preview_crop = rect
        cx, cy, cw, ch = rect
        self._lbl_preview_crop.setText(f"Crop {cw}×{ch} @({cx},{cy})")
        # Keep the button checked to signal an active crop.
        if not self._btn_preview_crop.isChecked():
            self._btn_preview_crop.blockSignals(True)
            self._btn_preview_crop.setChecked(True)
            self._btn_preview_crop.blockSignals(False)
        self._on_crop_changed()

    def _show_preview_crop_dialog(
        self, x: int, y: int, w: int, h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Dialog to confirm/edit the crop rectangle, with a live preview of the
        cropped region and a jog pad (▲▼◀▶ + step) to nudge the whole region.
        Returns ``(x, y, w, h)`` in raw-image pixels, or None if cancelled."""
        record = self._active_record()
        if record is None:
            return None
        shape = self._raw_frame_shape(record)
        if shape is None:
            return None
        img_h, img_w = shape
        full_rgb = self._composite_full_frame_rgb(record)  # (H, W, 3) uint8 or None
        preview_px = scaled(240)

        dlg = QDialog(self)
        dlg.setWindowTitle("Preview Crop")
        outer = QVBoxLayout(dlg)
        info = QLabel(f"Image: {img_w} × {img_h} px  "
                      f"(X = columns from left, Y = rows from top)")
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        outer.addWidget(info)

        body = QHBoxLayout()
        outer.addLayout(body)

        # ── Left: numeric fields + jog pad ──
        left = QVBoxLayout()
        body.addLayout(left)

        form = QFormLayout()
        sp_x = QSpinBox(); sp_x.setRange(0, max(0, img_w - 1))
        sp_x.setValue(max(0, min(x, img_w - 1)))
        sp_x.setToolTip("Left edge of crop (pixels from image left)")
        sp_y = QSpinBox(); sp_y.setRange(0, max(0, img_h - 1))
        sp_y.setValue(max(0, min(y, img_h - 1)))
        sp_y.setToolTip("Top edge of crop (pixels from image top)")
        sp_w = QSpinBox(); sp_w.setRange(1, img_w)
        sp_w.setValue(w if w > 0 else max(1, img_w - int(sp_x.value())))
        sp_w.setToolTip("Width of crop in pixels")
        sp_h = QSpinBox(); sp_h.setRange(1, img_h)
        sp_h.setValue(h if h > 0 else max(1, img_h - int(sp_y.value())))
        sp_h.setToolTip("Height of crop in pixels")
        form.addRow("X (left corner):", sp_x)
        form.addRow("Y (top corner):", sp_y)
        form.addRow("Width (px):", sp_w)
        form.addRow("Height (px):", sp_h)
        left.addLayout(form)

        # Jog pad — a symmetric 4-way cross of arrow buttons around a central
        # step field; each arrow shifts the whole region by ``step`` px (clamped
        # to the image). Equal-sized cells (min width/height on every row/column)
        # keep the cross symmetric regardless of the step field's width.
        jog_box = QGroupBox("Jog region")
        # Hug the content so the parent column can't stretch the grid and skew the
        # cross by rounding leftover width into one column.
        jog_box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        jog = QGridLayout(jog_box)
        jog.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        jog.setSpacing(scaled(4))
        btn_up = icon_button("fa5s.chevron-up", "Up — shift the region up by the step",
                             object_name="pipelineToolBtn", icon_px=12, button_px=30)
        btn_dn = icon_button("fa5s.chevron-down", "Down — shift the region down by the step",
                             object_name="pipelineToolBtn", icon_px=12, button_px=30)
        btn_lf = icon_button("fa5s.chevron-left", "Left — shift the region left by the step",
                             object_name="pipelineToolBtn", icon_px=12, button_px=30)
        btn_rt = icon_button("fa5s.chevron-right", "Right — shift the region right by the step",
                             object_name="pipelineToolBtn", icon_px=12, button_px=30)
        # Not auto-default, so Enter accepts the dialog rather than jogging.
        for _b in (btn_up, btn_dn, btn_lf, btn_rt):
            _b.setAutoDefault(False)
        sp_step = QSpinBox(); sp_step.setRange(1, max(1, max(img_w, img_h)))
        sp_step.setValue(10)
        # Wide enough that the value (up to 4 digits) is fully visible next to the
        # spin arrows — a too-narrow box hid the number.
        sp_step.setFixedSize(scaled(58), scaled(26))
        sp_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sp_step.setToolTip("Jog step (pixels) — how far each arrow shifts the region")
        # Fixed, equal cell sizes on every row/column (no stretch) so the cross is
        # exactly symmetric — stretch would distribute odd leftover width unevenly.
        # The center column is as wide as the step field so the value shows in full.
        cell_w, cell_h = scaled(62), scaled(32)
        for c in range(3):
            jog.setColumnMinimumWidth(c, cell_w)
        for r in range(3):
            jog.setRowMinimumHeight(r, cell_h)
        A = Qt.AlignmentFlag.AlignCenter
        jog.addWidget(btn_up, 0, 1, A)
        jog.addWidget(btn_lf, 1, 0, A)
        jog.addWidget(sp_step, 1, 1, A)
        jog.addWidget(btn_rt, 1, 2, A)
        jog.addWidget(btn_dn, 2, 1, A)
        left.addWidget(jog_box, 0, Qt.AlignmentFlag.AlignHCenter)
        left.addStretch(1)

        # ── Right: live preview of the cropped region ──
        right = QVBoxLayout()
        body.addLayout(right)
        right.addWidget(QLabel("Crop preview:"))
        preview = QLabel()
        preview.setFixedSize(preview_px, preview_px)
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setStyleSheet(
            f"background-color: {Settings.BG_SECONDARY}; border: 1px solid #555;")
        right.addWidget(preview)
        dims_lbl = QLabel("")
        dims_lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        dims_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(dims_lbl)
        right.addStretch(1)

        def _clamped() -> Tuple[int, int, int, int]:
            cx = max(0, min(int(sp_x.value()), img_w - 1))
            cy = max(0, min(int(sp_y.value()), img_h - 1))
            cw = max(1, min(int(sp_w.value()), img_w - cx))
            ch = max(1, min(int(sp_h.value()), img_h - cy))
            return cx, cy, cw, ch

        def _refresh_preview() -> None:
            cx, cy, cw, ch = _clamped()
            dims_lbl.setText(f"{cw} × {ch} px  @ ({cx}, {cy})")
            if full_rgb is None:
                preview.setText("(no image)")
                return
            sub = np.ascontiguousarray(full_rgb[cy:cy + ch, cx:cx + cw])
            if sub.size == 0:
                preview.clear()
                return
            sh, sw = sub.shape[:2]
            qimg = QImage(sub.data, sw, sh, sw * 3, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(
                preview_px, preview_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            preview.setPixmap(pm)

        def _jog(dx: int, dy: int) -> None:
            step = int(sp_step.value())
            cw = min(int(sp_w.value()), img_w)
            ch = min(int(sp_h.value()), img_h)
            nx = min(max(0, int(sp_x.value()) + dx * step), max(0, img_w - cw))
            ny = min(max(0, int(sp_y.value()) + dy * step), max(0, img_h - ch))
            sp_x.setValue(nx)
            sp_y.setValue(ny)

        for sp in (sp_x, sp_y, sp_w, sp_h):
            sp.valueChanged.connect(lambda _v: _refresh_preview())
        btn_up.clicked.connect(lambda: _jog(0, -1))
        btn_dn.clicked.connect(lambda: _jog(0, 1))
        btn_lf.clicked.connect(lambda: _jog(-1, 0))
        btn_rt.clicked.connect(lambda: _jog(1, 0))
        _refresh_preview()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        outer.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        cx, cy, cw, ch = _clamped()
        if cw < 1 or ch < 1:
            return None
        return (int(cx), int(cy), int(cw), int(ch))

    def _composite_full_frame_rgb(self, record) -> Optional[np.ndarray]:
        """An ``(H, W, 3)`` uint8 RGB of the current (m, t, z) raw frame,
        composited across enabled channels — the full (un-cropped) frame the crop
        dialog previews from. None if no frame is available."""
        try:
            from nd2studios.widgets.image_viewer import (
                CHANNEL_COLORS, composite_channels,
            )
        except Exception:  # noqa: BLE001
            return None
        m, t, _ = self.viewer.coords()
        z_mode = getattr(record, "z_view_mode", None) or "max"
        z_index = int(getattr(record, "z_view_index", 0) or 0)
        vol = getattr(record, "_raw_volume", None)
        frames: Dict[str, np.ndarray] = {}
        if vol is not None:
            for c, name in enumerate(getattr(vol, "channel_names", [])):
                try:
                    frames[name] = np.asarray(
                        vol.get_frame(c=c, m=m, t=t, z=z_index, z_mode=z_mode))
                except Exception:  # noqa: BLE001
                    continue
        else:
            for name, arr in (getattr(record, "_raw_channels", None) or {}).items():
                a = arr.materialize() if hasattr(arr, "materialize") else np.asarray(arr)
                frames[name] = (a[min(int(t), a.shape[0] - 1)] if a.ndim == 3
                                else np.asarray(a))
        if not frames:
            return None
        cd = getattr(record, "channel_display", {}) or {}
        colors: Dict[str, Tuple[int, int, int]] = {}
        enabled: Dict[str, bool] = {}
        for name in frames:
            disp = cd.get(name, {})
            cname = str(disp.get("color", "") or "").lower()
            colors[name] = CHANNEL_COLORS.get(cname, (255, 255, 255))
            enabled[name] = bool(disp.get("enabled", True))
        if not any(enabled.values()):
            enabled = {k: True for k in frames}
        try:
            return composite_channels(frames, colors, enabled, auto_contrast=True)
        except Exception:  # noqa: BLE001
            return None

    def _raw_frame_shape(self, record) -> Optional[Tuple[int, int]]:
        """(H, W) of the raw (un-cropped) frames for ``record``, or None."""
        vol = getattr(record, "_raw_volume", None)
        if vol is not None:
            h = int(getattr(vol, "height", 0) or 0)
            w = int(getattr(vol, "width", 0) or 0)
            if h > 0 and w > 0:
                return (h, w)
        for arr in (getattr(record, "_raw_channels", None) or {}).values():
            shp = getattr(arr, "shape", None) or np.asarray(arr).shape
            if len(shp) >= 2:
                return (int(shp[-2]), int(shp[-1]))
        meta = getattr(record, "nd2_metadata", {}) or {}
        h = int(meta.get("height", 0) or 0)
        w = int(meta.get("width", 0) or 0)
        return (h, w) if h > 0 and w > 0 else None

    def _on_crop_changed(self) -> None:
        """The preview crop was set or cleared — rebuild the processed-plane
        volume against the new geometry and re-render the base + preview.

        Works whether or not live Preview is on: with Preview on it re-runs the
        pipeline preview; with Preview off it still refreshes the displayed base
        image so the crop visibly takes effect (or clears)."""
        self._proc_volume = None
        self._analysis_screen_results = {}
        self._update_run_button()
        if self._btn_preview.isChecked():
            if self._stage is Stage.ANALYSIS:
                self._show_base_image()
            self._request_preview()
        else:
            # Preview compute is off — still update what the viewer shows so
            # enabling/disabling the crop is visible.
            self._show_base_image(force=True)

    def _read_planes_frames(self, record, planes: List[tuple]) -> Dict[tuple, Dict[str, Any]]:
        """Raw ``{(m, t): {channel: (1, H, W)}}`` for ``planes``, read on the GUI
        thread (one frame per channel per plane — cheap per plane). Sliced to the
        preview crop when one is active."""
        z_mode = getattr(record, "z_view_mode", None) or "max"
        z_index = int(getattr(record, "z_view_index", 0) or 0)
        vol = getattr(record, "_raw_volume", None)
        out: Dict[tuple, Dict[str, Any]] = {}
        for (m, t) in planes:
            chans: Dict[str, Any] = {}
            if vol is not None:
                for c, name in enumerate(getattr(vol, "channel_names", [])):
                    try:
                        f = vol.get_frame(c=c, m=m, t=t, z=z_index, z_mode=z_mode)
                        chans[name] = self._crop_frame(np.asarray(f)[None, ...])
                    except Exception:  # noqa: BLE001
                        continue
            else:
                for name, arr in (record._raw_channels or {}).items():
                    a = (arr.materialize() if hasattr(arr, "materialize")
                         else np.asarray(arr))
                    frame = (a[[min(t, a.shape[0] - 1)]] if a.ndim == 3
                             else a[None, ...])
                    chans[name] = self._crop_frame(frame)
            if chans:
                out[(int(m), int(t))] = chans
        return out

    # ── Analysis / Results base image ───────────────────────────────────────────
    def _show_base_image(self, force: bool = False) -> None:
        """Show the navigable base image the analysis/results overlay sits on.

        When a Processing recipe has been committed (``record.recipe``), the base
        is the **processed** data — applied per displayed frame on demand via
        :class:`ProcessedFrameVolume` (so e.g. a Background Subtract shows in the
        Analysis sub-tab, not raw) — otherwise the raw volume. Either way the
        whole stack stays navigable and nothing is materialized up front.

        ``force`` bypasses the Preview-toggle gate so a crop change can update the
        displayed (cropped) base image even when live Preview is off.
        """
        if not force and not self._btn_preview.isChecked():
            return
        record = self._active_record()
        if record is None:
            return
        recipe = list(getattr(record, "recipe", []) or [])
        rbc = getattr(record, "recipe_by_channel", None)
        normalized = bool(getattr(record, "recipe_normalized", False))
        vol = getattr(record, "_raw_volume", None)
        if vol is not None:
            base = (ProcessedFrameVolume(vol, recipe, normalized,
                                         recipe_by_channel=rbc)
                    if (recipe or rbc) else vol)
            # V1.59: keep the base drift-corrected once a Registration node has run,
            # so arming the crop tool doesn't revert the viewer to the un-registered
            # image. Register first, then crop → the crop is a region of the
            # registered image (matching what the user sees).
            base = self._maybe_register_volume(record, base)
            # Preview crop: show only the cropped sub-region (analysis, overlays
            # and measurements all run in this same crop space).
            base = self._maybe_crop_volume(base)
            self._show_preview_volume(base)
            return
        # Fallback: in-RAM channels (small files).
        if not record._raw_channels:
            return
        m, _, _ = self.viewer.coords()
        self._preview_m = m
        # V1.59: after a Registration write-back the aligned channels live in
        # ``_processed_channels`` — prefer them so the in-RAM base is drift-corrected.
        by_m = getattr(record, "_registration_by_m", None)
        src_channels = record._raw_channels
        if (isinstance(by_m, dict) and by_m
                and getattr(record, "_processed_channels", None)):
            src_channels = record._processed_channels
        base_channels = src_channels
        rect = self._crop_rect()
        if rect is not None:
            base_channels = {name: self._crop_frame(arr)
                             for name, arr in src_channels.items()}
        self.viewer.set_channels(
            base_channels, channel_display=record.channel_display,
            n_multipoints=self._record_n_multipoints(record), m=m,
        )
        if self._stage is Stage.ANALYSIS:
            self.viewer.set_frame_post_process(self._composite_analysis_overlay)
        elif self._stage is Stage.RESULTS:
            self.viewer.set_frame_post_process(self._composite_results_overlay)
        self.viewer.invalidate_post_process_cache()

    def _do_analysis_preview(self) -> None:
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        if getattr(self.main_window, "heavy_ops_blocked", lambda: False)():
            self._set_status("Preview paused — low memory")
            return
        self._clear_preview_shading()  # analysis-node preview: no walk shading
        action = self._resolve_analysis_action(self._preview_target(Stage.ANALYSIS))
        if action is None:
            return
        name = analysis_pipeline_name_for_op_key(action.op_key)
        cls = AnalysisPipeline.get_pipeline(name)
        if cls is None:
            return
        names = self._current_channel_names()
        if not names:
            return
        # V1.48: preview on the channel(s) wired into this node (rainbow ports),
        # replacing the channel_name param (2nd wired channel = counterstain).
        extra, seg_channels = self._analysis_run_spec(action, names)
        if not seg_channels:
            return
        params = dict(action.params)
        params.update(extra)
        m, t, _ = self.viewer.coords()
        # Screen the current frame, or every frame of a multi-frame selection —
        # one processed frame per wired channel per plane.
        planes = self._selected_planes(m, t)
        frames_by_mt: Dict[tuple, Dict[str, Any]] = {}
        for (pm, pt) in planes:
            chan_frames: Dict[str, Any] = {}
            for ch in seg_channels:
                frame = self._extract_processed_frame(record, ch, pm, pt)
                if frame is not None:
                    chan_frames[ch] = self._crop_frame(frame)
            if chan_frames:
                frames_by_mt[(int(pm), int(pt))] = chan_frames
        if not frames_by_mt:
            return
        metadata = self._preview_metadata(record)
        job = _AnalysisPreviewJob(
            _ANALYSIS_PREVIEW_KEY, cls, seg_channels, frames_by_mt, metadata, params,
        )
        self._set_preview_progress(visible=True, value=0)
        self._runner.submit(job)

    def _analysis_channels_for(self, node, names: List[str]) -> List[str]:
        """Ordered channel(s) an analysis ``node`` runs on (V1.48/49).

        Priority:
        1. **Explicit Analysis-slice wiring** — the node's effective channel set
           (channels wired into its rainbow ports + propagated).
        2. **Channels that were processed** in the Processing slice
           (``record.recipe_by_channel``) — so channel-specific *processing* flows
           through to the analysis even when the analysis node isn't separately
           wired (this is what "channel-specific processing goes through to the
           analysis" means; otherwise the analysis would default to the first
           channel, which may be a *raw* one that was never processed).
        3. The old ``channel_name`` param / the first channel — legacy graphs
           behave exactly as before."""
        names = list(names or [])
        if not names:
            return []
        sl = self._doc.analysis
        if has_channel_wiring(sl) and node is not None:
            chset = channel_sets(sl, names).get(node.id) or set()
            seg = [c for c in names if c in chset]
            if seg:
                return seg
        # No explicit analysis wiring → prefer the channel(s) processed upstream,
        # so a channel-specific Processing recipe carries into the analysis.
        record = self._active_record()
        rbc = getattr(record, "recipe_by_channel", None) if record is not None else None
        if rbc:
            proc = [c for c in names if c in rbc]
            if proc:
                return proc
        action = self._resolve_analysis_action(node.id) if node is not None else None
        ch = (action.params.get("channel_name") if action else "") or names[0]
        if ch not in names:
            ch = names[0]
        return [ch]

    def _op_counterstain(self, op_key: str) -> bool:
        """True if the analysis op supports a counterstain channel (tear
        detection). A 2nd wired channel is then used as the counterstain."""
        try:
            pcls = AnalysisPipeline.get_pipeline(
                analysis_pipeline_name_for_op_key(op_key))
            if pcls is None:
                return False
            return any(getattr(s, "name", "") == "counterstain_channel"
                       for s in pcls().get_params())
        except Exception:  # noqa: BLE001
            return False

    def _analysis_run_spec(self, action_node, names: List[str]):
        """``(extra_params, seg_channels)`` for running an analysis node on its
        wired channels (V1.48). For a counterstain-capable pipeline with ≥2 wired
        channels, the 2nd wired channel becomes the counterstain and only the
        primary is segmented; otherwise every wired channel is segmented in turn."""
        seg = self._analysis_channels_for(action_node, names)
        extra: Dict[str, Any] = {}
        op = getattr(action_node, "op_key", "") if action_node is not None else ""
        if op and len(seg) >= 2 and self._op_counterstain(op):
            extra["counterstain_channel"] = seg[1]
            seg = [seg[0]]
        return extra, seg

    def _resolve_analysis_action(self, node_id: str):
        """The analysis ACTION node for a preview target (resolve OUTPUT → its
        feeding action)."""
        sl = self._doc.analysis
        node = sl.nodes.get(node_id)
        if node is None:
            return None
        if node.role is NodeRole.ACTION:
            return node
        if node.role is NodeRole.OUTPUT:
            pred = predecessor(sl, node.id)
            if pred is not None and pred.role is NodeRole.ACTION:
                return pred
        return None

    def _extract_processed_frame(
        self, record, channel: str, m: int, t: int
    ) -> Optional[np.ndarray]:
        """A single processed ``(1, H, W)`` frame for ``channel`` at ``(m, t)``.

        Mirrors ``analysis_page._screen_current_frame``: read one raw frame from
        the lazy volume and apply the committed recipe to just that frame (so the
        preview matches the processed data the full run would use), or index a
        materialized channel directly.
        """
        z_mode = getattr(record, "z_view_mode", None) or "max"
        z_index = int(getattr(record, "z_view_index", 0) or 0)
        vol = getattr(record, "_raw_volume", None)
        if vol is not None and channel in getattr(vol, "channel_names", []):
            c_idx = vol.channel_names.index(channel)
            try:
                frame_2d = vol.get_frame(c_idx, m=m, t=t, z=z_index, z_mode=z_mode)
            except Exception:  # noqa: BLE001
                return None
            frame_arr = np.asarray(frame_2d)[np.newaxis]
            recipe = list(getattr(record, "recipe", []) or [])
            rbc = getattr(record, "recipe_by_channel", None)
            # Per-channel (V1.48): an unwired channel stays raw; a wired one gets
            # its own recipe. Legacy (rbc None) applies the single recipe.
            ch_recipe = (rbc.get(channel, []) if rbc is not None else recipe)
            if ch_recipe:
                out = apply_recipe({channel: frame_arr}, ch_recipe,
                                   bool(getattr(record, "recipe_normalized", False)))
                frame_arr = out.get(channel, frame_arr)
            # V1.59: a Registration node upstream publishes per-M drift transforms;
            # apply them to this freshly-recipe'd frame so a paused preview runs on
            # the aligned image (the ``_processed_channels`` fallback below is
            # already registered by the write-back, so it isn't re-registered).
            return self._register_frame(record, frame_arr, m, t)
        src = record._processed_channels or record._raw_channels or {}
        arr = src.get(channel)
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            if arr.ndim == 3:
                return arr[[min(t, arr.shape[0] - 1)]]
            return np.asarray(arr)[np.newaxis]
        try:
            return np.asarray(arr[t])[np.newaxis]
        except Exception:  # noqa: BLE001
            return None

    def _overlay_result_for(self, m: int, t: int):
        """Resolve the AnalysisResult + frame index whose masks overlay ``(m, t)``.

        Previewed planes (``_analysis_screen_results``, single-frame results keyed
        by ``(m, t)``) take priority — they reflect the latest interactive preview.
        Otherwise the full per-M Run result (``_run_results_by_m[m]``, a whole
        ``(T, H, W)`` stack) is used and indexed at frame ``t`` — so after a Run the
        overlay appears on **every** frame, not just the one shown when Run started.

        Returns ``(result, frame_index)`` or ``(None, 0)``.
        """
        res = self._analysis_screen_results.get((m, t))
        if res is not None:
            return res, 0
        # Run masks only overlay when their geometry matches the current display:
        # a cropped preview can't paint full-frame Run masks, and a full display
        # can't paint crop-sized Run masks. ``_run_results_crop`` is the geometry
        # the committed masks were computed at.
        if self._run_results_crop != self._crop_rect():
            return None, 0
        res = self._run_results_by_m.get(m)
        if res is not None:
            return res, int(t)
        return None, 0

    def _overlay_style(self) -> Dict[str, Any]:
        """The viewer's overlay-style settings (enabled / color / multicolor /
        weight / alpha), or a disabled default if the viewer has none."""
        try:
            return self.viewer.overlay_style()
        except Exception:  # noqa: BLE001
            return {"enabled": False}

    def _paint_segmentation_overlay(self, rgb: np.ndarray, t: int, m: int) -> np.ndarray:
        """Paint the segmentation label overlay for ``(m, t)`` (live / preview / Run)."""
        from nd2studios.pages.analysis_page import _overlay_labels
        style = self._overlay_style()
        st_on = bool(style.get("enabled"))
        # Custom-style overrides (else fall back to each result's built-in look).
        ov_color = (None if style.get("multicolor") else style.get("color"))
        ov_alpha = float(style.get("alpha", 1.0))
        ov_weight = int(style.get("weight", 1))

        # Live streaming during a Run: the just-finished frame's labels paint with
        # CellTracker's red outline (the committed result takes over afterwards).
        live = self._live_seg.get((m, t))
        if live is not None:
            return _overlay_labels(
                rgb, live,
                alpha=ov_alpha if st_on else 1.0,
                color=ov_color if st_on else (255, 50, 50),
                outline=True, thickness=ov_weight if st_on else 1)
        res, fidx = self._overlay_result_for(m, t)
        if res is None:
            return rgb
        ch = next(iter(res.label_masks), "")
        if not ch:
            return rgb
        masks = res.label_masks[ch]
        if masks.shape[0] == 0:
            return rgb
        fi = min(int(fidx), masks.shape[0] - 1)
        out = _overlay_labels(
            rgb, masks[fi],
            alpha=ov_alpha if st_on else res.overlay_alpha,
            color=ov_color if st_on else res.overlay_color,
            outline=getattr(res, "overlay_outline", False),
            thickness=ov_weight if st_on else 1)
        for sec in res.secondary_label_masks.values():
            if sec.shape[0] > 0:
                si = min(fi, sec.shape[0] - 1)
                out = _overlay_labels(out, sec[si],
                                      alpha=res.secondary_overlay_alpha,
                                      color=res.secondary_overlay_color)
        return out

    def _paint_tracks_overlay(self, rgb: np.ndarray, t: int, m: int) -> np.ndarray:
        """Solid track-colored cells for ``(m, t)`` (CellTracker make_colored_overlay)."""
        if not self._track_colormap:
            return rgb
        res, fidx = self._overlay_result_for(m, t)
        if res is None:
            return rgb
        ch = next(iter(res.label_masks), "")
        if not ch:
            return rgb
        masks = res.label_masks[ch]
        if masks.shape[0] == 0:
            return rgb
        fi = min(int(fidx), masks.shape[0] - 1)
        label_frame = np.asarray(masks[fi])
        label_to_track = {
            int(r["label_id"]): r.get("track_id")
            for r in self._overlay_rows()
            if int(r.get("m_position", 0)) == m and int(r.get("frame", 0)) == t
            and r.get("label_id") is not None
        }
        from nd2studios.backend.track_overlays import make_colored_overlay
        # Overlay style: uniform color (custom style, multicolor off) or the
        # per-track palette; opacity from the style when enabled.
        style = self._overlay_style()
        colormap = self._track_colormap
        alpha = 1.0
        if style.get("enabled"):
            alpha = float(style.get("alpha", 1.0))
            if not style.get("multicolor"):
                col = tuple(style.get("color", (255, 50, 50)))
                colormap = {tid: col for tid in self._track_long_ids}
        return make_colored_overlay(rgb, label_frame, label_to_track,
                                    colormap, self._track_long_ids, alpha=alpha)

    def _overlay_rows(self) -> List[Dict[str, Any]]:
        """Row source for the tracks / vectors overlays: the tracked rows frozen
        at Track-Objects time (so a downstream Dismiss / if-else filter doesn't
        blank the visualization), falling back to the live results rows."""
        return self._track_overlay_rows or self._results_rows

    def _frame_centroids(self, rows: List[Dict[str, Any]], m: int, frame: int) -> np.ndarray:
        """``(N, 2)`` array of ``(y, x)`` px centroids at ``(m, frame)``."""
        pts = [(float(r.get("centroid_y_px", 0.0)), float(r.get("centroid_x_px", 0.0)))
               for r in rows
               if int(r.get("m_position", 0)) == m and int(r.get("frame", 0)) == frame
               and r.get("centroid_y_px") is not None]
        return np.asarray(pts, dtype=float) if pts else np.empty((0, 2))

    def _paint_vectors_overlay(self, rgb: np.ndarray, t: int, m: int,
                               gridded: bool) -> np.ndarray:
        """Motion vectors for ``(m, t)`` — **outgoing**: each arrow's tail sits on
        the object's centroid and the head points to where it moves on the *next*
        frame (t → t+1). ``gridded=False`` draws one arrow per cell;
        ``gridded=True`` draws the interpolated velocity field, masked to the
        regions around objects so no arrows appear in empty space. The last frame
        (no t+1) draws nothing."""
        rows = self._overlay_rows()
        if not rows:
            return rgb
        if gridded:
            shape = self._field_shape_for(self._active_record())
            if shape is None:
                return rgb
            try:
                from nd2studios.backend.celltracker_bridge import compute_field_for_frame
                from nd2studios.backend.track_overlays import draw_grid_velocity
            except Exception:  # noqa: BLE001
                return rgb
            params = {"field": "speed", "grid_step": 24, "sigma": 1.0}
            # The field at frame t is the outgoing motion t → t+1 (computed as
            # pos(t+1) − pos(t) at frame t+1), matching the per-cell arrows.
            _arr, vel = compute_field_for_frame(rows, m, t + 1, shape, params)
            if not vel:
                return rgb
            H, W = shape
            # The velocity field is now NaN outside the cell footprint; this path
            # zeroes dead space itself (the "near a centroid" mask below), so drop
            # the NaNs to 0 first.
            vy = np.nan_to_num(np.asarray(vel["velocity_y"], dtype=float))
            vx = np.nan_to_num(np.asarray(vel["velocity_x"], dtype=float))
            gh, gw = vy.shape
            ys = np.linspace(0, H, gh)
            xs = np.linspace(0, W, gw)
            gx, gy = np.meshgrid(xs, ys)
            # Suppress arrows in dead space: keep only grid nodes near an object
            # centroid (within ~1.5 grid steps), so the field hugs the cells.
            cents = self._frame_centroids(rows, m, t)
            if len(cents):
                radius = 1.5 * float(params["grid_step"])
                d2 = ((gy.ravel()[:, None] - cents[:, 0][None, :]) ** 2 +
                      (gx.ravel()[:, None] - cents[:, 1][None, :]) ** 2)
                near = (d2.min(axis=1) <= radius ** 2).reshape(gh, gw)
                vy = np.where(near, vy, 0.0)
                vx = np.where(near, vx, 0.0)
            style = self._overlay_style()
            color = (tuple(style.get("color", (0, 255, 255)))
                     if style.get("enabled") and not style.get("multicolor")
                     else (0, 255, 255))
            weight = int(style.get("weight", 1)) if style.get("enabled") else 1
            return draw_grid_velocity(rgb, gy, gx, vy, vx, scale=3.0,
                                      color=color, thickness=weight)
        # Per-cell outgoing arrows: tail at the current centroid, head at the
        # same track's position on the next frame.
        def _by_track(frame: int):
            return {r.get("track_id"): r for r in rows
                    if int(r.get("m_position", 0)) == m
                    and int(r.get("frame", 0)) == frame
                    and r.get("track_id") is not None}

        cur = _by_track(t)
        nxt = _by_track(t + 1)
        vecs = []
        for tid, r in cur.items():
            n = nxt.get(tid)
            if n is None:
                continue
            vecs.append((float(r.get("centroid_y_px", 0.0)),
                         float(r.get("centroid_x_px", 0.0)),
                         float(n.get("centroid_y_px", 0.0)),
                         float(n.get("centroid_x_px", 0.0))))
        if not vecs:
            return rgb
        from nd2studios.backend.track_overlays import draw_cell_vectors
        style = self._overlay_style()
        color = (tuple(style.get("color", (255, 255, 0)))
                 if style.get("enabled") and not style.get("multicolor")
                 else (255, 255, 0))
        weight = int(style.get("weight", 1)) if style.get("enabled") else 1
        return draw_cell_vectors(rgb, vecs, color=color, thickness=weight)

    def _composite_pipeline_overlay(self, rgb: np.ndarray, t: int, m: int) -> np.ndarray:
        """Single post-process hook: paint the layer for the active overlay tab.

        Dispatches on ``self._overlay_mode`` and draws over the cached processed
        base for ``(m, t)``. Switching tabs only recomputes this layer
        (``invalidate_post_process_cache``) — the base image is never re-rendered.
        """
        mode = self._overlay_mode
        if mode == "image":
            return rgb
        if mode == "segmentation":
            return self._paint_segmentation_overlay(rgb, t, m)
        if mode == "tracks":
            return self._paint_tracks_overlay(rgb, t, m)
        if mode == "vectors_cells":
            return self._paint_vectors_overlay(rgb, t, m, gridded=False)
        if mode == "vectors_field":
            return self._paint_vectors_overlay(rgb, t, m, gridded=True)
        return rgb

    def _ensure_spatial_panel(self):
        """Lazily build the Spatial Maps panel and add it to the viewer stack."""
        if self._spatial_panel is None:
            from nd2studios.widgets.spatial_maps_panel import SpatialMapsPanel
            self._spatial_panel = SpatialMapsPanel(self)
            self._spatial_panel.m_change_requested.connect(
                self._populate_spatial_panel)
            self._spatial_panel.status_message.connect(self._set_status)
            self._viewer_stack.addWidget(self._spatial_panel)   # index 1
            # Match the current docked / popped-out state of the viewer.
            self._spatial_panel.set_compact(
                self._popout_windows.get("viewer") is None)
        return self._spatial_panel

    def _ensure_serialtrack_panel(self):
        """Lazily build the SerialTrack PTV panel and add it to the viewer stack."""
        if self._serialtrack_panel is None:
            from nd2studios.widgets.serialtrack_panel import SerialTrackPanel
            self._serialtrack_panel = SerialTrackPanel(self)
            self._serialtrack_panel.m_change_requested.connect(
                self._populate_serialtrack_panel)
            self._serialtrack_panel.status_message.connect(self._set_status)
            self._viewer_stack.addWidget(self._serialtrack_panel)
        return self._serialtrack_panel

    def _populate_serialtrack_panel(self, m: Optional[int] = None) -> None:
        """Feed the current multipoint's tracked rows to the SerialTrack panel.

        Reuses the same row source (``_overlay_rows``), channel materialization,
        frame-count and pixel-size logic as :meth:`_populate_spatial_panel`, but
        passes the raw rows (the panel builds its own :class:`TrackData`)."""
        panel = getattr(self, "_serialtrack_panel", None)
        if panel is None:
            return
        record = self._active_record()
        rows = self._overlay_rows()
        if m is None:
            cm, _, _ = self.viewer.coords()
            m = int(cm)
        m = int(m)
        ms_present = sorted({int(r.get("m_position", 0)) for r in rows})
        if ms_present and m not in ms_present:
            m = ms_present[0]

        field_shape = self._field_shape_for(record)
        if field_shape is None:
            self._set_status("SerialTrack: frame size unavailable.")
            return

        raw_channels = self._materialize_channels_for_m(record, m) if record else {}
        channels = raw_channels
        recipe = list(getattr(record, "recipe", []) or []) if record else []
        rbc = getattr(record, "recipe_by_channel", None) if record else None
        if (recipe or rbc) and raw_channels:
            try:
                channels = apply_recipe(
                    raw_channels, recipe,
                    bool(getattr(record, "recipe_normalized", False)),
                    recipe_by_channel=rbc)
            except Exception:  # noqa: BLE001 — fall back to raw for the backdrop
                channels = raw_channels

        if channels:
            n_frames = int(next(iter(channels.values())).shape[0])
        else:
            n_frames = max((int(r.get("frame", 0)) for r in rows
                            if int(r.get("m_position", 0)) == m), default=0) + 1

        pixel = getattr(record, "pixel_size_um", None) if record else None
        z_step = getattr(record, "z_step_um", None) if record else None
        n_multipoints = max(
            (int(r.get("m_position", 0)) for r in rows), default=0) + 1

        panel.set_data(
            [r for r in rows if int(r.get("m_position", 0)) == m],
            channels or None, field_shape, pixel, n_frames,
            m=m, n_multipoints=n_multipoints, z_step_um=z_step)
        n_rows = sum(1 for r in rows if int(r.get("m_position", 0)) == m)
        self._set_status(
            f"SerialTrack: M{m + 1} · {n_rows} objects · shape {field_shape} · "
            f"{n_frames} frame(s) · {len(channels or {})} channel(s)")

    def _ensure_dvc_panel(self):
        """Lazily build the DVC viewer panel and add it to the viewer stack."""
        if getattr(self, "_dvc_panel", None) is None:
            from nd2studios.widgets.dvc_panel import DVCPanel
            self._dvc_panel = DVCPanel(self)
            self._dvc_panel.m_change_requested.connect(self._populate_dvc_panel)
            self._dvc_panel.status_message.connect(self._set_status)
            self._viewer_stack.addWidget(self._dvc_panel)
        return self._dvc_panel

    def _populate_dvc_panel(self, m: Optional[int] = None) -> None:
        """Feed the current multipoint's DVC field **series** to the DVC panel so
        it can play through frames. ``series`` is ``{frame(t): DVCResult}`` with a
        per-frame backdrop; frames without a result (e.g. the reference frame) are
        simply absent from the series."""
        panel = getattr(self, "_dvc_panel", None)
        if panel is None:
            return
        by_m = getattr(self, "_dvc_series_by_m", {}) or {}
        if m is None:
            cm, _, _ = self.viewer.coords()
            m = int(cm)
        m = int(m)
        if m not in by_m and by_m:
            m = sorted(by_m)[0]
        series = by_m.get(m, {})
        increments = (getattr(self, "_dvc_incr_by_m", {}) or {}).get(m, {})
        bg_map = (getattr(self, "_dvc_bg_by_mt", {}) or {}).get(m, {})
        record = self._active_record()
        n_mp = self._record_n_multipoints(record) if record is not None else 1
        # Total frame count for the strip (spans the whole timelapse).
        vol = getattr(record, "_raw_volume", None) if record is not None else None
        n_t = int(getattr(vol, "n_timepoints", 0) or 0)
        frames = sorted(series)
        if n_t <= 0:
            n_t = (max(frames) + 1) if frames else 1
        # Effective (possibly downsampled) XY pixel size + field size from a result.
        eff_px = None
        field_shape = None
        if frames:
            r0 = series[frames[0]]
            if r0.voxel_size_um:
                eff_px = float(r0.voxel_size_um[-1])
            bg0 = bg_map.get(frames[0])
            if bg0 is not None and np.asarray(bg0).ndim >= 2:
                field_shape = tuple(int(v) for v in np.asarray(bg0).shape[-2:])
        panel.set_data(series, bg_map, frames, pixel_size=eff_px,
                       field_shape=field_shape, m=m, n_multipoints=n_mp,
                       n_frames_total=n_t, increments=increments or None)
        if frames:
            self._set_status(
                f"DVC: M{m + 1} · {len(frames)} field(s) across {n_t} frame(s)")

    # ── Registration viewer panel (V1.56) ──────────────────────────────────
    def _ensure_registration_panel(self):
        """Lazily build the Registration viewer panel and add it to the stack."""
        if getattr(self, "_registration_panel", None) is None:
            from nd2studios.widgets.registration_panel import RegistrationPanel
            self._registration_panel = RegistrationPanel(self)
            self._registration_panel.m_change_requested.connect(
                self._populate_registration_panel)
            self._registration_panel.status_message.connect(self._set_status)
            self._registration_panel.write_requested.connect(
                self._apply_registration_writeback)
            self._viewer_stack.addWidget(self._registration_panel)
        return self._registration_panel

    def _populate_registration_panel(self, m: Optional[int] = None) -> None:
        """Feed one multipoint's registration bundle (reference-channel raw +
        aligned series + per-frame shifts/confidence) to the Registration panel."""
        panel = getattr(self, "_registration_panel", None)
        if panel is None:
            return
        by_m = getattr(self, "_reg_by_m", {}) or {}
        if m is None:
            cm, _, _ = self.viewer.coords()
            m = int(cm)
        m = int(m)
        if m not in by_m and by_m:
            m = sorted(by_m)[0]
        bundle = by_m.get(m)
        if bundle is None:
            return
        aligned = bundle.get("aligned", {}) or {}
        ref_channel = bundle.get("channel", "")
        raw_ref = bundle.get("raw_ref")
        aligned_ref = aligned.get(ref_channel)
        record = self._active_record()
        # If a common-region crop was published, show the cropped FOV in the
        # before/after so the panel matches what downstream nodes / export receive.
        crop = getattr(record, "_registration_crop", None) if record is not None else None
        if crop and raw_ref is not None and aligned_ref is not None:
            y0, y1, x0, x1 = (int(v) for v in crop)
            raw_ref = np.asarray(raw_ref)[..., y0:y1, x0:x1]
            aligned_ref = np.asarray(aligned_ref)[..., y0:y1, x0:x1]
        px = bundle.get("pixel_size_um") or ()
        pixel_size = float(px[-1]) if px else None
        n_mp = self._record_n_multipoints(record) if record is not None else 1
        panel.set_data(
            raw_ref, aligned_ref, bundle.get("shifts"), bundle.get("confidence"),
            pixel_size=pixel_size, m=m, n_multipoints=n_mp,
            model=bundle.get("model", "translation"),
            reference_mode=bundle.get("reference_mode", "first"),
            channel=ref_channel, n_channels=len(aligned),
            gated=bundle.get("gated"),
            min_confidence=float(bundle.get("min_confidence", 0.0) or 0.0))
        self._set_status(
            f"Registration: M{m + 1} · '{ref_channel}' · {len(aligned)} "
            "channel(s) aligned")

    def _apply_registration_to_channels(self, record, channels: Dict[str, Any],
                                        m: int) -> Dict[str, Any]:
        """Apply the published per-M registration transforms to already-processed
        ``{channel: (T,H,W)}`` (register once, apply to all).

        No-op when no Registration node has run for multipoint ``m`` (so a graph
        without registration is unaffected). Applied *after* the recipe in
        ``_processed_channels_for_m`` so every downstream analysis / tracking node
        sees the drift-corrected image."""
        by_m = getattr(record, "_registration_by_m", None)
        tf = by_m.get(int(m)) if isinstance(by_m, dict) else None
        if not tf:
            return channels
        order = int(getattr(record, "_registration_interp_order", 1) or 1)
        from nd2studios.backend.registration import estimate as _est
        out: Dict[str, Any] = {}
        for name, arr in channels.items():
            a = np.asarray(arr)
            try:
                out[name] = (_est.apply_series(a, tf, interp_order=order)
                             if a.ndim == 3 else a)
            except Exception:  # noqa: BLE001 — never break a Run on a bad transform
                out[name] = a
        return out

    def _apply_registration_writeback(self, m: int) -> None:
        """Write the aligned channels for multipoint ``m`` into the record's
        processed view so the image viewer (and in-RAM single-M files) show the
        drift-corrected base. Multi-M analysis Runs get the same correction via
        ``_processed_channels_for_m`` → ``_apply_registration_to_channels``."""
        record = self._active_record()
        if record is None:
            return
        bundle = (getattr(self, "_reg_by_m", {}) or {}).get(int(m))
        aligned = (bundle or {}).get("aligned", {}) if bundle else {}
        if not aligned:
            self._set_status("Registration: nothing to write.")
            return
        # Crop to the common region if one was published (V1.60) — via _crop_frame,
        # so the in-RAM display matches the cropped lazy base.
        record._processed_channels = {name: self._crop_frame(np.asarray(arr))
                                      for name, arr in aligned.items()}
        # Display: for a lazy-volume file, drive the viewer through the cached lazy
        # RegisteredFrameVolume (set_volume) instead of pushing the full-res
        # materialized stack via set_channels — the lazy path is pyramid/cache-backed
        # so registered playback is as fast as the raw (V1.60 perf fix). The
        # materialized ``_processed_channels`` is still set above for downstream /
        # processed_view() compatibility. In-RAM files (no lazy volume) use the
        # small materialized dict directly.
        if getattr(record, "_raw_volume", None) is not None:
            self._show_base_image(force=True)
        else:
            try:
                self.viewer.set_channels(
                    record._processed_channels,
                    channel_display=getattr(record, "channel_display", None),
                    n_multipoints=self._record_n_multipoints(record), m=int(m))
                self.viewer.invalidate_post_process_cache()
            except Exception:  # noqa: BLE001 — write-back must never crash the GUI
                pass
        self._set_status(
            f"Registration: M{int(m) + 1} shown drift-corrected · "
            f"{len(aligned)} channel(s) written to processed data.")

    def _on_overlay_tab_changed(self, idx: int) -> None:
        """Switch the active overlay layer; recompute only the overlay (cheap).

        The "Spatial Maps" tab swaps the viewer stack to the spatial panel; every
        other tab swaps back to the image viewer and repaints its overlay."""
        if not (0 <= idx < len(self._overlay_tab_keys)):
            return
        self._overlay_mode = self._overlay_tab_keys[idx]
        if self._overlay_mode == "spatial":
            panel = self._ensure_spatial_panel()
            self._viewer_stack.setCurrentWidget(panel)
            self._populate_spatial_panel()
            return
        if self._overlay_mode == "serialtrack":
            panel = self._ensure_serialtrack_panel()
            self._viewer_stack.setCurrentWidget(panel)
            self._populate_serialtrack_panel()
            return
        if self._overlay_mode == "dvc":
            panel = self._ensure_dvc_panel()
            self._viewer_stack.setCurrentWidget(panel)
            self._populate_dvc_panel()
            return
        if self._overlay_mode == "registration":
            panel = self._ensure_registration_panel()
            self._viewer_stack.setCurrentWidget(panel)
            self._populate_registration_panel()
            return
        self._viewer_stack.setCurrentWidget(self.viewer)
        # Re-applying the same hook invalidates the overlay cache + repaints
        # the current frame, keeping the base render cache intact.
        self.viewer.set_frame_post_process(self._composite_pipeline_overlay)

    def _select_overlay_tab(self, mode: str) -> None:
        """Programmatically select an overlay tab (no user signal) and set mode.

        Keeps the viewer stack in sync: "spatial" shows the Spatial Maps panel,
        every other mode shows the image viewer."""
        if mode not in self._overlay_tab_keys:
            return
        self._overlay_mode = mode
        tb = getattr(self, "_overlay_tabbar", None)
        idx = self._overlay_tab_keys.index(mode)
        if tb is not None and tb.currentIndex() != idx:
            tb.blockSignals(True)
            tb.setCurrentIndex(idx)
            tb.blockSignals(False)
        stack = getattr(self, "_viewer_stack", None)
        if stack is not None:
            if mode == "spatial":
                stack.setCurrentWidget(self._ensure_spatial_panel())
            elif mode == "serialtrack":
                stack.setCurrentWidget(self._ensure_serialtrack_panel())
            elif mode == "dvc":
                stack.setCurrentWidget(self._ensure_dvc_panel())
            elif mode == "registration":
                stack.setCurrentWidget(self._ensure_registration_panel())
            else:
                stack.setCurrentWidget(self.viewer)

    def _update_overlay_tabs_available(self) -> None:
        """Show only the overlay tabs the current pipeline actually produced:
        Image always; Segmentation once an analysis result exists (or a run is
        live); Tracks / Vectors only after a Track Objects node has linked."""
        tb = getattr(self, "_overlay_tabbar", None)
        if tb is None:
            return
        has_seg = bool(self._run_results_by_m or self._analysis_screen_results
                       or self._live_seg or self._run_active)
        has_tracks = bool(self._track_colormap)
        has_rows = bool(self._track_overlay_rows or self._results_rows)
        vis = {
            "image": True,
            "segmentation": has_seg,
            "tracks": has_tracks,
            "vectors_cells": has_tracks,
            "vectors_field": has_tracks,
            # Spatial Maps needs measured objects (centroids) to map.
            "spatial": has_seg or has_rows,
            # SerialTrack PTV plots need tracked rows (track_id chaining).
            "serialtrack": has_tracks or has_rows,
            # DVC shows a dense field — available once a DVC node has run.
            "dvc": bool(self._dvc_series_by_m),
            # Registration before/after + drift plot — once a Registration node ran.
            "registration": bool(self._reg_by_m),
        }
        for i, key in enumerate(self._overlay_tab_keys):
            tb.setTabVisible(i, bool(vis.get(key, True)))
        # If the active tab just became hidden, fall back to a visible one.
        cur = tb.currentIndex()
        if 0 <= cur < len(self._overlay_tab_keys):
            if not vis.get(self._overlay_tab_keys[cur], True):
                self._select_overlay_tab("segmentation" if has_seg else "image")

    # Back-compat aliases — both legacy hooks now route through the dispatcher.
    def _composite_analysis_overlay(self, rgb: np.ndarray, t: int, m: int) -> np.ndarray:
        return self._composite_pipeline_overlay(rgb, t, m)

    # ── Results preview ───────────────────────────────────────────────────────
    def _start_preview_walk(self, target: str, interactive: bool) -> None:
        """Preview = a scoped Run on the selected planes, up to ``target``.

        Shades the chain feeding ``target``, screens + measures the analysis /
        results portion on the selected (M × T) planes (using the previewed
        Compute Measurements node's metric selection), then walks the remaining
        logic / special nodes (branch at if-else; pop Validate/Review when
        ``interactive``). Re-runs non-interactively as you navigate."""
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        if getattr(self.main_window, "heavy_ops_blocked", lambda: False)():
            self._set_status("Preview paused — low memory")
            return
        ana = self._upstream_analysis_node(target)
        names = self._current_channel_names()
        if ana is None or not names:
            self._analysis_screen_results = {}
            self._results_rows = []
            self._populate_results_table([])
            self.viewer.invalidate_post_process_cache()
            self._set_status("Wire an analysis node upstream to measure objects.")
            return
        cls = AnalysisPipeline.get_pipeline(analysis_pipeline_name_for_op_key(ana.op_key))
        if cls is None:
            return
        # V1.48: segment the channel(s) wired into the upstream analysis node
        # (2nd wired channel = counterstain).
        extra, seg_channels = self._analysis_run_spec(ana, names)
        if not seg_channels:
            seg_channels = [names[0]]
        aparams = dict(ana.params)
        aparams.update(extra)
        up_nodes, _ = self._upstream_subgraph(Stage.ANALYSIS, target)
        metrics = self._chain_measure_metrics(up_nodes)
        m, t, _ = self.viewer.coords()
        planes_frames = self._read_processed_planes(
            record, self._selected_planes(m, t), names)
        if not planes_frames:
            return
        # Shade EVERY reachable node; only the chain feeding the previewed node
        # turns "done" as it executes (downstream / other branches stay shaded).
        # The screen+measure covers this chain's input/analysis/results nodes.
        self._pv_target = target
        self._pv_interactive = bool(interactive)
        self._pv_screen_set = {
            nid for nid in up_nodes
            if (self._doc.analysis.nodes[nid].role is NodeRole.INPUT
                or self._doc.analysis.nodes[nid].category
                in (NodeCategory.ANALYSIS, NodeCategory.RESULTS))}
        reachable = GraphRunner(self._doc.analysis).reachable_nodes() | up_nodes
        self._pv_states = {nid: "shaded" for nid in reachable}
        for nid in self._pv_screen_set:
            self._pv_states[nid] = "current"
        self._scenes[Stage.ANALYSIS].set_run_states(self._pv_states)
        metadata = self._preview_metadata(record)
        job = _ResultsScreenMeasureJob(
            _PV_SCREEN_KEY, cls, seg_channels, planes_frames, metadata, aparams,
            metrics,
        )
        self._set_preview_progress(visible=True, value=0)
        self._runner.submit(job)

    def _chain_measure_metrics(self, node_ids: set):
        """The metric selection of the Compute Measurements node in ``node_ids``
        (None = all metrics) — so the measurement honors the picker."""
        for nid in node_ids:
            n = self._doc.analysis.nodes.get(nid)
            if (n is not None and n.op_key.startswith(RESULTS_PREFIX)
                    and results_op_name_for_op_key(n.op_key) == "Compute Measurements"):
                sel = n.params.get("metrics")
                return set(sel) if sel else None
        return None

    def _preview_walk_downstream(self) -> None:
        """After the screen+measure, walk the logic/special nodes feeding the
        target in topological order — shading each, branching at if-else, and
        popping interactive nodes when the walk was triggered deliberately."""
        if self._pv_walking:
            return
        self._pv_walking = True
        try:
            up_nodes, _ = self._upstream_subgraph(Stage.ANALYSIS, self._pv_target)
            order = [nid for nid in topological_order(self._doc.analysis)
                     if nid in up_nodes and nid not in self._pv_screen_set]
            scene = self._scenes[Stage.ANALYSIS]
            for nid in order:
                node = self._doc.analysis.nodes.get(nid)
                if node is None:
                    continue
                self._pv_states[nid] = "current"
                scene.set_run_states(self._pv_states)
                QApplication.processEvents()
                self._preview_execute_walk_node(node)
                self._pv_states[nid] = "done"
                scene.set_run_states(self._pv_states)
                if nid == self._pv_target:
                    break
            scene.set_run_states(self._pv_states)
        finally:
            self._pv_walking = False

    def _preview_execute_walk_node(self, node) -> None:
        """Execute one logic/special node during a preview walk (on the screened
        rows). If-else reports its branch; Validate/Review pop up only when the
        walk is interactive; the rest report their action."""
        op = node.op_key
        if op == IF_ELSE_OP_KEY:
            cond = Condition.from_dict(node.params.get("condition"))
            if str(node.params.get("lens")) == LENS_OBJECT:
                group_by = str(node.params.get("group_by") or "")
                group_by = GROUP_ROW if group_by == GROUP_ROW else "track"
                keep, drop = partition_rows(cond, self._results_rows, group_by=group_by)
                # Keep only the subset on the branch that feeds the previewed node,
                # so a downstream Spatial-Maps / Dismiss preview reflects the split.
                up_nodes, _ = self._upstream_subgraph(Stage.ANALYSIS, self._pv_target)
                sl = self._doc.analysis
                true_port = next((p.id for p in node.outputs if p.name == "true"), None)
                false_port = next((p.id for p in node.outputs if p.name == "false"), None)
                subset, seen = [], set()
                for e in sl.outgoing(node.id):
                    if e.dst_node not in up_nodes:
                        continue
                    branch = keep if e.src_port == true_port else \
                        (drop if e.src_port == false_port else [])
                    for r in branch:
                        if id(r) not in seen:
                            seen.add(id(r))
                            subset.append(r)
                self._results_rows = subset
                self._populate_results_table(self._results_rows)
                self._set_status(
                    f"{node.title} (object lens): {len(keep)} → TRUE, "
                    f"{len(drop)} → FALSE; previewing {len(subset)} on this branch")
            else:
                take = evaluate_condition(cond, self._results_rows)
                self._set_status(
                    f"{node.title} → {'TRUE' if take else 'FALSE'} branch "
                    f"({len(self._results_rows)} objects)")
        elif op == SPECIAL_TRACK_OP_KEY:
            record = self._active_record()
            px = (getattr(record, "pixel_size_um", None)
                  if record is not None else None)
            link_objects_with_params(
                self._results_rows, node.params, pixel_size_um=px)
            self._populate_results_table(self._results_rows)
            # Light up the Tracks / Vectors overlays + tabs in preview exactly as a
            # Run does (previously preview linked ids but never built this state,
            # so the tabs stayed hidden).
            self._set_track_overlay_state(self._results_rows)
            n = len({r.get("track_id") for r in self._results_rows
                     if r.get("track_id") is not None})
            self._set_status(f"{node.title}: linked {n} track(s).")
        elif op in (SPECIAL_VALIDATE_OP_KEY, SPECIAL_REVIEW_OP_KEY):
            if self._pv_interactive:
                self._preview_validate(node)
            else:
                self._set_status(f"{node.title} (skipped — navigate is automatic)")
        elif op == SPECIAL_DISMISS_OP_KEY:
            # Terminal discard — erase the objects on this branch (in preview the
            # current rows are the branch subset) from the table + overlays, so
            # preview matches a Run.
            dropped = self._discard_objects(list(self._results_rows))
            self._set_status(f"Dismiss: discarded {dropped} object row(s).")
        elif op == SPECIAL_CT_METRICS_OP_KEY:
            record = self._active_record()
            px = (getattr(record, "pixel_size_um", None)
                  if record is not None else None)
            augment_rows_with_metrics(
                self._results_rows, node.params, pixel_size_um=px)
            self._populate_results_table(self._results_rows)
            tracked = any(r.get("track_id") is not None for r in self._results_rows)
            note = "" if tracked else " (no tracks — neighbor distance only)"
            self._set_status(f"{node.title}: per-cell metrics added{note}.")
        elif op in (SPECIAL_CT_FIELDS_OP_KEY, SPECIAL_INTERP_MAP_OP_KEY):
            self._open_spatial_maps_tab(node)
        elif op == SPECIAL_DVC_OP_KEY:
            self._set_status(f"{node.title} runs on the full file via Run "
                             "(DVC is not computed in preview).")
        elif op == SPECIAL_REGISTER_OP_KEY:
            self._set_status(f"{node.title} runs on the full file via Run "
                             "(registration is not computed in preview).")
        elif op in (SPECIAL_EXPORT_OP_KEY, SPECIAL_SEND_RESULTS_OP_KEY):
            self._set_status(f"{node.title} runs on the full file via Run.")
        elif op == SPECIAL_PAUSE_OP_KEY:
            self._set_status("Pause — preview stops here (Run executes the rest).")

    def _field_shape_for(self, record) -> Optional[tuple]:
        """(H, W) of the previewed frames — the crop dims when a preview crop is
        active, otherwise the record metadata frame size, or None."""
        rect = self._crop_rect()
        if rect is not None:
            _, _, w, h = rect
            return (int(h), int(w))
        if record is None:
            return None
        meta = getattr(record, "nd2_metadata", {}) or {}
        h = int(meta.get("height", 0) or 0)
        w = int(meta.get("width", 0) or 0)
        return (h, w) if h > 0 and w > 0 else None

    def _open_spatial_maps_tab(self, node) -> None:
        """Open the interactive Spatial Maps tab and load the node's template(s).

        Both spatial-map nodes route here (preview and Run): instead of a one-off
        heatmap dialog or a folder export, they bring up the in-viewer Spatial
        Maps tab (faithful Cell-Tracker spatial page). Saved templates attached to
        the node pre-configure it; the tab's own Save button handles export."""
        if not self._overlay_rows():
            self._set_status(
                f"{node.title}: measure objects first (wire an analysis node "
                "upstream and preview / run it).")
            return
        self._ensure_spatial_panel()
        self._update_overlay_tabs_available()
        self._select_overlay_tab("spatial")
        self._populate_spatial_panel()
        self._spatial_panel.set_node_templates(
            list(node.params.get("templates") or []))
        self._set_status(f"{node.title}: Spatial Maps tab ready.")

    def _label_stack_for_m(self, m: int):
        """``((T, H, W) label stack, channel)`` for multipoint ``m`` — the whole
        run result when present (absolute-frame indexed), else None."""
        # Run masks only apply when their geometry matches the current display
        # (see _overlay_result_for); otherwise fall back to crop-space rows +
        # cropped channels for the Spatial Maps preview.
        if self._run_results_crop != self._crop_rect():
            return None, ""
        res = self._run_results_by_m.get(m)
        if res is not None:
            masks = getattr(res, "label_masks", {}) or {}
            ch = next(iter(masks), "")
            if ch:
                arr = np.asarray(masks[ch])
                if arr.ndim >= 3:
                    return arr, ch
        return None, ""

    def _populate_spatial_panel(self, m: Optional[int] = None) -> None:
        """Assemble the current multipoint's data and feed the Spatial Maps panel.

        ``tracked_df`` ← measured rows (CellTracker columns); ``label_stack`` ←
        the per-M run segmentation masks (for cell-mask / border modes);
        ``channels`` ← processed stacks (recipe applied), ``raw_channels`` ← the
        raw stacks; ``field_shape`` / ``pixel_size`` ← masks or record metadata."""
        panel = getattr(self, "_spatial_panel", None)
        if panel is None:
            return
        record = self._active_record()
        # Use the same row source as the (working) tracks / vectors overlays:
        # the tracked rows frozen at Track-Objects time, falling back to the live
        # results rows. ``_results_rows`` alone can be empty / scoped after a run
        # or a downstream Dismiss, while ``_track_overlay_rows`` still holds them.
        rows = self._overlay_rows()
        if m is None:
            cm, _, _ = self.viewer.coords()
            m = int(cm)
        m = int(m)
        # The viewer may sit on a multipoint with no measured objects — fall back
        # to the first M that actually has rows so the tab isn't blank.
        ms_present = sorted({int(r.get("m_position", 0)) for r in rows})
        if ms_present and m not in ms_present:
            m = ms_present[0]

        tracked_df = build_tracked_df(rows, m=m)
        label_stack, _seg_ch = self._label_stack_for_m(m)

        field_shape = None
        if label_stack is not None and label_stack.ndim >= 3:
            field_shape = (int(label_stack.shape[-2]), int(label_stack.shape[-1]))
        if field_shape is None:
            field_shape = self._field_shape_for(record)
        if field_shape is None:
            self._set_status("Spatial Maps: frame size unavailable.")
            return

        # Raw + processed channel stacks for the background image / intensity.
        raw_channels = self._materialize_channels_for_m(record, m) if record else {}
        channels = raw_channels
        recipe = list(getattr(record, "recipe", []) or []) if record else []
        rbc = getattr(record, "recipe_by_channel", None) if record else None
        if (recipe or rbc) and raw_channels:
            try:
                channels = apply_recipe(
                    raw_channels, recipe,
                    bool(getattr(record, "recipe_normalized", False)),
                    recipe_by_channel=rbc)
            except Exception:  # noqa: BLE001 — fall back to raw for the backdrop
                channels = raw_channels

        # Frame count: prefer the run stack, else the channel stack, else rows.
        if label_stack is not None and label_stack.ndim >= 3:
            n_frames = int(label_stack.shape[0])
        elif channels:
            n_frames = int(next(iter(channels.values())).shape[0])
        else:
            n_frames = max((int(r.get("frame", 0)) for r in rows
                            if int(r.get("m_position", 0)) == m), default=0) + 1

        pixel = getattr(record, "pixel_size_um", None) if record else None
        n_multipoints = max(
            (int(r.get("m_position", 0)) for r in rows), default=0) + 1

        panel.set_data(
            tracked_df, label_stack, channels, raw_channels or None,
            field_shape, pixel, n_frames, m=m, n_multipoints=n_multipoints)
        self._set_status(
            f"Spatial Maps: M{m + 1} · {len(tracked_df)} objects · "
            f"shape {field_shape} · {n_frames} frame(s) · "
            f"labels {'yes' if label_stack is not None else 'no'} · "
            f"{len(channels)} channel(s)")

    def _read_processed_planes(self, record, planes: List[tuple],
                               names: List[str]) -> Dict[tuple, Dict[str, Any]]:
        """Processed ``{(m, t): {channel: (1, H, W)}}`` for ``planes`` (all
        channels), read on the GUI thread for the results screen+measure job.
        Sliced to the preview crop when one is active."""
        out: Dict[tuple, Dict[str, Any]] = {}
        for (m, t) in planes:
            chans: Dict[str, Any] = {}
            for ch in names:
                f = self._extract_processed_frame(record, ch, m, t)
                if f is not None:
                    chans[ch] = self._crop_frame(f)
            if chans:
                out[(int(m), int(t))] = chans
        return out

    def _upstream_analysis_node(self, node_id: str):
        """The nearest analysis ACTION node feeding ``node_id`` (BFS upstream)."""
        sl = self._doc.analysis
        seen: set = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            n = sl.nodes.get(cur)
            if (n is not None and n.role is NodeRole.ACTION
                    and n.op_key.startswith("analysis:")):
                return n
            for e in sl.incoming(cur):
                stack.append(e.src_node)
        return None

    def _resolve_results_action(self, node_id: str):
        """The Results ACTION node for a preview target (resolve OUTPUT → its
        feeding action). Lives in the merged Analysis slice (V1.45)."""
        sl = self._doc.analysis
        node = sl.nodes.get(node_id)
        if node is None:
            return None
        if node.role is NodeRole.ACTION:
            return node
        if node.role is NodeRole.OUTPUT:
            pred = predecessor(sl, node.id)
            if pred is not None and pred.role is NodeRole.ACTION:
                return pred
        return None

    def _get_analysis_result_names(self) -> List[str]:
        """Pipeline names with at least one committed :class:`AnalysisResult`."""
        record = self._active_record()
        if record is None:
            return []
        out: List[str] = []
        for name, store in (getattr(record, "analysis_results", {}) or {}).items():
            if isinstance(store, AnalysisResult):
                out.append(name)
            elif isinstance(store, dict) and any(
                isinstance(r, AnalysisResult) for r in store.values()
            ):
                out.append(name)
        return out

    def _get_analysis_result_for(self, name: str, m: int) -> Optional[AnalysisResult]:
        record = self._active_record()
        if record is None:
            return None
        store = (getattr(record, "analysis_results", {}) or {}).get(name)
        if isinstance(store, AnalysisResult):
            return store
        if isinstance(store, dict):
            r = store.get(m)
            if isinstance(r, AnalysisResult):
                return r
            for _mm, rr in sorted(store.items(), key=lambda kv: kv[0]):
                if isinstance(rr, AnalysisResult):
                    return rr
        return None

    def _composite_results_overlay(self, rgb: np.ndarray, t: int, m: int) -> np.ndarray:
        return self._composite_pipeline_overlay(rgb, t, m)

    def _populate_results_table(self, rows: List[Dict[str, Any]]) -> None:
        from nd2studios.pages.results_page import _MeasurementsModel

        self._results_table.setModel(_MeasurementsModel(list(rows), self))
        if not rows:
            self._results_summary.setText("No objects measured.")
            return
        frames = len({r.get("frame") for r in rows})
        areas = [r.get("area_um2") for r in rows
                 if isinstance(r.get("area_um2"), (int, float))]
        mean_area = (sum(areas) / len(areas)) if areas else 0.0
        self._results_summary.setText(
            f"{len(rows)} objects · {frames} frame(s) · mean area {mean_area:.3g} µm²"
        )

    # ── Plot/data tabs (bottom panel) ───────────────────────────────────────────

    def _plot_canvas(self, name: str):
        """Get (or lazily create) the MplCanvas behind data tab ``name``."""
        cv = self._plot_canvases.get(name)
        if cv is None:
            from nd2studios.widgets.common import MplCanvas
            cv = MplCanvas(self._data_tabs, width=4, height=2.4, dpi=90)
            self._plot_canvases[name] = cv
            self._data_tabs.addTab(cv, name)
        return cv

    def _clear_plot_tabs(self) -> None:
        """Drop every plot tab, keeping the Measurements table (tab 0)."""
        while self._data_tabs.count() > 1:
            w = self._data_tabs.widget(1)
            self._data_tabs.removeTab(1)
            if w is not None:
                w.deleteLater()
        self._plot_canvases = {}

    def _loop_iter_label(self, i: int, plan: list) -> str:
        """A short legend label for loop iteration ``i`` — its swept params, e.g.
        ``"#2  Scale=0.5"`` (falls back to just the index for a plain repeat)."""
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)
        label = f"#{i + 1}"
        if 0 <= i < len(plan):
            parts = []
            for _nid, ov in (plan[i] or {}).items():
                for k, v in (ov or {}).items():
                    parts.append(f"{k}={_fmt(v)}")
            if parts:
                label += "  " + ", ".join(parts)
        return label

    def _iteration_colors(self, n: int) -> list:
        """``n`` distinct hex colors for per-iteration plot series (tab10 for a
        handful, viridis for many)."""
        import matplotlib
        import matplotlib.colors as mcolors
        try:
            cmap = matplotlib.colormaps["tab10" if n <= 10 else "viridis"]
        except Exception:  # noqa: BLE001 — older matplotlib
            cmap = matplotlib.cm.get_cmap("tab10" if n <= 10 else "viridis")
        if n <= 10:
            return [mcolors.to_hex(cmap(i % 10)) for i in range(n)]
        return [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]

    def _update_analysis_plots_loop(self, series: list,
                                    combined: List[Dict[str, Any]]) -> None:
        """Overlay each loop iteration's Cells/frame line + Area step-histogram in a
        distinct color with a legend, plus a dashed 'Combined' series (V1.49)."""
        from collections import Counter
        colors = self._iteration_colors(len(series))

        cv = self._plot_canvas("Cells/frame")
        cv.fig.clear()
        ax = cv.add_subplot(111)
        for i, s in enumerate(series):
            counts = Counter(int(r.get("frame", 0)) for r in s["rows"]
                             if r.get("label_id") is not None)
            fr = sorted(counts)
            if fr:
                ax.plot(fr, [counts[f] for f in fr], color=colors[i],
                        label=s["label"], linewidth=1.2)
        if combined:
            counts = Counter(int(r.get("frame", 0)) for r in combined
                             if r.get("label_id") is not None)
            fr = sorted(counts)
            if fr:
                ax.plot(fr, [counts[f] for f in fr], color="#111111",
                        linestyle="--", linewidth=2.0, label="Combined")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Objects", fontsize=8)
        ax.legend(fontsize=6, loc="best", framealpha=0.4)
        cv.fig.tight_layout()
        cv.draw()

        # Area: overlaid step-histograms (readable with many iterations).
        all_areas = [float(r["area_px"]) for s in series for r in s["rows"]
                     if isinstance(r.get("area_px"), (int, float))]
        cv2 = self._plot_canvas("Area")
        cv2.fig.clear()
        ax2 = cv2.add_subplot(111)
        if all_areas:
            hi = float(np.quantile(all_areas, 0.99)) or max(all_areas)
            rng = (0.0, max(hi, 1.0))
            for i, s in enumerate(series):
                areas = [float(r["area_px"]) for r in s["rows"]
                         if isinstance(r.get("area_px"), (int, float))]
                if areas:
                    ax2.hist(areas, bins=40, range=rng, histtype="step",
                             color=colors[i], label=s["label"], linewidth=1.2)
            ax2.legend(fontsize=6, loc="best", framealpha=0.4)
        ax2.set_xlabel("Area (px²)", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        cv2.fig.tight_layout()
        cv2.draw()

    def _update_analysis_plots(self, rows: List[Dict[str, Any]]) -> None:
        """Cells-per-frame line + area histogram from the run's measurement rows
        (CellTracker's two Live Stats plots). During a loop run, overlays every
        iteration as its own colored, legended series (V1.49)."""
        series = getattr(self, "_loop_plot_series", None)
        if series:
            self._update_analysis_plots_loop(series, rows)
            return
        if not rows:
            return
        from collections import Counter
        counts = Counter(int(r.get("frame", 0)) for r in rows
                         if r.get("label_id") is not None)
        frames = sorted(counts)
        cv = self._plot_canvas("Cells/frame")
        cv.fig.clear()
        ax = cv.add_subplot(111)
        ax.plot(frames, [counts[f] for f in frames], color=Settings.ACCENT_CYAN)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Objects", fontsize=8)
        cv.fig.tight_layout()
        cv.draw()

        areas = [float(r["area_px"]) for r in rows
                 if isinstance(r.get("area_px"), (int, float))]
        cv2 = self._plot_canvas("Area")
        cv2.fig.clear()
        ax2 = cv2.add_subplot(111)
        if areas:
            hi = float(np.quantile(areas, 0.99)) or max(areas)
            ax2.hist(areas, bins=50, range=(0.0, max(hi, 1.0)),
                     color=Settings.ACCENT_GREEN, alpha=0.7)
        ax2.set_xlabel("Area (px²)", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        cv2.fig.tight_layout()
        cv2.draw()

    def _update_track_plots_loop(self, series: list,
                                 combined: List[Dict[str, Any]]) -> None:
        """Overlay each loop iteration's Tracks/frame line + Track-length
        step-histogram in a distinct color with a legend, plus a dashed 'Combined'
        series (V1.49.x — mirrors :meth:`_update_analysis_plots_loop`)."""
        from collections import Counter, defaultdict
        colors = self._iteration_colors(len(series))

        def _tracks_per_frame(rows):
            pf: Dict[int, set] = defaultdict(set)
            for r in rows:
                if r.get("track_id") is not None:
                    pf[int(r.get("frame", 0))].add(r.get("track_id"))
            return pf

        cv = self._plot_canvas("Tracks/frame")
        cv.fig.clear()
        ax = cv.add_subplot(111)
        drew = False
        for i, s in enumerate(series):
            pf = _tracks_per_frame(s["rows"])
            fr = sorted(pf)
            if fr:
                ax.plot(fr, [len(pf[f]) for f in fr], color=colors[i],
                        label=s["label"], linewidth=1.2)
                drew = True
        if combined:
            pf = _tracks_per_frame(combined)
            fr = sorted(pf)
            if fr:
                ax.plot(fr, [len(pf[f]) for f in fr], color="#111111",
                        linestyle="--", linewidth=2.0, label="Combined")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Tracks", fontsize=8)
        if drew:
            ax.legend(fontsize=6, loc="best", framealpha=0.4)
        cv.fig.tight_layout()
        cv.draw()

        all_lengths = [c for s in series
                       for c in Counter(r.get("track_id") for r in s["rows"]
                                        if r.get("track_id") is not None).values()]
        cv2 = self._plot_canvas("Track length")
        cv2.fig.clear()
        ax2 = cv2.add_subplot(111)
        if all_lengths:
            hi = max(all_lengths)
            rng = (0.5, hi + 0.5)
            bins = min(50, max(5, hi))
            for i, s in enumerate(series):
                lens = list(Counter(r.get("track_id") for r in s["rows"]
                                    if r.get("track_id") is not None).values())
                if lens:
                    ax2.hist(lens, bins=bins, range=rng, histtype="step",
                             color=colors[i], label=s["label"], linewidth=1.2)
            ax2.legend(fontsize=6, loc="best", framealpha=0.4)
        ax2.set_xlabel("Track length (frames)", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        cv2.fig.tight_layout()
        cv2.draw()

    def _update_track_plots(self, rows: List[Dict[str, Any]]) -> None:
        """Tracks-per-frame line + track-length histogram (CellTracker track stats).
        During a loop run, overlays every iteration as its own colored series."""
        series = getattr(self, "_loop_plot_series", None)
        if series:
            self._update_track_plots_loop(series, rows)
            return
        from collections import Counter, defaultdict
        tracked = [r for r in rows if r.get("track_id") is not None]
        if not tracked:
            return
        per_frame: Dict[int, set] = defaultdict(set)
        for r in tracked:
            per_frame[int(r.get("frame", 0))].add(r.get("track_id"))
        frames = sorted(per_frame)
        cv = self._plot_canvas("Tracks/frame")
        cv.fig.clear()
        ax = cv.add_subplot(111)
        ax.plot(frames, [len(per_frame[f]) for f in frames],
                color=Settings.ACCENT_PURPLE)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Tracks", fontsize=8)
        cv.fig.tight_layout()
        cv.draw()

        lengths = list(Counter(r.get("track_id") for r in tracked).values())
        cv2 = self._plot_canvas("Track length")
        cv2.fig.clear()
        ax2 = cv2.add_subplot(111)
        if lengths:
            ax2.hist(lengths, bins=min(50, max(5, len(set(lengths)))),
                     color=Settings.ACCENT_PINK, alpha=0.75)
        ax2.set_xlabel("Track length (frames)", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        cv2.fig.tight_layout()
        cv2.draw()

    def _update_live_cells_plot(self) -> None:
        """Refresh the live Cells/frame plot from the streamed per-frame counts."""
        if not self._live_counts:
            return
        frames = sorted(self._live_counts)
        cv = self._plot_canvas("Cells/frame")
        cv.fig.clear()
        ax = cv.add_subplot(111)
        ax.plot(frames, [self._live_counts[f] for f in frames],
                color=Settings.ACCENT_CYAN)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Cells", fontsize=8)
        cv.fig.tight_layout()
        cv.draw()

    def _update_preview_plots(self) -> None:
        """Rebuild the bottom-panel plots (Cells/frame, Area, Tracks/frame,
        Track length) from the current preview rows (V1.46).

        Preview previously drove only the overlays + measurements table; the
        matplotlib plots were Run-only. This mirrors them for the preview walk,
        scoped to the selected frames + crop. Inert during a Run (that path owns
        the plots) and when Preview is off."""
        if self._run_active or not self._btn_preview.isChecked():
            _log.debug("preview plots skipped (run_active=%s, preview_on=%s)",
                       self._run_active, self._btn_preview.isChecked())
            return
        rows = self._results_rows or []
        # A preview walk is a plain single-series redraw — drop any loop overlay.
        self._loop_plot_series = None
        # Rebuild from scratch so a removed Track-Objects node drops the track
        # plots (the builders update in place and would otherwise leave stale tabs).
        self._clear_plot_tabs()
        if not rows:
            _log.debug("preview plots: no rows to plot")
            return
        _log.info("preview plots: building from %d row(s)", len(rows))
        self._update_analysis_plots(rows)   # Cells/frame + Area
        self._update_track_plots(rows)      # Tracks/frame + Track length (if tracked)

    def _on_run_frame(self, key: str, m: int, t: int, labels) -> None:
        """Live per-frame stream from the analysis worker: paint the overlay on the
        just-finished frame and extend the Cells/frame plot (CellTracker's live
        per-frame overlay + Live Stats)."""
        if not self._run_active or key != _RUN_ANALYSIS_KEY:
            return
        arr = np.asarray(labels)
        self._live_seg[(int(m), int(t))] = arr
        self._live_counts[int(t)] = int(arr.max())
        self._update_live_cells_plot()
        # Navigate to + repaint the finished frame so its overlay appears live.
        self.viewer.invalidate_post_process_cache()
        self.viewer.set_current_frame(m=int(m), t=int(t))

    def _set_results_view_mode(self, mode: str) -> None:
        if mode not in ("image", "table", "split"):
            mode = "split"
        self._results_view_mode = mode
        btn = self._mode_btns.get(mode)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._set_panel_visible("viewer", mode in ("image", "split"))
        self._set_panel_visible("plots", mode in ("table", "split"))

    def _inject_result_choices(self, specs, names: List[str]) -> None:
        """Populate the ``analysis_result`` choice with committed-result names."""
        for spec in specs:
            if spec.name == "analysis_result" and names:
                spec.choices = list(names)
                if spec.default not in names:
                    spec.default = names[0]

    # ── runner callbacks ──────────────────────────────────────────────────────
    def _on_runner_done(self, result: JobResult) -> None:
        if result.key == _PREVIEW_KEY:
            # Processing preview: the job processed the selected plane(s). Show
            # them on the raw, fully-navigable volume so ONLY those (M, T) planes
            # are recipe-processed and every other frame stays raw.
            self._set_preview_progress(visible=False)
            # A processing-preview result must only touch the viewer while the
            # Processing tab is active. If the user Applied then switched to the
            # Analysis tab before this (debounced, async) job finished, applying
            # it here would overwrite the Analysis base image — which
            # ``_show_base_image`` set to a fully-processed ``ProcessedFrameVolume``
            # — with the processing-preview ``PinnedProcessedVolume`` (processed
            # only on the pinned plane, raw everywhere else). That is the
            # "Analysis viewer reverts to raw after Apply + tab switch, especially
            # under a preview crop" bug. Drop the stale result.
            if self._stage is not Stage.PROCESSING:
                return
            if not result.ok or not result.value:
                self._set_status("Preview error")
                return
            record = self._active_record()
            cd = record.channel_display if record is not None else {}
            # result.value: {(m, t): {name: (1, H, W)}} -> {(m, t): {name: (H, W)}}
            planes = {}
            for mt, chans in result.value.items():
                plane = {}
                for name, arr in chans.items():
                    a = np.asarray(arr)
                    plane[name] = a[0] if a.ndim == 3 and a.shape[0] else a
                planes[mt] = plane
            vol = getattr(record, "_raw_volume", None) if record is not None else None
            if vol is not None and planes:
                if self._proc_volume is not None and self.viewer._volume is self._proc_volume:
                    # Move the processed plane set in place + re-render — keeps the
                    # frame selection, LUTs and slider positions intact (a fresh
                    # set_volume would reset them).
                    self._proc_volume.set_planes(planes)
                    self.viewer.refresh()
                else:
                    # Pin the processed (already crop-sized) planes over a base
                    # volume matching the same geometry — cropped when a preview
                    # crop is active so raw (non-pinned) frames line up.
                    self._proc_volume = PinnedProcessedVolume(
                        self._maybe_crop_volume(vol), planes)
                    self._show_preview_volume(self._proc_volume)
            else:
                # Small in-RAM file (no lazy volume): show the single processed
                # plane directly.
                first = next(iter(result.value.values()), {})
                self.viewer.set_channels(first, channel_display=cd,
                                         n_multipoints=1, m=0)
                self.viewer.set_frame_post_process(None)
                self.viewer.invalidate_post_process_cache()
        elif result.key == _ANALYSIS_PREVIEW_KEY:
            self._set_preview_progress(visible=False)
            if not result.ok:
                # Surface the real reason (e.g. a pipeline error on the cropped
                # region) instead of a generic message, and log it — a silent
                # "preview ran but nothing appeared" is otherwise a dead end.
                msg = f"Analysis preview failed: {result.error or 'unknown error'}"
                _log.warning(msg)
                self._set_status(msg)
                return
            if not result.value:
                self._set_status(
                    "Analysis preview produced no result on the previewed "
                    "frame(s)/crop — nothing to overlay.")
                return
            # {(m, t): AnalysisResult} — one per previewed/selected plane.
            self._analysis_screen_results = dict(result.value)
            # The preview reset cleared prior results, which hides the Segmentation
            # tab and can drop the viewer to the (overlay-less) Image tab. Now that
            # a result exists, re-enable the tabs and show the segmentation overlay
            # if the viewer fell back to Image (mirrors the Run path).
            self._update_overlay_tabs_available()
            if self._overlay_mode == "image":
                self._select_overlay_tab("segmentation")
            # Re-render so the overlay actually appears: invalidate alone clears
            # the cache but does not repaint, so the auto-refresh after a screen
            # (navigation / param edit / node change) would otherwise not show.
            self.viewer.refresh()
        elif result.key == _ANALYSIS_COMMIT_KEY:
            self._set_preview_progress(visible=False)
            if not result.ok or not result.value:
                self._set_status(result.error or "Analysis error")
                return
            record = self._active_record()
            name = self._analysis_commit_pipeline
            m = self._analysis_commit_m
            if record is not None and name:
                store = record.analysis_results.setdefault(name, {})
                store[m] = result.value
            self._analysis_results_per_m[m] = result.value
            self.viewer.invalidate_post_process_cache()
            self._set_status(f"Applied '{name}' → analysis result (M{m + 1})")
            # Refresh the single-frame overlay so the current frame reflects the
            # just-applied result (the committed full stack is reviewed in
            # Results). Harmless no-op if Preview is off.
            self._request_preview()
        elif result.key == _PV_SCREEN_KEY:
            # Preview walk: the analysis/results portion finished. Show the
            # per-plane overlay + aggregated table, mark those nodes done, then
            # walk the remaining logic/special nodes up to the previewed node.
            self._set_preview_progress(visible=False)
            if not result.ok or not isinstance(result.value, dict):
                self._set_status("Measurement error")
                return
            self._analysis_screen_results = dict(result.value.get("results") or {})
            self._results_rows = result.value.get("rows") or []
            _log.info(
                "Preview screen+measure done: %d plane result(s), %d row(s)",
                len(self._analysis_screen_results), len(self._results_rows))
            # Reset the track-overlay scratch; the downstream walk rebuilds it if a
            # Track Objects node runs, so a graph *without* tracking clears any
            # stale track tabs from a previous preview.
            self._track_colormap = None
            self._track_overlay_rows = []
            self._populate_results_table(self._results_rows)
            # As with the analysis preview, the reset can drop the viewer to the
            # Image tab; restore the segmentation overlay now results exist.
            self._update_overlay_tabs_available()
            if self._overlay_mode == "image" and self._analysis_screen_results:
                self._select_overlay_tab("segmentation")
            self.viewer.refresh()
            scene = self._scenes.get(Stage.ANALYSIS)
            if scene is not None:
                for nid in self._pv_screen_set:
                    self._pv_states[nid] = "done"
                scene.set_run_states(self._pv_states)
            self._preview_walk_downstream()
            # Build the bottom-panel plots from the previewed rows (the downstream
            # walk may have added track_id / dismissed rows first), scoped to the
            # selected frames + crop — previously these were Run-only.
            self._update_preview_plots()
            # Reveal the Segmentation / Spatial overlay tabs now that results
            # exist — the async preview result lands *after* the node-promotion
            # tab refresh, so without this the tabs stayed hidden (the Tracks /
            # Vectors tabs are handled by the walk's _set_track_overlay_state).
            self._update_overlay_tabs_available()
        elif result.key == _RUN_ANALYSIS_KEY:
            # Run: one multipoint's analysis finished — store it, then measure that
            # M on a worker (chained), so the if-else / specials see whole-file data.
            if not self._run_active:
                return
            m = self._run_current_m
            ctx = self._run_analysis_ctx or {}
            name = ctx.get("name") or self._analysis_commit_pipeline
            if result.ok and result.value is not None:
                record = self._active_record()
                if record is not None and name:
                    record.analysis_results.setdefault(name, {})[m] = result.value
                self._analysis_results_per_m[m] = result.value
                self._run_results_by_m[m] = result.value
                metadata = dict(ctx.get("metadata") or {})
                # Measure on THIS M's processed channels (the same data the
                # analysis just ran on), not the single-M processed_view() (V1.48).
                src = self._run_channels_m
                if not src:
                    src = record.processed_view() if record is not None else {}
                job = _ResultsMeasureJob(
                    _RUN_MEASURE_KEY, result.value.label_masks, src, metadata, m,
                    getattr(result.value, "volumetric_voxel_counts", None))
                self._runner.submit(job)
            else:
                # Analysis failed for this M — surface why (was silent), then skip.
                msg = (f"Run: '{name or 'analysis'}' failed on "
                       f"M{self._run_current_m + 1}: {result.error}")
                _log.warning(msg)
                self._set_status(msg)
                self._advance_run_m()
        elif result.key == _RUN_MEASURE_KEY:
            # Run: one multipoint's measurements finished — tag with m_position,
            # then track within that M *on a worker* (the linker is O(n) heavy and
            # would freeze the GUI here), accumulating + advancing once it returns.
            if not self._run_active:
                return
            rows = (result.value or []) if result.ok else []
            if rows:
                for r in rows:
                    r["m_position"] = self._run_current_m
                self._run_pending_track_rows = rows
                self._runner.submit(_TrackJob(_RUN_TRACK_KEY, rows))
                return  # accumulate + advance happen on the track result
            self._advance_run_m()
        elif result.key == _RUN_TRACK_KEY:
            # Run: per-M default tracking finished (off-thread) — accumulate the now
            # track-tagged rows and advance. On failure the rows survive untracked
            # (link_objects pre-initialises track_id=None), so keep them either way.
            if not self._run_active:
                return
            rows = (result.value if result.ok else None) or self._run_pending_track_rows or []
            self._run_pending_track_rows = None
            if rows:
                self._run_all_rows.extend(rows)
            self._advance_run_m()
        elif result.key == _RUN_TRACKOBJ_KEY:
            # Run: the Track Objects node's linker finished (off-thread).
            if not self._run_active:
                return
            self._finish_track_objects(result)
        elif result.key == _RUN_DVC_KEY:
            # Run: the DVC node's ALDVC engine finished (off-thread).
            if not self._run_active:
                return
            self._finish_dvc(result)
        elif result.key == _RUN_REGISTER_KEY:
            # Run: the Registration node finished (off-thread).
            if not self._run_active:
                return
            self._finish_register(result)
        elif result.key == _RUN_LOOPCOMBINE_KEY:
            # Run: the loop's off-thread union-dedup combine finished.
            if not self._run_active:
                return
            self._set_preview_progress(visible=False)
            if result.ok and result.value is not None:
                rows, merged = result.value
                self._loop_finalize_union(rows, merged)
            else:
                # Combine failed — fall back to the last iteration so the Run still
                # produces a result rather than silently stalling.
                msg = f"Loop combine failed: {result.error}"
                _log.warning(msg)
                self._set_status(msg)
                ctx = getattr(self, "_loop_ctx", None)
                its = (ctx.get("iterations") if ctx else None) or []
                if its:
                    self._run_all_rows = list(its[-1]["rows"])
                    self._run_results_by_m = dict(its[-1]["results_by_m"])
                    self._results_rows = self._run_all_rows
                self._loop_finish_cleanup()
            self._finalize_analysis_publish(self._run_analysis_ctx or {})
        elif result.key == _RUN_RESULTS_KEY:
            # Legacy single-M measurement path (retained for safety).
            self._set_preview_progress(visible=False)
            if not self._run_active:
                return
            rows = (result.value or []) if result.ok else []
            if rows:
                try:
                    link_objects(rows)
                except Exception:  # noqa: BLE001
                    pass
            self._results_rows = rows
            self._run_context["rows"] = rows
            self._populate_results_table(rows)
            self._run_finish_node(self._run_pending)

    def _on_runner_cancelled(self, key: str) -> None:
        # A superseded preview — a newer one is already in flight (its own
        # submit re-showed the bar), so leave the progress bar as-is. A
        # cancelled Run job (e.g. file changed mid-run) aborts the walk cleanly.
        if key in (_RUN_ANALYSIS_KEY, _RUN_MEASURE_KEY, _RUN_TRACK_KEY,
                   _RUN_TRACKOBJ_KEY, _RUN_DVC_KEY, _RUN_REGISTER_KEY,
                   _RUN_RESULTS_KEY, _RUN_LOOPCOMBINE_KEY) and self._run_active:
            if getattr(self, "_loop_ctx", None) is not None:
                self._loop_finish_cleanup()  # V1.49: restore swept params
            self._run_active = False
            self._run_paused = False
            self._runner_obj = None
            self._run_pending = ""
            self._run_analysis_ctx = None
            self._run_trackobj_node = None
            self._run_dvc_node = None
            self._run_register_node = None
            self._run_pending_track_rows = None
            self._run_states = {}
            scene = self._scenes.get(Stage.ANALYSIS)
            if scene is not None:
                scene.clear_run_states()
            self._btn_run.setEnabled(self._stage is Stage.ANALYSIS)
            self._set_preview_progress(visible=False)

    def _on_runner_progress(self, key: str, fraction: float, _message: str) -> None:
        if key in (_PREVIEW_KEY, _ANALYSIS_PREVIEW_KEY, _ANALYSIS_COMMIT_KEY,
                   _RESULTS_PREVIEW_KEY, _PV_SCREEN_KEY, _RUN_ANALYSIS_KEY,
                   _RUN_MEASURE_KEY, _RUN_TRACK_KEY, _RUN_TRACKOBJ_KEY,
                   _RUN_DVC_KEY, _RUN_REGISTER_KEY, _RUN_RESULTS_KEY):
            self._set_preview_progress(visible=True, value=int(fraction * 100))

    def _set_preview_progress(self, *, visible: bool, value: int = 0) -> None:
        bar = getattr(self, "_preview_progress", None)
        if bar is None:
            return
        if visible:
            bar.setValue(max(0, min(100, value)))
        bar.setVisible(visible)

    # ── apply ────────────────────────────────────────────────────────────────
    def _on_apply(self) -> None:
        # Apply only commits a Processing recipe now; the merged Analysis tab
        # executes via Run (the Apply button is hidden there).
        if self._stage is Stage.PROCESSING:
            self._apply_processing()

    # ── Run (merged Analysis tab) ───────────────────────────────────────────
    def _on_run_button(self) -> None:
        """Run-button click. With no preview crop it starts a full-file Run. With
        a crop set it drops a menu to choose full vs cropped."""
        if self._preview_crop is not None and not self._run_active:
            self._show_run_menu()
        else:
            self._start_run(cropped=False)

    def _show_run_menu(self) -> None:
        """Dropdown offering full-file vs cropped-region Run (crop is set)."""
        menu = QMenu(self)
        act_full = menu.addAction("Run full (uncropped) file")
        rect = self._preview_crop
        label = "Run cropped region"
        if rect is not None:
            label += f"  ({rect[2]}×{rect[3]} @ {rect[0]},{rect[1]})"
        act_crop = menu.addAction(label)
        act_full.triggered.connect(lambda: self._start_run(cropped=False))
        act_crop.triggered.connect(lambda: self._start_run(cropped=True))
        menu.exec(self._btn_run.mapToGlobal(self._btn_run.rect().bottomLeft()))

    def _start_run(self, cropped: bool) -> None:
        """Launch a Run, optionally scoped to the preview crop."""
        self._run_cropped = bool(cropped and self._preview_crop is not None)
        self._on_run()

    def _update_run_button(self) -> None:
        """Add a ▾ affordance + tooltip when a crop makes Run a full/cropped
        choice; plain 'Run' otherwise."""
        btn = getattr(self, "_btn_run", None)
        if btn is None:
            return
        if self._preview_crop is not None:
            btn.setText(" Run ▾")
            btn.setToolTip("Run the pipeline — choose full file or the cropped "
                           "region (a preview crop is active).")
        else:
            btn.setText(" Run")
            btn.setToolTip("Run the pipeline (executes the graph)")

    def _on_run(self) -> None:
        """Execute the merged graph. A :class:`GraphRunner` walks it in topo
        order; the page paints each node shaded (pending / un-taken branch),
        gold (executing), then normal (done), branching at if-else nodes and
        parking on async compute. Pause returns to editor mode."""
        if self._stage is not Stage.ANALYSIS or self._run_active:
            return
        record = self._active_record()
        if record is None or not record._raw_channels:
            QMessageBox.information(self, "Run", "Import a file first.")
            return
        sl = self._doc.analysis
        if input_node(sl) is None or not sl.nodes:
            QMessageBox.information(self, "Run", "Add nodes and wire them first.")
            return
        # Resume a paused run from where it left off — but only if the graph's
        # node set is unchanged; otherwise restart cleanly.
        if (self._run_paused and self._runner_obj is not None
                and set(sl.nodes) == self._run_paused_nodes):
            self._run_paused = False
            self._run_active = True
            reach = self._runner_obj.reachable_nodes()
            done = self._runner_obj.done
            self._run_states = {nid: ("done" if nid in done else "shaded")
                                for nid in reach}
            self._scenes[Stage.ANALYSIS].set_run_states(self._run_states)
            self._btn_run.setEnabled(False)
            self._set_status("Resuming run…")
            self._run_pump()
            return
        self._run_paused = False
        self._loop_ctx = None  # V1.49: no loop in flight at run start
        self._loop_plot_series = None  # per-iteration plot overlay (set at combine)
        self._loop_saved = None        # saved iterations for the viewer dropdown
        # V1.56: drop any prior run's registration transforms so this Run only
        # applies drift correction if a Registration node runs in it again.
        # V1.60: also drop the common-region crop it published.
        self._reg_by_m = {}
        if record is not None:
            record._registration_by_m = {}
            record._registration_crop = None
        self._populate_iteration_selector()  # hide the combo for a fresh run
        self._run_pending = ""
        # Per-branch row scoping for object-lens if-else: rows carried on each
        # output port, and the set of ports holding a branch subset.
        self._run_port_rows = {}
        self._run_scoped_ports = set()
        # V1.53: resume from the deepest valid checkpoint — its frozen upstream
        # work (segmentation / tracking) is skipped and its data restored, so only
        # the downstream pipeline runs. Falls back to a normal full Run when no
        # checkpoint is valid.
        ckpt_id = self._checkpoint_resume_target()
        if ckpt_id is not None:
            frozen = self._checkpoint_ancestors(ckpt_id)
            self._runner_obj = GraphRunner(sl, frozen=frozen)
            self._run_active = True
            self._run_context = {"rows": [], "result": None}
            self._restore_checkpoint(ckpt_id)
            reachable = self._runner_obj.reachable_nodes()
            self._run_states = {
                nid: ("cached" if nid in frozen else "shaded")
                for nid in reachable | frozen
            }
            self._scenes[Stage.ANALYSIS].set_run_states(self._run_states)
            self._btn_run.setEnabled(False)
            title = sl.nodes[ckpt_id].title if ckpt_id in sl.nodes else "checkpoint"
            self._set_status(
                f"Resuming from '{title}' — upstream frozen, running downstream…")
            self._run_pump()
            return
        self._runner_obj = GraphRunner(sl)
        self._run_active = True
        # Record the geometry this Run's masks will be computed at, so overlays
        # know whether the committed masks match the current display (full vs crop).
        self._run_results_crop = self._crop_rect()
        self._run_context = {"rows": list(self._results_rows or []), "result": None}
        self._run_states = {nid: "shaded" for nid in self._runner_obj.reachable_nodes()}
        self._scenes[Stage.ANALYSIS].set_run_states(self._run_states)
        self._btn_run.setEnabled(False)
        self._set_status("Running pipeline…")
        self._run_pump()

    def _run_pump(self) -> None:
        """Advance to the next ready node; dispatch it. Async nodes re-enter via
        their job-done handler; nothing ready + finished ends the run."""
        if not self._run_active or self._runner_obj is None:
            return
        nid = self._runner_obj.next_ready()
        if nid is None:
            if self._run_pending == "" and self._runner_obj.is_finished():
                self._finish_run()
            return  # else: waiting on an in-flight async node
        self._run_states[nid] = "current"
        self._scenes[Stage.ANALYSIS].set_run_states(self._run_states)
        node = self._doc.analysis.nodes.get(nid)
        if node is None:
            self._run_finish_node(nid)
            return
        self._apply_branch_scope(node)
        self._run_execute_node(node)

    def _apply_branch_scope(self, node) -> None:
        """Scope the active rows to whichever branch feeds ``node``.

        For nodes downstream of an object-lens if-else split, the node's live
        incoming edges carry a row subset on their source ports; set
        ``_run_context["rows"]`` to the union of those (dedup by identity) so the
        node operates only on its branch's objects. Sets ``scoped_now`` so
        ``_ensure_run_rows`` won't re-derive the full set for an empty branch."""
        self._run_context["scoped_now"] = False
        if self._runner_obj is None:
            return
        parts = [self._run_port_rows[e.src_port]
                 for e in self._runner_obj.active_incoming(node.id)
                 if e.src_port in self._run_scoped_ports
                 and e.src_port in self._run_port_rows]
        if not parts:
            return
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for rows in parts:
            for r in rows:
                if id(r) not in seen:
                    seen.add(id(r))
                    merged.append(r)
        self._run_context["rows"] = merged
        self._run_context["scoped_now"] = True

    def _run_finish_node(self, nid: str, prune_ports=None, port_rows=None) -> None:
        if not self._run_active or self._runner_obj is None:
            return
        self._run_states[nid] = "done"
        # Record each output port's rows so a downstream object-lens branch can
        # scope to them. ``port_rows`` (from an object-lens if-else) overrides per
        # port; otherwise every output carries the node's current rows. If this
        # node ran on a scoped branch, its outputs stay scoped too (propagation).
        node = self._doc.analysis.nodes.get(nid)
        if node is not None:
            explicit = port_rows or {}
            cur = self._run_context.get("rows")
            inherited_scope = bool(self._run_context.get("scoped_now"))
            for p in node.outputs:
                self._run_port_rows[p.id] = explicit.get(p.id, cur)
                if p.id in explicit or inherited_scope:
                    self._run_scoped_ports.add(p.id)
        self._runner_obj.complete(nid, prune_ports=prune_ports)
        self._scenes[Stage.ANALYSIS].set_run_states(self._run_states)
        self._run_pending = ""
        QTimer.singleShot(0, self._run_pump)  # yield so the repaint lands

    def _run_execute_node(self, node) -> None:
        op = node.op_key
        if node.role is NodeRole.INPUT or node.role is NodeRole.OUTPUT:
            self._run_finish_node(node.id)
        elif op == IF_ELSE_OP_KEY:
            cond = Condition.from_dict(node.params.get("condition"))
            true_port = next((p.id for p in node.outputs if p.name == "true"), None)
            false_port = next((p.id for p in node.outputs if p.name == "false"), None)
            rows = self._ensure_run_rows()
            if str(node.params.get("lens")) == LENS_OBJECT:
                # Object lens: split the objects — passing → true branch, failing
                # → false branch. Both branches run; each downstream node is later
                # scoped to its branch's rows (see _apply_branch_scope).
                group_by = str(node.params.get("group_by") or "")
                group_by = GROUP_ROW if group_by == GROUP_ROW else "track"
                keep, drop = partition_rows(cond, rows, group_by=group_by)
                self._set_status(
                    f"{node.title}: {len(keep)} object(s) → TRUE, "
                    f"{len(drop)} → FALSE")
                self._run_finish_node(
                    node.id, prune_ports=set(),
                    port_rows={p: r for p, r in
                               ((true_port, keep), (false_port, drop))
                               if p is not None})
            else:
                take_true = evaluate_condition(cond, rows)
                prune = {false_port if take_true else true_port}
                prune.discard(None)
                self._set_status(
                    f"{node.title}: {'TRUE' if take_true else 'FALSE'} branch")
                self._run_finish_node(node.id, prune_ports=prune)
        elif op == SPECIAL_PAUSE_OP_KEY:
            self._pause_run(node)
        elif op == SPECIAL_CHECKPOINT_OP_KEY:
            self._run_checkpoint(node)
        elif op.startswith("analysis:"):
            self._run_analysis_node(node)
        elif op.startswith("results:"):
            self._run_results_node(node)
        elif op == SPECIAL_TRACK_OP_KEY:
            self._run_track_objects(node)
        elif op == SPECIAL_VALIDATE_OP_KEY:
            self._run_open_validation(node, "Validate Tracked Objects")
        elif op == SPECIAL_REVIEW_OP_KEY:
            self._run_review_objects(node)
        elif op == SPECIAL_DISMISS_OP_KEY:
            self._run_dismiss(node)
        elif op == SPECIAL_EXPORT_OP_KEY:
            self._run_export(node)
        elif op == SPECIAL_CT_METRICS_OP_KEY:
            self._run_ct_metrics(node)
        elif op in (SPECIAL_CT_FIELDS_OP_KEY, SPECIAL_INTERP_MAP_OP_KEY):
            self._open_spatial_maps_tab(node)
            self._run_finish_node(node.id)
        elif op == SPECIAL_DVC_OP_KEY:
            self._run_dvc(node)
        elif op == SPECIAL_REGISTER_OP_KEY:
            self._run_register(node)
        elif op == SPECIAL_SEND_RESULTS_OP_KEY:
            self._run_send_results(node)
        else:
            self._run_finish_node(node.id)

    def _skip_analysis_node(self, node, reason: str) -> None:
        """Complete an analysis node without running it, but say **why**.

        Every precondition failure in :meth:`_run_analysis_node` used to bail
        silently via ``_run_finish_node`` — the node went shaded→done and the Run
        produced nothing, with no clue in the UI or logs. Route them here so the
        reason lands in the status bar and the log (a Run must never silently do
        nothing)."""
        msg = f"Run: '{node.title}' skipped — {reason}"
        _log.warning(msg)
        self._set_status(msg)
        self._run_finish_node(node.id)

    def _run_analysis_node(self, node) -> None:
        """Run the analysis pipeline over the WHOLE file — every multipoint, not
        just the current frame. Sets up the per-M loop; ``_advance_run_m`` submits
        one analysis job per M (each followed by a measurement job), accumulating
        per-M results + rows before the node completes."""
        record = self._active_record()
        action = self._resolve_analysis_action(node.id)
        names = self._current_channel_names()
        if record is None or action is None or not names:
            reason = ("no file imported" if record is None
                      else "node has no analysis action"
                      if action is None else "no channels available")
            self._skip_analysis_node(node, reason)
            return
        name = analysis_pipeline_name_for_op_key(action.op_key)
        cls = AnalysisPipeline.get_pipeline(name)
        if cls is None:
            self._skip_analysis_node(
                node, f"analysis pipeline {name!r} is not registered")
            return
        # V1.49: if this analysis node is the entry of a loop region, begin the
        # iteration loop (sets up _loop_ctx + applies iteration 0's param
        # overrides). Re-entry for later iterations finds _loop_ctx already set and
        # skips this, so the node runs with the overridden params.
        if getattr(self, "_loop_ctx", None) is None:
            self._loop_maybe_begin(node)
        # V1.48: the channel(s) this analysis node runs on come from its wired
        # rainbow ports (channel_sets), replacing the old channel_name param; a
        # 2nd wired channel counterstains a counterstain-capable pipeline.
        extra, seg_channels = self._analysis_run_spec(node, names)
        if not seg_channels:
            self._skip_analysis_node(node, "no channel wired to this analysis node")
            return
        params = dict(action.params)
        params.update(extra)
        # Channels are read + processed PER M inside ``_advance_run_m``
        # (``_processed_channels_for_m``) rather than captured once here from the
        # single-M ``processed_view()`` — the old capture made every M analyze
        # M0's pixels.
        metadata = self._preview_metadata(record)
        self._analysis_commit_pipeline = name
        self._run_analysis_ctx = {
            "node_id": node.id, "cls": cls, "name": name, "params": params,
            "metadata": metadata, "record": record, "seg_channels": seg_channels,
        }
        self._run_m_total = max(1, self._record_n_multipoints(record))
        self._run_m_queue = list(range(self._run_m_total))
        # V1.49: a loop scoped to the current multipoint restricts every iteration
        # to the viewed M (sweeps × all-M can be very expensive).
        lctx = getattr(self, "_loop_ctx", None)
        if lctx is not None and lctx.get("multipoint") == "current":
            cur_m, _, _ = self.viewer.coords()
            cur_m = max(0, min(int(cur_m), self._run_m_total - 1))
            self._run_m_queue = [cur_m]
        self._run_results_by_m = {}
        self._run_all_rows = []
        # Live streaming: reset per-frame buffers + Cells/frame plot, and show the
        # Segmentation overlay so frames light up as they're computed.
        self._live_seg = {}
        self._live_counts = {}
        self._track_colormap = None       # a fresh run hasn't tracked yet
        self._track_long_ids = set()
        self._track_overlay_rows = []
        self._clear_plot_tabs()
        self._select_overlay_tab("segmentation")
        self._update_overlay_tabs_available()
        self._run_pending = node.id
        self._advance_run_m()

    def _advance_run_m(self) -> None:
        """Submit the next multipoint's analysis job, or finalize the node when
        every M has been processed."""
        if not self._run_active or self._runner_obj is None:
            return
        ctx = self._run_analysis_ctx or {}
        if not self._run_m_queue:
            node_id = ctx.get("node_id", "")
            # V1.49: this iteration of a loop finished — record it, then either
            # start the next iteration (return), kick off the (heavy) combine on a
            # worker (return; publish happens on its job-done), or, for the cheap
            # rules, combine synchronously and fall through to publish.
            lctx = getattr(self, "_loop_ctx", None)
            if lctx is not None and lctx.get("node_id") == node_id:
                self._loop_record_iteration()
                if self._loop_should_continue():
                    self._loop_start_next_iteration()
                    return
                if self._loop_begin_combine():  # async union-dedup started
                    return
            # All M done — publish aggregated results/rows and finish the node.
            self._finalize_analysis_publish(ctx)
            return
        m = self._run_m_queue.pop(0)
        self._run_current_m = m
        params = dict(ctx["params"])
        self._inject_label_streaming(params, ctx["record"], ctx["cls"], m)
        # Read + process THIS multipoint's channels (V1.48 fix): previously every
        # M reused the single-M ``processed_view()`` so all M analyzed M0's pixels.
        # V1.49: within a loop the channels for a given M don't change between
        # iterations (only the node's params sweep), so a current-M loop caches the
        # processed channels once instead of re-materializing + re-applying the
        # recipe every iteration. (All-M loops skip the cache to bound RAM.)
        lctx = getattr(self, "_loop_ctx", None)
        chan_cache = lctx.get("chan_cache") if lctx is not None else None
        channels = chan_cache.get(m) if chan_cache is not None else None
        if channels is None:
            try:
                channels = self._processed_channels_for_m(ctx["record"], m)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Run: could not read channels for M%d: %s", m, exc)
                channels = {}
            if (chan_cache is not None and channels
                    and lctx.get("multipoint") == "current"):
                chan_cache[m] = channels
        self._run_channels_m = channels
        if not channels:
            self._set_status(f"Run: no channels for M{m + 1} — skipping.")
            self._advance_run_m()
            return
        done = self._run_m_total - len(self._run_m_queue) - 1
        self._set_preview_progress(
            visible=True, value=int(done / max(1, self._run_m_total) * 100))
        seg_channels = ctx.get("seg_channels") or list(channels.keys())
        self._set_status(
            f"Run: '{ctx['name']}' M{m + 1}/{self._run_m_total} "
            f"on {', '.join(seg_channels)}…")
        job = _MultiChannelCommitJob(
            _RUN_ANALYSIS_KEY, ctx["cls"], channels, ctx["metadata"], params, m,
            seg_channels,
        )
        self._runner.submit(job)

    def _finalize_analysis_publish(self, ctx: dict) -> None:
        """Publish the finished analysis node's aggregated results (table, plots,
        overlay) and complete the node. Shared by the normal all-M finish and the
        loop's combined finish (which may arrive asynchronously)."""
        results = self._run_results_by_m
        cur_m, _, _ = self.viewer.coords()
        self._run_context["results_by_m"] = results
        self._run_context["result"] = (
            results.get(cur_m) or next(iter(results.values()), None))
        self._run_context["rows"] = self._run_all_rows
        self._results_rows = self._run_all_rows
        self._populate_results_table(self._run_all_rows)
        # Live frames done — drop them so the committed per-M result (with its
        # real overlay style) takes over on every frame.
        self._live_seg = {}
        self._update_analysis_plots(self._run_all_rows)
        # Full per-M masks now exist for every frame — re-apply the overlay hook +
        # drop stale (preview-empty) overlay tiles so navigating any T repaints
        # from the committed result (not just the Run-start frame).
        self._update_merged_view_mode()
        self._set_preview_progress(visible=False)
        node_id = ctx.get("node_id", "")
        self._run_analysis_ctx = None
        self._run_channels_m = None  # free the last M's channels
        self._set_status(
            f"Run: '{ctx.get('name','analysis')}' done on "
            f"{self._run_m_total} multipoint(s) → {len(self._run_all_rows)} objects.")
        self._populate_iteration_selector()  # V1.49.x: show saved-iteration combo
        self._run_finish_node(node_id)

    def _run_results_node(self, node) -> None:
        """Explicit Compute Measurements node. The analysis loop already measured
        every M, so just ensure the aggregated rows exist and continue."""
        self._ensure_run_rows()
        self._run_finish_node(node.id)

    # ── loop / iteration driver (V1.49) ────────────────────────────────────
    # A loop edge whose entry is an analysis node re-runs that node's per-M
    # computation across an iteration plan (parameter sweep / count / until),
    # accumulating each iteration's rows + label masks, then combines them (union
    # + overlap-dedup / best / last / keep-all) and publishes the combined result
    # so the rest of the pipeline runs once on the comprehensive result.
    # ── loop-entry registry (V1.49.x) ───────────────────────────────────────
    # Which nodes can anchor an iteration loop, and how the driver re-runs each.
    # To wire a NEW node into the loop, add its op_key → kind here, map the kind to
    # its run method in ``_LOOP_RUN_BY_KIND``, call ``_loop_maybe_begin(node)`` at
    # the top of the node's run handler, and route its finish through
    # ``_loop_step`` (see docs/DEVELOPING_LOOP_NODES.md).
    #   op_key match → kind
    _LOOP_KIND_BY_OP = {
        SPECIAL_TRACK_OP_KEY: "track",
        SPECIAL_DVC_OP_KEY: "dvc",
        SPECIAL_REGISTER_OP_KEY: "register",
        SPECIAL_CT_METRICS_OP_KEY: "ct_metrics",
    }
    #   kind → run-handler method name (re-run for the next iteration)
    _LOOP_RUN_BY_KIND = {
        "analysis": "_run_analysis_node",
        "track": "_run_track_objects",
        "dvc": "_run_dvc",
        "register": "_run_register",
        "ct_metrics": "_run_ct_metrics",
    }
    #   kinds whose per-iteration product is measurement rows / label masks — these
    #   drive the combine rules, the per-iteration plots and the saved-iteration
    #   viewer selector. Other kinds (dvc / register) sweep params and keep the
    #   final result in their own tab (combine falls back to "last").
    _LOOP_ROW_KINDS = {"analysis", "track", "ct_metrics"}

    def _loop_entry_kind(self, node) -> str:
        """Which loop executor drives an entry node (e.g. ``"analysis"`` /
        ``"track"`` / ``"dvc"`` / ``"register"`` / ``"ct_metrics"``), or ``""``
        when the node type can't anchor a loop."""
        op = getattr(node, "op_key", "") or ""
        if op.startswith("analysis:"):
            return "analysis"
        return self._LOOP_KIND_BY_OP.get(op, "")

    def _loop_maybe_begin(self, node) -> None:
        """Start a loop if ``node`` is the entry of a loop region (else no-op).

        Only analysis and Track-Objects entries are executable in V1.49.x — a loop
        anchored on any other node runs the node once with a status note (rather
        than silently doing nothing)."""
        from nd2studios.pipeline_graph.conditions import Condition
        from nd2studios.pipeline_graph.loop import (
            iteration_plan, loop_entry_map, loop_region,
        )
        sl = self._doc.analysis
        edge = loop_entry_map(sl).get(node.id)
        if edge is None:
            return
        kind = self._loop_entry_kind(node)
        if not kind:
            self._set_status(
                f"Loop on '{node.title}' isn't supported (loopable nodes: Analysis, "
                "Track Objects, Cell-Tracker Metrics, DVC, Registration) — "
                "running once.")
            return
        cfg = dict(edge.params or {})
        region = loop_region(sl, edge)
        plan = iteration_plan(cfg)
        # Keep only overrides that target nodes inside the loop body.
        plan = [{nid: ov for nid, ov in a.items() if nid in region.body}
                for a in plan]
        if not plan:
            self._set_status("Loop: no iterations configured — running once.")
            return
        stop_cfg = cfg.get("stop")
        stop_cond = Condition.from_dict(stop_cfg) if stop_cfg else None
        orig = {nid: dict(sl.nodes[nid].params)
                for nid in region.body if nid in sl.nodes}
        self._loop_ctx = {
            "edge_id": edge.id, "node_id": node.id, "region": region,
            "config": cfg, "plan": plan, "index": 0, "iterations": [],
            "orig_params": orig, "stop": stop_cond, "kind": kind,
            "multipoint": cfg.get("multipoint", "current"),
            "chan_cache": {},  # processed channels reused across iterations
        }
        self._loop_apply_overrides(plan[0])
        self._set_status(
            f"Loop: {len(plan)} iteration(s) over {len(region.body)} node(s)…")

    def _loop_apply_overrides(self, assignment) -> None:
        sl = self._doc.analysis
        for nid, ov in (assignment or {}).items():
            node = sl.nodes.get(nid)
            if node is None:
                continue
            for k, v in (ov or {}).items():
                node.params[k] = v

    def _loop_restore_params(self) -> None:
        ctx = getattr(self, "_loop_ctx", None)
        if not ctx:
            return
        sl = self._doc.analysis
        for nid, params in (ctx.get("orig_params") or {}).items():
            node = sl.nodes.get(nid)
            if node is not None:
                node.params = dict(params)

    def _loop_record_iteration(self) -> None:
        ctx = self._loop_ctx
        masks: Dict[Any, Any] = {}
        for m, res in (self._run_results_by_m or {}).items():
            for ch, arr in (getattr(res, "label_masks", {}) or {}).items():
                masks[(int(m), ch)] = arr
        ctx["iterations"].append({
            "rows": [dict(r) for r in (self._run_all_rows or [])],
            "results_by_m": dict(self._run_results_by_m or {}),
            "masks": masks,
        })
        self._set_status(
            f"Loop iteration {ctx['index'] + 1}/{len(ctx['plan'])}: "
            f"{len(self._run_all_rows or [])} objects.")

    def _loop_should_continue(self) -> bool:
        from nd2studios.pipeline_graph.loop import MODE_UNTIL
        ctx = self._loop_ctx
        if ctx["index"] + 1 >= len(ctx["plan"]):
            return False
        if ctx["config"].get("mode") == MODE_UNTIL and ctx.get("stop") is not None:
            if self._loop_stop_met():
                return False
        return True

    def _loop_stop_met(self) -> bool:
        from nd2studios.pipeline_graph.conditions import evaluate_condition
        from nd2studios.pipeline_graph.loop import (
            IterationResult, RULE_UNION_DEDUP, deduplicate_objects,
        )
        ctx = self._loop_ctx
        its = ctx.get("iterations") or []
        if not its:
            return False
        combine = ctx["config"].get("combine", {}) or {}
        d = combine.get("dedup", {}) or {}
        has_masks = any(it.get("masks") for it in its)
        try:
            if combine.get("rule") == RULE_UNION_DEDUP and has_masks:
                # Object-detection sweep: evaluate on the deduped union so
                # ``nondup_object_count`` counts unique objects.
                irs = [IterationResult(i, {}, it["rows"], it["masks"])
                       for i, it in enumerate(its)]
                rows, _ = deduplicate_objects(
                    irs, metric=str(d.get("metric", "iou")),
                    iou_threshold=float(d.get("iou_threshold", 0.3)),
                    centroid_distance=float(d.get("centroid_distance", 10.0)))
            else:
                # Tracking / non-dedup sweep: evaluate on the latest iteration's
                # rows directly (they carry ``track_id`` for tracking_coverage).
                rows = its[-1]["rows"]
            return bool(evaluate_condition(ctx["stop"], rows))
        except Exception:  # noqa: BLE001 — a bad stop block never wedges a Run
            return False

    def _loop_rerun_entry(self, node) -> None:
        """Re-run the loop entry for the next iteration, dispatched by kind."""
        kind = (self._loop_ctx or {}).get("kind", "analysis")
        getattr(self, self._LOOP_RUN_BY_KIND.get(kind, "_run_analysis_node"))(node)

    def _loop_step(self, node, publish) -> bool:
        """Drive one loop iteration for a **synchronously-combined** entry node
        (track / ct_metrics / dvc / register). Record this iteration; if more
        remain, apply the next overrides and re-run (returns True — the caller must
        return); otherwise combine synchronously, call ``publish`` (a 0-arg
        finalize) and return True. Returns False when no loop is active for this
        node (the caller should publish itself).

        Analysis is *not* routed here — it keeps its bespoke per-M + off-thread
        union-dedup path in ``_advance_run_m``."""
        ctx = getattr(self, "_loop_ctx", None)
        if not (ctx and ctx.get("node_id") == node.id):
            return False
        self._loop_record_iteration()
        if self._loop_should_continue():
            self._loop_start_next_iteration()
            return True
        self._loop_begin_combine(force_sync=True)  # never off-thread for these
        publish()
        return True

    def _loop_start_next_iteration(self) -> None:
        ctx = self._loop_ctx
        node_id = ctx.get("node_id", "")
        ctx["index"] += 1
        self._loop_apply_overrides(ctx["plan"][ctx["index"]])
        node = self._doc.analysis.nodes.get(node_id)
        if node is None:  # node vanished mid-loop — combine what we have and finish
            kind = ctx.get("kind")
            if kind == "analysis":
                if not self._loop_begin_combine():
                    self._finalize_analysis_publish(self._run_analysis_ctx or {})
            elif kind == "track":
                self._loop_begin_combine(force_sync=True)
                self._finalize_track_publish(None, self._run_all_rows or [])
            else:  # generic (ct_metrics / dvc / register) — best-effort finish
                self._loop_begin_combine(force_sync=True)
                self._run_finish_node(node_id)
            return
        self._loop_rerun_entry(node)

    def _loop_stash_plot_series(self) -> None:
        """Stash each iteration's rows + a readable label so the finalize plot pass
        overlays them as distinct colored, legended series (V1.49)."""
        ctx = self._loop_ctx
        its_data = ctx.get("iterations") or []
        plan = ctx.get("plan") or []
        self._loop_plot_series = [
            {"rows": it["rows"], "label": self._loop_iter_label(i, plan)}
            for i, it in enumerate(its_data)
        ]

    def _loop_begin_combine(self, force_sync: bool = False) -> bool:
        """Combine the loop's iterations. Returns True when a background combine
        was started (the caller must return and let the job-done handler publish);
        False when combining finished synchronously (cheap rules). ``force_sync``
        (for track / ct_metrics / dvc / register) never launches the off-thread
        union-dedup job."""
        from nd2studios.pipeline_graph.loop import RULE_UNION_DEDUP
        ctx = self._loop_ctx
        its_data = ctx.get("iterations") or []
        if not its_data:
            self._loop_finish_cleanup()
            return False
        self._loop_stash_plot_series()
        combine = ctx["config"].get("combine", {}) or {}
        rule = combine.get("rule", "last")
        has_masks = any(it.get("masks") for it in its_data)
        if rule == RULE_UNION_DEDUP and has_masks and not force_sync:
            # Overlap-dedup is the heavy path — run it off the GUI thread. Only
            # applicable when there are masks (a Track sweep produces none).
            dedup = combine.get("dedup", {}) or {}
            self._set_status(
                f"Loop: combining {len(its_data)} iterations (overlap dedup)…")
            self._set_preview_progress(visible=True, value=0)
            self._run_pending = ctx.get("node_id", "")
            self._runner.submit(_LoopCombineJob(
                _RUN_LOOPCOMBINE_KEY,
                [{"rows": it["rows"], "masks": it["masks"]} for it in its_data],
                dedup))
            return True
        # Cheap rules (best / last / keep-all), and union-dedup with no masks
        # (a tracking sweep) which falls back to "last" — all synchronous.
        if rule == RULE_UNION_DEDUP:
            rule = "last"
        self._loop_apply_combined_sync(rule, combine, its_data)
        return False

    def _loop_apply_combined_sync(self, rule: str, combine: dict,
                                  its_data: list) -> None:
        """Best / Last / Keep-all combining (no heavy NumPy — safe on the GUI
        thread). Sets ``_run_all_rows`` / ``_run_results_by_m`` then cleans up."""
        from nd2studios.pipeline_graph.loop import (
            BEST_TRACKING_RATIO, RULE_BEST, RULE_KEEP_ALL, tracking_ratio,
        )
        if rule == RULE_BEST:
            metric = combine.get("best_metric", "object_count")

            def _score(it):
                if metric == BEST_TRACKING_RATIO:
                    r, _ = tracking_ratio(it["rows"], 0)
                    return r
                return len(it["rows"])
            best = max(its_data, key=_score)
            self._run_all_rows = list(best["rows"])
            self._run_results_by_m = dict(best["results_by_m"])
        elif rule == RULE_KEEP_ALL:
            rows = []
            for i, it in enumerate(its_data):
                for r in it["rows"]:
                    rr = dict(r)
                    rr["loop_iteration"] = i
                    rows.append(rr)
            self._run_all_rows = rows
            self._run_results_by_m = dict(its_data[-1]["results_by_m"])
        else:  # last
            last = its_data[-1]
            self._run_all_rows = list(last["rows"])
            self._run_results_by_m = dict(last["results_by_m"])
        self._results_rows = self._run_all_rows
        self._loop_finish_cleanup()

    def _loop_finalize_union(self, rows: list, merged_masks: dict) -> None:
        """Apply the off-thread union-dedup result (from :class:`_LoopCombineJob`):
        rebuild the per-M results from the merged masks (light, reference-only),
        set the combined rows, then clean up the loop context."""
        import copy
        ctx = getattr(self, "_loop_ctx", None)
        its_data = (ctx.get("iterations") if ctx else None) or []
        by_m: Dict[int, Dict[str, Any]] = {}
        for (m, ch), arr in (merged_masks or {}).items():
            by_m.setdefault(int(m), {})[ch] = arr
        template = its_data[-1]["results_by_m"] if its_data else {}
        new_results: Dict[int, Any] = {}
        for m, chmasks in by_m.items():
            base = template.get(m) or next(iter(template.values()), None)
            if base is not None:
                res = copy.copy(base)
                res.label_masks = chmasks
                try:
                    res.volumetric_voxel_counts = None
                except Exception:  # noqa: BLE001
                    pass
                new_results[m] = res
        self._run_all_rows = list(rows or [])
        if new_results:
            self._run_results_by_m = new_results
        self._results_rows = self._run_all_rows
        self._loop_finish_cleanup()

    def _loop_finish_cleanup(self) -> None:
        ctx = getattr(self, "_loop_ctx", None)
        n = len(ctx.get("iterations", [])) if ctx else 0
        # V1.49.x: retain every iteration for the viewer dropdown (opt-in). Only
        # row/mask-producing kinds populate the selector — dvc/register keep their
        # result in their own tab, not as swappable rows/masks. Built here (before
        # the ctx is dropped) so both combine paths get it.
        if (ctx and (ctx.get("config") or {}).get("save_iterations")
                and ctx.get("kind") in self._LOOP_ROW_KINDS):
            self._loop_build_saved_store(ctx)
        else:
            self._loop_saved = None
        try:
            self._loop_restore_params()
        except Exception:  # noqa: BLE001
            pass
        self._loop_ctx = None
        self._set_status(
            f"Loop done: {n} iteration(s) → "
            f"{len(self._run_all_rows or [])} combined objects.")

    def _loop_build_saved_store(self, ctx: dict) -> None:
        """Snapshot the combined result + each iteration's result so the viewer's
        iteration dropdown can step through them. Index 0 is always 'Combined'."""
        plan = ctx.get("plan") or []
        its = ctx.get("iterations") or []
        labels = ["Combined"] + [self._loop_iter_label(i, plan)
                                 for i in range(len(its))]
        results = [dict(self._run_results_by_m or {})] + [
            dict(it.get("results_by_m") or {}) for it in its]
        rows = [list(self._run_all_rows or [])] + [
            list(it.get("rows") or []) for it in its]
        self._loop_saved = {"labels": labels, "results": results, "rows": rows}

    def _populate_iteration_selector(self) -> None:
        """Show/hide + fill the viewer's iteration dropdown from ``_loop_saved``."""
        combo = getattr(self, "_iter_combo", None)
        if combo is None:
            return
        saved = getattr(self, "_loop_saved", None)
        labels = (saved or {}).get("labels") or []
        show = len(labels) > 1  # "Combined" + at least one iteration
        combo.blockSignals(True)
        combo.clear()
        if show:
            combo.addItems(labels)
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        combo.setVisible(show)
        self._iter_label.setVisible(show)

    def _on_iteration_selected(self, idx: int) -> None:
        """Re-point the displayed result (overlay/table/plots) to the chosen saved
        iteration (index 0 = the Combined result)."""
        saved = getattr(self, "_loop_saved", None)
        if not saved or self._run_active:
            return
        results = saved.get("results") or []
        rows_list = saved.get("rows") or []
        if not (0 <= idx < len(results)):
            return
        self._run_results_by_m = dict(results[idx])
        self._results_rows = list(rows_list[idx])
        self._run_context["rows"] = self._results_rows
        self._run_context["results_by_m"] = self._run_results_by_m
        cur_m, _, _ = self.viewer.coords()
        self._run_context["result"] = (
            self._run_results_by_m.get(cur_m)
            or next(iter(self._run_results_by_m.values()), None))
        # Refresh table, plots (single-series for a chosen iteration) and overlays.
        self._loop_plot_series = None
        self._populate_results_table(self._results_rows)
        self._update_analysis_plots(self._results_rows)
        self._update_track_plots(self._results_rows)
        self._build_track_overlay(self._results_rows)
        self._update_merged_view_mode()

    def _processed_channels_for_m(self, record, m: int) -> Dict[str, Any]:
        """Recipe-processed ``{channel: (T, H, W)}`` for multipoint ``m``.

        Reads that M's raw channels from the lazy volume
        (``all_channels_as_lazy(m=m)``) and applies the committed recipe, so a
        Run analyzes the *correct* multipoint — not the single M that
        ``record._raw_channels`` (and therefore ``processed_view()``) happens to
        hold. That stale reuse was the V1.48 bug: every M was analyzed on M0's
        pixels. Each channel is materialized one M at a time (peak RAM = one M's
        channels, as before). Sliced to the crop only for a cropped Run. Falls
        back to ``processed_view()`` for in-RAM files with no lazy volume
        (``n_multipoints == 1``, so single-M is already correct)."""
        vol = getattr(record, "_raw_volume", None)
        recipe = list(getattr(record, "recipe", []) or [])
        rbc = getattr(record, "recipe_by_channel", None)
        normalized = bool(getattr(record, "recipe_normalized", False))
        cropped = self._crop_rect() is not None
        out: Dict[str, Any] = {}
        if vol is not None and hasattr(vol, "all_channels_as_lazy"):
            z_mode = getattr(record, "z_view_mode", None) or "max"
            z_index = int(getattr(record, "z_view_index", 0) or 0)
            raw_m = vol.all_channels_as_lazy(m=m, z_mode=z_mode, z_index=z_index)
            from nd2studios.pipeline.stages.recipe_stage import EnhancedDataset
            ds = EnhancedDataset(raw_m, recipe, normalized,
                                 pixel_size_um=record.pixel_size_um,
                                 recipe_by_channel=rbc)
            for nm in ds.keys():
                out[nm] = ds[nm]  # materialize this M's channel (all T), recipe applied
            # V1.56: a Registration node upstream publishes per-M drift transforms;
            # apply them after the recipe so this analysis runs on aligned images.
            # V1.59: register the *full* frame first, then crop — so a cropped Run is
            # a region of the registered image (register → crop), not the reverse.
            out = self._apply_registration_to_channels(record, out, m)
            if cropped:
                out = {nm: self._crop_channel_for_run(arr)
                       for nm, arr in out.items()}
            return out
        view = record.processed_view()
        for nm in view.keys():
            arr = view[nm]
            out[nm] = (self._crop_channel_for_run(arr) if cropped
                       else np.asarray(arr))
        return out

    def _materialize_channels_for_m(self, record, m: int) -> Dict[str, Any]:
        """``{channel: (T, H, W)}`` for multipoint ``m`` — image data the
        validation / export / send-to-results / Spatial Maps steps need. Reads
        the raw volume frame-by-frame (or indexes in-RAM channels for small
        files). Sliced to the preview crop when one is active (Spatial Maps
        preview) — inert during a full Run."""
        out: Dict[str, np.ndarray] = {}
        vol = getattr(record, "_raw_volume", None)
        if vol is not None:
            z_mode = getattr(record, "z_view_mode", None) or "max"
            z_index = int(getattr(record, "z_view_index", 0) or 0)
            nt = int(getattr(vol, "n_timepoints", 1))
            full: Dict[str, np.ndarray] = {}
            for c, name in enumerate(getattr(vol, "channel_names", [])):
                frames = []
                for t in range(nt):
                    try:
                        frames.append(np.asarray(
                            vol.get_frame(c=c, m=m, t=t, z=z_index, z_mode=z_mode)))
                    except Exception:  # noqa: BLE001
                        pass
                if frames:
                    full[name] = np.stack(frames, axis=0)
            # V1.59: apply registration to the full frame, then crop → a registered
            # crop for the Spatial Maps / validation preview.
            full = self._apply_registration_to_channels(record, full, m)
            for name, arr in full.items():
                out[name] = self._crop_frame(arr)
            return out
        full = {}
        for name, arr in (record._raw_channels or {}).items():
            a = arr.materialize() if hasattr(arr, "materialize") else np.asarray(arr)
            full[name] = a if a.ndim == 3 else a[None, ...]
        full = self._apply_registration_to_channels(record, full, m)
        for name, arr in full.items():
            out[name] = self._crop_frame(arr)
        return out

    def _ensure_run_rows(self) -> List[Dict[str, Any]]:
        """Measurement rows for the Run's condition / special nodes.

        If a measurement node already produced rows, reuse them. Otherwise derive
        them on demand from the most recent analysis result's label masks — so an
        if-else can branch on object count / properties **directly after analysis**
        without the user wiring an explicit Compute Measurements node. Tracks the
        objects too (so timelapse conditions + Validate work). Cached in the run
        context so it computes at most once per result.
        """
        # On an object-lens branch the rows are the scoped subset (possibly empty
        # for a fully-dismissed branch) — use them as-is, never re-derive the full
        # set from masks.
        if self._run_context.get("scoped_now"):
            return self._run_context.get("rows") or []
        rows = self._run_context.get("rows")
        if rows:
            return rows
        record = self._active_record()
        # Prefer the whole-file per-M results from the analysis loop; fall back to
        # a single result for safety. (Normally the loop already filled 'rows'.)
        results = self._run_context.get("results_by_m")
        if not results:
            r = self._run_context.get("result")
            if r is not None:
                m, _, _ = self.viewer.coords()
                results = {m: r}
        if not results or record is None:
            return []
        self._set_status("Measuring objects…")
        metadata = dict(record.nd2_metadata)
        metadata["pixel_size_um"] = record.pixel_size_um
        all_rows: List[Dict[str, Any]] = []
        for m, res in sorted(results.items()):
            masks = getattr(res, "label_masks", {}) or {}
            if not masks:
                continue
            try:
                channels = self._materialize_channels_for_m(record, m)
                rows_m = compute_measurements(
                    masks, channels, metadata, m_index=m,
                    volumetric_voxel_counts=getattr(res, "volumetric_voxel_counts", None))
                for r in rows_m:
                    r["m_position"] = m
                link_objects(rows_m)
                all_rows.extend(rows_m)
            except Exception:  # noqa: BLE001 — measurement must never break a Run
                continue
        self._run_context["rows"] = all_rows
        self._results_rows = all_rows
        self._populate_results_table(all_rows)
        return all_rows

    def _run_track_objects(self, node) -> None:
        """Track Objects: re-link the accumulated rows into tracks using the
        node's chosen method + thresholds (authoritative over the default per-M
        tracking done at measurement time). The linker runs **on a worker** (it can
        take many seconds — minutes for SerialTrack — on a dense field and would
        otherwise freeze the GUI); ``_finish_track_objects`` resumes on its result."""
        # V1.49.x: a loop anchored on this Track node re-runs tracking across the
        # iteration plan (sweeping tracking params). Begin it here (applies
        # iteration-0 overrides); re-entry for later iterations finds _loop_ctx set.
        if getattr(self, "_loop_ctx", None) is None:
            self._loop_maybe_begin(node)
        rows = self._ensure_run_rows()
        if not rows:
            self._set_status(f"{node.title}: no measured objects yet — skipped.")
            self._run_finish_node(node.id)
            return
        record = self._active_record()
        px = getattr(record, "pixel_size_um", None) if record is not None else None
        # Mask-overlap (IoU) linker needs the per-(channel, m) StarDist label
        # images. Build a light reference map from the per-M analysis results;
        # the other linkers ignore it. Arrays are passed by reference (no copy),
        # read-only on the worker.
        label_masks: Dict[Tuple[str, int], Any] = {}
        for mm, res in (self._run_context.get("results_by_m") or {}).items():
            for ch, arr in (getattr(res, "label_masks", {}) or {}).items():
                label_masks[(str(ch), int(mm))] = arr
        self._run_trackobj_node = node
        self._run_pending = node.id  # async node — gate the walk until the job lands
        self._set_status(
            f"{node.title}: tracking {len(rows)} objects "
            f"({node.params.get('method', '')})…")
        self._runner.submit(
            _TrackJob(_RUN_TRACKOBJ_KEY, rows, dict(node.params), px,
                      label_masks=label_masks or None))

    def _finish_track_objects(self, result) -> None:
        """Resume the Run after the Track Objects worker job finishes (success or
        failure). For a loop on this node, record the iteration and either re-run
        with the next params or combine; otherwise publish + complete the node."""
        node = self._run_trackobj_node
        title = node.title if node is not None else "Track Objects"
        rows = (result.value if result.ok else None) or self._run_context.get("rows") or []
        if not result.ok:
            self._set_status(f"{title} failed: {result.error}")
        self._run_context["rows"] = rows
        self._run_all_rows = rows
        # V1.49.x: Track-loop driver — iterate the tracking params, then combine.
        lctx = getattr(self, "_loop_ctx", None)
        if (lctx is not None and lctx.get("kind") == "track"
                and node is not None and lctx.get("node_id") == node.id):
            self._loop_record_iteration()
            if self._loop_should_continue():
                self._loop_start_next_iteration()
                return
            self._loop_begin_combine()  # track combines are always synchronous
            rows = self._run_all_rows
        self._run_trackobj_node = None
        self._finalize_track_publish(node, rows)

    def _finalize_track_publish(self, node, rows) -> None:
        """Publish a finished Track Objects node: table, plots, overlay, complete."""
        title = node.title if node is not None else "Track Objects"
        self._run_context["rows"] = rows
        self._results_rows = rows
        self._populate_results_table(rows)
        self._update_track_plots(rows)
        self._build_track_overlay(rows)
        n_tracks = len({r.get("track_id") for r in rows
                        if r.get("track_id") is not None})
        method = node.params.get("method", "") if node is not None else ""
        self._set_status(
            f"{title}: {n_tracks} track(s) across {len(rows)} objects ({method}).")
        self._populate_iteration_selector()  # V1.49.x: show saved-iteration combo
        self._run_finish_node(node.id if node is not None else self._run_pending)

    def _run_dvc(self, node) -> None:
        """DVC (ALDVC): compute the dense displacement + strain field **series**
        over the whole timelapse of the wired channel.

        Tracking mode (a node param, faithful to ``main_ALDVC.m``):
        **cumulative** correlates the fixed reference frame against every other
        frame; **incremental** correlates each frame against the previous one.
        Runs true 3D DVC on the full ``(Z,H,W)`` volumes via ``get_volume``
        (``z_start``/``z_end`` limit Z; ``downsample`` XY-bins) — 2D DIC when
        there is no Z. Volumes are read inside the worker (``_DVCJob``, like the
        app's other volume workers), one Z-stack at a time; ``_finish_dvc``
        resumes on the series. Computes the viewed M, or every M with
        "All multipoints"."""
        # V1.49.x: a loop anchored here sweeps DVC params (e.g. smoothness,
        # downsample). Begin before reading params so this iteration's overrides
        # apply; the field series in the DVC tab reflects the final ("last")
        # iteration.
        if getattr(self, "_loop_ctx", None) is None:
            self._loop_maybe_begin(node)
        record = self._active_record()
        vol = getattr(record, "_raw_volume", None) if record is not None else None
        if record is None or vol is None or not hasattr(vol, "get_volume"):
            self._set_status(f"{node.title}: no file imported — skipped.")
            self._run_finish_node(node.id)
            return
        names = list(getattr(vol, "channel_names", []) or [])
        if not names:
            self._set_status(f"{node.title}: no channels — skipped.")
            self._run_finish_node(node.id)
            return
        # Channel: the one wired into the node's rainbow port (V1.48), else the 1st.
        _extra, seg = self._analysis_run_spec(node, names)
        channel = seg[0] if seg else names[0]
        if channel not in names:
            channel = names[0]
        c_idx = names.index(channel)
        n_t = int(getattr(vol, "n_timepoints", 1) or 1)
        if n_t < 2:
            self._set_status(f"{node.title}: needs ≥2 timepoints — skipped.")
            self._run_finish_node(node.id)
            return
        mode = str(node.params.get("tracking_mode", "cumulative") or "cumulative")
        ref_frame = max(0, min(int(node.params.get("ref_frame", 0) or 0), n_t - 1))
        # Deformed frames to compute (a field per frame → a playable series).
        if mode == "incremental":
            frames = list(range(1, n_t))                     # frame vs previous
        else:
            frames = [t for t in range(n_t) if t != ref_frame]   # frame vs reference
        n_z = int(getattr(vol, "n_zslices", 1) or 1)
        z0 = max(0, int(node.params.get("z_start", 0) or 0))
        z1_raw = int(node.params.get("z_end", 0) or 0)
        z1 = None if z1_raw <= 0 else min(z1_raw, n_z)
        z_count = (n_z if z1 is None else z1) - z0
        is_3d = z_count > 1
        down = max(1, int(node.params.get("downsample", 1) or 1))
        px = float(getattr(record, "pixel_size_um", 1.0) or 1.0) * down
        zs = float(getattr(record, "z_step_um", 1.0) or 1.0)
        voxel = (zs, px, px) if is_3d else (px, px)
        n_m = max(1, self._record_n_multipoints(record))
        cur_m, _, _ = self.viewer.coords()
        cur_m = max(0, min(int(cur_m), n_m - 1))
        all_m = bool(node.params.get("all_multipoints", False))
        m_list = list(range(n_m)) if all_m else [cur_m]

        self._run_dvc_node = node
        self._run_dvc_display_m = cur_m
        self._run_pending = node.id  # async node — gate the walk until the job lands
        dim = "3D" if is_3d else "2D"
        scope = f"all {n_m} M" if all_m else f"M{cur_m + 1}"
        binned = f" · ÷{down} XY" if down > 1 else ""
        zinfo = f" · Z[{z0}:{z1 if z1 is not None else n_z}]" if is_3d else ""
        self._set_status(
            f"{node.title}: {dim} {mode} DVC on '{channel}' · {scope} · "
            f"{len(frames)} frame(s){zinfo}{binned}…")
        self._runner.submit(_DVCJob(
            _RUN_DVC_KEY, vol, c_idx, m_list, frames, ref_frame, mode,
            z0, z1, down, voxel, dict(node.params)))

    def _finish_dvc(self, result) -> None:
        """Resume the Run after the DVC worker finishes: store the field series,
        open + populate the (playable) DVC tab, and complete the node."""
        node = getattr(self, "_run_dvc_node", None)
        self._run_dvc_node = None
        title = node.title if node is not None else "DVC"
        nid = node.id if node is not None else self._run_pending
        if not result.ok or not result.value:
            self._set_status(f"{title} failed: {result.error}")
            self._run_finish_node(nid)
            return
        by_m = result.value  # {m: {"primary": {t:(res,bg)}, "increment": {t:res}}}
        n_fields = 0
        for m, bundle in by_m.items():
            primary = bundle.get("primary", {}) if isinstance(bundle, dict) else {}
            increment = bundle.get("increment", {}) if isinstance(bundle, dict) else {}
            res_map = {int(t): res for t, (res, _bg) in primary.items()}
            bg_map = {int(t): np.asarray(bg) for t, (_res, bg) in primary.items()}
            self._dvc_series_by_m[int(m)] = res_map
            self._dvc_bg_by_mt[int(m)] = bg_map
            self._dvc_incr_by_m[int(m)] = {int(t): r for t, r in increment.items()}
            n_fields += len(res_map)
        # V1.49.x: a loop anchored on this node sweeps DVC params — record the
        # iteration, re-run with the next params, or (final) finish. The DVC tab
        # shows the last iteration's field series (combine falls back to "last").
        if not self._loop_step(node, lambda: self._finalize_dvc_publish(
                node, nid, by_m, n_fields)):
            self._finalize_dvc_publish(node, nid, by_m, n_fields)

    def _finalize_dvc_publish(self, node, nid, by_m, n_fields) -> None:
        """Open + populate the DVC tab for the final field series and complete."""
        title = node.title if node is not None else "DVC"
        self._ensure_dvc_panel()
        self._update_overlay_tabs_available()
        self._select_overlay_tab("dvc")
        show_m = getattr(self, "_run_dvc_display_m", None)
        if show_m is None or int(show_m) not in by_m:
            show_m = sorted(by_m)[0] if by_m else 0
        self._populate_dvc_panel(int(show_m))
        self._set_status(
            f"{title}: {n_fields} field(s) across {len(by_m)} multipoint(s).")
        self._run_finish_node(nid)

    # ── Registration node (V1.56) ──────────────────────────────────────────
    def _run_register(self, node) -> None:
        """Registration: estimate a spatial transform aligning each frame of the
        wired reference channel onto a reference frame (temporal drift correction;
        translation / rigid / affine), and apply the SAME transform to every
        channel ("register once, apply to all" — preserves colocalization).

        Operates on the projected ``(T,H,W)`` frames (like analysis nodes) via
        ``get_frame``, read inside the worker (:class:`_RegisterJob`). Registers
        the currently-viewed M, or every M with "All multipoints"."""
        # V1.49.x: a loop anchored here sweeps registration params (model, etc.);
        # the Registration tab / downstream alignment uses the final iteration.
        if getattr(self, "_loop_ctx", None) is None:
            self._loop_maybe_begin(node)
        record = self._active_record()
        vol = getattr(record, "_raw_volume", None) if record is not None else None
        if record is None or vol is None or not hasattr(vol, "get_frame"):
            self._set_status(f"{node.title}: no file imported — skipped.")
            self._run_finish_node(node.id)
            return
        names = list(getattr(vol, "channel_names", []) or [])
        if not names:
            self._set_status(f"{node.title}: no channels — skipped.")
            self._run_finish_node(node.id)
            return
        # Reference channel: the one wired into the rainbow port (V1.48), else 1st.
        _extra, seg = self._analysis_run_spec(node, names)
        ref_channel = seg[0] if seg else names[0]
        if ref_channel not in names:
            ref_channel = names[0]
        ref_c = names.index(ref_channel)
        n_t = int(getattr(vol, "n_timepoints", 1) or 1)
        if n_t < 2:
            self._set_status(f"{node.title}: needs ≥2 timepoints — skipped.")
            self._run_finish_node(node.id)
            return
        apply_all = bool(node.params.get("apply_to_all_channels", True))
        if apply_all:
            apply_c = list(range(len(names)))
        else:
            apply_c = [names.index(c) for c in seg if c in names] or [ref_c]
        z_index = int(getattr(record, "z_view_index", 0) or 0)
        z_mode = getattr(record, "z_view_mode", None) or "max"
        px = float(getattr(record, "pixel_size_um", 1.0) or 1.0)
        pixel_size_um = (px, px)
        n_m = max(1, self._record_n_multipoints(record))
        cur_m, _, _ = self.viewer.coords()
        cur_m = max(0, min(int(cur_m), n_m - 1))
        all_m = bool(node.params.get("all_multipoints", False))
        m_list = list(range(n_m)) if all_m else [cur_m]

        self._run_register_node = node
        self._run_register_display_m = cur_m
        self._run_pending = node.id  # async node — gate the walk until the job lands
        model = str(node.params.get("model", "translation"))
        scope = f"all {n_m} M" if all_m else f"M{cur_m + 1}"
        tgt = (f"all {len(names)} channels" if apply_all
               else f"{len(apply_c)} channel(s)")
        self._set_status(
            f"{node.title}: {model} registration on '{ref_channel}' · {scope} · "
            f"{n_t} frame(s) · apply to {tgt}…")
        self._runner.submit(_RegisterJob(
            _RUN_REGISTER_KEY, vol, ref_c, apply_c, names, m_list, n_t,
            z_index, z_mode, pixel_size_um, dict(node.params)))

    def _finish_register(self, result) -> None:
        """Resume the Run after the Registration worker finishes: store the
        per-multipoint bundles, open + populate the Registration tab, complete."""
        node = getattr(self, "_run_register_node", None)
        self._run_register_node = None
        title = node.title if node is not None else "Registration"
        nid = node.id if node is not None else self._run_pending
        if not result.ok or not result.value:
            self._set_status(f"{title} failed: {result.error}")
            self._run_finish_node(nid)
            return
        by_m = result.value  # {m: {channel, raw_ref, aligned, transforms, ...}}
        self._reg_by_m = dict(by_m)
        # Publish the per-M transforms on the record so EVERY downstream node that
        # reads processed channels (`_processed_channels_for_m`) gets the
        # drift-corrected image — analyses/tracking then run on the aligned data.
        record = self._active_record()
        if record is not None:
            record._registration_by_m = {int(m): b.get("transforms")
                                         for m, b in by_m.items()}
            record._registration_interp_order = int(
                node.params.get("interp_order", 1) if node is not None else 1)
            record._registration_crop = self._compute_common_crop(node, by_m)
        # V1.49.x: a loop anchored here sweeps registration params — record the
        # iteration, re-run with the next params, or (final) finish. The
        # Registration tab / downstream alignment use the final ("last") iteration.
        if not self._loop_step(node, lambda: self._finalize_register_publish(
                node, nid, by_m)):
            self._finalize_register_publish(node, nid, by_m)

    def _finalize_register_publish(self, node, nid, by_m) -> None:
        """Open + populate the Registration tab for the final transforms, redraw
        the registered image, and complete the node."""
        title = node.title if node is not None else "Registration"
        self._ensure_registration_panel()
        self._update_overlay_tabs_available()
        show_m = getattr(self, "_run_register_display_m", None)
        if show_m is None or int(show_m) not in by_m:
            show_m = sorted(by_m)[0] if by_m else 0
        # Redraw the main image viewer to the registered image for the shown M.
        self._apply_registration_writeback(int(show_m))
        self._select_overlay_tab("registration")
        self._populate_registration_panel(int(show_m))
        rec = self._active_record()
        crop = getattr(rec, "_registration_crop", None) if rec is not None else None
        crop_txt = ""
        if crop:
            y0, y1, x0, x1 = crop
            crop_txt = f" · cropped to common {x1 - x0}×{y1 - y0} px"
        self._set_status(
            f"{title}: registered {len(by_m)} multipoint(s) — downstream nodes now "
            f"run on the drift-corrected image{crop_txt} (see the Registration tab).")
        self._run_finish_node(nid)

    def _compute_common_crop(self, node, by_m):
        """The largest border-free rectangle common to ALL registered frames of ALL
        multipoints, as ``(y0, y1, x0, x1)`` — or None when the node's "Crop to
        common region" is off, the model isn't translation, or there is no overlap.

        A single region across all M (so every M + frame ends up identical size).
        Translation only: the valid footprint is an axis-aligned rectangle."""
        if node is None or not bool(node.params.get("crop_to_common", False)):
            return None
        if str(node.params.get("model", "translation")) != "translation":
            return None
        shifts, shape = [], None
        for b in by_m.values():
            sh = b.get("shifts")
            if sh is not None and np.asarray(sh).size:
                shifts.append(np.asarray(sh, dtype=np.float64).reshape(-1, 2))
            raw = b.get("raw_ref")
            if shape is None and raw is not None and np.asarray(raw).ndim == 3:
                shape = tuple(int(v) for v in np.asarray(raw).shape[1:])
        if not shifts or shape is None:
            return None
        from nd2studios.backend.registration import estimate as _est
        return _est.common_translation_crop(np.concatenate(shifts, axis=0), shape)

    def _set_track_overlay_state(self, rows: List[Dict[str, Any]]) -> bool:
        """Build the track colormap + long-track set + frozen overlay rows from
        ``rows`` and refresh overlay-tab availability. Returns True if any tracks.

        Shared by the Run (``_build_track_overlay``) and the preview walk so the
        Tracks / Vectors overlays and their tabs light up in **both** paths — the
        preview path previously linked track ids without ever building this state,
        so ``has_tracks`` stayed False and the tabs never appeared."""
        from collections import Counter
        from nd2studios.backend.track_overlays import generate_track_colormap
        cnt = Counter(r.get("track_id") for r in rows
                      if r.get("track_id") is not None)
        if not cnt:
            return False
        T = max((int(r.get("frame", 0)) for r in rows), default=0) + 1
        min_long = max(10, T // 10)
        long_ids = {tid for tid, c in cnt.items() if c >= min_long}
        if not long_ids:                      # nothing meets the threshold → color all
            long_ids = set(cnt)
        self._track_long_ids = long_ids
        self._track_colormap = generate_track_colormap(long_ids)
        # Freeze the tracked rows for the tracks / vectors overlays so they keep
        # rendering even after a downstream Dismiss empties ``_results_rows``.
        self._track_overlay_rows = list(rows)
        self._update_overlay_tabs_available()
        return True

    def _build_track_overlay(self, rows: List[Dict[str, Any]]) -> None:
        """Build the track colormap + long-track set, switch to the Tracks tab,
        and play through the frames so the colored cells build up 'as we go',
        ending on frame 0 with every tracked cell shown as a solid color."""
        if not self._set_track_overlay_state(rows):
            return
        T = max((int(r.get("frame", 0)) for r in rows), default=0) + 1
        self._select_overlay_tab("tracks")
        self._start_track_playthrough(T)

    def _start_track_playthrough(self, n_frames: int) -> None:
        """Step the viewer through every frame (then land on 0) so the colored
        cells fill in over time, like CellTracker's end-of-tracking reveal."""
        if n_frames <= 1:
            self.viewer.invalidate_post_process_cache()
            self.viewer.set_current_frame(t=0)
            return
        self._pt_frames = list(range(n_frames)) + [0]
        timer = getattr(self, "_pt_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.timeout.connect(self._pt_step)
            self._pt_timer = timer
        timer.start(120)

    def _pt_step(self) -> None:
        if not getattr(self, "_pt_frames", None):
            self._pt_timer.stop()
            return
        t = self._pt_frames.pop(0)
        self.viewer.invalidate_post_process_cache()
        self.viewer.set_current_frame(t=t)

    def _run_review_objects(self, node) -> None:
        """Review Objects: dispatch on the node's ``mode`` param — cropped
        per-object panels ('Single objects') or a whole-frame label viewer."""
        mode = (node.params or {}).get("mode", "Single objects")
        if mode == "Whole frame":
            self._run_review_whole_frame(node)
        else:
            self._run_open_validation(node, node.title, include_untracked=True)

    def _run_open_validation(self, node, title: str,
                             include_untracked: bool = False) -> None:
        """Single-object review: open ``TrackValidationDialog`` once per
        multipoint (whole file), then apply accept/reject across all M and
        continue with the kept rows. Decisions come back as row references, so
        tracked *and* untracked objects (``include_untracked``) are handled."""
        rows = self._ensure_run_rows()
        record = self._active_record()
        results = self._run_context.get("results_by_m") or {}
        if not rows or not results or record is None:
            self._set_status(f"{title}: no measured objects yet — skipped.")
            self._run_finish_node(node.id)
            return
        cd = getattr(record, "channel_display", {}) or {}
        rejected_ids: set = set()
        accepted_ids: set = set()
        try:
            from nd2studios.widgets.track_validation_dialog import (
                TrackValidationDialog,
            )
            for m, res in sorted(results.items()):
                rows_m = [r for r in rows if int(r.get("m_position", m) or 0) == m]
                masks = getattr(res, "label_masks", {}) or {}
                if not rows_m or not masks:
                    continue
                has_track = any(r.get("track_id") is not None for r in rows_m)
                if not has_track and not include_untracked:
                    continue  # nothing trackable to validate for this M
                channels = self._materialize_channels_for_m(record, m)
                self._set_status(f"{title}: M{m + 1}/{len(results)}…")
                dlg = TrackValidationDialog(
                    rows_m, masks, channels, cd, parent=self,
                    include_untracked=include_untracked,
                    overlay_style=self._overlay_style())
                dlg.exec()
                rejected_ids |= {id(r) for r in dlg.rejected_rows}
                accepted_ids |= {id(r) for r in dlg.accepted_rows}
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"{title} failed: {exc}")
            self._run_finish_node(node.id)
            return
        rows = self._apply_review_decisions(rows, rejected_ids, accepted_ids)
        self._set_status(
            f"{title}: kept {len(rows)} rows "
            f"({len(accepted_ids)} accepted, {len(rejected_ids)} rejected) across "
            f"{len(results)} multipoint(s).")
        self._run_finish_node(node.id)

    def _run_review_whole_frame(self, node) -> None:
        """Whole-frame review: one viewer across all M (T slider + M selector),
        label ids drawn on each object; click-reject + per-frame accept/reject."""
        rows = self._ensure_run_rows()
        record = self._active_record()
        results = self._run_context.get("results_by_m") or {}
        if not rows or not results or record is None:
            self._set_status(f"{node.title}: no measured objects yet — skipped.")
            self._run_finish_node(node.id)
            return
        rows_by_m: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            rows_by_m.setdefault(int(r.get("m_position", 0) or 0), []).append(r)
        masks_by_m = {m: (getattr(res, "label_masks", {}) or {})
                      for m, res in results.items()}
        cd = getattr(record, "channel_display", {}) or {}
        try:
            from nd2studios.widgets.whole_frame_review_dialog import (
                WholeFrameReviewDialog,
            )
            dlg = WholeFrameReviewDialog(
                rows_by_m, masks_by_m,
                lambda mm: self._materialize_channels_for_m(record, mm),
                cd, parent=self)
            dlg.exec()
            rejected_ids = {id(r) for r in dlg.rejected_rows}
            accepted_ids = {id(r) for r in dlg.accepted_rows}
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"{node.title} failed: {exc}")
            self._run_finish_node(node.id)
            return
        rows = self._apply_review_decisions(rows, rejected_ids, accepted_ids)
        self._set_status(
            f"{node.title} (whole frame): kept {len(rows)} rows "
            f"({len(accepted_ids)} accepted, {len(rejected_ids)} rejected).")
        self._run_finish_node(node.id)

    def _apply_review_decisions(self, rows, rejected_ids: set,
                                accepted_ids: set) -> List[Dict[str, Any]]:
        """Drop rejected rows (by identity), tag accepted ones, and republish."""
        if rejected_ids:
            rows = [r for r in rows if id(r) not in rejected_ids]
        for r in rows:
            if id(r) in accepted_ids:
                r["track_validation"] = "accepted"
        self._run_context["rows"] = rows
        self._results_rows = rows
        self._populate_results_table(rows)
        return rows

    def _discard_objects(self, drop_rows: List[Dict[str, Any]]) -> int:
        """Terminal discard: erase every object in ``drop_rows`` — its label, its
        track id, and all of its frames — from the results table, the frozen
        tracks/vectors overlay rows, and the committed segmentation label masks.
        The object then vanishes from every viewer tab and never reaches a
        downstream node. Returns the number of object rows removed.

        The Run masks are edited in place (the object's label pixels are zeroed
        per frame); intentional for a terminal discard, and it keeps exports and
        the segmentation overlay consistent with what the viewer shows. Row stores
        share dict identity, so dropping by ``id()`` reaches every store at once."""
        drop_ids = {id(r) for r in drop_rows}
        if not drop_ids:
            return 0
        # 1. Zero the dropped labels out of the primary segmentation masks so the
        #    segmentation/masks overlay and exports stop showing them.
        for r in drop_rows:
            res = self._run_results_by_m.get(int(r.get("m_position", 0) or 0))
            masks = getattr(res, "label_masks", None) if res is not None else None
            if not masks:
                continue
            seg = r.get("segmentation_channel") or next(iter(masks), None)
            arr = masks.get(seg) if seg is not None else None
            lid = r.get("label_id")
            f = int(r.get("frame", 0) or 0)
            if arr is None or lid is None or not (0 <= f < arr.shape[0]):
                continue
            frame = np.asarray(arr[f])
            frame[frame == int(lid)] = 0
        # 2. Drop the rows from every row store (results table + frozen overlays).
        def _survivors(rows):
            return [r for r in (rows or []) if id(r) not in drop_ids]
        self._run_context["rows"] = _survivors(self._run_context.get("rows"))
        self._results_rows = _survivors(self._results_rows)
        self._track_overlay_rows = _survivors(
            getattr(self, "_track_overlay_rows", None))
        # 3. Rebuild the track colormap from the survivors and refresh the views.
        if not self._set_track_overlay_state(self._track_overlay_rows):
            self._track_colormap = {}
            self._track_long_ids = set()
        self._populate_results_table(self._results_rows)
        try:
            self.viewer.invalidate_post_process_cache()
            _, t, _ = self.viewer.coords()
            self.viewer.set_current_frame(t=t)
        except Exception:  # noqa: BLE001 — a repaint hiccup must not break a Run
            pass
        return len(drop_ids)

    def _run_dismiss(self, node) -> None:
        """Dismiss — terminal discard: every object routed to this node (its
        track and all its frames) is erased from the results table, the overlays
        and the segmentation masks, so it disappears from every viewer/results tab
        and never reaches a downstream node. On an object-lens branch the incoming
        rows are the branch's objects; with no upstream split it discards the whole
        current set."""
        rows = list(self._ensure_run_rows())
        dropped = self._discard_objects(rows)
        self._set_status(f"Dismissed {dropped} object row(s).")
        self._run_finish_node(node.id)

    def _run_export(self, node) -> None:
        """Export every multipoint's objects / frames to a chosen folder, laid out
        per the node's **frame organization** (which of M / T / Z stack inside a
        file vs split into separate files)."""
        record = self._active_record()
        results = self._run_context.get("results_by_m")
        if not results:
            r = self._run_context.get("result")
            if r is not None:
                m, _, _ = self.viewer.coords()
                results = {m: r}
        if not results or record is None:
            self._set_status("Export: nothing to export — skipped.")
            self._run_finish_node(node.id)
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not out_dir:
            self._set_status("Export cancelled.")
            self._run_finish_node(node.id)
            return
        params = dict(node.params)
        content = str(params.get("content", "objects"))
        fmt = str(params.get("image_format", "TIFF"))
        overlay = bool(params.get("include_overlay", False)) or content == "frames_with_objects"
        organization = EXPORT_ORG_MAP.get(
            str(params.get("frame_organization", "")), "fold_T")
        # Per-M channels (only needed when rendering overlays) + masks.
        frames_by_m = {}
        masks_by_m = {}
        for m, res in sorted(results.items()):
            masks = getattr(res, "label_masks", {}) or {}
            if not masks:
                continue
            masks_by_m[m] = masks
            if overlay:
                frames_by_m[m] = self._materialize_channels_for_m(record, m)
        # Honor the boundary-outline display hint from the analysis result(s).
        outline = any(getattr(r, "overlay_outline", False) for r in results.values())
        try:
            written = export_organized(
                frames_by_m, masks_by_m, out_dir,
                fmt=fmt, organization=organization,
                channel_display=getattr(record, "channel_display", {}) or {},
                pixel_size_um=record.pixel_size_um, overlay=overlay,
                basename="pipeline", outline=outline)
            self._set_status(
                f"Exported {len(written)} file(s) from {len(masks_by_m)} "
                f"multipoint(s) to {out_dir}.")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Export failed: {exc}")
        self._run_finish_node(node.id)

    def _run_ct_metrics(self, node) -> None:
        """Cell-Tracker Metrics: augment the accumulated rows with per-cell
        spatial metrics (neighbor distance, local divergence / curl) and, when an
        intensity channel is chosen, self-fold-change. Mirrors the synchronous
        Track Objects handler — needs ``track_id`` from an upstream Track Objects
        node for the velocity / fold metrics (neighbor distance works without)."""
        # V1.49.x: a loop anchored here sweeps the metric params (e.g. neighbor
        # count) and combines the per-iteration augmented rows.
        if getattr(self, "_loop_ctx", None) is None:
            self._loop_maybe_begin(node)
        rows = self._ensure_run_rows()
        if not rows:
            self._set_status(f"{node.title}: no measured objects yet — skipped.")
            self._run_finish_node(node.id)
            return
        record = self._active_record()
        px = getattr(record, "pixel_size_um", None) if record is not None else None
        try:
            augment_rows_with_metrics(rows, node.params, pixel_size_um=px)
        except Exception as exc:  # noqa: BLE001 — analysis must never break a Run
            self._set_status(f"{node.title} failed: {exc}")
            self._run_finish_node(node.id)
            return
        self._run_context["rows"] = rows
        self._run_all_rows = rows
        if not self._loop_step(node, lambda: self._finalize_ct_metrics_publish(
                node, self._run_all_rows)):
            self._finalize_ct_metrics_publish(node, rows)

    def _finalize_ct_metrics_publish(self, node, rows) -> None:
        """Publish a finished Cell-Tracker Metrics node (table, status, complete)."""
        self._run_context["rows"] = rows
        self._results_rows = rows
        self._populate_results_table(rows)
        tracked = any(r.get("track_id") is not None for r in rows)
        note = "" if tracked else " (no tracks — neighbor distance only)"
        title = node.title if node is not None else "Cell-Tracker Metrics"
        self._set_status(
            f"{title}: metrics added for {len(rows)} object(s){note}.")
        self._populate_iteration_selector()
        self._run_finish_node(node.id if node is not None else self._run_pending)

    def _run_send_results(self, node) -> None:
        """Send to Results: push the current rows (+ masks/channels) to the
        top-level Results page and navigate there for plotting / processing."""
        rows = self._ensure_run_rows()
        result = self._run_context.get("result")
        mw = self.main_window
        rp = getattr(mw, "pages", {}).get("results") if mw is not None else None
        if rp is None:
            self._set_status("Send to Results: Results page unavailable — skipped.")
            self._run_finish_node(node.id)
            return
        try:
            rp._measurements = list(rows)
            if result is not None:
                rp._label_masks_for_validation = getattr(result, "label_masks", {}) or {}
            record = self._active_record()
            if record is not None:
                m, _, _ = self.viewer.coords()
                rp._channels_for_validation = self._materialize_channels_for_m(record, m)
            rp._update_table(rows)
            rp._update_summary(rows)
            nav = getattr(mw, "_navigate", None)
            if callable(nav):
                nav("results")
            self._set_status(f"Sent {len(rows)} rows to the Results page.")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Send to Results failed: {exc}")
        self._run_finish_node(node.id)

    # ── checkpoint (V1.53) ─────────────────────────────────────────────────
    # A Checkpoint node freezes everything computed upstream when the Run reaches
    # it. A later Run whose upstream graph is unchanged resumes *from* the
    # checkpoint (see ``_checkpoint_resume_target`` / ``_on_run``) with the frozen
    # data restored, so segmentation / tracking never re-run.
    def _checkpoint_ancestors(self, node_id: str) -> set:
        """Structural ancestors of ``node_id`` in the Analysis slice — every node
        whose output feeds the checkpoint (the nodes whose work it freezes)."""
        sl = self._doc.analysis
        anc: set = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for e in sl.structural_incoming(cur):
                if e.src_node not in anc:
                    anc.add(e.src_node)
                    stack.append(e.src_node)
        return anc

    def _checkpoint_upstream_hash(self, node_id: str) -> str:
        """Stable hash of everything that determines the checkpoint's frozen data.

        Covers the structural ancestors (op_key + params), every edge feeding them
        or the checkpoint (structural + channel wiring, so the wired segmentation
        channel counts), any loop-edge config touching them, the Processing
        recipe / per-channel recipes / normalized flag, the Run crop geometry, and
        a **source-file signature** (so a persisted cache reloaded against a
        different ND2 file invalidates through the same gate). A change to any of
        these invalidates the freeze and forces a full re-run.
        """
        sl = self._doc.analysis
        anc = self._checkpoint_ancestors(node_id)
        scope = anc | {node_id}
        nodes_desc = sorted(
            (nid, sl.nodes[nid].op_key,
             json.dumps(sl.nodes[nid].params, sort_keys=True, default=str))
            for nid in anc if nid in sl.nodes
        )
        edges_desc = sorted(
            (e.src_node, e.src_port, e.dst_node, e.dst_port, e.kind,
             json.dumps(e.params, sort_keys=True, default=str))
            for e in sl.edges.values() if e.dst_node in scope
        )
        record = self._active_record()
        payload = {
            "nodes": nodes_desc,
            "edges": edges_desc,
            "recipe": self._graph_recipe(),
            "recipe_by_channel": self._graph_channel_recipes(record),
            "normalized": bool(self._normalized),
            "crop": self._crop_rect(),
            "file": self._record_signature(record),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _record_signature(self, record) -> Dict[str, Any]:
        """Identity fingerprint of the active file's pixels, folded into the
        checkpoint hash so frozen masks are only ever reused with the file they
        were computed from (name + channel set + shape + pixel size)."""
        if record is None:
            return {}
        names = sorted((getattr(record, "_raw_channels", None) or {}).keys())
        return {
            "name": getattr(record, "name", ""),
            "channels": names,
            "n_frames": int(getattr(record, "n_frames", 0) or 0),
            "n_multipoints": self._record_n_multipoints(record),
            "n_zslices": int(getattr(record, "n_zslices", 1) or 1),
            "height": int(getattr(record, "frame_height", 0) or 0),
            "width": int(getattr(record, "frame_width", 0) or 0),
            "pixel_size_um": round(
                float(getattr(record, "pixel_size_um", 1.0) or 1.0), 6),
        }

    def _run_checkpoint(self, node) -> None:
        """Freeze the accumulated Run state at this checkpoint, then pass through.

        Ensures measurement rows exist, snapshots the per-M analysis results,
        rows, run context and track overlay state (arrays held by reference —
        session RAM), and stamps the current upstream hash so a later Run can tell
        the freeze is still valid. Re-freezing on every full Run keeps the snapshot
        current after any upstream edit."""
        self._ensure_run_rows()
        snapshot = {
            "results_by_m": dict(self._run_results_by_m or {}),
            "all_rows": [dict(r) for r in (self._run_all_rows or [])],
            "results_rows": [dict(r) for r in (self._results_rows or [])],
            "ctx_rows": [dict(r) for r in (self._run_context.get("rows") or [])],
            "ctx_result": self._run_context.get("result"),
            "ctx_results_by_m": dict(self._run_context.get("results_by_m") or {}),
            "results_crop": self._run_results_crop,
            "track_colormap": (dict(self._track_colormap)
                               if self._track_colormap else None),
            "track_long_ids": set(self._track_long_ids or set()),
            "track_overlay_rows": [dict(r) for r in
                                   (self._track_overlay_rows or [])],
        }
        self._checkpoint_store[node.id] = {
            "hash": self._checkpoint_upstream_hash(node.id),
            "data": snapshot,
        }
        n_obj = len(snapshot["all_rows"])
        self._set_status(
            f"Checkpoint '{node.title}': froze {n_obj} object(s) across "
            f"{len(snapshot['results_by_m'])} multipoint(s). Downstream edits now "
            "re-run instantly.")
        self._run_finish_node(node.id)

    def _checkpoint_resume_target(self) -> Optional[str]:
        """The furthest-downstream checkpoint whose frozen data is still valid.

        A checkpoint is valid when it has a stored snapshot and its stored upstream
        hash equals the current upstream hash. Among valid checkpoints, pick the
        one deepest in topological order (skips the most upstream work). ``None``
        when no checkpoint can be resumed (⇒ a normal full Run)."""
        sl = self._doc.analysis
        order = {nid: i for i, nid in enumerate(topological_order(sl))}
        best: Optional[str] = None
        best_rank = -1
        for nid, node in sl.nodes.items():
            if node.op_key != SPECIAL_CHECKPOINT_OP_KEY:
                continue
            entry = self._checkpoint_store.get(nid)
            if not entry:
                continue
            if entry.get("hash") != self._checkpoint_upstream_hash(nid):
                continue
            rank = order.get(nid, -1)
            if rank > best_rank:
                best_rank, best = rank, nid
        return best

    def _restore_checkpoint(self, node_id: str) -> None:
        """Load a checkpoint's frozen snapshot back into the Run accumulators."""
        snap = self._checkpoint_store[node_id]["data"]
        self._run_results_by_m = dict(snap["results_by_m"])
        self._run_all_rows = [dict(r) for r in snap["all_rows"]]
        self._results_rows = [dict(r) for r in snap["results_rows"]]
        self._run_results_crop = snap["results_crop"]
        self._run_context["rows"] = [dict(r) for r in snap["ctx_rows"]]
        self._run_context["result"] = snap["ctx_result"]
        self._run_context["results_by_m"] = dict(snap["ctx_results_by_m"])
        self._track_colormap = (dict(snap["track_colormap"])
                                if snap["track_colormap"] else None)
        self._track_long_ids = set(snap["track_long_ids"])
        self._track_overlay_rows = [dict(r) for r in snap["track_overlay_rows"]]
        # Reflect the restored state in the table / overlays immediately.
        self._populate_results_table(self._run_all_rows)
        self._update_analysis_plots(self._run_all_rows)
        self._update_overlay_tabs_available()
        self._update_merged_view_mode()

    def _pause_run(self, node) -> None:
        """Pause node: halt the run and drop the whole page back to editor mode
        (all nodes un-shaded) so the user can modify the downstream pipeline.
        The runner state is retained — clicking Run resumes from here (unless the
        graph's node set changed, in which case Run restarts)."""
        if self._runner_obj is not None:
            self._runner_obj.complete(node.id)  # so resume continues past Pause
        self._run_active = False
        self._run_paused = True
        self._run_paused_nodes = set(self._doc.analysis.nodes)
        self._run_pending = ""
        self._run_states = {}
        self._scenes[Stage.ANALYSIS].clear_run_states()
        self._btn_run.setEnabled(True)
        self._set_preview_progress(visible=False)
        self._set_status(
            f"Paused at '{node.title}' — editor mode. Edit downstream or use "
            "Preview Crop to troubleshoot a sub-region (e.g. of a registered "
            "image), then Run to resume.")

    def _finish_run(self) -> None:
        # V1.49: if a loop was still mid-flight (e.g. an early skip), restore the
        # body's original params so the graph isn't left with swept values.
        if getattr(self, "_loop_ctx", None) is not None:
            self._loop_finish_cleanup()
        self._run_active = False
        self._run_paused = False
        self._runner_obj = None
        self._run_pending = ""
        self._run_channels_m = None  # free any lingering per-M channels
        self._btn_run.setEnabled(True)
        self._set_preview_progress(visible=False)
        self._set_status("Run complete.")
        # Hold the final shading briefly, then return to editor mode.
        QTimer.singleShot(1400, self._clear_run_states_if_idle)

    def _clear_run_states_if_idle(self) -> None:
        if not self._run_active:
            self._run_states = {}
            scene = self._scenes.get(Stage.ANALYSIS)
            if scene is not None:
                scene.clear_run_states()

    # ── graph-authoritative recipe sync (V1.46) ─────────────────────────────
    def _graph_recipe(self) -> List:
        """The recipe the Processing node graph currently represents.

        Linearizes the primary OUTPUT node's chain. Returns ``[]`` when there is
        no OUTPUT node, when the OUTPUT is not connected back to the INPUT
        (``recipe_for_node`` raises), or for a direct Input→Output wire (empty
        chain) — i.e. the graph says "no processing".
        """
        outs = output_nodes(self._doc.processing)
        if not outs:
            return []
        try:
            return recipe_for_node(self._doc.processing, outs[0].id)
        except ValueError:
            return []

    def _graph_channel_recipes(self, record) -> Optional[Dict[str, List]]:
        """Per-channel recipes from the Processing graph (V1.48).

        ``None`` when the graph has no channel wiring — the legacy single recipe
        then applies to every channel. Otherwise ``{channel: recipe}`` for the
        primary Output's chain; channels absent from the dict stay raw."""
        sl = self._doc.processing
        if not has_channel_wiring(sl):
            return None
        outs = output_nodes(sl)
        names = list((record._raw_channels or {}).keys())
        if not outs:
            return {}
        try:
            return channel_recipes(sl, outs[0].id, names)
        except Exception:  # noqa: BLE001
            return None

    def _sync_committed_recipe_from_graph(self) -> None:
        """Make the node graph authoritative: re-derive ``record.recipe`` from
        the Processing graph so the Analysis base image / exports never show a
        stale committed recipe that the (possibly empty) graph no longer holds.

        Mirrors the record-side of :meth:`_apply_processing` (recipe, normalized
        flag, processed view/channels) but does **not** persist to the session
        workspace — that stays the explicit Apply action. No-ops when the
        effective recipe is unchanged, so the lazy ``EnhancedDataset`` is only
        rebuilt on a real change.
        """
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        recipe = self._graph_recipe()
        rbc = self._graph_channel_recipes(record)
        normalized = bool(self._normalized)
        if (list(getattr(record, "recipe", []) or []) == recipe
                and bool(getattr(record, "recipe_normalized", False)) == normalized
                and getattr(record, "recipe_by_channel", None) == rbc):
            return
        from nd2studios.pipeline.stages.recipe_stage import EnhancedDataset
        record.recipe = recipe
        record.recipe_normalized = normalized
        record.recipe_by_channel = rbc
        record._processed_channels = None
        record._processed_view = (
            EnhancedDataset(
                record._raw_channels, recipe, normalized,
                pixel_size_um=record.pixel_size_um, recipe_by_channel=rbc,
            )
            if (recipe or rbc) else None
        )

    def _apply_processing(self) -> None:
        record = self._active_record()
        if record is None or not record._raw_channels:
            QMessageBox.information(self, "Apply", "Import a file first.")
            return
        outs = output_nodes(self._doc.processing)
        if not outs:
            QMessageBox.information(
                self, "Apply",
                "Add an Output node and connect it to commit a recipe.",
            )
            return
        try:
            recipe = recipe_for_node(self._doc.processing, outs[0].id)
        except ValueError as exc:
            QMessageBox.warning(self, "Apply", str(exc))
            return

        from nd2studios.pipeline.stages.recipe_stage import EnhancedDataset

        # Commit the recipe to the record. It is dataset-agnostic and applied
        # lazily to the **entire file** (every M/T/Z) wherever the processed data
        # is read — ``EnhancedDataset`` for Export, and per-frame in the Analysis
        # sub-tab. No full-stack materialization, so it scales to multi-GB files.
        # V1.48: with channel wiring, each channel gets its own recipe and unwired
        # channels stay raw (``recipe_by_channel``).
        rbc = self._graph_channel_recipes(record)
        record.recipe = recipe
        record.recipe_normalized = self._normalized
        record.recipe_by_channel = rbc
        record._processed_channels = None
        record._processed_view = EnhancedDataset(
            record._raw_channels, recipe, self._normalized,
            pixel_size_um=record.pixel_size_um, recipe_by_channel=rbc,
        )
        # Persist to the session workspace if one is attached (best-effort).
        mw = self.main_window
        if mw is not None:
            stage = getattr(mw, "recipe_stage", lambda: None)()
            if stage is not None:
                try:
                    stage.set_recipe(
                        recipe, self._normalized,
                        list(record._raw_channels.keys()),
                    )
                    stage.commit()
                except Exception:  # noqa: BLE001 — commit must never crash Apply
                    pass
            if getattr(mw, "exp_manager", None) is not None:
                mw.exp_manager.set_status("preprocessed")
        self._set_status(
            f"Applied {len(recipe)}-step recipe to the file → Analysis input ready"
        )
        # Stay on the Processing tab and refresh its preview (progress bar). The
        # Analysis sub-tab's input now reads the processed data on demand (it
        # applies record.recipe per frame), so the output is available there
        # without leaving Processing.
        self._request_preview()

    def _apply_analysis(self) -> None:
        record = self._active_record()
        if record is None or not record._raw_channels:
            QMessageBox.information(self, "Apply", "Import a file first.")
            return
        if getattr(self.main_window, "heavy_ops_blocked", lambda: False)():
            QMessageBox.information(self, "Apply",
                                    "Paused — system memory is low.")
            return
        action = self._resolve_analysis_action(self._preview_target(Stage.ANALYSIS))
        if action is None:
            QMessageBox.information(
                self, "Apply",
                "Add an analysis node (and an Output) to run a pipeline.",
            )
            return
        name = analysis_pipeline_name_for_op_key(action.op_key)
        cls = AnalysisPipeline.get_pipeline(name)
        if cls is None:
            return
        names = self._current_channel_names()
        if not names:
            return
        params = dict(action.params)
        ch = params.get("channel_name") or names[0]
        if ch not in names:
            ch = names[0]
        params["channel_name"] = ch

        # V1.46 — keep the processed channels LAZY (no np.asarray); the pipeline
        # reads frames on demand and (adaptively) streams its label masks to a
        # disk scratch, so Apply on a constrained machine doesn't materialize the
        # whole stack. processed_view() is dict-like over channels.
        view = record.processed_view()
        try:
            channels = {nm: view[nm] for nm in view.keys()}
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Apply", f"Could not read channels: {exc}")
            return
        if not channels:
            return
        metadata = dict(record.nd2_metadata)
        metadata["pixel_size_um"] = record.pixel_size_um
        m, _, _ = self.viewer.coords()
        self._analysis_commit_pipeline = name
        self._analysis_commit_m = m
        self._inject_label_streaming(params, record, cls, m)
        job = PipelineCommitJob(
            key=_ANALYSIS_COMMIT_KEY,
            pipeline_cls=cls,
            channels=channels,
            metadata=metadata,
            params=params,
            m_index=m,
        )
        self._set_preview_progress(visible=True, value=0)
        self._set_status(f"Running '{name}' on M{m + 1}…")
        self._runner.submit(job)

    def _apply_results(self) -> None:
        """Compute (refresh) the measurements for the previewed Results node and
        retain the source pipeline on the record. (Export source-selector wiring
        is deferred.)"""
        record = self._active_record()
        if record is None:
            QMessageBox.information(self, "Apply", "Import a file first.")
            return
        names = self._get_analysis_result_names()
        if not names:
            QMessageBox.information(
                self, "Apply",
                "Run & Apply an analysis pipeline before computing results.",
            )
            return
        self._do_results_preview()  # recompute the table for the previewed node
        action = self._resolve_results_action(self._preview_target(Stage.ANALYSIS))
        name = (action.params.get("analysis_result") if action else "") or ""
        if name not in names:
            name = names[0]
        record.results_config = dict(getattr(record, "results_config", {}) or {})
        record.results_config["pipelines_last_measured"] = name
        self._set_status(f"Computed measurements for '{name}' (Export wiring pending).")

    def _inject_label_streaming(self, params: Dict[str, Any], record,
                                pipeline_cls, m: int) -> None:
        """Adaptively stream the pipeline's label masks to a disk scratch (V1.46).

        The Pipelines tab keeps analysis results in RAM for overlay display, so
        the scratch persists for the session (cleared on file change / close);
        the streamed readers in ``record.analysis_results`` point into it.
        """
        from nd2studios.utils.resource_strategy import should_stream_analysis
        from nd2studios.pipeline.storage import LabelStackWriter
        try:
            from nd2studios.core.memory_monitor import global_monitor
            monitor = global_monitor()
        except Exception:  # noqa: BLE001
            monitor = None

        force = (False if getattr(pipeline_cls, "needs_full_stack", False)
                 else None)
        vol = getattr(record, "_raw_volume", None)
        if not should_stream_analysis(vol, monitor=monitor, force=force):
            params.pop("_stream_labels", None)
            params.pop("_label_sink_factory", None)
            return

        if self._analysis_scratch is None:
            self._analysis_scratch = tempfile.mkdtemp(prefix="nd2s_pipe_labels_")
        scratch = self._analysis_scratch

        def _sink_factory(nm: str, shape):
            safe = "".join(c if c.isalnum() else "_" for c in str(nm))
            base = os.path.join(scratch, f"m{m:03d}_{safe}")
            return LabelStackWriter(base, shape)

        params["_stream_labels"] = True
        params["_label_sink_factory"] = _sink_factory

    def _clear_analysis_scratch(self) -> None:
        if self._analysis_scratch:
            shutil.rmtree(self._analysis_scratch, ignore_errors=True)
            self._analysis_scratch = None

    # ── save / load ──────────────────────────────────────────────────────────
    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pipeline", "", f"ND2Studios pipeline (*{PIPELINE_EXTENSION})"
        )
        if not path:
            return
        if not path.endswith(PIPELINE_EXTENSION):
            path += PIPELINE_EXTENSION
        try:
            save_pipeline(path, self._doc)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save pipeline", str(exc))
            return
        # V1.53: persist frozen Checkpoint data alongside the pipeline so a later
        # session can resume from a checkpoint without re-running the upstream work.
        n_ck = self._save_checkpoint_cache(path)
        self._set_status(
            f"Pipeline saved (+ {n_ck} checkpoint{'s' if n_ck != 1 else ''})"
            if n_ck else "Pipeline saved")

    def _save_checkpoint_cache(self, path: str) -> int:
        """Write the frozen checkpoint store to ``<pipeline>.checkpoints/``.

        Only checkpoints whose node opted in (``persist_to_disk`` param) are
        written; the rest stay session-only RAM. Returns the number of
        checkpoints written (0 when none are frozen / opted-in or on failure — a
        cache-write failure never blocks the pipeline save)."""
        sl = self._doc.analysis
        to_persist = {
            nid: entry for nid, entry in self._checkpoint_store.items()
            if nid in sl.nodes
            and bool(sl.nodes[nid].params.get("persist_to_disk"))
        }
        try:
            from nd2studios.pipeline_graph.checkpoint_io import (
                checkpoints_dir_for, save_checkpoints,
            )
            cdir = checkpoints_dir_for(path)
            if not to_persist:
                # Nothing opted in — remove any stale cache from a prior save so a
                # checkpoint toggled back to session-only doesn't linger on disk.
                if os.path.isdir(cdir):
                    shutil.rmtree(cdir, ignore_errors=True)
                return 0
            sig = self._record_signature(self._active_record())
            return save_checkpoints(cdir, to_persist, sig)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not save checkpoint cache: %s", exc)
            return 0

    def _load_checkpoint_cache(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Read the companion ``<pipeline>.checkpoints/`` cache for ``path``.

        Returns an empty store on absence / failure. If the cache was built for a
        different file (its saved signature ≠ the current record's), it is still
        loaded but a note is logged — the Run-time upstream hash will decline to
        resume from it, so it re-runs and re-freezes harmlessly."""
        try:
            from nd2studios.pipeline_graph.checkpoint_io import (
                checkpoints_dir_for, load_checkpoints,
            )
            store, sig = load_checkpoints(checkpoints_dir_for(path))
            if store and sig and sig != self._record_signature(self._active_record()):
                _log.info(
                    "Checkpoint cache was built for a different file — it will "
                    "re-run on the next Run unless the active file matches.")
            return store
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not load checkpoint cache: %s", exc)
            return {}

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load pipeline", "", f"ND2Studios pipeline (*{PIPELINE_EXTENSION})"
        )
        if not path:
            return
        try:
            doc = load_pipeline(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Load pipeline", str(exc))
            return
        self._doc = doc
        self._bridge_counter = len(output_nodes(doc.processing))
        self._analysis_bridge_counter = len(output_nodes(doc.analysis))
        self._results_bridge_counter = len(output_nodes(doc.results))
        self._selected_node_id = ""
        self._preview_node_ids = {
            Stage.PROCESSING: "", Stage.ANALYSIS: "", Stage.RESULTS: "",
        }
        self._analysis_screen_results = {}
        self._analysis_results_per_m = {}
        # V1.53: restore any frozen Checkpoint data saved alongside this pipeline
        # (node ids round-trip, so the store keys still match). Validity is gated
        # at Run time by the upstream hash — which includes the source-file
        # signature — so a cache built for a different file simply won't resume.
        self._checkpoint_store = self._load_checkpoint_cache(path)
        self._results_overlay_result = None
        self._results_rows = []
        self._popup.hide()
        for stage in self._stages:
            self._scenes[stage].slice = doc.slice_for(stage)
            self._scenes[stage]._rebuild_from_slice()
        # V1.48: reconcile channel pills to the current file + migrate rainbow
        # ports on loaded process nodes, and recolor channel wires.
        for stage in self._stages:
            self._ensure_channel_nodes(stage)
        n_ck = len(self._checkpoint_store)
        self._set_status(
            f"Pipeline loaded (+ {n_ck} checkpoint{'s' if n_ck != 1 else ''})"
            if n_ck else "Pipeline loaded")
        self._update_preview_highlight()
        self._update_merged_view_mode()
        if self._stage is Stage.ANALYSIS:
            self._show_base_image()
        self._request_preview()

    # ── lifecycle (called by MainWindow._navigate) ──────────────────────────
    def on_activated(self) -> None:
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        self._ensure_input_node(self._stage)
        self._refresh_active_view()

    def load_from_experiment(self, exp) -> None:
        """A new file (or session) became the active record — rewire the board.

        Called by ``MainWindow._on_experiment_changed`` whenever the active
        experiment changes, including when a file is loaded *on top of* an
        existing one while the Pipelines tab is showing (``on_activated`` only
        fires on page navigation, so without this the tab kept the old file).

        The pipeline graph (nodes / wires / params) is intentionally preserved —
        a built pipeline carries over to the new file — but every piece of
        per-file preview state is dropped so nothing leaks from the old record,
        the source node's channel count is refreshed, and (if we're the visible
        page) the viewer + preview re-run against the new data.
        """
        # Drop stale per-plane preview / overlay state from the previous file.
        self._analysis_screen_results = {}
        self._analysis_results_per_m = {}
        # V1.53: frozen checkpoint data belongs to the previous file — drop it so
        # a Run on the new file recomputes from scratch (and re-freezes).
        self._checkpoint_store = {}
        self._processing_planes = []
        self._proc_volume = None
        # Preview crop is per-file — clear it (dims won't match a new file).
        self._preview_crop = None
        self._crop_selecting = False
        self._crop_history = []
        self._run_cropped = False
        self._run_results_crop = None
        if getattr(self, "_btn_preview_crop", None) is not None:
            self._btn_preview_crop.blockSignals(True)
            self._btn_preview_crop.setChecked(False)
            self._btn_preview_crop.blockSignals(False)
            self._lbl_preview_crop.setText("")
            self.viewer.set_crop_mode(False)
        self._update_crop_undo_enabled()
        self._update_run_button()
        self._results_overlay_result = None
        self._results_rows = []
        self._populate_results_table([])
        # Refresh (or create) the source node title for the new channel count.
        for stage in self._stages:
            self._refresh_input_node(stage)
        # When we're the page on screen, on_activated won't fire — rewire now.
        # Otherwise the upcoming on_activated (on navigation) does it.
        if (self.main_window is not None
                and getattr(self.main_window, "_current_page_key", None)
                == "pipelines"):
            self._refresh_active_view()

    def _refresh_active_view(self) -> None:
        """Rewire the viewer + preview to the active record for the current
        sub-tab (shared by ``on_activated`` and ``load_from_experiment``)."""
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        self._update_preview_highlight()
        self._update_merged_view_mode()
        if self._stage is Stage.ANALYSIS:
            self._show_base_image()
        self._request_preview()
        QTimer.singleShot(0, self._frame_all_nodes)

    def on_close(self) -> None:
        self._preview_debounce.stop()
        self._popup.hide()
        self._clear_analysis_scratch()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ensure_input_node(self, stage: Stage) -> None:
        """Create the stage's source node from the active record, once."""
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        sl = self._doc.slice_for(stage)
        if input_node(sl) is None:
            scene = self._scenes[stage]
            if stage is Stage.PROCESSING:
                n_ch = len(record._raw_channels)
                node = scene.add_node_from_spec(processing_input_spec(), (40.0, 80.0))
                node.title = f"Input ({n_ch} ch)"
            elif stage is Stage.ANALYSIS:
                n_ch = len(self._current_channel_names())
                node = scene.add_node_from_spec(analysis_input_spec(), (40.0, 80.0))
                node.title = f"Processed ({n_ch} ch)"
            else:
                return
            item = scene.node_item(node.id)
            if item is not None:
                item.update()
        # V1.48: per-channel source pills under the Input node (+ "All").
        self._ensure_channel_nodes(stage)

    def _refresh_input_node(self, stage: Stage) -> None:
        """Update the existing source node's title to the active record's channel
        count (creating it if the stage has none yet). Lets a new file's channel
        count replace the previous file's on the carried-over graph."""
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        sl = self._doc.slice_for(stage)
        node = input_node(sl)
        if node is None:
            self._ensure_input_node(stage)
            return
        if stage is Stage.PROCESSING:
            node.title = f"Input ({len(record._raw_channels)} ch)"
        elif stage is Stage.ANALYSIS:
            node.title = f"Processed ({len(self._current_channel_names())} ch)"
        item = self._scenes[stage].node_item(node.id)
        if item is not None:
            item.update()
        # Reconcile channel pills to the new file's channel set.
        self._ensure_channel_nodes(stage)

    # ── channel-flow (V1.48) ────────────────────────────────────────────────
    def _channel_colors_map(self) -> Dict[str, str]:
        """``{channel_name: '#rrggbb'}`` from the record's channel display colors
        (falls back to a neutral color)."""
        record = self._active_record()
        out: Dict[str, str] = {}
        cd = getattr(record, "channel_display", {}) if record is not None else {}
        for name in self._current_channel_names():
            color_name = (cd.get(name, {}) or {}).get("color") or "gray"
            rgb = CHANNEL_COLORS.get(color_name, (255, 255, 255))
            out[name] = "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return out

    def _push_channel_context(self, stage: Stage) -> None:
        """Push live channel names + colors to the stage's scene so it can color
        channel wires, propagation strands and channel-source pills."""
        scene = self._scenes.get(stage)
        if scene is not None:
            scene.set_channel_context(self._current_channel_names(),
                                      self._channel_colors_map())

    def _ensure_channel_nodes(self, stage: Stage) -> None:
        """Create / reconcile the channel-source pills under the Input node — one
        per live channel plus an "All" node — then push the channel context.

        Stale pills (a channel no longer in the file) are removed; the graph's
        real work (processes / wires) is preserved. Only the Processing and
        Analysis slices get them (they operate on image channels)."""
        if stage not in (Stage.PROCESSING, Stage.ANALYSIS):
            return
        record = self._active_record()
        if record is None or not record._raw_channels:
            return
        sl = self._doc.slice_for(stage)
        scene = self._scenes[stage]
        inp = input_node(sl)
        base_x = (inp.pos[0] - 172.0) if inp is not None else -140.0
        base_y = (inp.pos[1]) if inp is not None else 80.0
        names = self._current_channel_names()
        desired = [CHANNEL_ALL_OP_KEY] + [f"{CHANNEL_PREFIX}{n}" for n in names]
        existing = {n.op_key: n for n in sl.nodes.values()
                    if is_channel_source_op(n.op_key)}
        # Drop stale channel pills (their channel is gone from this file).
        for opk, node in list(existing.items()):
            if opk not in desired:
                scene.delete_node(node.id)
                existing.pop(opk, None)
        # Add any missing pills, stacked in a column beside the Input node.
        for i, opk in enumerate(desired):
            if opk in existing:
                continue
            pos = (base_x, base_y + i * 34.0)
            spec = (channel_all_spec() if opk == CHANNEL_ALL_OP_KEY
                    else channel_source_spec(channel_name_for_op_key(opk)))
            scene.add_node_from_spec(spec, pos)
        self._push_channel_context(stage)

    def _current_channel_names(self) -> List[str]:
        record = self._active_record()
        if record is None:
            return []
        try:
            view = record.processed_view()
            return list(view.keys())
        except Exception:  # noqa: BLE001
            return list((record._raw_channels or {}).keys())

    def _record_n_multipoints(self, record) -> int:
        vol = getattr(record, "_raw_volume", None)
        if vol is not None:
            return max(1, int(getattr(vol, "n_multipoints", 1)))
        return max(1, int(getattr(record, "n_multipoints", 1)))

    def _frame_all_nodes(self) -> None:
        """Fit the view to the whole graph for the current sub-tab (all nodes),
        capping zoom at 1:1 so a sparse graph isn't magnified."""
        scene = self._scenes.get(self._stage)
        if scene is None:
            return
        rect = scene.itemsBoundingRect()
        if rect.isNull() or rect.isEmpty():
            return
        margin = scaled(48)
        rect = rect.adjusted(-margin, -margin, margin, margin)
        self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        scale = self._view.transform().m11()
        if scale > 1.0:
            self._view.scale(1.0 / scale, 1.0 / scale)
            self._view.centerOn(rect.center())

    def _inject_channel_choices(self, specs, names: List[str]) -> None:
        """Populate channel 'choice' params with live channel names in place —
        mirrors ``analysis_page._load_params``."""
        for spec in specs:
            if spec.name == "channel_name" and names:
                spec.choices = list(names)
                if spec.default not in names:
                    spec.default = names[0]
            elif spec.name in ("counterstain_channel", "intensity_channel") and names:
                # Optional channels — a "None" sentinel means "skip this channel".
                spec.choices = ["None"] + list(names)
                if spec.default not in spec.choices:
                    spec.default = "None"

    def _active_record(self):
        mw = self.main_window
        if mw is None or getattr(mw, "exp_manager", None) is None:
            return None
        return mw.exp_manager.active

    def _current_slice(self):
        return self._doc.slice_for(self._stage)

    def _set_status(self, text: str) -> None:
        mw = self.main_window
        setter = getattr(mw, "set_status_text", None)
        if callable(setter):
            setter(text)
