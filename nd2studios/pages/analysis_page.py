"""
General Analysis page — extensible pipeline runner.

Hosts the AnalysisPipeline registry: one pipeline is selected from a
drop-down, its parameters are rendered via ParamEditor, and the work is
submitted to the project-wide :class:`~nd2studios.compute.JobRunner` as
either a :class:`PipelinePreviewJob` (single frame, ``"analysis_preview"``
key) or a :class:`PipelineCommitJob` (full stack per M position,
``"analysis_commit"`` key). Results (label masks + measurements +
summary) are displayed with a label-mask overlay on the live
MultiAxisViewer and a metrics panel. The viewer is wired to the
experiment on tab entry so the user can navigate the file and tweak
parameters before pressing Run.

**Screening mode:** a "Screen frame" button (or "Auto-screen" checkbox)
runs the pipeline on the *single frame currently visible in the viewer*
and overlays the result immediately. When Auto-screen is enabled,
moving the T/M/Z sliders **or** editing any parameter restarts a 300 ms
debounce timer; firing submits a preview job. The runner cancels any
in-flight predecessor under the same key, so rapid edits collapse to
exactly one in-flight preview. Screen results are overlaid only for
their specific frame; full-analysis results (from "Run Analysis") cover
all other frames.

Export buttons write label masks as int32 TIFF and measurements as CSV
(full-analysis results only).
"""
from __future__ import annotations

import copy
import csv
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from skimage.transform import resize as sk_resize
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from nd2studios.backend.analysis.manual_mask import (
    DEFAULT_EDIT_VERTEX_COUNT,
    expand_polygon_uniformly,
    rasterize_shapes,
    shape_to_editable_polygon,
)
from nd2studios.backend.exporters.tiff_exporter import export_tiff_stack
from nd2studios.compute import (
    JobResult,
    PipelineCommitJob,
    PipelinePreviewJob,
)
from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import ParamEditor
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer


MANUAL_MASK_PIPELINE_NAME = "Manual Mask"

# V1.37 Phase 5 — coalescing keys for the project-wide JobRunner. Each
# value is shared across every submission of the same tier on this
# page, so a fresh submission cancels the in-flight predecessor.
_PREVIEW_KEY = "analysis_preview"
_COMMIT_KEY = "analysis_commit"


class AnalysisPage(QWidget):
    """Page 4: General Analysis — select and run analysis pipelines."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window

        # V1.37 Phase 5 — both preview and commit work go through the
        # project-wide JobRunner instead of per-page QThread workers.
        # ``job_runner`` is built in :class:`MainWindow.__init__` before
        # any page is constructed, so it's safe to read here. We tolerate
        # ``main_window is None`` for headless unit tests by falling
        # back to ``None`` — those paths simply never submit jobs.
        self._runner = getattr(main_window, "job_runner", None)
        if self._runner is not None:
            self._runner.job_done.connect(self._on_runner_done)
            self._runner.job_cancelled.connect(self._on_runner_cancelled)
            self._runner.job_progress.connect(self._on_runner_progress)

        # Full-analysis state
        self._result: Optional[AnalysisResult] = None
        self._current_channel_names: List[str] = []

        # Per-M-position full-analysis results {m_index: AnalysisResult}
        self._results_per_m: Dict[int, AnalysisResult] = {}

        # Multi-M sequential run state
        self._run_queue: List[int] = []
        self._run_total: int = 0
        self._current_run_m: int = 0
        self._run_pipeline_cls = None
        self._run_params: Dict[str, Any] = {}

        # Screening state (single-frame preview). _screen_frame /
        # _screen_m are also embedded in each preview job's ``tag`` so
        # stale completions are filtered in ``_on_runner_done``.
        self._screen_result: Optional[AnalysisResult] = None
        self._screen_frame: int = -1   # viewer T index of the last screen run
        self._screen_m: int = 0        # viewer M index of the last screen run

        # Overlay visibility toggle
        self._overlay_visible: bool = True

        # Manual-mask drawing state.
        # {m: {frame_idx: [{"type": str, "vertices": [[y, x], ...]}, ...]}}
        self._manual_shapes: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        self._current_draw_mode: Optional[str] = None

        # Mask-edit state — when active, one shape on the current (m, t) frame
        # is being edited with draggable vertex handles.
        #   _edit_shape_idx is the *combo* index; _edit_shape_keys is the
        #   parallel list mapping each combo entry to ``(z_key, local_idx)``
        #   inside ``self._manual_shapes[m][t][z_key]``.
        self._edit_active: bool = False
        self._edit_shape_idx: int = -1
        self._edit_shape_keys: List[Tuple[Any, int]] = []

        # Rasterized-mask cache: keyed by (m, t, z_slot, raster_version).
        # Avoids re-running rasterize_shapes() on every frame refresh when
        # shapes haven't changed (e.g., T-slider scrubbing with masks drawn).
        self._raster_cache: Dict[Tuple[int, int, Any, int], np.ndarray] = {}
        self._raster_version: int = 0

        # Multi-file: currently selected record (overrides exp_manager.active).
        self._selected_rec: Optional[ND2StudiosRecord] = None

        self._screen_debounce = QTimer(self)
        self._screen_debounce.setSingleShot(True)
        self._screen_debounce.setInterval(300)
        self._screen_debounce.timeout.connect(self._screen_current_frame)

        self._build_ui()
        self._populate_pipeline_combo()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        # ── Left panel: controls + results ───────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(8)

        # File selector — hidden when only one confirmed file is loaded.
        self._file_selector_row = QWidget()
        fs_layout = QHBoxLayout(self._file_selector_row)
        fs_layout.setContentsMargins(0, 0, 0, 0)
        fs_layout.setSpacing(4)
        fs_layout.addWidget(QLabel("File:"))
        self._combo_file = QComboBox()
        self._combo_file.setSizePolicy(
            self._combo_file.sizePolicy().horizontalPolicy(),
            self._combo_file.sizePolicy().verticalPolicy(),
        )
        self._combo_file.currentIndexChanged.connect(self._on_file_selected)
        fs_layout.addWidget(self._combo_file, stretch=1)
        self._file_selector_row.hide()
        left_layout.addWidget(self._file_selector_row)

        # Pipeline selector
        pipeline_group = QGroupBox("Pipeline")
        pg_layout = QVBoxLayout(pipeline_group)
        pg_layout.setContentsMargins(8, 8, 8, 8)
        pg_layout.setSpacing(4)

        self.combo_pipeline = QComboBox()
        self.combo_pipeline.currentTextChanged.connect(self._on_pipeline_changed)
        pg_layout.addWidget(self.combo_pipeline)

        self.lbl_pipeline_desc = QLabel()
        self.lbl_pipeline_desc.setWordWrap(True)
        self.lbl_pipeline_desc.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        pg_layout.addWidget(self.lbl_pipeline_desc)
        left_layout.addWidget(pipeline_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        param_inner = QWidget()
        param_inner_layout = QVBoxLayout(param_inner)
        param_inner_layout.setContentsMargins(4, 4, 4, 4)
        self.param_editor = ParamEditor()
        # V1.37 Phase 5 — let parameter edits drive a debounced preview
        # so dragging a threshold / sigma slider re-renders the overlay
        # without forcing the user to nudge the T/M/Z slider to trigger
        # a re-screen. The same debounce timer is shared with viewer
        # coords changes so two adjacent events collapse into one
        # submission.
        self.param_editor.params_changed.connect(self._on_params_changed)
        param_inner_layout.addWidget(self.param_editor)
        param_inner_layout.addStretch(1)

        param_scroll.setWidget(param_inner)
        pg_l = QVBoxLayout(param_group)
        pg_l.setContentsMargins(6, 6, 6, 6)
        pg_l.addWidget(param_scroll)
        left_layout.addWidget(param_group, stretch=1)

        # ── Drawing tools (visible only when Manual Mask pipeline is active) ──
        self._draw_group = QGroupBox("Drawing tools")
        dg = QVBoxLayout(self._draw_group)
        dg.setContentsMargins(8, 8, 8, 8)
        dg.setSpacing(4)

        tools_row = QHBoxLayout()
        self._btn_draw_rect = QPushButton("Rectangle")
        self._btn_draw_rect.setCheckable(True)
        self._btn_draw_ellipse = QPushButton("Ellipse")
        self._btn_draw_ellipse.setCheckable(True)
        self._btn_draw_polygon = QPushButton("Polygon")
        self._btn_draw_polygon.setCheckable(True)
        self._btn_draw_polygon.setToolTip(
            "Click and drag to freehand-draw a polygon. The vertices are "
            "captured along the path; on release the polygon auto-closes."
        )
        self._draw_btn_group = QButtonGroup(self)
        self._draw_btn_group.setExclusive(False)  # we handle exclusivity manually
        for b, mode in (
            (self._btn_draw_rect, "rect"),
            (self._btn_draw_ellipse, "ellipse"),
            (self._btn_draw_polygon, "polygon"),
        ):
            self._draw_btn_group.addButton(b)
            b.toggled.connect(lambda checked, m=mode, btn=b:
                              self._on_draw_tool_toggled(m, btn, checked))
            tools_row.addWidget(b)
        tools_row.addStretch(1)
        dg.addLayout(tools_row)

        actions_row = QHBoxLayout()
        self._btn_clear_frame = QPushButton("Clear frame")
        self._btn_clear_frame.setToolTip("Remove all shapes from the current frame.")
        self._btn_clear_frame.clicked.connect(self._on_clear_current_frame)
        self._btn_copy_prev = QPushButton("Copy from previous frame")
        self._btn_copy_prev.setToolTip(
            "Replace the current frame's shapes with a copy of the "
            "previous frame's shapes."
        )
        self._btn_copy_prev.clicked.connect(self._on_copy_from_prev_frame)
        self._btn_apply_all = QPushButton("Apply to all frames")
        self._btn_apply_all.setToolTip(
            "Copy the current frame's shapes to every frame in this M position."
        )
        self._btn_apply_all.clicked.connect(self._on_apply_to_all_frames)
        actions_row.addWidget(self._btn_clear_frame)
        actions_row.addWidget(self._btn_copy_prev)
        actions_row.addWidget(self._btn_apply_all)
        actions_row.addStretch(1)
        dg.addLayout(actions_row)

        self._lbl_draw_status = QLabel("—")
        self._lbl_draw_status.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        dg.addWidget(self._lbl_draw_status)

        # ── Edit panel — appears within the Drawing tools group ──
        edit_divider = QFrame()
        edit_divider.setFrameShape(QFrame.Shape.HLine)
        edit_divider.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        dg.addWidget(edit_divider)

        edit_row1 = QHBoxLayout()
        self._btn_edit = QPushButton("Edit shape")
        self._btn_edit.setCheckable(True)
        self._btn_edit.setToolTip(
            "Convert the selected shape into a polygon with evenly-spaced "
            "vertices, then drag the handles to deform it locally. "
            "Use 'Expand by' below for a uniform morphological grow/shrink."
        )
        self._btn_edit.toggled.connect(self._on_edit_toggled)
        edit_row1.addWidget(self._btn_edit)

        edit_row1.addWidget(QLabel("Shape:"))
        self._combo_edit_shape = QComboBox()
        self._combo_edit_shape.setMinimumWidth(110)
        self._combo_edit_shape.setEnabled(False)
        self._combo_edit_shape.currentIndexChanged.connect(self._on_edit_shape_combo_changed)
        edit_row1.addWidget(self._combo_edit_shape)
        edit_row1.addStretch(1)
        dg.addLayout(edit_row1)

        edit_row2 = QHBoxLayout()
        edit_row2.addWidget(QLabel("Expand by (px):"))
        self._spin_expand = QDoubleSpinBox()
        self._spin_expand.setRange(-200.0, 200.0)
        self._spin_expand.setDecimals(1)
        self._spin_expand.setSingleStep(1.0)
        self._spin_expand.setValue(1.0)
        self._spin_expand.setFixedWidth(80)
        self._spin_expand.setEnabled(False)
        edit_row2.addWidget(self._spin_expand)
        self._btn_apply_expand = QPushButton("Apply")
        self._btn_apply_expand.setEnabled(False)
        self._btn_apply_expand.setToolTip(
            "Uniformly grow (positive) or shrink (negative) the selected "
            "shape by the given pixel amount, then re-derive its outline."
        )
        self._btn_apply_expand.clicked.connect(self._on_apply_expand)
        edit_row2.addWidget(self._btn_apply_expand)
        edit_row2.addStretch(1)
        dg.addLayout(edit_row2)

        self._draw_group.hide()
        left_layout.addWidget(self._draw_group)

        # Run row (full analysis)
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.progress_bar)
        run_row.addWidget(self.lbl_status)
        run_row.addStretch(1)
        left_layout.addLayout(run_row)

        # Screen row (single-frame preview)
        screen_row = QHBoxLayout()
        self.btn_screen = QPushButton("Screen frame")
        self.btn_screen.setEnabled(False)
        self.btn_screen.setToolTip(
            "Run the pipeline on the currently visible frame only\n"
            "and preview the overlay immediately."
        )
        self.btn_screen.clicked.connect(self._screen_current_frame)
        self.chk_auto_screen = QCheckBox("Auto-screen")
        self.chk_auto_screen.setToolTip(
            "Automatically re-screen the current frame whenever\n"
            "you move the T / M / Z slider (300 ms debounce)."
        )
        self.lbl_screen_status = QLabel("—")
        self.lbl_screen_status.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        screen_row.addWidget(self.btn_screen)
        screen_row.addWidget(self.chk_auto_screen)
        screen_row.addWidget(self.lbl_screen_status)
        screen_row.addStretch(1)
        left_layout.addLayout(screen_row)

        # Results widget (hidden until full analysis completes)
        self._results_widget = QWidget()
        self._results_widget.hide()
        results_layout = QVBoxLayout(self._results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        # Summary group
        summary_group = QGroupBox("Summary")
        self._summary_form = QFormLayout(summary_group)
        self._summary_form.setContentsMargins(8, 8, 8, 8)
        self._summary_labels: Dict[str, QLabel] = {}
        for key, display in [
            ("total_objects", "Total objects"),
            ("n_frames_with_objects", "Frames with objects"),
            ("mean_area_px", "Mean area (px²)"),
            ("std_area_px", "Std area (px²)"),
            ("mean_area_um2", "Mean area (µm²)"),
            ("std_area_um2", "Std area (µm²)"),
        ]:
            lbl = QLabel("—")
            self._summary_labels[key] = lbl
            self._summary_form.addRow(display + ":", lbl)
        results_layout.addWidget(summary_group)

        # Per-frame table
        table_group = QGroupBox("Per-frame")
        tg_layout = QVBoxLayout(table_group)
        tg_layout.setContentsMargins(4, 4, 4, 4)
        self.tbl_results = QTableWidget(0, 3)
        self.tbl_results.setHorizontalHeaderLabels(["Frame", "Objects", "Mean area (px²)"])
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_results.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tg_layout.addWidget(self.tbl_results)
        results_layout.addWidget(table_group, stretch=1)

        # Export buttons
        export_row = QHBoxLayout()
        self.btn_export_tiff = QPushButton("Export Label Masks (TIFF)")
        self.btn_export_tiff.setEnabled(False)
        self.btn_export_tiff.clicked.connect(self._on_export_tiff)
        self.btn_export_csv = QPushButton("Export Measurements (CSV)")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        export_row.addWidget(self.btn_export_tiff)
        export_row.addWidget(self.btn_export_csv)
        export_row.addStretch(1)
        results_layout.addLayout(export_row)

        left_layout.addWidget(self._results_widget, stretch=1)

        main_splitter.addWidget(left)

        # ── Right panel: live MultiAxisViewer ────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(0)

        overlay_row = QHBoxLayout()
        overlay_row.addStretch(1)
        self.btn_toggle_overlay = QPushButton("Hide Overlays")
        self.btn_toggle_overlay.setCheckable(True)
        self.btn_toggle_overlay.setFixedWidth(130)
        self.btn_toggle_overlay.clicked.connect(self._on_toggle_overlay)
        overlay_row.addWidget(self.btn_toggle_overlay)
        right_layout.addLayout(overlay_row)

        self.viewer = MultiAxisViewer(self, show_tile_preview=False)
        self.viewer.coords_changed.connect(self._on_viewer_coords_changed)
        self.viewer.shape_drawn.connect(self._on_shape_drawn)
        self.viewer.vertex_moved.connect(self._on_vertex_moved)
        right_layout.addWidget(self.viewer, stretch=1)

        main_splitter.addWidget(right)
        main_splitter.setSizes([380, 720])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        outer.addWidget(main_splitter, stretch=1)

    def _populate_pipeline_combo(self) -> None:
        """Fill the pipeline combo from the registry."""
        self.combo_pipeline.blockSignals(True)
        self.combo_pipeline.clear()
        for pipeline_cls in AnalysisPipeline.get_pipelines():
            self.combo_pipeline.addItem(pipeline_cls.name)
        self.combo_pipeline.blockSignals(False)
        if self.combo_pipeline.count() > 0:
            self._on_pipeline_changed(self.combo_pipeline.currentText())

    # ── Lifecycle hooks (called by MainWindow) ────────────────────────────────

    def on_activated(self) -> None:
        """Wire the viewer to the current experiment and refresh channel choices."""
        self._rebuild_file_selector()
        exp = self._active_exp()
        if exp is None:
            return

        # Adopt the record's manual-mask shapes so live overlay + drawing
        # tools operate on the right per-experiment state.
        self._manual_shapes = copy.deepcopy(getattr(exp, "manual_mask_shapes", {}) or {})

        # V1.38 Phase 6 — if the Recipe page released
        # ``_processed_channels`` after committing the recipe, ask the
        # ``EnhancedDataset`` proxy to materialize once now so the
        # viewer + downstream analysis sees a normal dict. The proxy
        # caches, so re-entering Analysis is cheap.
        if (
            exp._processed_channels is None
            and exp._processed_view is not None
        ):
            try:
                exp._processed_channels = exp._processed_view.materialize_all()
            except Exception as exc:  # noqa: BLE001
                if self.main_window is not None:
                    self.main_window.set_status_text(
                        f"Recipe rehydrate failed: {exc}"
                    )

        channels = exp._processed_channels or exp._raw_channels or {}

        # Wire viewer with volume (ND2) or flat channels (TIFF / post-recipe).
        if exp._raw_volume is not None:
            self.viewer.set_volume(
                exp._raw_volume,
                channel_display=exp.channel_display,
                z_mode=exp.z_view_mode or "max",
                z_index=exp.z_view_index,
                m=exp.m_index, t=0, z=exp.z_view_index,
            )
        elif channels:
            self.viewer.set_channels(channels, channel_display=exp.channel_display)

        has_data = bool(channels)
        self.btn_screen.setEnabled(has_data)

        names = list(channels.keys())
        if names != self._current_channel_names:
            self._current_channel_names = names
            self._reload_params(names)

        # Restore full-analysis results if they exist for the active pipeline.
        pipeline_name = self.combo_pipeline.currentText()
        self._restore_results_from_exp(exp, pipeline_name)
        if self._result is not None:
            self._show_result(self._result, exp)

        # Register the composite overlay hook (handles both full + screen results).
        self._update_overlay()

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        """Restore pipeline selection and params from a loaded session."""
        # Restore manual-mask shapes first so they are ready before the pipeline
        # change handler triggers a status / overlay refresh.
        self._manual_shapes = copy.deepcopy(getattr(exp, "manual_mask_shapes", {}) or {})

        cfg = exp.analysis_config if hasattr(exp, "analysis_config") else {}
        if not cfg:
            return
        pipeline_name = cfg.get("pipeline", "")
        if pipeline_name and self.combo_pipeline.findText(pipeline_name) >= 0:
            self.combo_pipeline.blockSignals(True)
            self.combo_pipeline.setCurrentText(pipeline_name)
            self.combo_pipeline.blockSignals(False)
            self._on_pipeline_changed(pipeline_name, restore_values=cfg.get("params"))

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        """Persist pipeline selection and params."""
        exp.analysis_config = {
            "pipeline": self.combo_pipeline.currentText(),
            "params": self.param_editor.get_values(),
        }
        # Manual-mask drawings live alongside analysis_config so they survive
        # session save/load. Pruned of empty buckets at every level to keep
        # the manifest tidy.
        cleaned: Dict[int, Dict[int, Dict[Any, List[Dict[str, Any]]]]] = {}
        for m, per_t in self._manual_shapes.items():
            per_t_clean: Dict[int, Dict[Any, List[Dict[str, Any]]]] = {}
            for t, by_z in per_t.items():
                by_z_clean = {k: v for k, v in by_z.items() if v}
                if by_z_clean:
                    per_t_clean[t] = by_z_clean
            if per_t_clean:
                cleaned[m] = per_t_clean
        exp.manual_mask_shapes = cleaned

    # ── Pipeline / param wiring ───────────────────────────────────────────────

    def _on_pipeline_changed(self, name: str, restore_values: Optional[Dict] = None) -> None:
        # Cancel any in-flight screen; clear both result sets (pipeline-specific).
        if self._runner is not None:
            self._runner.cancel(_PREVIEW_KEY)
        self._screen_debounce.stop()
        self._result = None
        self._results_per_m.clear()
        self._screen_result = None
        self._screen_frame = -1
        self._screen_m = 0
        self.lbl_screen_status.setText("—")
        self._results_widget.hide()
        self.btn_export_tiff.setEnabled(False)
        self.btn_export_csv.setEnabled(False)

        # Drawing tools are pipeline-conditional. When leaving Manual Mask,
        # also tell the viewer to drop any active draw mode so the cursor
        # returns to normal.
        is_manual = (name == MANUAL_MASK_PIPELINE_NAME)
        self._draw_group.setVisible(is_manual)
        if not is_manual:
            self._set_draw_mode(None)
            if self._edit_active:
                self._btn_edit.setChecked(False)
        else:
            self._update_draw_status()

        self._update_overlay()

        pipeline_cls = AnalysisPipeline.get_pipeline(name)
        if pipeline_cls is None:
            return
        instance = pipeline_cls()
        self.lbl_pipeline_desc.setText(instance.description)
        self._load_params(instance, restore_values)

        # Restore cached full-analysis results for the newly selected pipeline.
        exp = self._active_exp()
        if exp is not None:
            self._restore_results_from_exp(exp, name)
            if self._result is not None:
                self._show_result(self._result, exp)
                self._update_overlay()

    def _reload_params(self, channel_names: List[str]) -> None:
        """Re-build params for the current pipeline with updated channel list."""
        name = self.combo_pipeline.currentText()
        pipeline_cls = AnalysisPipeline.get_pipeline(name)
        if pipeline_cls is None:
            return
        current_values = self.param_editor.get_values()
        instance = pipeline_cls()
        self._load_params(instance, current_values, channel_names)

    def _load_params(
        self,
        instance: AnalysisPipeline,
        restore_values: Optional[Dict] = None,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        specs = instance.get_params()
        names = channel_names or self._current_channel_names
        for spec in specs:
            if spec.name == "channel_name" and names:
                spec.choices = names
                if spec.default not in names:
                    spec.default = names[0]
            elif spec.name == "counterstain_channel" and names:
                spec.choices = ["None"] + names
                if spec.default not in spec.choices:
                    spec.default = "None"
        self.param_editor.set_params(specs)
        if restore_values:
            self.param_editor.set_values(restore_values)

    # ── Full analysis: Run / Cancel ───────────────────────────────────────────

    def _on_run(self) -> None:
        exp = self._active_exp()
        if exp is None:
            QMessageBox.warning(self, "No data", "Import a file before running analysis.")
            return

        channels_raw = exp._processed_channels or exp._raw_channels
        if not channels_raw:
            QMessageBox.warning(self, "No data", "No channel data available. Load a file first.")
            return

        pipeline_name = self.combo_pipeline.currentText()
        pipeline_cls = AnalysisPipeline.get_pipeline(pipeline_name)
        if pipeline_cls is None:
            return

        params = self.param_editor.get_values()

        # Determine all M positions in the file to analyze.
        vol = getattr(exp, "_raw_volume", None)
        if vol is not None and exp._processed_channels is None:
            m_positions = list(range(vol.n_multipoints))
        else:
            m_positions = [0]

        # Memory size warning — estimate for one M, scaled by total count.
        sample_ch = self._build_channels_for_m(exp, m_positions[0], params)
        if sample_ch:
            total_bytes = sum(a.nbytes for a in sample_ch.values()) * len(m_positions)
            if total_bytes > 2 * 1024 ** 3:
                reply = QMessageBox.question(
                    self, "Large dataset",
                    f"The selected data is ~{total_bytes / 1024**3:.1f} GB. "
                    "Label masks will use similar memory. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

        # Store run context for sequential multi-M execution.
        self._run_pipeline_cls = pipeline_cls
        self._run_params = params
        self._results_per_m.clear()
        self._run_queue = m_positions[:]
        self._run_total = len(m_positions)

        # Cancel any in-flight screen; clear stale overlay while run executes.
        if self._runner is not None:
            self._runner.cancel(_PREVIEW_KEY)
        self._screen_debounce.stop()
        self._screen_result = None
        self._screen_frame = -1
        self.lbl_screen_status.setText("—")
        self._results_widget.hide()
        self._update_overlay()

        self.btn_run.setEnabled(False)
        self.btn_screen.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export_tiff.setEnabled(False)
        self.btn_export_csv.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting…")
        self._start_next_m_run()

    def _build_channels_for_m(
        self, exp: "ND2StudiosRecord", m: int, params: Dict[str, Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Build per-channel (T, H, W) arrays for a single M position."""
        channels_raw = exp._processed_channels or exp._raw_channels
        if not channels_raw:
            return None
        selected_ch = params.get("channel_name") or next(iter(channels_raw))
        channels: Dict[str, np.ndarray] = {}
        vol = getattr(exp, "_raw_volume", None)
        if vol is not None and exp._processed_channels is None:
            z_mode = getattr(exp, "z_view_mode", None) or "max"
            z_index = int(getattr(exp, "z_view_index", None) or 0)
            for c_idx, ch_name in enumerate(vol.channel_names):
                if ch_name == selected_ch:
                    channels[ch_name] = vol.to_lazy_channel(
                        c_idx, m=m, z_mode=z_mode, z_index=z_index
                    ).materialize()
                    break
            if not channels:
                channels = {
                    ch_name: vol.to_lazy_channel(
                        c_idx, m=m, z_mode=z_mode, z_index=z_index
                    ).materialize()
                    for c_idx, ch_name in enumerate(vol.channel_names)
                }
        else:
            for ch, arr in channels_raw.items():
                if ch == selected_ch:
                    channels[ch] = np.asarray(arr)
            if not channels:
                channels = {k: np.asarray(v) for k, v in channels_raw.items()}
        return channels or None

    def _start_next_m_run(self) -> None:
        """Pop the next M position from the queue and submit a commit job."""
        exp = self._active_exp()
        if exp is None or not self._run_queue:
            return
        if self._runner is None:
            self._set_idle("Error: job runner unavailable.")
            return
        m = self._run_queue.pop(0)
        self._current_run_m = m

        channels = self._build_channels_for_m(exp, m, self._run_params)
        if not channels:
            self._set_idle("Error: no channel data for M position.")
            return

        metadata = dict(exp.nd2_metadata)
        metadata["pixel_size_um"] = exp.pixel_size_um

        label = (
            f"M {self._run_total - len(self._run_queue)}/{self._run_total}: Running…"
            if self._run_total > 1 else "Running…"
        )
        self.lbl_status.setText(label)

        # Per-M shape injection for the Manual Mask pipeline. Each M has its
        # own drawn-shape dict; we splice it into params just before the job
        # picks it up so the pipeline can stay pure.
        params_for_job = dict(self._run_params)
        pipeline_name = self.combo_pipeline.currentText()
        if pipeline_name == MANUAL_MASK_PIPELINE_NAME:
            params_for_job["frame_shapes"] = copy.deepcopy(
                self._manual_shapes.get(m, {})
            )

        # V1.37 Phase 5 — replaces the legacy AnalysisWorker QThread.
        # The runner coalesces by key, so submitting a new commit while
        # one is in flight cancels the predecessor (used by the Cancel
        # button via runner.cancel(_COMMIT_KEY)).
        job = PipelineCommitJob(
            key=_COMMIT_KEY,
            pipeline_cls=self._run_pipeline_cls,
            channels=channels,
            metadata=metadata,
            params=params_for_job,
            m_index=m,
        )
        self._runner.submit(job)

    def _on_m_progress(self, p: int) -> None:
        """Scale single-M worker progress (0–100) into overall multi-M progress."""
        n_done = self._run_total - len(self._run_queue) - 1
        overall = int((n_done * 100 + p) / max(self._run_total, 1))
        self.progress_bar.setValue(overall)

    def _on_cancel(self) -> None:
        self._run_queue.clear()
        if self._runner is not None:
            self._runner.cancel(_COMMIT_KEY)
        self._set_idle("Cancelled.")

    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _on_finished(self, result: AnalysisResult) -> None:
        m = self._current_run_m
        self._results_per_m[m] = result
        self._result = result

        exp = self._active_exp()
        if exp is not None:
            pipeline_name = self.combo_pipeline.currentText()
            stored = exp.analysis_results.get(pipeline_name)
            if not isinstance(stored, dict):
                stored = {}
                exp.analysis_results[pipeline_name] = stored
            stored[m] = result
            # V1.38 Phase 6 — stream this M's label masks to the
            # workspace so multi-M runs do not pile up GiB of int32
            # arrays in RAM. The page-leave hook in
            # ``MainWindow._navigate`` then drops the arrays once the
            # user navigates away.
            self._commit_analysis_m(pipeline_name, m, result)

        self._update_overlay()

        if self._run_queue:
            n_done = self._run_total - len(self._run_queue)
            self.lbl_status.setText(f"M {n_done}/{self._run_total} done…")
            self._start_next_m_run()
        else:
            self._set_idle("Done.")
            self.progress_bar.setValue(100)
            self.btn_export_tiff.setEnabled(True)
            self.btn_export_csv.setEnabled(True)
            self._show_result(result, exp)
            self._finalize_analysis_stage()

    def _commit_analysis_m(
        self, pipeline_name: str, m: int, result: AnalysisResult,
    ) -> None:
        """Persist one M's label masks to the workspace."""
        if self.main_window is None:
            return
        stage = self.main_window.analysis_stage(pipeline_name)
        if stage is None:
            return
        try:
            stage.commit_m(m, result)
        except OSError as exc:
            self.main_window.set_status_text(
                f"Workspace write failed for M={m} ({exc.__class__.__name__})"
            )

    def _finalize_analysis_stage(self) -> None:
        """Stamp the manifest record after a multi-M run drains."""
        if self.main_window is None:
            return
        pipeline_name = self.combo_pipeline.currentText()
        stage = self.main_window.analysis_stage(pipeline_name)
        if stage is None:
            return
        try:
            stage.commit()
        except OSError:
            pass

    def _on_error(self, msg: str) -> None:
        self._set_idle("Error.")
        if "Cellpose is not installed" in msg or "cellpose" in msg.lower():
            QMessageBox.warning(
                self, "Cellpose not installed",
                "This pipeline requires Cellpose.\n\n"
                "Install with:\n    pip install cellpose\n\n"
                "Then restart ND2Studios.",
            )
        else:
            QMessageBox.critical(self, "Analysis error", msg)

    def _set_idle(self, status: str = "Ready") -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(status)
        self.btn_screen.setEnabled(self._has_data())

    # ── V1.37 Phase 5 — JobRunner signal dispatch ────────────────────────────

    def _on_runner_done(self, result: JobResult) -> None:
        """Dispatch a completed job back to the matching handler.

        The runner is shared with the rest of the app, so every page
        listening to ``job_done`` sees every result. We filter on
        :attr:`JobResult.key` before doing anything.
        """
        if result.key == _PREVIEW_KEY:
            if not result.ok:
                self.lbl_screen_status.setText("Screen error")
                return
            # The runner cancels any in-flight predecessor on each
            # submit, so any ``job_done`` we see for the preview key
            # was the latest submission. ``_screen_frame`` /
            # ``_screen_m`` are the (m, t) it ran on; overlay code
            # uses those to decide where to paint.
            self._screen_result = result.value
            self.lbl_screen_status.setText(
                f"Frame {self._screen_frame + 1} screened"
            )
            self._update_overlay()
        elif result.key == _COMMIT_KEY:
            if not result.ok:
                self._on_error(result.error or "Unknown error")
                return
            self._on_finished(result.value)

    def _on_runner_cancelled(self, key: str) -> None:
        """Quietly absorb a cancellation.

        Cancellations happen for two reasons:
          (1) the user clicked Cancel — UI state already moved to
              idle in :meth:`_on_cancel`.
          (2) a fresh submission under the same key superseded the
              previous one — there is by design nothing to update.
        Either way, no UI change is needed here.
        """
        # Defensive — if a commit cancellation arrives while the run
        # queue is empty (user clicked Cancel mid-flight), make sure
        # the buttons are in the idle state.
        if key == _COMMIT_KEY and not self._run_queue:
            self._set_idle("Cancelled.")

    def _on_runner_progress(self, key: str, fraction: float, message: str) -> None:
        """Translate runner progress into the existing progress widgets."""
        if key == _COMMIT_KEY:
            # PipelineCommitJob reports 0–1 for one M; scale into the
            # multi-M progress bar exactly like ``_on_m_progress`` did
            # for the legacy AnalysisWorker.
            self._on_m_progress(int(fraction * 100))
            if message:
                self._on_status(message)
        # Preview progress is sub-second; surfacing it would just
        # cause label thrash. Status was already set on submit.

    # ── Screening: single-frame preview ──────────────────────────────────────

    def _on_viewer_coords_changed(self, m: int, t: int, z: int) -> None:
        """Restart the debounce timer when the viewer position changes."""
        if self.chk_auto_screen.isChecked() and self._has_data():
            self._screen_debounce.start()
        if self.combo_pipeline.currentText() == MANUAL_MASK_PIPELINE_NAME:
            self._update_draw_status()
            # Edit targets are per-frame — leaving the frame must close the
            # edit session cleanly. The user can re-enter on the new frame.
            if self._edit_active:
                self._btn_edit.setChecked(False)

    def _on_params_changed(self, _values: Dict[str, Any]) -> None:
        """Restart the preview debounce when any parameter changes.

        Mirrors :meth:`_on_viewer_coords_changed`. Auto-screen is the
        gate — without it on, the user uses the explicit Screen frame
        button. With it on, parameter edits behave the same way slider
        moves do.
        """
        if self.chk_auto_screen.isChecked() and self._has_data():
            self._screen_debounce.start()

    def _screen_current_frame(self) -> None:
        """Submit a single-frame preview job to the V1.37 JobRunner."""
        self._screen_debounce.stop()

        exp = self._active_exp()
        if exp is None or self._runner is None:
            return

        channels_raw = exp._processed_channels or exp._raw_channels
        if not channels_raw:
            return

        pipeline_name = self.combo_pipeline.currentText()
        pipeline_cls = AnalysisPipeline.get_pipeline(pipeline_name)
        if pipeline_cls is None:
            return

        params = self.param_editor.get_values()
        selected_ch = params.get("channel_name") or next(iter(channels_raw))

        m, t, _ = self.viewer.coords()

        # Extract the single (m, t) frame without materialising the whole stack.
        vol = getattr(exp, "_raw_volume", None)
        if vol is not None and exp._processed_channels is None and selected_ch in vol.channel_names:
            z_mode = getattr(exp, "z_view_mode", None) or "max"
            z_index = int(getattr(exp, "z_view_index", None) or 0)
            c_idx = vol.channel_names.index(selected_ch)
            frame_2d = vol.get_frame(c_idx, m=m, t=t, z=z_index, z_mode=z_mode)
            frame_arr = frame_2d[np.newaxis]  # (1, H, W)
        elif selected_ch in channels_raw:
            arr = channels_raw[selected_ch]
            if isinstance(arr, np.ndarray):
                frame_arr = arr[[min(t, arr.shape[0] - 1)]] if arr.ndim == 3 else arr[np.newaxis]
            else:
                # LazyND2Channel: __getitem__(t) reads only the one frame
                frame_2d = arr[t]
                frame_arr = frame_2d[np.newaxis]
        else:
            return

        metadata = dict(exp.nd2_metadata)
        metadata["pixel_size_um"] = exp.pixel_size_um

        # Track which (m, t) this preview belongs to so stale completions
        # are silently discarded. The tag travels on the job so
        # _on_runner_done can verify without an extra signal hop.
        self._screen_frame = t
        self._screen_m = m

        # V1.37 Phase 5 — replaces the hand-rolled cancel-and-replace
        # logic. The runner cancels any in-flight preview under the
        # same key before scheduling this one.
        job = PipelinePreviewJob(
            key=_PREVIEW_KEY,
            pipeline_cls=pipeline_cls,
            channel_name=selected_ch,
            frame=frame_arr,
            metadata=metadata,
            params=params,
            tag=(m, t),
        )
        self.lbl_screen_status.setText(f"Screening frame {t + 1}…")
        self._runner.submit(job)

    # ── Manual-mask drawing tools ────────────────────────────────────────────

    def _on_draw_tool_toggled(self, mode: str, btn: QPushButton, checked: bool) -> None:
        """Handle one of the three draw-tool buttons being toggled."""
        if not checked:
            # User un-toggled the active tool — disable drawing entirely.
            if self._current_draw_mode == mode:
                self._set_draw_mode(None)
            return
        # Untoggle the other two tools so exactly one is active.
        for other in (self._btn_draw_rect, self._btn_draw_ellipse, self._btn_draw_polygon):
            if other is not btn and other.isChecked():
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
        self._set_draw_mode(mode)

    def _set_draw_mode(self, mode: Optional[str]) -> None:
        """Set the active draw mode and propagate it to the viewer."""
        self._current_draw_mode = mode
        self.viewer.set_draw_mode(mode)
        if mode is None:
            # Untoggle every tool button.
            for b in (self._btn_draw_rect, self._btn_draw_ellipse, self._btn_draw_polygon):
                if b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)

    def _on_shape_drawn(self, mode: str, vertices: list) -> None:
        """Record a drawn shape onto the current (m, t, z-slot) and redraw."""
        if self.combo_pipeline.currentText() != MANUAL_MASK_PIPELINE_NAME:
            return
        # New shapes invalidate any in-progress edit (shape index ordering
        # may be off, and the user is clearly adding rather than editing).
        if self._edit_active:
            self._btn_edit.setChecked(False)
        m, t, _z = self.viewer.coords()
        z_slot = self._current_z_slot()
        shape = {"type": mode, "vertices": [[float(y), float(x)] for y, x in vertices]}
        per_t = self._manual_shapes.setdefault(m, {})
        by_z = per_t.setdefault(t, {})
        by_z.setdefault(z_slot, []).append(shape)
        self._update_draw_status()
        self._invalidate_mask_cache()
        self._update_overlay()

    def _on_clear_current_frame(self) -> None:
        if self.combo_pipeline.currentText() != MANUAL_MASK_PIPELINE_NAME:
            return
        if self._edit_active:
            self._btn_edit.setChecked(False)
        m, t, _ = self.viewer.coords()
        per_m = self._manual_shapes.get(m)
        if per_m and t in per_m:
            del per_m[t]
            self._update_draw_status()
            self._invalidate_mask_cache()
            self._update_overlay()

    def _on_copy_from_prev_frame(self) -> None:
        if self.combo_pipeline.currentText() != MANUAL_MASK_PIPELINE_NAME:
            return
        if self._edit_active:
            self._btn_edit.setChecked(False)
        m, t, _ = self.viewer.coords()
        if t <= 0:
            return
        per_m = self._manual_shapes.setdefault(m, {})
        prev = per_m.get(t - 1)
        if not prev:
            return
        per_m[t] = copy.deepcopy(prev)
        self._update_draw_status()
        self._update_overlay()

    def _on_apply_to_all_frames(self) -> None:
        if self.combo_pipeline.currentText() != MANUAL_MASK_PIPELINE_NAME:
            return
        if self._edit_active:
            self._btn_edit.setChecked(False)
        m, t, _ = self.viewer.coords()
        per_m = self._manual_shapes.setdefault(m, {})
        current = per_m.get(t) or {}
        if not current:
            return
        T = self._n_timepoints()
        if T <= 0:
            return
        for ft in range(T):
            per_m[ft] = copy.deepcopy(current)
        self._update_draw_status()
        self._update_overlay()

    def _update_draw_status(self) -> None:
        m, t, z = self.viewer.coords()
        z_slot = self._current_z_slot()
        per_m = self._manual_shapes.get(m, {})
        by_z = per_m.get(t, {})
        n_here = len(by_z.get(z_slot, []))
        n_all = len(by_z.get("all", []))
        n_total_frame = sum(len(v) for v in by_z.values())
        if z_slot == "all":
            target = "Z=all"
            piece = f"{n_here} all-Z"
        else:
            target = f"Z={int(z) + 1}"
            piece = f"{n_here} at this Z + {n_all} all-Z"
        n_total_m = sum(
            sum(len(v) for v in d.values())
            for d in per_m.values()
        )
        self._lbl_draw_status.setText(
            f"Frame T={t + 1}, {target}, M={m}: {piece} · "
            f"{n_total_frame} on this frame · {n_total_m} total in M"
        )

    def _current_z_slot(self) -> Any:
        """Return the z-slot key new shapes should be filed under.

        For projected viewing (z_mode != "none" or n_z == 1) the user can
        only see a single composite slice; shapes go under the string
        ``"all"`` and the pipeline replicates them across every Z.  In
        per-Z viewing the slot is the current integer Z index so each
        slice can carry its own shapes.
        """
        z_mode = getattr(self.viewer, "_z_mode", None) or "max"
        vol = getattr(self.viewer, "_volume", None)
        n_z = int(getattr(vol, "n_zslices", 1)) if vol is not None else 1
        if z_mode == "none" and n_z > 1:
            _, _, z = self.viewer.coords()
            return int(z)
        return "all"

    def _shapes_visible_at(self, m: int, t: int, z: int) -> List[Dict[str, Any]]:
        """Concatenate the shapes that should appear on the overlay at (m, t, z).

        "all" shapes are always visible.  Per-Z shapes only appear when the
        viewer is showing that Z slice.  In projected viewing ``z`` is the
        viewer's stored z (typically 0); per-Z shapes from a previous per-Z
        editing session would be hidden in projected view, which is the
        intended behavior.
        """
        by_z = self._manual_shapes.get(m, {}).get(t, {})
        out: List[Dict[str, Any]] = list(by_z.get("all", []))
        out.extend(by_z.get(int(z), []))
        return out

    def _n_timepoints(self) -> int:
        """Return T for the active viewer/experiment, or 0 if unknown."""
        vol = getattr(self.viewer, "_volume", None)
        if vol is not None:
            return int(vol.n_timepoints)
        exp = self._active_exp()
        if exp is None:
            return 0
        channels = exp._processed_channels or exp._raw_channels or {}
        for arr in channels.values():
            try:
                if arr.ndim == 3:
                    return int(arr.shape[0])
            except AttributeError:
                continue
        return 0

    def _frame_hw(self) -> Optional[Tuple[int, int]]:
        """Return (H, W) of the displayed frame, or None if unavailable."""
        vol = getattr(self.viewer, "_volume", None)
        if vol is not None:
            return int(vol.height), int(vol.width)
        exp = self._active_exp()
        if exp is None:
            return None
        channels = exp._processed_channels or exp._raw_channels or {}
        for arr in channels.values():
            try:
                if arr.ndim == 3:
                    return int(arr.shape[1]), int(arr.shape[2])
            except AttributeError:
                continue
        return None

    # ── Edit-mode handlers ───────────────────────────────────────────────────

    def _on_edit_toggled(self, checked: bool) -> None:
        """Enter or leave vertex-edit mode for a Manual Mask shape."""
        if self.combo_pipeline.currentText() != MANUAL_MASK_PIPELINE_NAME:
            # Defensive — the Edit button is only inside the Manual Mask group,
            # but guard in case of unusual signal ordering.
            if checked:
                self._btn_edit.blockSignals(True)
                self._btn_edit.setChecked(False)
                self._btn_edit.blockSignals(False)
            return

        if checked:
            # Editing is exclusive with drawing — drop any active draw tool.
            self._set_draw_mode(None)
            self._enter_edit_mode()
        else:
            self._exit_edit_mode()

    def _enter_edit_mode(self) -> None:
        m, t, _ = self.viewer.coords()
        by_z = self._manual_shapes.get(m, {}).get(t, {})
        flat = self._flatten_edit_shapes(by_z)
        if not flat:
            QMessageBox.information(
                self, "No shape to edit",
                "Draw a shape on this frame first, then click Edit shape."
            )
            self._btn_edit.blockSignals(True)
            self._btn_edit.setChecked(False)
            self._btn_edit.blockSignals(False)
            return

        self._edit_active = True
        self._edit_shape_keys = [(z_key, local_idx) for z_key, local_idx, _ in flat]
        labels = [self._edit_label_for(z_key, local_idx, n)
                  for n, (z_key, local_idx, _) in enumerate(flat)]
        self._populate_edit_shape_combo(labels)
        # Default to the last shape (most recent within whichever slot).
        self._edit_shape_idx = len(flat) - 1
        self._combo_edit_shape.blockSignals(True)
        self._combo_edit_shape.setCurrentIndex(self._edit_shape_idx)
        self._combo_edit_shape.blockSignals(False)
        self._combo_edit_shape.setEnabled(True)
        self._spin_expand.setEnabled(True)
        self._btn_apply_expand.setEnabled(True)
        # Disable the three draw tool buttons while editing (mutex).
        for b in (self._btn_draw_rect, self._btn_draw_ellipse, self._btn_draw_polygon):
            b.setEnabled(False)
        self._push_edit_vertices_to_canvas()

    def _exit_edit_mode(self) -> None:
        self._edit_active = False
        self._edit_shape_idx = -1
        self._edit_shape_keys = []
        self._combo_edit_shape.clear()
        self._combo_edit_shape.setEnabled(False)
        self._spin_expand.setEnabled(False)
        self._btn_apply_expand.setEnabled(False)
        for b in (self._btn_draw_rect, self._btn_draw_ellipse, self._btn_draw_polygon):
            b.setEnabled(True)
        self.viewer.set_edit_vertices(None)
        self._update_overlay()

    def _populate_edit_shape_combo(self, labels: List[str]) -> None:
        self._combo_edit_shape.blockSignals(True)
        self._combo_edit_shape.clear()
        for lbl in labels:
            self._combo_edit_shape.addItem(lbl)
        self._combo_edit_shape.blockSignals(False)

    def _on_edit_shape_combo_changed(self, index: int) -> None:
        if not self._edit_active or index < 0:
            return
        self._edit_shape_idx = index
        self._push_edit_vertices_to_canvas()

    def _push_edit_vertices_to_canvas(self) -> None:
        """Convert the active shape to a polygon (if needed) and show handles."""
        shape = self._active_edit_shape()
        if shape is None:
            return
        # Force-convert any rect / ellipse to a polygon with evenly-spaced
        # vertices the first time it's edited so the handle behavior is uniform.
        if shape.get("type") != "polygon" or len(shape.get("vertices") or []) < 3:
            new_verts = shape_to_editable_polygon(
                shape, n_vertices=DEFAULT_EDIT_VERTEX_COUNT
            )
            shape["type"] = "polygon"
            shape["vertices"] = new_verts
            self._update_overlay()
        verts = [(float(v[0]), float(v[1])) for v in shape["vertices"]]
        self.viewer.set_edit_vertices(verts)

    def _active_edit_shape(self) -> Optional[Dict[str, Any]]:
        if not (0 <= self._edit_shape_idx < len(self._edit_shape_keys)):
            return None
        z_key, local_idx = self._edit_shape_keys[self._edit_shape_idx]
        m, t, _ = self.viewer.coords()
        shapes = self._manual_shapes.get(m, {}).get(t, {}).get(z_key, [])
        if 0 <= local_idx < len(shapes):
            return shapes[local_idx]
        return None

    @staticmethod
    def _flatten_edit_shapes(
        by_z: Dict[Any, List[Dict[str, Any]]],
    ) -> List[Tuple[Any, int, Dict[str, Any]]]:
        """Enumerate frame shapes in canonical (all, then z=0,1,...) order."""
        out: List[Tuple[Any, int, Dict[str, Any]]] = []
        if "all" in by_z:
            for i, sh in enumerate(by_z["all"]):
                out.append(("all", i, sh))
        for z_key in sorted(k for k in by_z.keys() if k != "all"):
            for i, sh in enumerate(by_z[z_key]):
                out.append((z_key, i, sh))
        return out

    @staticmethod
    def _edit_label_for(z_key: Any, _local_idx: int, n: int) -> str:
        slot = "all-Z" if z_key == "all" else f"Z={int(z_key) + 1}"
        return f"Shape {n + 1} ({slot})"

    def _on_vertex_moved(self, idx: int, iy: float, ix: float) -> None:
        """Live update from the canvas — write the new vertex into the shape."""
        if not self._edit_active:
            return
        shape = self._active_edit_shape()
        if shape is None:
            return
        verts = shape.get("vertices") or []
        if 0 <= idx < len(verts):
            verts[idx] = [float(iy), float(ix)]
            self._invalidate_mask_cache()
            self._update_overlay()

    def _on_apply_expand(self) -> None:
        """Uniformly grow / shrink the active shape via morphological dilation."""
        if not self._edit_active:
            return
        shape = self._active_edit_shape()
        if shape is None:
            return
        hw = self._frame_hw()
        if hw is None:
            QMessageBox.warning(self, "No image",
                                "Load a file before editing masks.")
            return
        H, W = hw
        expand_px = float(self._spin_expand.value())
        new_verts = expand_polygon_uniformly(
            shape["vertices"], expand_px, H, W,
            n_vertices=DEFAULT_EDIT_VERTEX_COUNT,
        )
        shape["vertices"] = new_verts
        self._push_edit_vertices_to_canvas()
        self._invalidate_mask_cache()
        self._update_overlay()

    # ── Composite overlay ─────────────────────────────────────────────────────

    def _composite_overlay(self, rgb: np.ndarray, t: int, m: int) -> np.ndarray:
        """Overlay function registered on the viewer via set_frame_post_process.

        Screen result takes priority for its specific frame; full-analysis
        results (keyed by M position) cover all other frames.  Secondary masks
        (e.g. background) are composited after the primary overlay.

        When the active pipeline is Manual Mask, an extra live preview of the
        drawn-but-not-yet-rasterized shapes is overlaid in yellow so the user
        sees their edits before pressing Run.
        """
        if not self._overlay_visible:
            return rgb

        # Live manual-mask preview takes over completely while Manual Mask is
        # the active pipeline — we render directly from the editable shape
        # state so the canvas tracks edits without waiting for Run. The
        # rasterized result cached in _results_per_m would otherwise re-paint
        # stale shapes on top.  Per-Z shapes are filtered to only show
        # those visible at the current Z slice.
        if self.combo_pipeline.currentText() == MANUAL_MASK_PIPELINE_NAME:
            _, _, current_z = self.viewer.coords()
            shapes = self._shapes_visible_at(m, t, int(current_z))
            if not shapes:
                return rgb
            h, w = rgb.shape[:2]
            rkey = (m, t, int(current_z), self._raster_version)
            live_mask = self._raster_cache.get(rkey)
            if live_mask is None:
                live_mask = np.zeros((h, w), dtype=np.int32)
                rasterize_shapes(live_mask, shapes, h, w)
                if len(self._raster_cache) >= 200:
                    self._raster_cache.pop(next(iter(self._raster_cache)))
                self._raster_cache[rkey] = live_mask
            if live_mask.max() == 0:
                return rgb
            return _overlay_labels(
                rgb, live_mask, alpha=0.45, color=(255, 220, 0)
            )

        # Single-frame screen result — only shown for the exact (m, t) analyzed.
        if (self._screen_result is not None
                and t == self._screen_frame and m == self._screen_m):
            result = self._screen_result
            ch = next(iter(result.label_masks), "")
            if ch and result.label_masks[ch].shape[0] > 0:
                frame_idx = 0
                out = _overlay_labels(
                    rgb,
                    result.label_masks[ch][frame_idx],
                    alpha=result.overlay_alpha,
                    color=result.overlay_color,
                )
                for sec_mask in result.secondary_label_masks.values():
                    if sec_mask.shape[0] > frame_idx:
                        out = _overlay_labels(
                            out,
                            sec_mask[frame_idx],
                            alpha=result.secondary_overlay_alpha,
                            color=result.secondary_overlay_color,
                        )
                return out

        # Full-analysis result — look up by current M position.
        result = self._results_per_m.get(m)
        if result is not None:
            ch = next(iter(result.label_masks), "")
            if ch and t < result.label_masks[ch].shape[0]:
                out = _overlay_labels(
                    rgb,
                    result.label_masks[ch][t],
                    alpha=result.overlay_alpha,
                    color=result.overlay_color,
                )
                for sec_mask in result.secondary_label_masks.values():
                    if t < sec_mask.shape[0]:
                        out = _overlay_labels(
                            out,
                            sec_mask[t],
                            alpha=result.secondary_overlay_alpha,
                            color=result.secondary_overlay_color,
                        )
                return out

        return rgb

    def _invalidate_mask_cache(self) -> None:
        """Discard cached rasterized masks and the viewer's overlay composites.

        Call whenever drawn shapes change so stale masks are not displayed.
        """
        self._raster_version += 1
        self._raster_cache.clear()
        self.viewer.invalidate_post_process_cache()

    def _update_overlay(self) -> None:
        """Re-register the composite overlay hook to trigger a viewer re-render."""
        self.viewer.set_frame_post_process(self._composite_overlay)

    def _on_toggle_overlay(self, checked: bool) -> None:
        """Toggle visibility of all overlays (full-analysis and screen results)."""
        self._overlay_visible = not checked
        self.btn_toggle_overlay.setText("Show Overlays" if checked else "Hide Overlays")
        self._update_overlay()

    def _restore_results_from_exp(self, exp: "ND2StudiosRecord", pipeline_name: str) -> None:
        """Populate _results_per_m and _result from the experiment's cached results."""
        self._results_per_m.clear()
        self._result = None
        stored = getattr(exp, "analysis_results", {}).get(pipeline_name)
        if isinstance(stored, dict):
            self._results_per_m = {
                m: r for m, r in stored.items() if isinstance(r, AnalysisResult)
            }
            self._result = next(iter(self._results_per_m.values()), None)
        elif isinstance(stored, AnalysisResult):
            self._results_per_m = {0: stored}
            self._result = stored
        if self._result is not None:
            self.btn_export_tiff.setEnabled(True)
            self.btn_export_csv.setEnabled(True)

    # ── Results display ───────────────────────────────────────────────────────

    def _show_result(
        self, result: AnalysisResult, exp: Optional[ND2StudiosRecord]
    ) -> None:
        # Summary labels
        s = result.summary
        for key, lbl in self._summary_labels.items():
            val = s.get(key, 0)
            if isinstance(val, float):
                lbl.setText(f"{val:.2f}")
            else:
                lbl.setText(str(val))

        # Per-frame table
        channel_name = next(iter(result.label_masks), "")
        if channel_name:
            label_stack = result.label_masks[channel_name]
            T = label_stack.shape[0]
            frame_stats: Dict[int, Dict] = defaultdict(lambda: {"count": 0, "areas": []})
            for m in result.measurements:
                t = m["frame"]
                frame_stats[t]["count"] += 1
                frame_stats[t]["areas"].append(m["area_px"])

            self.tbl_results.setRowCount(T)
            for t in range(T):
                st = frame_stats.get(t, {"count": 0, "areas": []})
                count = st["count"]
                mean_area = float(np.mean(st["areas"])) if st["areas"] else 0.0
                self.tbl_results.setItem(t, 0, QTableWidgetItem(str(t + 1)))
                self.tbl_results.setItem(t, 1, QTableWidgetItem(str(count)))
                self.tbl_results.setItem(t, 2, QTableWidgetItem(f"{mean_area:.1f}"))

        self._results_widget.show()

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export_tiff(self) -> None:
        if self._result is None:
            return
        channel_name = next(iter(self._result.label_masks), "")
        if not channel_name:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Label Masks",
            os.path.expanduser("~/labels.tif"),
            "TIFF files (*.tif *.tiff)",
        )
        if not path:
            return

        exp = self._active_exp()
        pixel_size = exp.pixel_size_um if exp is not None else None

        try:
            export_tiff_stack(
                self._result.label_masks[channel_name],
                path,
                bit_depth="passthrough",
                pixel_size_um=pixel_size,
            )
            QMessageBox.information(self, "Export complete", f"Label masks saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_export_csv(self) -> None:
        if self._result is None or not self._result.measurements:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Measurements",
            os.path.expanduser("~/measurements.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        fieldnames = list(self._result.measurements[0].keys())
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._result.measurements)
            QMessageBox.information(
                self, "Export complete",
                f"{len(self._result.measurements)} rows saved to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def on_close(self) -> None:
        """Called by :meth:`MainWindow.closeEvent` on app exit.

        Stop the debounce timer so it can't fire during shutdown and
        push a fresh preview submission past the runner's drain
        deadline. The runner itself drains the actual job pool.
        """
        try:
            self._screen_debounce.stop()
        except Exception:  # noqa: BLE001
            pass

    def _has_data(self) -> bool:
        exp = self._active_exp()
        if exp is None:
            return False
        return bool(exp._processed_channels or exp._raw_channels)

    def _active_exp(self) -> Optional[ND2StudiosRecord]:
        if self._selected_rec is not None:
            return self._selected_rec
        if self.main_window is None:
            return None
        return self.main_window.exp_manager.active

    def _reload_for_selected_record(self) -> None:
        """Refresh the viewer and result state for self._selected_rec."""
        exp = self._active_exp()
        if exp is None:
            return
        # Rehydrate processed channels if needed.
        if exp._processed_channels is None and exp._processed_view is not None:
            try:
                exp._processed_channels = exp._processed_view.materialize_all()
            except Exception:  # noqa: BLE001
                pass
        channels = exp._processed_channels or exp._raw_channels or {}
        if exp._raw_volume is not None:
            self.viewer.set_volume(
                exp._raw_volume,
                channel_display=exp.channel_display,
                z_mode=exp.z_view_mode or "max",
                z_index=exp.z_view_index,
                m=exp.m_index, t=0, z=exp.z_view_index,
            )
        elif channels:
            self.viewer.set_channels(channels, channel_display=exp.channel_display)
        self.btn_screen.setEnabled(bool(channels))
        names = list(channels.keys())
        if names != self._current_channel_names:
            self._current_channel_names = names
            self._reload_params(names)
        # Restore analysis results for this record.
        pipeline_name = self.combo_pipeline.currentText()
        self._restore_results_from_exp(exp, pipeline_name)
        if self._result is not None:
            self._show_result(self._result, exp)
        self._update_overlay()

    def _rebuild_file_selector(self) -> None:
        """Populate the file-selector combo from currently confirmed records."""
        if self.main_window is None:
            return
        fn = getattr(self.main_window, "get_confirmed_records", None)
        records: List[ND2StudiosRecord] = fn() if callable(fn) else []
        if not records:
            exp = self.main_window.exp_manager.active
            if exp is not None:
                records = [exp]

        self._combo_file.blockSignals(True)
        self._combo_file.clear()
        for rec in records:
            import os as _os
            label = _os.path.basename(
                (getattr(rec, "import_config", {}) or {}).get("filepath", "")
                or (getattr(rec, "nd2_metadata", {}) or {}).get("filepath", "")
            ) or "Untitled"
            self._combo_file.addItem(label)
        self._combo_file.blockSignals(False)

        # Show row only when multiple files are confirmed.
        self._file_selector_row.setVisible(len(records) > 1)

        # Set selected record to whichever the combo is pointing at.
        self._selected_rec = records[self._combo_file.currentIndex()] if records else None

    def _on_file_selected(self, index: int) -> None:
        """Switch the active record when the user picks a different file."""
        if self.main_window is None:
            return
        fn = getattr(self.main_window, "get_confirmed_records", None)
        records: List[ND2StudiosRecord] = fn() if callable(fn) else []
        if not records:
            exp = self.main_window.exp_manager.active
            if exp is not None:
                records = [exp]
        if 0 <= index < len(records):
            self._selected_rec = records[index]
        else:
            self._selected_rec = None
        # Reload viewer for the newly selected record.
        self._reload_for_selected_record()


# ── Module-level helpers (no Qt state) ───────────────────────────────────────

# Deterministic color palette: 20 visually distinct colors for label overlay.
_LABEL_PALETTE: List[tuple] = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
    (255, 100, 255), (100, 255, 255), (255, 165, 0),   (180, 255, 130),
    (130, 180, 255), (255, 130, 180), (200, 200, 100), (100, 200, 200),
    (200, 100, 200), (150, 255, 200), (255, 200, 150), (200, 150, 255),
    (80, 180, 80),   (180, 80, 80),   (80, 80, 180),   (180, 180, 80),
]


def _overlay_labels(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: Optional[tuple] = None,
) -> np.ndarray:
    """Blend integer label mask onto an RGB image.

    When *color* is given, every non-zero label is painted that single (R,G,B)
    color (binary-mask style).  Otherwise each label cycles through the palette.
    Label 0 is background and is always left transparent.
    """
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        mask = sk_resize(mask, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
    result = rgb.astype(np.float32)
    unique_labels = np.unique(mask)
    for label_id in unique_labels:
        if label_id == 0:
            continue
        c = color if color is not None else _LABEL_PALETTE[(int(label_id) - 1) % len(_LABEL_PALETTE)]
        region = mask == label_id
        for c_idx, c_val in enumerate(c):
            result[region, c_idx] = (1 - alpha) * result[region, c_idx] + alpha * c_val
    return np.clip(result, 0, 255).astype(np.uint8)
