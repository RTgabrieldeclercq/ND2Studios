"""
Export page — four tabs covering V1.0 outputs:

1. **Z-Projection TIFF** — single-channel multi-page TIFF, one file per
   enabled channel. Bit-depth selector. Source: raw or processed.
2. **RGB Composite TIFF** — multi-channel additive composite TIFF.
   Source: raw or processed.
3. **Movie** — MP4 / GIF time-lapse with optional scale bar, timestamp,
   and channel-label overlays. After the user clicks "Export Movie…" an
   :class:`ExportPreviewDialog` pops up so they can scrub T, tweak
   brightness / contrast / saturation / hue / fade, and confirm before
   the export worker starts.
4. **Image Sequence** — one PNG per (T, M, Z) frame with the same
   preview / adjustment workflow as the movie tab. Filename suffix
   (``_T01_M02_Z03``) only includes axes that have more than one frame.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QSpinBox, QStackedWidget, QTabWidget,
    QVBoxLayout, QWidget,
)

from nd2studios.backend.exporters.composite_exporter import ImageAdjustments
from nd2studios.backend.exporters.movie_exporter import MovieOptions
from nd2studios.backend.results_engine import (
    compute_measurements, export_label_masks_tiff, export_overlay_frames,
)
from nd2studios.core.analysis_registry import AnalysisResult
from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.widgets.export_preview_dialog import ExportPreviewDialog
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.workers.export_worker import ExportRequest, ExportWorker


class ExportPage(QWidget):
    """Page 3: export TIFF stacks, RGB composites, and movies."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._worker: Optional[ExportWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Top bar: export type selector + summary label.
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Export type:"))
        self.combo_export_type = QComboBox()
        self.combo_export_type.addItems(
            ["Raw Image", "Processed Image", "Tracked Objects", "Pipeline Results"]
        )
        self.combo_export_type.setToolTip(
            "Raw Image: export from the original ND2 data.\n"
            "Processed Image: export after the recipe pipeline is applied.\n"
            "Tracked Objects: export per-object crops from the Results tracking workflow.\n"
            "Pipeline Results: export measurements / overlays / label masks from a\n"
            "committed analysis result (Analysis or Pipelines tab)."
        )
        self.combo_export_type.currentIndexChanged.connect(self._on_export_type_changed)
        top_bar.addWidget(self.combo_export_type)
        top_bar.addStretch(1)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        top_bar.addWidget(self.lbl_summary)
        outer.addLayout(top_bar)

        # Stacked widget — index 0: image export tabs; index 1: tracked objects panel.
        self._stacked = QStackedWidget()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_tiff(), "Z-Projection TIFF")
        self.tabs.addTab(self._build_tab_composite(), "RGB Composite")
        self.tabs.addTab(self._build_tab_movie(), "Movie")
        self.tabs.addTab(self._build_tab_image_sequence(), "Image Sequence")
        self._stacked.addWidget(self.tabs)

        self._stacked.addWidget(self._build_tracked_objects_panel())
        self._stacked.addWidget(self._build_pipeline_results_panel())
        outer.addWidget(self._stacked, stretch=1)

    # ── Tab 1: TIFF stack ──
    def _build_tab_tiff(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        self.combo_tiff_bitdepth = QComboBox()
        self.combo_tiff_bitdepth.addItems(["passthrough", "uint16", "uint8"])
        form.addRow("Bit depth", self.combo_tiff_bitdepth)

        self.lbl_tiff_zproj = QLabel("Z-projection")
        self.combo_tiff_zproj = QComboBox()
        self.combo_tiff_zproj.addItems(["max", "mean", "min"])
        self.combo_tiff_zproj.setToolTip(
            "Re-project the Z stack using the chosen method at export time.\n"
            "Only available when the file has more than one Z slice."
        )
        form.addRow(self.lbl_tiff_zproj, self.combo_tiff_zproj)
        self.lbl_tiff_zproj.setVisible(False)
        self.combo_tiff_zproj.setVisible(False)

        layout.addLayout(form)

        info = QLabel(
            "Writes a single ImageJ TZCYX hyperstack TIFF containing all\n"
            "enabled channels with channel names in Labels metadata. Pixel\n"
            "size is written into the resolution + spacing tags. Same file\n"
            "construction as the Stitch dialog output."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.btn_export_tiff = QPushButton("Export TIFF Stack…")
        self.btn_export_tiff.setObjectName("primaryBtn")
        self.btn_export_tiff.clicked.connect(self._on_export_tiff)
        layout.addWidget(self.btn_export_tiff)
        layout.addStretch(1)
        return w

    # ── Tab 2: RGB composite ──
    def _build_tab_composite(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel(
            "Writes a single multi-page RGB TIFF (uint8) where each enabled\n"
            "channel is mapped to its assigned color and additively blended."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.btn_export_composite = QPushButton("Export RGB Composite TIFF…")
        self.btn_export_composite.setObjectName("primaryBtn")
        self.btn_export_composite.clicked.connect(self._on_export_composite)
        layout.addWidget(self.btn_export_composite)
        layout.addStretch(1)
        return w

    # ── Tab 3: Movie ──
    def _build_tab_movie(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Format / FPS
        basics = QGroupBox("Basics")
        bf = QFormLayout(basics)
        self.combo_movie_fmt = QComboBox()
        self.combo_movie_fmt.addItems(["mp4", "gif"])
        bf.addRow("Format", self.combo_movie_fmt)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(0.5, 60.0)
        self.spin_fps.setValue(10.0)
        self.spin_fps.setSingleStep(0.5)
        bf.addRow("FPS", self.spin_fps)
        layout.addWidget(basics)

        # Scale bar
        sb = QGroupBox("Scale bar")
        sf = QFormLayout(sb)
        self.cb_scalebar = QCheckBox("Show scale bar")
        self.cb_scalebar.setChecked(True)
        sf.addRow(self.cb_scalebar)
        self.spin_scalebar_um = QDoubleSpinBox()
        self.spin_scalebar_um.setRange(0.1, 10000.0)
        self.spin_scalebar_um.setValue(50.0)
        self.spin_scalebar_um.setSuffix(" µm")
        sf.addRow("Length", self.spin_scalebar_um)
        self.combo_scalebar_pos = QComboBox()
        self.combo_scalebar_pos.addItems(
            ["bottom-right", "bottom-left", "top-right", "top-left"])
        sf.addRow("Position", self.combo_scalebar_pos)
        self.combo_scalebar_color = QComboBox()
        self.combo_scalebar_color.addItems(["white", "yellow", "black", "magenta"])
        sf.addRow("Color", self.combo_scalebar_color)
        layout.addWidget(sb)

        # Timestamp
        ts = QGroupBox("Timestamp")
        tf = QFormLayout(ts)
        self.cb_timestamp = QCheckBox("Show timestamp")
        self.cb_timestamp.setChecked(True)
        tf.addRow(self.cb_timestamp)
        self.combo_ts_pos = QComboBox()
        self.combo_ts_pos.addItems(
            ["bottom-left", "bottom-right", "top-left", "top-right"])
        tf.addRow("Position", self.combo_ts_pos)
        self.combo_ts_color = QComboBox()
        self.combo_ts_color.addItems(["white", "yellow", "black"])
        tf.addRow("Color", self.combo_ts_color)
        self.cb_ts_use_synth = QCheckBox(
            "Use synthetic dt instead of ND2 timestamps")
        tf.addRow(self.cb_ts_use_synth)
        self.spin_ts_dt = QDoubleSpinBox()
        self.spin_ts_dt.setRange(0.001, 10000.0)
        self.spin_ts_dt.setValue(1.0)
        self.spin_ts_dt.setSuffix(" s")
        tf.addRow("dt (synthetic)", self.spin_ts_dt)
        layout.addWidget(ts)

        # Channel labels
        cl = QGroupBox("Channel labels")
        cf = QFormLayout(cl)
        self.cb_channel_labels = QCheckBox("Show channel labels")
        self.cb_channel_labels.setChecked(True)
        cf.addRow(self.cb_channel_labels)
        self.combo_chl_pos = QComboBox()
        self.combo_chl_pos.addItems(
            ["top-left", "top-right", "bottom-left", "bottom-right"])
        cf.addRow("Position", self.combo_chl_pos)
        layout.addWidget(cl)

        self.btn_export_movie = QPushButton("Export Movie…")
        self.btn_export_movie.setObjectName("primaryBtn")
        self.btn_export_movie.clicked.connect(self._on_export_movie)
        layout.addWidget(self.btn_export_movie)
        layout.addStretch(1)
        return w

    # ── Tab 4: Image Sequence ──
    def _build_tab_image_sequence(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel(
            "Writes one PNG per frame. Filenames are "
            "<basename>_TXX[_MXX][_ZXX].png — index suffixes are only "
            "added for axes with more than one frame. Overlays (scale "
            "bar, timestamp, channel labels) reuse the Movie tab settings."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QGroupBox("Output")
        ff = QFormLayout(form)
        self.le_seq_basename = QLineEdit()
        self.le_seq_basename.setPlaceholderText("Auto: derived from filename")
        ff.addRow("Base name", self.le_seq_basename)
        layout.addWidget(form)

        # Iterate-all-axes toggle (only useful when the volume has M>1 or Z>1).
        self.cb_seq_iterate_volume = QCheckBox(
            "Iterate all M / Z positions (raw — recipe not applied)"
        )
        self.cb_seq_iterate_volume.setToolTip(
            "When checked, the exporter walks every (M, T, Z) position in "
            "the file using raw pixel data. When unchecked, only the "
            "current M / Z view is exported and the recipe is preserved."
        )
        self.cb_seq_iterate_volume.toggled.connect(self._refresh_seq_summary)
        layout.addWidget(self.cb_seq_iterate_volume)

        self.lbl_seq_summary = QLabel("")
        self.lbl_seq_summary.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        self.lbl_seq_summary.setWordWrap(True)
        layout.addWidget(self.lbl_seq_summary)

        self.btn_export_seq = QPushButton("Preview & Export Image Sequence…")
        self.btn_export_seq.setObjectName("primaryBtn")
        self.btn_export_seq.clicked.connect(self._on_export_image_sequence)
        layout.addWidget(self.btn_export_seq)
        layout.addStretch(1)
        return w

    # ── Tracked objects panel ──
    def _build_tracked_objects_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel(
            "Exports each tracked object as a cropped image (3× object size), "
            "optionally with image channels and mask highlight overlay. "
            "Multiple objects can be tiled side-by-side per output file."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        options = QGroupBox("Options")
        form = QFormLayout(options)

        self.combo_tracked_pipeline = QComboBox()
        self.combo_tracked_pipeline.addItem("(no tracked objects)")
        self.combo_tracked_pipeline.setToolTip(
            "Filter by segmentation channel, or choose 'All' to export every "
            "tracked object regardless of which pipeline produced it."
        )
        form.addRow("Pipeline", self.combo_tracked_pipeline)

        self.spin_objects_per_m = QSpinBox()
        self.spin_objects_per_m.setRange(1, 100)
        self.spin_objects_per_m.setValue(1)
        self.spin_objects_per_m.setToolTip(
            "Number of tracked objects tiled side-by-side per output TIFF/PNG file.")
        form.addRow("Objects per frame", self.spin_objects_per_m)

        self.cb_tracked_include_image = QCheckBox("Include image channels")
        self.cb_tracked_include_image.setChecked(True)
        form.addRow(self.cb_tracked_include_image)

        self.cb_tracked_mask_overlay = QCheckBox("Overlay mask highlight")
        self.cb_tracked_mask_overlay.setChecked(True)
        form.addRow(self.cb_tracked_mask_overlay)

        self.combo_tracked_fmt = QComboBox()
        self.combo_tracked_fmt.addItems(["TIFF", "PNG"])
        form.addRow("Format", self.combo_tracked_fmt)

        layout.addWidget(options)

        self.btn_export_tracked = QPushButton("Export Tracked Objects…")
        self.btn_export_tracked.setObjectName("primaryBtn")
        self.btn_export_tracked.clicked.connect(self._on_export_tracked_objects)
        layout.addWidget(self.btn_export_tracked)

        layout.addStretch(1)
        return w

    def _on_export_type_changed(self, index: int) -> None:
        # 0/1 = Raw/Processed Image → image tabs (0); 2 = Tracked Objects → tracked
        # panel (1); 3 = Pipeline Results → results panel (2).
        if index < 2:
            self._stacked.setCurrentIndex(0)
        elif index == 2:
            self._stacked.setCurrentIndex(1)
        else:
            self._stacked.setCurrentIndex(2)
            self._refresh_pipeline_results()

    def _refresh_tracked_pipelines(self) -> None:
        self.combo_tracked_pipeline.blockSignals(True)
        self.combo_tracked_pipeline.clear()
        results_page = self.main_window.pages.get("results") if self.main_window else None
        measurements = getattr(results_page, "_measurements", []) if results_page else []
        tracked = [r for r in measurements if r.get("track_id") is not None]
        seg_channels = sorted({
            str(r.get("segmentation_channel", ""))
            for r in tracked
            if r.get("segmentation_channel")
        })
        if seg_channels:
            self.combo_tracked_pipeline.addItem("All")
            for ch in seg_channels:
                self.combo_tracked_pipeline.addItem(ch)
        else:
            self.combo_tracked_pipeline.addItem("(no tracked objects)")
        self.combo_tracked_pipeline.blockSignals(False)

    def _on_export_tracked_objects(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        results_page = self.main_window.pages.get("results")
        measurements = getattr(results_page, "_measurements", []) if results_page else []
        seg_filter = self.combo_tracked_pipeline.currentText()
        tracked = [
            r for r in measurements
            if r.get("track_id") is not None
            and (seg_filter in ("All", "(no tracked objects)")
                 or str(r.get("segmentation_channel", "")) == seg_filter)
        ]
        if not tracked:
            QMessageBox.information(
                self, "No tracked objects",
                "No tracked objects found. Go to the Results page, run analysis, "
                "click Compute Measurements, and ensure at least one valid track exists."
            )
            return
        label_masks = getattr(results_page, "_label_masks_for_validation", {})
        channels_data = getattr(results_page, "_channels_for_validation", {})
        exp = self.main_window.exp_manager.active
        channel_display = dict(getattr(exp, "channel_display", None) or {})

        _rp = getattr(self, "_replay_export_path", "")
        out_dir = _rp or QFileDialog.getExistingDirectory(self, "Choose output folder", "")
        if not out_dir:
            return

        self._record_macro(
            "export_tracked_objects",
            f"Export Tracked Objects → {out_dir}",
            export_dir=out_dir,
        )

        from nd2studios.backend.exporters.tracked_objects_exporter import export_tracked_objects
        objects_per_m = int(self.spin_objects_per_m.value())
        with_image = self.cb_tracked_include_image.isChecked()
        with_mask = self.cb_tracked_mask_overlay.isChecked()
        fmt = self.combo_tracked_fmt.currentText().lower()

        _replaying = getattr(
            getattr(self.main_window, "macro_recorder", None), "replaying", False
        )
        try:
            if self.main_window:
                self.main_window.set_status_text("Exporting tracked objects…")
            paths = export_tracked_objects(
                measurements=tracked,
                label_masks=label_masks,
                channels=channels_data,
                channel_display=channel_display,
                output_dir=out_dir,
                objects_per_m=objects_per_m,
                with_image=with_image,
                with_mask_overlay=with_mask,
                fmt=fmt,
                progress_cb=lambda p: self.main_window.set_progress(p) if self.main_window else None,
                basename=self._export_basename(),
            )
            if self.main_window:
                self.main_window.set_progress(0)
                self.main_window.set_status_text(
                    f"Exported {len(paths)} tracked object file(s) to {out_dir}")
            if not _replaying:
                QMessageBox.information(
                    self, "Export complete",
                    f"Wrote {len(paths)} file(s) to:\n{out_dir}"
                )
        except Exception as e:
            if self.main_window:
                self.main_window.set_progress(0)
            QMessageBox.warning(self, "Export failed", str(e))

    # ── Pipeline Results panel ──
    def _build_pipeline_results_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel(
            "Export deliverables from a committed analysis result — the output of "
            "an Analysis pipeline (Analysis tab) or the Pipelines tab's Analysis "
            "sub-tab (Apply). Run & Apply a pipeline first to populate this list."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        options = QGroupBox("Result")
        form = QFormLayout(options)
        self.combo_pipe_result = QComboBox()
        self.combo_pipe_result.addItem("(no results)")
        self.combo_pipe_result.currentIndexChanged.connect(self._refresh_pipe_summary)
        form.addRow("Pipeline result", self.combo_pipe_result)

        self.lbl_pipe_summary = QLabel("")
        self.lbl_pipe_summary.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        form.addRow(self.lbl_pipe_summary)

        self.combo_pipe_overlay_fmt = QComboBox()
        self.combo_pipe_overlay_fmt.addItems(["tiff", "jpg"])
        form.addRow("Overlay format", self.combo_pipe_overlay_fmt)
        self.cb_pipe_scalebar = QCheckBox("Scale bar on overlay frames")
        self.cb_pipe_scalebar.setChecked(True)
        form.addRow(self.cb_pipe_scalebar)
        layout.addWidget(options)

        self.btn_pipe_csv = QPushButton("Export Measurements (CSV)…")
        self.btn_pipe_csv.setObjectName("primaryBtn")
        self.btn_pipe_csv.clicked.connect(self._on_export_pipe_measurements)
        layout.addWidget(self.btn_pipe_csv)

        self.btn_pipe_overlay = QPushButton("Export Overlay Frames…")
        self.btn_pipe_overlay.clicked.connect(self._on_export_pipe_overlay)
        layout.addWidget(self.btn_pipe_overlay)

        self.btn_pipe_masks = QPushButton("Export Label Masks (TIFF)…")
        self.btn_pipe_masks.clicked.connect(self._on_export_pipe_masks)
        layout.addWidget(self.btn_pipe_masks)

        layout.addStretch(1)
        return w

    def _committed_result_names(self, exp: ND2StudiosRecord) -> List[str]:
        """Pipeline names in ``analysis_results`` with at least one result."""
        out: List[str] = []
        for name, store in (getattr(exp, "analysis_results", {}) or {}).items():
            if isinstance(store, AnalysisResult):
                out.append(name)
            elif isinstance(store, dict) and any(
                isinstance(r, AnalysisResult) for r in store.values()
            ):
                out.append(name)
        return out

    def _results_by_m(self, exp: ND2StudiosRecord, name: str) -> Dict[int, AnalysisResult]:
        store = (getattr(exp, "analysis_results", {}) or {}).get(name)
        if isinstance(store, AnalysisResult):
            return {0: store}
        if isinstance(store, dict):
            return {m: r for m, r in store.items() if isinstance(r, AnalysisResult)}
        return {}

    def _pipe_channels(self, exp: ND2StudiosRecord) -> Dict[str, np.ndarray]:
        """Materialized processed (or raw) channels for measurements / overlays."""
        if hasattr(exp, "processed_view"):
            src = exp.processed_view()
        else:
            src = exp._processed_channels or exp._raw_channels or {}
        out: Dict[str, np.ndarray] = {}
        for name in src.keys():
            try:
                out[name] = np.asarray(src[name])
            except Exception:  # noqa: BLE001
                continue
        return out

    def _refresh_pipeline_results(self) -> None:
        """Repopulate the pipeline-result combo from the active record."""
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        names = self._committed_result_names(exp)
        self.combo_pipe_result.blockSignals(True)
        current = self.combo_pipe_result.currentText()
        self.combo_pipe_result.clear()
        self.combo_pipe_result.addItems(names or ["(no results)"])
        if current in names:
            self.combo_pipe_result.setCurrentText(current)
        self.combo_pipe_result.blockSignals(False)
        enabled = bool(names)
        for b in (self.btn_pipe_csv, self.btn_pipe_overlay, self.btn_pipe_masks):
            b.setEnabled(enabled)
        self._refresh_pipe_summary()

    def _refresh_pipe_summary(self) -> None:
        exp = self.main_window.exp_manager.active if self.main_window else None
        name = self.combo_pipe_result.currentText()
        if exp is None or name in ("", "(no results)"):
            self.lbl_pipe_summary.setText("")
            return
        rbm = self._results_by_m(exp, name)
        if not rbm:
            self.lbl_pipe_summary.setText("")
            return
        n_labels = 0
        for res in rbm.values():
            for masks in res.label_masks.values():
                try:
                    n_labels += int(np.max(np.asarray(masks[0]))) if masks.shape[0] else 0
                except Exception:  # noqa: BLE001
                    pass
        self.lbl_pipe_summary.setText(
            f"{len(rbm)} M position(s) committed · ~{n_labels} labels (frame 1)"
        )

    def _selected_pipe(self) -> Tuple[Optional[ND2StudiosRecord], str, Dict[int, AnalysisResult]]:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return None, "", {}
        exp = self.main_window.exp_manager.active
        name = self.combo_pipe_result.currentText()
        rbm = self._results_by_m(exp, name)
        if not rbm:
            QMessageBox.information(
                self, "No analysis result",
                "Run an analysis pipeline and click Apply (Analysis or Pipelines "
                "tab) before exporting results."
            )
            return None, "", {}
        return exp, name, rbm

    def _on_export_pipe_measurements(self) -> None:
        exp, name, rbm = self._selected_pipe()
        if exp is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save measurements CSV", f"{self._export_basename()}_measurements.csv",
            "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        metadata = dict(exp.nd2_metadata or {})
        metadata.setdefault("pixel_size_um", exp.pixel_size_um)
        channels = self._pipe_channels(exp)
        multi_m = len(rbm) > 1
        all_rows: List[Dict[str, Any]] = []
        try:
            if self.main_window:
                self.main_window.set_status_text("Computing measurements…")
            for m, res in sorted(rbm.items()):
                rows = compute_measurements(
                    res.label_masks, channels, metadata, m_index=m,
                    volumetric_voxel_counts=res.volumetric_voxel_counts,
                )
                if multi_m:
                    for r in rows:
                        r["m_position"] = m
                all_rows.extend(rows)
            n = self._write_measurements_csv(path, all_rows)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        finally:
            if self.main_window:
                self.main_window.set_progress(0)
        if self.main_window:
            self.main_window.set_status_text(f"Wrote {n} measurement row(s) → {path}")
        QMessageBox.information(self, "Export complete", f"Wrote {n} row(s) to:\n{path}")

    def _on_export_pipe_overlay(self) -> None:
        exp, name, rbm = self._selected_pipe()
        if exp is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder", "")
        if not out_dir:
            return
        metadata = dict(exp.nd2_metadata or {})
        metadata.setdefault("pixel_size_um", exp.pixel_size_um)
        channels = self._pipe_channels(exp)
        channel_display = dict(getattr(exp, "channel_display", None) or {})
        fmt = self.combo_pipe_overlay_fmt.currentText()
        safe = name.replace(" ", "_").replace("/", "_")
        multi_m = len(rbm) > 1
        paths: List[str] = []
        try:
            for m, res in sorted(rbm.items()):
                base = f"{self._export_basename()}_{safe}"
                if multi_m:
                    base += f"_M{m + 1:02d}"
                paths += export_overlay_frames(
                    channels=channels, label_masks=res.label_masks, metadata=metadata,
                    output_dir=out_dir, fmt=fmt, channel_display=channel_display,
                    pixel_size_um=exp.pixel_size_um,
                    show_scale_bar=self.cb_pipe_scalebar.isChecked(),
                    scale_bar_um=50.0, show_channel_labels=True, mask_alpha=0.5,
                    frame_timestamps=getattr(exp, "_frame_timestamps", None),
                    progress_cb=lambda p: self.main_window.set_progress(p) if self.main_window else None,
                    basename=base,
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        finally:
            if self.main_window:
                self.main_window.set_progress(0)
        if self.main_window:
            self.main_window.set_status_text(f"Exported {len(paths)} overlay frame(s)")
        QMessageBox.information(self, "Export complete",
                                f"Wrote {len(paths)} file(s) to:\n{out_dir}")

    def _on_export_pipe_masks(self) -> None:
        exp, name, rbm = self._selected_pipe()
        if exp is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder", "")
        if not out_dir:
            return
        safe = name.replace(" ", "_").replace("/", "_")
        multi_m = len(rbm) > 1
        paths: List[str] = []
        try:
            for m, res in sorted(rbm.items()):
                base = f"{self._export_basename()}_{safe}"
                if multi_m:
                    base += f"_M{m + 1:02d}"
                paths += export_label_masks_tiff(res.label_masks, out_dir, basename=base)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        if self.main_window:
            self.main_window.set_status_text(f"Exported {len(paths)} label-mask TIFF(s)")
        QMessageBox.information(self, "Export complete",
                                f"Wrote {len(paths)} file(s) to:\n{out_dir}")

    def _write_measurements_csv(self, path: str, rows: List[Dict[str, Any]]) -> int:
        """Write rows to CSV with an ordered union of all keys."""
        headers: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    headers.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return len(rows)

    # ── Helpers ──
    def _export_basename(self) -> str:
        """Return the stem of the imported filename for auto-naming exports."""
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return "export"
        exp = self.main_window.exp_manager.active
        fp = (exp.import_config or {}).get("filepath", "")
        if fp:
            return os.path.splitext(os.path.basename(fp))[0]
        return exp.name or "export"

    # ── Page lifecycle ──
    def on_activated(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        has_processed = getattr(exp, "has_processed", lambda: bool(exp._processed_channels))()

        # Enable/disable the "Processed Image" combo option.
        model = self.combo_export_type.model()
        proc_item = model.item(1)
        if proc_item is not None:
            proc_item.setEnabled(has_processed)
        if not has_processed and self.combo_export_type.currentIndex() == 1:
            self.combo_export_type.setCurrentIndex(0)

        # Refresh tracked objects pipeline selector.
        self._refresh_tracked_pipelines()

        # Refresh the Pipeline-Results selector; enable that source only when at
        # least one analysis result is committed (Analysis or Pipelines tab).
        self._refresh_pipeline_results()
        pipe_names = self._committed_result_names(exp)
        pipe_item = self.combo_export_type.model().item(3)
        if pipe_item is not None:
            pipe_item.setEnabled(bool(pipe_names))
        if not pipe_names and self.combo_export_type.currentIndex() == 3:
            self.combo_export_type.setCurrentIndex(0)

        # Z stack mode: z_mode="none" with a multi-Z volume.
        is_zstack = (
            exp._raw_volume is not None
            and exp.n_zslices > 1
            and exp.z_view_mode == "none"
        )
        self.btn_export_tiff.setText(
            "Export Z Stack TIFF…" if is_zstack else "Export TIFF Stack…"
        )

        # Show a quick summary.
        n = exp.n_frames or 0
        ch = len(exp._raw_channels or {})
        z_note = f" · {exp.n_zslices} Z slices (Z stack)" if is_zstack else ""
        self.lbl_summary.setText(
            f"{n} frames · {ch} channels · {exp.frame_width}×{exp.frame_height} px"
            f" · {exp.pixel_size_um:.3f} µm/px{z_note}"
        )

        # Z-projection combo: show when file has multi-Z with an active projection.
        has_z_proj = (
            exp._raw_volume is not None
            and exp.n_zslices > 1
            and exp.z_view_mode in ("max", "mean", "min")
        )
        self.lbl_tiff_zproj.setVisible(has_z_proj)
        self.combo_tiff_zproj.setVisible(has_z_proj)
        if has_z_proj:
            self.combo_tiff_zproj.setCurrentText(exp.z_view_mode)

        # Image-sequence tab defaults — derive base name from filename, and
        # only enable the multi-axis checkbox when the file actually has
        # multiple M positions or unprojected Z slices.
        if not self.le_seq_basename.text():
            fp = exp.import_config.get("filepath") if exp.import_config else None
            stem = (os.path.splitext(os.path.basename(fp))[0]
                    if fp else (exp.name or "frame"))
            self.le_seq_basename.setText(f"{stem}_seq")
        has_multi_m = exp._raw_volume is not None and exp.n_multipoints > 1
        has_multi_z_unprojected = (
            exp._raw_volume is not None
            and exp.n_zslices > 1
            and exp.z_view_mode == "none"
        )
        can_iterate = has_multi_m or has_multi_z_unprojected
        self.cb_seq_iterate_volume.setEnabled(can_iterate)
        if not can_iterate:
            self.cb_seq_iterate_volume.setChecked(False)

        self._refresh_seq_summary()

    def _refresh_seq_summary(self) -> None:
        """Update the image-sequence file-count label live."""
        if self.main_window is None or self.main_window.exp_manager.active is None:
            self.lbl_seq_summary.setText("")
            return
        exp = self.main_window.exp_manager.active
        has_multi_m = exp._raw_volume is not None and exp.n_multipoints > 1
        has_multi_z_unprojected = (
            exp._raw_volume is not None
            and exp.n_zslices > 1
            and exp.z_view_mode == "none"
        )
        n_t = exp.n_frames or 1
        n_m = exp.n_multipoints if has_multi_m else 1
        n_z = exp.n_zslices if has_multi_z_unprojected else 1
        if self.cb_seq_iterate_volume.isChecked() and (
                has_multi_m or has_multi_z_unprojected):
            total = n_t * n_m * n_z
            self.lbl_seq_summary.setText(
                f"Will write {total} PNG files "
                f"(T={n_t} · M={n_m} · Z={n_z})."
            )
        else:
            self.lbl_seq_summary.setText(
                f"Will write {n_t} PNG file(s) (T only — current M/Z view)."
            )

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        if "fps" in exp.export_config:
            self.spin_fps.setValue(float(exp.export_config["fps"]))
        if "movie_format" in exp.export_config:
            self.combo_movie_fmt.setCurrentText(str(exp.export_config["movie_format"]))

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.export_config = {
            "fps": float(self.spin_fps.value()),
            "movie_format": self.combo_movie_fmt.currentText(),
            "tiff_bit_depth": self.combo_tiff_bitdepth.currentText(),
        }
        exp.fps = float(self.spin_fps.value())

    # ── Source picking ──
    def _channels_and_state(self) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, Tuple[int, int, int]],
        Dict[str, bool],
        Dict[str, Tuple[float, float, float]],
    ]:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return {}, {}, {}, {}
        exp = self.main_window.exp_manager.active
        has_processed = getattr(exp, "has_processed", lambda: bool(exp._processed_channels))()
        if self.combo_export_type.currentText() == "Processed Image" and has_processed:
            # V1.38 Phase 6 — ``processed_view()`` returns the in-RAM
            # dict when present, else an ``EnhancedDataset`` proxy.
            # Materialize each channel on access — exports stream
            # one channel at a time anyway, so peak RAM stays bounded.
            source = exp.processed_view() if hasattr(exp, "processed_view") else exp._processed_channels
            channels = {name: source[name] for name in source}
        else:
            # Materialize lazy proxies for export.
            channels = {}
            for name, data in (exp._raw_channels or {}).items():
                m = getattr(data, "materialize", None)
                channels[name] = m() if callable(m) else np.asarray(data)
        colors: Dict[str, Tuple[int, int, int]] = {}
        enabled: Dict[str, bool] = {}
        for name in channels:
            cd = exp.channel_display.get(name, {})
            colors[name] = CHANNEL_COLORS.get(cd.get("color", "gray"), (255, 255, 255))
            enabled[name] = bool(cd.get("enabled", True))
        lut_settings = self._lut_settings()
        return channels, colors, enabled, lut_settings

    def _lut_settings(self) -> Dict[str, Tuple[float, float, float]]:
        """Return current per-channel LUT (lo, hi, gamma) from the live viewer."""
        try:
            viewer = self.main_window.pages["import"].viewer
            state = viewer.channel_state()
            return {
                name: (
                    float(cfg["lut_lo"]),
                    float(cfg["lut_hi"]),
                    float(cfg.get("lut_gamma", 1.0)),
                )
                for name, cfg in state.items()
                if "lut_lo" in cfg and "lut_hi" in cfg
            }
        except Exception:
            pass
        # Fallback: read from the serialised experiment record.
        try:
            exp = self.main_window.exp_manager.active
            if exp is not None:
                return {
                    name: (
                        float(cd["lut_lo"]),
                        float(cd["lut_hi"]),
                        float(cd.get("lut_gamma", 1.0)),
                    )
                    for name, cd in exp.channel_display.items()
                    if "lut_lo" in cd and "lut_hi" in cd
                }
        except Exception:
            pass
        return {}

    # ── Export handlers ──
    def _on_export_tiff(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        exp = self.main_window.exp_manager.active

        is_zstack = (
            exp._raw_volume is not None
            and exp.n_zslices > 1
            and exp.z_view_mode == "none"
        )

        if is_zstack:
            # Export all Z slices as a (T, Z, H, W) ImageJ hyperstack per channel.
            enabled = {
                name: exp.channel_display.get(name, {}).get("enabled", True)
                for name in exp._raw_volume.channel_names
            }
            _rp = getattr(self, "_replay_export_path", "")
            if _rp:
                path = os.path.join(_rp, f"{self._export_basename()}_zstack.tif")
            else:
                path = QFileDialog.getSaveFileName(
                    self, "Export Z Stack TIFF",
                    f"{self._export_basename()}_zstack.tif",
                    "TIFF (*.tif *.tiff);;All files (*)",
                )[0]
            if not path:
                return
            export_dir = os.path.dirname(path)
            req = ExportRequest(
                mode="tiff_zstack",
                filepath=path,
                channels={},
                colors={},
                enabled=enabled,
                pixel_size_um=self._pixel_size_um(),
                bit_depth=self.combo_tiff_bitdepth.currentText(),
                raw_volume=exp._raw_volume,
                m_index=exp.m_index,
                crop_rect=exp.crop_rect,
            )
            self._record_macro("export_tiff",
                               f"Export TIFF Z-Stack → {export_dir}",
                               bit_depth=self.combo_tiff_bitdepth.currentText(),
                               export_dir=export_dir)
            self._run_export(req, "Building Z stack TIFF…")
            return

        # Standard (T, H, W) export — Z already projected (or re-projected).
        channels, colors, enabled, _lut = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return

        # Determine projection label and optionally re-project.
        has_z_proj = (
            exp._raw_volume is not None
            and exp.n_zslices > 1
            and exp.z_view_mode in ("max", "mean", "min")
        )
        if has_z_proj:
            chosen_mode = self.combo_tiff_zproj.currentText()
            if chosen_mode != exp.z_view_mode:
                vol = exp._raw_volume
                # Keep these lazy — the heavy per-frame read happens inside
                # ExportWorker (export_tiff_hyperstack calls np.asarray per
                # channel off the GUI thread). Materializing here would block
                # the GUI on a multi-GB read (V1.41 smoothness).
                channels = {
                    ch_name: vol.to_lazy_channel(
                        c_idx, m=exp.m_index,
                        z_mode=chosen_mode, z_index=0,
                    )
                    for c_idx, ch_name in enumerate(vol.channel_names)
                }
            suffix = f"_z{chosen_mode}"
        else:
            suffix = "_tiff"

        _rp = getattr(self, "_replay_export_path", "")
        if _rp:
            path = os.path.join(_rp, f"{self._export_basename()}{suffix}.tif")
        else:
            path = QFileDialog.getSaveFileName(
                self, "Export TIFF Stack",
                f"{self._export_basename()}{suffix}.tif",
                "TIFF (*.tif *.tiff);;All files (*)",
            )[0]
        if not path:
            return
        export_dir = os.path.dirname(path)
        req = ExportRequest(
            mode="tiff_stack",
            filepath=path,
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            bit_depth=self.combo_tiff_bitdepth.currentText(),
        )
        self._record_macro("export_tiff",
                           f"Export TIFF Stack → {export_dir}",
                           bit_depth=self.combo_tiff_bitdepth.currentText(),
                           export_dir=export_dir)
        self._run_export(req, "Writing TIFF…")

    def _on_export_composite(self) -> None:
        channels, colors, enabled, lut_settings = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        _rp = getattr(self, "_replay_export_path", "")
        if _rp:
            path = os.path.join(_rp, f"{self._export_basename()}_composite.tif")
        else:
            path = QFileDialog.getSaveFileName(
                self, "Export RGB Composite TIFF",
                f"{self._export_basename()}_composite.tif",
                "TIFF (*.tif *.tiff);;All files (*)",
            )[0]
        if not path:
            return
        export_dir = os.path.dirname(path)
        req = ExportRequest(
            mode="rgb_composite",
            filepath=path,
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            lut_settings=lut_settings,
        )
        self._record_macro("export_composite",
                           f"Export RGB Composite → {export_dir}",
                           export_dir=export_dir)
        self._run_export(req, "Writing RGB composite…")

    def _movie_options_from_ui(self) -> MovieOptions:
        """Build a :class:`MovieOptions` from the current Movie-tab widgets."""
        return MovieOptions(
            fps=float(self.spin_fps.value()),
            codec=self.combo_movie_fmt.currentText(),
            show_scale_bar=self.cb_scalebar.isChecked(),
            scale_bar_um=float(self.spin_scalebar_um.value()),
            scale_bar_color=self.combo_scalebar_color.currentText(),
            scale_bar_position=self.combo_scalebar_pos.currentText(),
            show_timestamp=self.cb_timestamp.isChecked(),
            timestamp_position=self.combo_ts_pos.currentText(),
            timestamp_color=self.combo_ts_color.currentText(),
            timestamp_dt_seconds=(float(self.spin_ts_dt.value())
                                  if self.cb_ts_use_synth.isChecked() else None),
            show_channel_labels=self.cb_channel_labels.isChecked(),
            channel_label_position=self.combo_chl_pos.currentText(),
        )

    def _frame_timestamps(self) -> Optional[np.ndarray]:
        if (self.main_window is None
                or self.main_window.exp_manager.active is None
                or self.main_window.exp_manager.active._frame_timestamps is None):
            return None
        return np.asarray(self.main_window.exp_manager.active._frame_timestamps)

    def _run_preview_dialog(
        self,
        channels: Dict[str, np.ndarray],
        colors: Dict[str, Tuple[int, int, int]],
        enabled: Dict[str, bool],
        lut_settings: Dict[str, Tuple[float, float, float]],
        opts: MovieOptions,
        title: str,
    ) -> Optional[ImageAdjustments]:
        """Open the preview dialog and return chosen adjustments, or None on cancel.

        During macro replay ``_replay_adjustments`` is set on self; in that case
        the dialog is skipped entirely and the stored adjustments are returned.
        """
        replay_adj = getattr(self, "_replay_adjustments", None)
        if replay_adj is not None:
            return replay_adj
        dlg = ExportPreviewDialog(
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(),
            lut_settings=lut_settings,
            movie_options=opts,
            title=title,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return dlg.adjustments()

    def _on_export_movie(self) -> None:
        channels, colors, enabled, lut_settings = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        fmt = self.combo_movie_fmt.currentText()
        ext = ".mp4" if fmt == "mp4" else ".gif"
        _rp = getattr(self, "_replay_export_path", "")
        if _rp:
            path = os.path.join(_rp, f"{self._export_basename()}_movie{ext}")
        else:
            path = QFileDialog.getSaveFileName(
                self, "Export Movie",
                f"{self._export_basename()}_movie{ext}",
                f"{fmt.upper()} (*{ext});;All files (*)",
            )[0]
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext

        opts = self._movie_options_from_ui()

        adjustments = self._run_preview_dialog(
            channels, colors, enabled, lut_settings, opts,
            title="Movie Export — Preview",
        )
        if adjustments is None:
            return

        req = ExportRequest(
            mode="movie",
            filepath=path,
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(),
            movie_options=opts,
            lut_settings=lut_settings,
            image_adjustments=adjustments,
        )
        export_dir = os.path.dirname(path)
        self._record_macro(
            "export_movie",
            f"Export Movie ({opts.codec}, {opts.fps:.0f} fps) → {export_dir}",
            fps=opts.fps,
            format=opts.codec,
            show_scale_bar=opts.show_scale_bar,
            scale_bar_um=opts.scale_bar_um,
            scale_bar_color=opts.scale_bar_color,
            scale_bar_position=opts.scale_bar_position,
            show_timestamp=opts.show_timestamp,
            timestamp_position=opts.timestamp_position,
            timestamp_color=opts.timestamp_color,
            show_channel_labels=opts.show_channel_labels,
            export_dir=export_dir,
            brightness=adjustments.brightness,
            contrast=adjustments.contrast,
            saturation=adjustments.saturation,
            hue=adjustments.hue,
            fade=adjustments.fade,
        )
        self._run_export(req, "Rendering movie…")

    def _on_export_image_sequence(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        exp = self.main_window.exp_manager.active

        channels, colors, enabled, lut_settings = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return

        basename = self.le_seq_basename.text().strip()
        if not basename:
            fp = exp.import_config.get("filepath") if exp.import_config else None
            basename = (os.path.splitext(os.path.basename(fp))[0]
                        if fp else (exp.name or "frame"))

        _rp = getattr(self, "_replay_export_path", "")
        out_dir = _rp or QFileDialog.getExistingDirectory(
            self, "Choose output folder for image sequence", ""
        )
        if not out_dir:
            return

        opts = self._movie_options_from_ui()
        adjustments = self._run_preview_dialog(
            channels, colors, enabled, lut_settings, opts,
            title="Image Sequence Export — Preview",
        )
        if adjustments is None:
            return

        iterate_volume = (
            self.cb_seq_iterate_volume.isChecked()
            and self.cb_seq_iterate_volume.isEnabled()
            and exp._raw_volume is not None
        )

        req = ExportRequest(
            mode="image_sequence",
            filepath=out_dir,
            channels=channels if not iterate_volume else {},
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(),
            movie_options=opts,
            lut_settings=lut_settings,
            image_adjustments=adjustments,
            basename=basename,
            iterate_volume=iterate_volume,
            raw_volume=exp._raw_volume if iterate_volume else None,
            z_mode=exp.z_view_mode,
            z_view_index=exp.z_view_index,
            crop_rect=exp.crop_rect,
        )
        self._record_macro("export_image_sequence",
                           f"Export Image Sequence → {out_dir}",
                           export_dir=out_dir, basename=basename,
                           brightness=adjustments.brightness,
                           contrast=adjustments.contrast,
                           saturation=adjustments.saturation,
                           hue=adjustments.hue,
                           fade=adjustments.fade,
                           )
        self._run_export(req, "Writing image sequence…")

    # ── Macro recording / replay ──────────────────────────────────────────────

    def _record_macro(self, action_type: str, label: str, **params) -> None:
        mw = self.main_window
        if mw is None:
            return
        from nd2studios.backend.macro_engine import MacroAction
        mw.upgrade_last_macro_action(MacroAction(action_type, label, params))

    def _replay_export(self, action: "MacroAction") -> None:  # type: ignore[name-defined]
        """Apply an export_* macro action to the current file.

        The stored ``path`` param bypasses the file-picker dialog so replay
        writes to the same location as the original recording.
        """
        from nd2studios.backend.exporters.composite_exporter import ImageAdjustments

        t = action.params if isinstance(action.params, dict) else {}
        action_type = action.action_type
        # _replay_export_path is set to the OUTPUT DIRECTORY.
        # Each export method then constructs the full filename from _export_basename().
        # Fallback: if an old macro stored a full "path", use its dirname.
        self._replay_export_path = (
            t.get("export_dir", "")
            or os.path.dirname(t.get("path", ""))
        )
        self._replay_adjustments = None
        try:
            if action_type == "export_tiff":
                bd = t.get("bit_depth", "uint16")
                if hasattr(self, "combo_tiff_bitdepth"):
                    idx = self.combo_tiff_bitdepth.findText(str(bd))
                    if idx >= 0:
                        self.combo_tiff_bitdepth.setCurrentIndex(idx)
                self._on_export_tiff()

            elif action_type == "export_composite":
                self._on_export_composite()

            elif action_type == "export_movie":
                if hasattr(self, "spin_fps") and "fps" in t:
                    self.spin_fps.setValue(float(t["fps"]))
                if hasattr(self, "combo_movie_fmt") and "format" in t:
                    idx = self.combo_movie_fmt.findText(str(t["format"]))
                    if idx >= 0:
                        self.combo_movie_fmt.setCurrentIndex(idx)
                if hasattr(self, "cb_scalebar"):
                    self.cb_scalebar.setChecked(bool(t.get("show_scale_bar", False)))
                if hasattr(self, "spin_scalebar_um") and "scale_bar_um" in t:
                    self.spin_scalebar_um.setValue(float(t["scale_bar_um"]))
                if hasattr(self, "cb_timestamp"):
                    self.cb_timestamp.setChecked(bool(t.get("show_timestamp", False)))
                if hasattr(self, "cb_channel_labels"):
                    self.cb_channel_labels.setChecked(
                        bool(t.get("show_channel_labels", False))
                    )
                self._replay_adjustments = ImageAdjustments(
                    brightness=float(t.get("brightness", 0.0)),
                    contrast=float(t.get("contrast", 0.0)),
                    saturation=float(t.get("saturation", 0.0)),
                    hue=float(t.get("hue", 0.0)),
                    fade=float(t.get("fade", 0.0)),
                )
                self._on_export_movie()

            elif action_type == "export_image_sequence":
                self._replay_adjustments = ImageAdjustments(
                    brightness=float(t.get("brightness", 0.0)),
                    contrast=float(t.get("contrast", 0.0)),
                    saturation=float(t.get("saturation", 0.0)),
                    hue=float(t.get("hue", 0.0)),
                    fade=float(t.get("fade", 0.0)),
                )
                self._on_export_image_sequence()

            elif action_type == "export_tracked_objects":
                self._on_export_tracked_objects()

        finally:
            self._replay_export_path = ""
            self._replay_adjustments = None

        # Wait for the export worker to finish before returning to the caller.
        import time as _time
        from PySide6.QtCore import QCoreApplication
        _time.sleep(0.2)
        QCoreApplication.processEvents()
        deadline = _time.time() + 600.0
        while (self._worker is not None and self._worker.isRunning()
               and _time.time() < deadline):
            QCoreApplication.processEvents()

    # ── Worker plumbing ──
    def _run_export(self, request: ExportRequest, label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy",
                                    "An export is already running.")
            return
        self._worker = ExportWorker(request, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        if self.main_window is not None:
            self.main_window.set_status_text(label)
        self._worker.start()

    def _pixel_size_um(self) -> float:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return 1.0
        return float(self.main_window.exp_manager.active.pixel_size_um or 1.0)

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_done(self, result: Any) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(0)
            self.main_window.set_status_text(f"Saved: {result}")
            self.main_window.exp_manager.set_status("ready_to_export")
        _replaying = getattr(
            getattr(self.main_window, "macro_recorder", None), "replaying", False
        )
        if not _replaying:
            QMessageBox.information(self, "Export complete",
                                    f"Wrote:\n{result}")

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Export failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
