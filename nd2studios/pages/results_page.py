"""
Results page (V1.22).

Displays extended per-object measurements computed from the AnalysisResult
label masks produced by the Analysis page.  Measurements are shown in a
sortable table alongside a column-selector sidebar and summary panel.
Measurements can be exported to CSV; images with label overlay can be
exported as TIFF or JPG.

Prerequisite: at least one analysis pipeline must have been run for the
active session (exp.analysis_results must be non-empty).
"""
from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from PySide6.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, QCoreApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QAbstractItemView,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.workers.base_worker import BaseWorker


# ── Column catalogue ──────────────────────────────────────────────────────────
# Each entry: (group_label, [(column_key, display_label), ...])
# Intensity columns (mean_intensity_*, std_intensity_*) are added dynamically
# into the sidebar when the first compute finishes.

_COLUMN_GROUPS: List[tuple] = [
    ("Identity", [
        ("segmentation_channel", "Segmentation channel"),
        ("frame", "Frame"),
        ("label_id", "Label ID"),
        ("m_position", "M position"),
    ]),
    ("Size", [
        ("area_px", "Area (px²)"),
        ("area_um2", "Area (µm²)"),
        ("volume_um3", "Volume (µm³)"),
    ]),
    ("Change (Δ)", [
        ("delta_area_px", "Δ Area (px²)"),
        ("delta_area_um2", "Δ Area (µm²)"),
        ("delta_volume_um3", "Δ Volume (µm³)"),
    ]),
    ("Position", [
        ("centroid_y_px", "Centroid Y (px)"),
        ("centroid_x_px", "Centroid X (px)"),
        ("centroid_y_um", "Centroid Y (µm)"),
        ("centroid_x_um", "Centroid X (µm)"),
        ("centroid_y_stage_um", "Centroid Y stage (µm)"),
        ("centroid_x_stage_um", "Centroid X stage (µm)"),
    ]),
    ("Shape", [
        ("perimeter", "Perimeter"),
        ("circularity", "Circularity"),
        ("eccentricity", "Eccentricity"),
        ("solidity", "Solidity"),
    ]),
    ("Bounding box", [
        ("bbox_min_row", "BBox min row"),
        ("bbox_min_col", "BBox min col"),
        ("bbox_max_row", "BBox max row"),
        ("bbox_max_col", "BBox max col"),
    ]),
    ("Tracking", [
        ("track_id",         "Track ID"),
        ("track_length",     "Track length"),
        ("track_validation", "Track validation"),
    ]),
]

_DEFAULT_COLUMNS: Set[str] = {
    "segmentation_channel", "frame", "label_id",
    "area_px", "area_um2",
    "delta_area_um2",
    "centroid_y_um", "centroid_x_um",
    "circularity", "eccentricity", "solidity",
    "track_id", "track_length", "track_validation",
}

# Uniform row height for every checkbox in the sidebar (px).
_CB_HEIGHT = 24


# ── Background worker for Results tab exports ─────────────────────────────────

class _ResultsExportWorker(BaseWorker):
    """Run a Results-tab export task in a background thread."""

    def __init__(self, fn: Callable[..., Any], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run_task(self) -> Any:
        return self._fn(self.set_progress, self.set_status)


# ── Minimal read-only table model ────────────────────────────────────────────

class _MeasurementsModel(QAbstractTableModel):
    """Flat list-of-dicts → sortable QAbstractTableModel."""

    def __init__(self, rows: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self._rows = rows
        self._headers: List[str] = list(rows[0].keys()) if rows else []

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return len(self._headers)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._headers[section]
            return str(section + 1)
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):  # noqa: N802
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.UserRole):
            return None
        val = self._rows[index.row()].get(self._headers[index.column()])
        if role == Qt.UserRole:
            return val
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if column >= len(self._headers):
            return
        key = self._headers[column]
        reverse = order == Qt.DescendingOrder
        self.layoutAboutToBeChanged.emit()
        self._rows.sort(
            key=lambda r: (r.get(key) is None, r.get(key) or 0),
            reverse=reverse,
        )
        self.layoutChanged.emit()


# ── Results page ──────────────────────────────────────────────────────────────

class ResultsPage(QWidget):
    """Page 5: Results — measurements from analysis binaries."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._measurements: List[Dict[str, Any]] = []
        self._current_pipeline: str = ""
        self._exp: Optional[ND2StudiosRecord] = None
        self._selected_rec: Optional[ND2StudiosRecord] = None
        self._worker: Optional[_ResultsExportWorker] = None

        # Cached data for the track-validation dialog.
        self._label_masks_for_validation: Dict[str, Any] = {}
        self._channels_for_validation: Dict[str, Any] = {}

        # key → QCheckBox for every column (static + dynamic intensity).
        self._col_checkboxes: Dict[str, QCheckBox] = {}
        # Layout and header label for the dynamic Intensity group.
        self._intensity_layout: Optional[QVBoxLayout] = None
        self._intensity_header: Optional[QLabel] = None
        self._intensity_sep: Optional[QFrame] = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── File selector (hidden when single file) ──
        self._file_selector_row = QWidget()
        fs_layout = QHBoxLayout(self._file_selector_row)
        fs_layout.setContentsMargins(0, 0, 0, 0)
        fs_layout.setSpacing(4)
        fs_layout.addWidget(QLabel("File:"))
        self._combo_file = QComboBox()
        self._combo_file.currentIndexChanged.connect(self._on_file_selected)
        fs_layout.addWidget(self._combo_file, stretch=1)
        self._file_selector_row.hide()
        root.addWidget(self._file_selector_row)

        # ── Top controls row ──
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Pipeline:"))
        self._combo_pipeline = QComboBox()
        self._combo_pipeline.setMinimumWidth(220)
        self._combo_pipeline.currentTextChanged.connect(self._on_pipeline_changed)
        ctrl_row.addWidget(self._combo_pipeline)

        self._btn_compute = QPushButton("Compute Measurements")
        self._btn_compute.setObjectName("primaryBtn")
        self._btn_compute.clicked.connect(self._on_compute)
        ctrl_row.addWidget(self._btn_compute)

        self._compute_progress_bar = QProgressBar()
        self._compute_progress_bar.setRange(0, 100)
        self._compute_progress_bar.setMaximumHeight(14)
        self._compute_progress_bar.setVisible(False)
        ctrl_row.addWidget(self._compute_progress_bar)

        self._cb_raw_intensity = QCheckBox("Use raw image for intensity")
        self._cb_raw_intensity.setToolTip(
            "When checked, intensity measurements are taken from the raw\n"
            "(unprocessed) image data. Mask/label positions still come\n"
            "from the analysis result, which may have been run on processed data."
        )
        ctrl_row.addWidget(self._cb_raw_intensity)
        ctrl_row.addStretch(1)

        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        ctrl_row.addWidget(self._lbl_count)
        root.addLayout(ctrl_row)

        # ── Object tracking parameters row ──
        track_row = QHBoxLayout()
        track_row.setSpacing(6)

        lbl_tracking = QLabel("Object Tracking:")
        lbl_tracking.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: bold 8pt; letter-spacing: 0.5px;"
        )
        track_row.addWidget(lbl_tracking)

        track_row.addWidget(QLabel("Link distance (px):"))
        self._spin_max_disp = QSpinBox()
        self._spin_max_disp.setRange(1, 9999)
        self._spin_max_disp.setValue(100)
        self._spin_max_disp.setFixedWidth(65)
        self._spin_max_disp.setToolTip(
            "Maximum centroid displacement (Euclidean, pixels) between\n"
            "consecutive frames to link two detections as the same object."
        )
        track_row.addWidget(self._spin_max_disp)

        track_row.addSpacing(12)
        track_row.addWidget(QLabel("Min. track length (frames):"))
        self._spin_min_track_len = QSpinBox()
        self._spin_min_track_len.setRange(2, 9999)
        self._spin_min_track_len.setValue(2)
        self._spin_min_track_len.setFixedWidth(65)
        self._spin_min_track_len.setToolTip(
            "Minimum number of consecutive frames an object must appear in\n"
            "to be treated as a tracked object.  Objects with fewer frames\n"
            "receive no track_id and are excluded from validation."
        )
        track_row.addWidget(self._spin_min_track_len)

        track_row.addSpacing(12)
        track_row.addWidget(QLabel("Min. circularity:"))
        self._spin_min_circ = QDoubleSpinBox()
        self._spin_min_circ.setRange(0.0, 1.0)
        self._spin_min_circ.setSingleStep(0.05)
        self._spin_min_circ.setDecimals(2)
        self._spin_min_circ.setValue(0.0)
        self._spin_min_circ.setFixedWidth(65)
        self._spin_min_circ.setToolTip(
            "Minimum circularity (4π·area/perimeter²) an object must have\n"
            "to be eligible for tracking.  0.0 = no filter (all shapes pass);\n"
            "1.0 = perfect circles only."
        )
        track_row.addWidget(self._spin_min_circ)

        track_row.addSpacing(12)
        track_row.addWidget(QLabel("Max. eccentricity:"))
        self._spin_max_ecc = QDoubleSpinBox()
        self._spin_max_ecc.setRange(0.0, 1.0)
        self._spin_max_ecc.setSingleStep(0.05)
        self._spin_max_ecc.setDecimals(2)
        self._spin_max_ecc.setValue(1.0)
        self._spin_max_ecc.setFixedWidth(65)
        self._spin_max_ecc.setToolTip(
            "Maximum eccentricity an object may have to be eligible for\n"
            "tracking.  0.0 = circles only; 1.0 = no filter (all shapes pass)."
        )
        track_row.addWidget(self._spin_max_ecc)

        track_row.addStretch(1)
        root.addLayout(track_row)

        # ── Three-pane splitter: columns | table | summary ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_columns_sidebar())

        # Centre: measurements table
        table_widget = QWidget()
        tl = QVBoxLayout(table_widget)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel("Measurements", objectName="sectionHeader"))
        self._table = QTableView()
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setStyleSheet(f"""
            QTableView {{
                background-color: {Settings.BG_PRIMARY};
                gridline-color: {Settings.BORDER_COLOR};
            }}
            QTableView::item {{
                background-color: {Settings.BG_PRIMARY};
                color: {Settings.FG_PRIMARY};
                border-bottom: 1px solid {Settings.BORDER_COLOR};
            }}
            QTableView::item:selected {{
                background-color: {Settings.BG_HOVER};
                color: {Settings.FG_PRIMARY};
            }}
        """)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        tl.addWidget(self._table, stretch=1)
        splitter.addWidget(table_widget)

        # Right: summary panel
        summary_group = QGroupBox("Summary")
        sl = QVBoxLayout(summary_group)
        self._summary_labels: Dict[str, QLabel] = {}
        for key in ("n_objects", "n_frames", "mean_area_um2", "std_area_um2"):
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl_key = QLabel(key.replace("_", " ").title() + ":")
            lbl_key.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
            lbl_key.setMinimumWidth(130)
            lbl_val = QLabel("—")
            rl.addWidget(lbl_key)
            rl.addWidget(lbl_val, stretch=1)
            self._summary_labels[key] = lbl_val
            sl.addWidget(row_w)
        self._dynamic_summary_container = QVBoxLayout()
        sl.addLayout(self._dynamic_summary_container)
        sl.addStretch(1)
        summary_group.setMinimumWidth(220)
        summary_group.setMaximumWidth(300)
        splitter.addWidget(summary_group)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([210, 600, 240])
        root.addWidget(splitter, stretch=1)

        # ── Export row ──
        export_row = QHBoxLayout()
        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        export_row.addWidget(self._btn_export_csv)

        self._btn_export_masks = QPushButton("Export Label Masks (TIFF)")
        self._btn_export_masks.setEnabled(False)
        self._btn_export_masks.clicked.connect(self._on_export_masks)
        export_row.addWidget(self._btn_export_masks)

        self._cb_mask_overlay = QCheckBox("Include image")
        self._cb_mask_overlay.setToolTip(
            "Overlay colored per-object masks on the image data.\n"
            "When unchecked, exports raw integer label TIFFs."
        )
        export_row.addWidget(self._cb_mask_overlay)

        self._combo_img_fmt = QComboBox()
        self._combo_img_fmt.addItems(["TIFF", "JPG"])
        export_row.addWidget(self._combo_img_fmt)
        self._btn_export_images = QPushButton("Export Overlay Images")
        self._btn_export_images.setEnabled(False)
        self._btn_export_images.clicked.connect(self._on_export_images)
        export_row.addWidget(self._btn_export_images)

        self._btn_validate_tracks = QPushButton("Validate Tracked Objects")
        self._btn_validate_tracks.setEnabled(False)
        self._btn_validate_tracks.setToolTip(
            "Review and accept/reject tracked objects that appear across\n"
            "multiple frames.  Rejected tracks are removed from the table."
        )
        self._btn_validate_tracks.clicked.connect(self._on_validate_tracked)
        export_row.addWidget(self._btn_validate_tracks)

        export_row.addStretch(1)
        root.addLayout(export_row)

        # ── Progress bar (hidden when idle) ──
        progress_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumHeight(14)
        progress_row.addWidget(self._progress_bar, stretch=1)
        self._btn_cancel_export = QPushButton("Cancel")
        self._btn_cancel_export.setVisible(False)
        self._btn_cancel_export.setMaximumWidth(70)
        self._btn_cancel_export.clicked.connect(self._on_cancel_export)
        progress_row.addWidget(self._btn_cancel_export)
        root.addLayout(progress_row)

        # ── Empty-state label ──
        self._lbl_empty = QLabel(
            "Run an analysis pipeline on the Analysis page first,\n"
            "then click 'Compute Measurements'."
        )
        self._lbl_empty.setAlignment(Qt.AlignCenter)
        self._lbl_empty.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 11pt;"
        )
        root.addWidget(self._lbl_empty)
        self._lbl_empty.setVisible(False)

    # ── Column selector sidebar ───────────────────────────────────────────────

    def _build_columns_sidebar(self) -> QWidget:
        """Build the left column-selector panel with uniform checkbox spacing."""
        outer = QWidget()
        outer.setMinimumWidth(180)
        outer.setMaximumWidth(240)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 4, 0)
        outer_layout.setSpacing(4)

        outer_layout.addWidget(QLabel("Columns", objectName="sectionHeader"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        self._col_inner_layout = QVBoxLayout(inner)
        self._col_inner_layout.setContentsMargins(2, 2, 4, 4)
        self._col_inner_layout.setSpacing(0)

        for group_name, cols in _COLUMN_GROUPS:
            self._add_col_group(group_name, cols)

        # Intensity placeholder (hidden until first compute).
        self._intensity_header = QLabel("INTENSITY")
        self._intensity_header.setFixedHeight(_CB_HEIGHT)
        self._intensity_header.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: bold 8pt; "
            f"letter-spacing: 1px; padding-left: 4px;"
        )
        self._intensity_header.hide()
        self._col_inner_layout.addWidget(self._intensity_header)

        intensity_container = QWidget()
        self._intensity_layout = QVBoxLayout(intensity_container)
        self._intensity_layout.setContentsMargins(0, 0, 0, 0)
        self._intensity_layout.setSpacing(0)
        intensity_container.hide()
        self._intensity_container = intensity_container
        self._col_inner_layout.addWidget(intensity_container)

        self._intensity_sep = QFrame()
        self._intensity_sep.setFrameShape(QFrame.Shape.HLine)
        self._intensity_sep.setFixedHeight(1)
        self._intensity_sep.setStyleSheet(
            f"background: {Settings.BORDER_COLOR}; margin: 4px 0;"
        )
        self._intensity_sep.hide()
        self._col_inner_layout.addWidget(self._intensity_sep)

        self._col_inner_layout.addStretch(1)
        scroll.setWidget(inner)
        outer_layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 2, 0, 0)
        btn_all = QPushButton("All")
        btn_all.setFixedWidth(50)
        btn_all.setToolTip("Check all columns")
        btn_all.clicked.connect(self._select_all_columns)
        btn_none = QPushButton("None")
        btn_none.setFixedWidth(50)
        btn_none.setToolTip("Uncheck all columns")
        btn_none.clicked.connect(self._clear_all_columns)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch(1)
        outer_layout.addLayout(btn_row)

        return outer

    def _add_col_group(self, group_name: str, cols: List[tuple]) -> None:
        """Add a section header + uniform-height checkboxes to the inner layout."""
        lbl = QLabel(group_name.upper())
        lbl.setFixedHeight(_CB_HEIGHT)
        lbl.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: bold 8pt; "
            f"letter-spacing: 1px; padding-left: 4px;"
        )
        self._col_inner_layout.addWidget(lbl)

        for key, label in cols:
            cb = QCheckBox(label)
            cb.setFixedHeight(_CB_HEIGHT)
            cb.setChecked(key in _DEFAULT_COLUMNS)
            cb.setStyleSheet("padding-left: 4px;")
            cb.toggled.connect(self._on_column_toggled)
            self._col_checkboxes[key] = cb
            self._col_inner_layout.addWidget(cb)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background: {Settings.BORDER_COLOR}; margin: 4px 0;"
        )
        self._col_inner_layout.addWidget(sep)

    # ── Column sidebar slots ──────────────────────────────────────────────────

    def _on_column_toggled(self, _checked: bool = False) -> None:
        if self._measurements:
            self._update_table(self._measurements)

    def _select_all_columns(self) -> None:
        for cb in self._col_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        if self._measurements:
            self._update_table(self._measurements)

    def _clear_all_columns(self) -> None:
        for cb in self._col_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        if self._measurements:
            self._update_table(self._measurements)

    def _get_selected_column_keys(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return []
        return [
            key for key in rows[0]
            if self._col_checkboxes.get(key, None) is None
            or self._col_checkboxes[key].isChecked()
        ]

    def _sync_intensity_columns(self, rows: List[Dict[str, Any]]) -> None:
        """Register sidebar checkboxes for any new intensity columns."""
        if not rows:
            return
        new_added = False
        for key in rows[0]:
            if key in self._col_checkboxes:
                continue
            if not (key.startswith("mean_intensity_") or key.startswith("std_intensity_")):
                continue
            if key.startswith("mean_intensity_"):
                ch = key[len("mean_intensity_"):].replace("_", " ")
                label = f"Mean intensity ({ch})"
            else:
                ch = key[len("std_intensity_"):].replace("_", " ")
                label = f"Std intensity ({ch})"
            cb = QCheckBox(label)
            cb.setFixedHeight(_CB_HEIGHT)
            cb.setChecked(True)
            cb.setStyleSheet("padding-left: 4px;")
            cb.toggled.connect(self._on_column_toggled)
            self._col_checkboxes[key] = cb
            self._intensity_layout.addWidget(cb)
            new_added = True
        if new_added:
            self._intensity_header.show()
            self._intensity_container.show()
            self._intensity_sep.show()

    # ── Page lifecycle ────────────────────────────────────────────────────────

    def on_activated(self) -> None:
        self._rebuild_file_selector()
        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        self._refresh_pipeline_combo(exp)

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        self._exp = exp
        self._selected_rec = exp
        cfg = exp.results_config
        if cfg.get("pipeline"):
            idx = self._combo_pipeline.findText(cfg["pipeline"])
            if idx >= 0:
                self._combo_pipeline.setCurrentIndex(idx)
        saved_cols: Optional[List[str]] = cfg.get("selected_columns")
        if saved_cols is not None:
            sel_set = set(saved_cols)
            for key, cb in self._col_checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(key in sel_set)
                cb.blockSignals(False)
        self._refresh_pipeline_combo(exp)

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.results_config = {
            "pipeline": self._combo_pipeline.currentText(),
            "image_format": self._combo_img_fmt.currentText().lower(),
            "selected_columns": [
                k for k, cb in self._col_checkboxes.items() if cb.isChecked()
            ],
        }

    # ── File selector ─────────────────────────────────────────────────────────

    def _rebuild_file_selector(self) -> None:
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
            label = os.path.basename(
                (getattr(rec, "import_config", {}) or {}).get("filepath", "")
                or (getattr(rec, "nd2_metadata", {}) or {}).get("filepath", "")
            ) or "Untitled"
            self._combo_file.addItem(label)
        self._combo_file.blockSignals(False)

        self._file_selector_row.setVisible(len(records) > 1)

        current_idx = self._combo_file.currentIndex()
        best_idx = 0
        for i, rec in enumerate(records):
            if getattr(rec, "analysis_results", None):
                best_idx = i
                break
        target_idx = current_idx if 0 <= current_idx < len(records) else best_idx
        self._combo_file.setCurrentIndex(target_idx)
        self._selected_rec = records[target_idx] if records else None

    def _on_file_selected(self, index: int) -> None:
        if self.main_window is None:
            return
        fn = getattr(self.main_window, "get_confirmed_records", None)
        records: List[ND2StudiosRecord] = fn() if callable(fn) else []
        if not records:
            exp = self.main_window.exp_manager.active
            if exp is not None:
                records = [exp]
        self._selected_rec = records[index] if 0 <= index < len(records) else None
        self._refresh_pipeline_combo(self._selected_rec)

    # ── Pipeline combo ────────────────────────────────────────────────────────

    def _refresh_pipeline_combo(self, exp: Optional[ND2StudiosRecord]) -> None:
        self._combo_pipeline.blockSignals(True)
        prev = self._combo_pipeline.currentText()
        self._combo_pipeline.clear()
        if exp is not None and exp.analysis_results:
            for key in exp.analysis_results:
                self._combo_pipeline.addItem(key)
            idx = self._combo_pipeline.findText(prev)
            if idx >= 0:
                self._combo_pipeline.setCurrentIndex(idx)
        self._combo_pipeline.blockSignals(False)
        has_results = bool(exp is not None and exp.analysis_results)
        self._btn_compute.setEnabled(has_results)
        self._lbl_empty.setVisible(not has_results)

    def _on_pipeline_changed(self, name: str) -> None:
        self._current_pipeline = name
        self._measurements = []
        self._update_table([])
        self._update_summary([])
        self._btn_export_csv.setEnabled(False)
        self._btn_export_masks.setEnabled(False)
        self._btn_export_images.setEnabled(False)
        self._btn_validate_tracks.setEnabled(False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_result(self, exp: ND2StudiosRecord, pipeline_name: str):
        from nd2studios.core.analysis_registry import AnalysisResult
        stored = exp.analysis_results.get(pipeline_name)
        if stored is None:
            return None
        if isinstance(stored, AnalysisResult):
            return stored
        if isinstance(stored, dict):
            m = int(exp.m_index)
            result = stored.get(m)
            if result is None:
                result = next(iter(stored.values()), None)
            return result
        return None

    def _channels_for_m(self, exp: ND2StudiosRecord, m: int) -> Dict[str, Any]:
        from nd2studios.pipeline import EnhancedDataset
        use_raw = self._cb_raw_intensity.isChecked()
        vol = getattr(exp, "_raw_volume", None)
        if vol is not None:
            z_mode = getattr(exp, "z_view_mode", None) or "max"
            z_index = int(getattr(exp, "z_view_index", None) or 0)
            raw_for_m = {
                ch_name: vol.to_lazy_channel(
                    c_idx, m=m, z_mode=z_mode, z_index=z_index
                ).materialize()
                for c_idx, ch_name in enumerate(vol.channel_names)
            }
            if use_raw:
                return raw_for_m
            # Apply committed recipe so intensity values match processed data.
            recipe = list(getattr(exp, "recipe", []) or [])
            normalized = bool(getattr(exp, "recipe_normalized", False))
            if recipe or normalized:
                enhanced = EnhancedDataset(
                    raw_channels=raw_for_m,
                    recipe=recipe,
                    normalized=normalized,
                    pixel_size_um=float(getattr(exp, "pixel_size_um", 1.0)),
                )
                return {name: enhanced.materialize_channel(name)
                        for name in raw_for_m}
            return raw_for_m
        source = exp.processed_view() if hasattr(exp, "processed_view") else (
            exp._processed_channels or exp._raw_channels or {}
        )
        result: Dict[str, Any] = {}
        for k in source:
            v = source[k]
            if hasattr(v, "materialize") and callable(v.materialize):
                result[k] = v.materialize()
            else:
                result[k] = np.asarray(v)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _export_basename(self) -> str:
        """Return the stem of the imported filename for auto-naming exports."""
        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return "export"
        fp = (getattr(exp, "import_config", {}) or {}).get("filepath", "")
        if fp:
            return os.path.splitext(os.path.basename(fp))[0]
        return getattr(exp, "name", None) or "export"

    # ── V1.38 Phase 6 — workspace rehydrate ──────────────────────────────────

    def _rehydrate_released_label_masks(
        self, pipeline_name: str, results_by_m: Dict[int, Any],
    ) -> None:
        if self.main_window is None:
            return
        stage = self.main_window.analysis_stage(pipeline_name)
        if stage is None or not stage.is_committed():
            return
        for m, result in results_by_m.items():
            if result.label_masks:
                continue
            fresh = stage.rehydrate_m(m)
            if fresh is None:
                continue
            result.label_masks = fresh.label_masks
            if fresh.secondary_label_masks:
                result.secondary_label_masks = fresh.secondary_label_masks

    # ── Compute ───────────────────────────────────────────────────────────────

    def _on_compute(self) -> None:
        from nd2studios.backend.results_engine import compute_measurements
        from nd2studios.core.analysis_registry import AnalysisResult

        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return

        pipeline_name = self._combo_pipeline.currentText()
        if not pipeline_name or pipeline_name not in exp.analysis_results:
            QMessageBox.information(
                self, "No Analysis Result",
                "Run an analysis pipeline on the Analysis page first."
            )
            return

        stored = exp.analysis_results.get(pipeline_name)
        if isinstance(stored, AnalysisResult):
            results_by_m = {0: stored}
        elif isinstance(stored, dict):
            results_by_m = {m: r for m, r in stored.items() if isinstance(r, AnalysisResult)}
        else:
            results_by_m = {}

        if not results_by_m:
            QMessageBox.information(
                self, "No Analysis Result",
                "No result found for the selected pipeline."
            )
            return

        self._rehydrate_released_label_masks(pipeline_name, results_by_m)

        metadata = dict(exp.nd2_metadata or {})
        metadata.setdefault("pixel_size_um", exp.pixel_size_um)

        multi_m = len(results_by_m) > 1
        total_m = max(len(results_by_m), 1)
        all_rows: List[Dict[str, Any]] = []

        self._compute_progress_bar.setValue(0)
        self._compute_progress_bar.setVisible(True)
        self._btn_compute.setEnabled(False)
        if self.main_window is not None:
            self.main_window.set_progress(1)

        try:
            for idx, (m, result) in enumerate(sorted(results_by_m.items())):
                channels_for_m = self._channels_for_m(exp, m)
                try:
                    rows = compute_measurements(
                        result.label_masks,
                        channels_for_m,
                        metadata,
                        m_index=m,
                        volumetric_voxel_counts=result.volumetric_voxel_counts,
                    )
                except Exception as exc:
                    QMessageBox.warning(self, "Compute Error", str(exc))
                    return
                if multi_m:
                    for row in rows:
                        row["m_position"] = m
                all_rows.extend(rows)
                pct = int((idx + 1) / total_m * 90)
                self._compute_progress_bar.setValue(pct)
                if self.main_window is not None:
                    self.main_window.set_progress(pct)
                QCoreApplication.processEvents()
        finally:
            self._compute_progress_bar.setValue(100)
            if self.main_window is not None:
                self.main_window.set_progress(0)
            self._compute_progress_bar.setVisible(False)
            self._btn_compute.setEnabled(True)

        self._measurements = all_rows

        # ── Object tracking ───────────────────────────────────────────────────
        from nd2studios.backend.object_tracker import link_objects
        link_objects(
            all_rows,
            max_displacement_px=float(self._spin_max_disp.value()),
            min_track_length=int(self._spin_min_track_len.value()),
            min_circularity=float(self._spin_min_circ.value()),
            max_eccentricity=float(self._spin_max_ecc.value()),
        )

        # Cache channel data and label masks for the validation dialog.
        # Only the first (or only) M position is cached; multi-M is V2.
        self._label_masks_for_validation = {}
        self._channels_for_validation = {}
        if results_by_m:
            m0 = next(iter(sorted(results_by_m.keys())))
            self._channels_for_validation = self._channels_for_m(exp, m0)
            self._label_masks_for_validation = dict(results_by_m[m0].label_masks)

        has_tracked = any(r.get("track_id") is not None for r in all_rows)
        self._btn_validate_tracks.setEnabled(has_tracked)
        # ─────────────────────────────────────────────────────────────────────

        self._sync_intensity_columns(all_rows)
        self._update_table(all_rows)
        self._update_summary(all_rows)

        first_result = next(iter(results_by_m.values()))
        has_data = bool(all_rows)
        self._btn_export_csv.setEnabled(has_data)
        self._btn_export_masks.setEnabled(bool(first_result.label_masks))
        self._btn_export_images.setEnabled(has_data)
        self._lbl_count.setText(f"{len(all_rows)} objects")

        if self.main_window is not None:
            self.main_window.set_status_text(
                f"Results: {len(all_rows)} objects from '{pipeline_name}'."
            )

    # ── Track validation ──────────────────────────────────────────────────────

    def _on_validate_tracked(self) -> None:
        """Open the TrackValidationDialog and apply accept/reject decisions."""
        from nd2studios.widgets.track_validation_dialog import TrackValidationDialog

        if not self._measurements:
            return
        if not any(r.get("track_id") is not None for r in self._measurements):
            return

        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        channel_display: Dict[str, Any] = dict(
            getattr(exp, "channel_display", None) or {}
        )

        dlg = TrackValidationDialog(
            measurements=self._measurements,
            label_masks=self._label_masks_for_validation,
            channels=self._channels_for_validation,
            channel_display=channel_display,
            parent=self,
        )
        dlg.exec()

        rejected = dlg.rejected_track_ids
        accepted = dlg.accepted_track_ids

        if rejected:
            self._measurements = [
                r for r in self._measurements
                if r.get("track_id") not in rejected
            ]

        for r in self._measurements:
            if r.get("track_id") in accepted:
                r["track_validation"] = "accepted"

        self._update_table(self._measurements)
        self._update_summary(self._measurements)
        self._lbl_count.setText(f"{len(self._measurements)} objects")

        if rejected and self.main_window is not None:
            self.main_window.set_status_text(
                f"Validation complete: {len(rejected)} track(s) rejected and removed."
            )

    # ── Table update ─────────────────────────────────────────────────────────

    def _update_table(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            self._table.setModel(None)
            return
        selected_keys = self._get_selected_column_keys(rows)
        if not selected_keys:
            self._table.setModel(None)
            return
        filtered = [{k: r.get(k) for k in selected_keys} for r in rows]
        model = _MeasurementsModel(filtered, parent=self)
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        self._table.setModel(proxy)
        self._table.resizeColumnsToContents()
        for c in range(proxy.columnCount()):
            w = self._table.columnWidth(c)
            self._table.setColumnWidth(c, min(w, 160))

    # ── Summary update ────────────────────────────────────────────────────────

    def _update_summary(self, rows: List[Dict[str, Any]]) -> None:
        while self._dynamic_summary_container.count():
            item = self._dynamic_summary_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not rows:
            for lbl in self._summary_labels.values():
                lbl.setText("—")
            return

        areas = [r.get("area_um2") for r in rows if r.get("area_um2") is not None]
        frames = {r.get("frame") for r in rows}

        self._summary_labels["n_objects"].setText(str(len(rows)))
        self._summary_labels["n_frames"].setText(str(len(frames)))
        if areas:
            arr = np.asarray(areas, dtype=float)
            self._summary_labels["mean_area_um2"].setText(f"{arr.mean():.3f} µm²")
            self._summary_labels["std_area_um2"].setText(f"{arr.std():.3f} µm²")
        else:
            self._summary_labels["mean_area_um2"].setText("—")
            self._summary_labels["std_area_um2"].setText("—")

        intensity_keys = sorted({
            k for r in rows for k in r if k.startswith("mean_intensity_")
        })
        for key in intensity_keys:
            ch_label = key[len("mean_intensity_"):].replace("_", " ")
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl_key = QLabel(f"Mean intensity ({ch_label}):")
            lbl_key.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
            lbl_val = QLabel(f"{mean_v:.3f}")
            rl.addWidget(lbl_key)
            rl.addWidget(lbl_val, stretch=1)
            self._dynamic_summary_container.addWidget(row_w)

    # ── Export: CSV ───────────────────────────────────────────────────────────

    def _record_macro(self, action_type: str, label: str, **params) -> None:
        mw = self.main_window
        if mw is None:
            return
        from nd2studios.backend.macro_engine import MacroAction
        mw.upgrade_last_macro_action(MacroAction(action_type, label, params))

    def _on_export_csv(self) -> None:
        if not self._measurements:
            return
        _rp = getattr(self, "_replay_result_export_path", "")
        if _rp:
            path = os.path.join(_rp, f"{self._export_basename()}_measurements.csv")
        else:
            path = QFileDialog.getSaveFileName(
                self, "Export Measurements CSV",
                f"{self._export_basename()}_measurements.csv",
                "CSV files (*.csv)",
            )[0]
        if not path:
            return
        export_dir = os.path.dirname(path)
        self._record_macro("export_csv", f"Export CSV → {export_dir}",
                           export_dir=export_dir)
        fieldnames = list(self._measurements[0].keys())
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._measurements)
            if self.main_window is not None:
                self.main_window.set_status_text(f"CSV saved: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    # ── Export: label masks ───────────────────────────────────────────────────

    def _on_export_masks(self) -> None:
        from nd2studios.backend.results_engine import (
            export_label_masks_tiff, export_label_masks_as_overlay,
        )
        from nd2studios.core.analysis_registry import AnalysisResult

        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return
        pipeline_name = self._combo_pipeline.currentText()
        stored = exp.analysis_results.get(pipeline_name) if exp else None
        if stored is None:
            return

        if isinstance(stored, AnalysisResult):
            results_by_m = {0: stored}
        elif isinstance(stored, dict):
            results_by_m = {m: r for m, r in stored.items() if isinstance(r, AnalysisResult)}
        else:
            return
        if not results_by_m:
            return

        self._rehydrate_released_label_masks(pipeline_name, results_by_m)

        has_any_masks = any(r.label_masks for r in results_by_m.values())
        if not has_any_masks:
            QMessageBox.information(
                self, "No Label Masks",
                "No label masks are available for this pipeline.\n"
                "Run 'Compute Measurements' first."
            )
            return

        _rp = getattr(self, "_replay_result_export_path", "")
        out_dir = _rp or QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return

        self._record_macro("export_label_masks", f"Export Label Masks → {out_dir}", export_dir=out_dir)

        include_image = self._cb_mask_overlay.isChecked()
        fmt = self._combo_img_fmt.currentText().lower()
        metadata = dict(exp.nd2_metadata or {})
        pixel_size_um = float(exp.pixel_size_um or 0.0)
        multi_m = len(results_by_m) > 1

        basename = self._export_basename()

        def _run(progress_cb, status_cb):
            total_paths: List[str] = []
            items = sorted(results_by_m.items())
            for idx, (m, result) in enumerate(items):
                if not result.label_masks:
                    continue
                target = os.path.join(out_dir, f"M{m:02d}") if multi_m else out_dir
                if multi_m:
                    os.makedirs(target, exist_ok=True)
                if include_image:
                    channels_for_m = self._channels_for_m(exp, m)
                    paths = export_label_masks_as_overlay(
                        channels=channels_for_m,
                        label_masks=result.label_masks,
                        metadata=metadata,
                        output_dir=target,
                        fmt=fmt,
                        channel_display=exp.channel_display or {},
                        pixel_size_um=pixel_size_um,
                        show_scale_bar=True,
                        show_channel_labels=True,
                        progress_cb=lambda p: progress_cb(
                            int(idx / len(items) * 100 + p / len(items))
                        ),
                        basename=basename,
                    )
                else:
                    paths = export_label_masks_tiff(result.label_masks, target,
                                                    basename=basename)
                total_paths.extend(paths)
            return total_paths

        self._run_export(_run, f"Exporting label masks → {out_dir}")

    # ── Export: overlay images ────────────────────────────────────────────────

    def _on_export_images(self) -> None:
        from nd2studios.backend.results_engine import export_overlay_frames
        from nd2studios.core.analysis_registry import AnalysisResult

        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return
        pipeline_name = self._combo_pipeline.currentText()
        stored = exp.analysis_results.get(pipeline_name) if exp else None
        if stored is None:
            return

        if isinstance(stored, AnalysisResult):
            results_by_m = {0: stored}
        elif isinstance(stored, dict):
            results_by_m = {m: r for m, r in stored.items() if isinstance(r, AnalysisResult)}
        else:
            return
        if not results_by_m:
            return

        _rp = getattr(self, "_replay_result_export_path", "")
        out_dir = _rp or QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return

        self._record_macro("export_overlay_images", f"Export Overlay Images → {out_dir}",
                           export_dir=out_dir)

        fmt = self._combo_img_fmt.currentText().lower()
        metadata = dict(exp.nd2_metadata or {})
        pixel_size_um = float(exp.pixel_size_um or 0.0)
        multi_m = len(results_by_m) > 1

        basename = self._export_basename()

        def _run(progress_cb, status_cb):
            total_paths: List[str] = []
            items = sorted(results_by_m.items())
            for idx, (m, result) in enumerate(items):
                if not result.label_masks:
                    continue
                channels_for_m = self._channels_for_m(exp, m)
                target = os.path.join(out_dir, f"M{m:02d}") if multi_m else out_dir
                if multi_m:
                    os.makedirs(target, exist_ok=True)
                paths = export_overlay_frames(
                    channels=channels_for_m,
                    label_masks=result.label_masks,
                    metadata=metadata,
                    output_dir=target,
                    fmt=fmt,
                    channel_display=exp.channel_display or {},
                    pixel_size_um=pixel_size_um,
                    show_scale_bar=True,
                    show_channel_labels=True,
                    progress_cb=lambda p: progress_cb(
                        int(idx / len(items) * 100 + p / len(items))
                    ),
                    basename=basename,
                )
                total_paths.extend(paths)
            return total_paths

        self._run_export(_run, f"Exporting overlay images → {out_dir}")

    # ── Worker plumbing ───────────────────────────────────────────────────────

    def _run_export(self, fn: Callable[..., Any], label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "An export is already running.")
            return

        self._worker = _ResultsExportWorker(fn, parent=self)
        self._worker.progress.connect(self._on_export_progress)
        self._worker.status.connect(self._on_export_status)
        self._worker.finished.connect(self._on_export_done)
        self._worker.error.connect(self._on_export_error)

        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._btn_cancel_export.setVisible(True)
        self._set_export_buttons_enabled(False)

        if self.main_window is not None:
            self.main_window.set_status_text(label)
        self._worker.start()

    def _set_export_buttons_enabled(self, enabled: bool) -> None:
        self._btn_export_csv.setEnabled(enabled and bool(self._measurements))
        first_result = self._first_result_with_masks()
        self._btn_export_masks.setEnabled(enabled and first_result is not None)
        self._btn_export_images.setEnabled(enabled and bool(self._measurements))

    def _first_result_with_masks(self):
        exp = self._selected_rec or self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return None
        pipeline_name = self._combo_pipeline.currentText()
        stored = exp.analysis_results.get(pipeline_name) if exp else None
        if stored is None:
            return None
        from nd2studios.core.analysis_registry import AnalysisResult
        if isinstance(stored, AnalysisResult):
            return stored if stored.label_masks else None
        if isinstance(stored, dict):
            for r in stored.values():
                if isinstance(r, AnalysisResult) and r.label_masks:
                    return r
        return None

    def _on_cancel_export(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_export_progress(self, p: int) -> None:
        self._progress_bar.setValue(p)

    def _on_export_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_export_done(self, result: Any) -> None:
        self._progress_bar.setVisible(False)
        self._btn_cancel_export.setVisible(False)
        self._set_export_buttons_enabled(True)
        n = len(result) if isinstance(result, list) else 1
        msg = f"Saved {n} file(s)."
        if self.main_window is not None:
            self.main_window.set_status_text(msg)
        _replaying = getattr(
            getattr(self.main_window, "macro_recorder", None), "replaying", False
        )
        if not _replaying:
            QMessageBox.information(self, "Export Complete", msg)

    def _on_export_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._btn_cancel_export.setVisible(False)
        self._set_export_buttons_enabled(True)
        QMessageBox.warning(self, "Export Failed", msg)

    # ── Macro replay ──────────────────────────────────────────────────────────

    def _replay_result_export(self, action: "MacroAction") -> None:  # type: ignore[name-defined]
        """Replay export_csv / export_label_masks / export_overlay_images actions."""
        import time as _time

        t = action.params if isinstance(action.params, dict) else {}
        # _replay_result_export_path is the output directory; each export method
        # constructs the full filename via _export_basename().
        # Fallback: old macros stored a full "path" — use its dirname.
        self._replay_result_export_path = (
            t.get("export_dir", "")
            or os.path.dirname(t.get("path", ""))
        )
        try:
            if action.action_type == "export_csv":
                self._on_export_csv()
            elif action.action_type == "export_label_masks":
                self._on_export_masks()
            elif action.action_type == "export_overlay_images":
                self._on_export_images()
        finally:
            self._replay_result_export_path = ""

        # Wait for the async worker (masks / overlay are threaded; CSV is not).
        _time.sleep(0.2)
        QCoreApplication.processEvents()
        deadline = _time.time() + 600.0
        while (self._worker is not None and self._worker.isRunning()
               and _time.time() < deadline):
            QCoreApplication.processEvents()
