"""
FilePanel widget — one per loaded ND2 / TIFF file.

Each panel bundles:
  * A collapsible controls sidebar (left, 220 px ↔ 0 px, instant via
    QSplitter.setSizes — no animation needed, never fights the splitter)
  * A :class:`MultiAxisViewer` (right, fills remaining space)
  * A header bar: [</>] collapse button | filename | [x] close button

Multiple FilePanels are placed side-by-side inside ImportPage's outer
QSplitter.  Both the inner (controls ↔ viewer) and outer (panel ↔ panel)
splitter moves auto-fit the viewer image via :meth:`fit_viewer`.

Implementation note on collapsing:
  ``setFixedWidth`` locks min *and* max simultaneously, so the splitter can
  never resize the widget and the collapse button has no visible effect.
  Instead we leave the controls widget with ``minimumWidth=0`` (so Qt can
  collapse it) and drive collapse/expand by calling
  ``_inner_splitter.setSizes()`` directly.  ``setChildrenCollapsible(True)``
  on the inner splitter allows programmatic and user-drag collapse.
"""
from __future__ import annotations

import gc
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import icon_button, scale_qss, scaled
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
from nd2studios.widgets.viewer3d import PyVista3DViewer
from nd2studios.workers.load_worker import LoadWorker

_SIDEBAR_EXPANDED_W: int = 220


def release_record_arrays(rec: ND2StudiosRecord) -> None:
    """Drop all in-memory image arrays from *rec* and run GC."""
    old_volume = rec._raw_volume
    if old_volume is not None:
        try:
            close = getattr(old_volume, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    rec._raw_channels = None
    rec._original_raw_channels = None
    rec._processed_channels = None
    rec._processed_view = None
    rec._raw_volume = None
    rec._frame_timestamps = None
    rec.crop_rect = None
    gc.collect()


class ChannelInfoRow(QWidget):
    """Read-only metadata row: ``Cn: name  exposure  ex  em``."""

    def __init__(
        self,
        idx: int,
        name: str,
        exposure_ms: Optional[float] = None,
        emission_nm: Optional[float] = None,
        excitation_nm: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        lbl = QLabel(f"C{idx}: {name}")
        lbl.setMinimumWidth(scaled(110))
        layout.addWidget(lbl)
        bits: List[str] = []
        if exposure_ms is not None:
            bits.append(f"{exposure_ms:.0f} ms")
        if excitation_nm is not None:
            bits.append(f"ex {excitation_nm:.0f}")
        if emission_nm is not None:
            bits.append(f"em {emission_nm:.0f}")
        info = QLabel("  ".join(bits) or "—")
        info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        layout.addWidget(info, stretch=1)


class FilePanel(QWidget):
    """Self-contained file panel: collapsible sidebar + MultiAxisViewer.

    Parameters
    ----------
    on_progress:
        Called with an int (0–100) while a file loads.
    on_status:
        Called with a status string while a file loads.
    on_confirm:
        Called with ``self`` when the Confirm Import button is clicked.
    on_stitch:
        Called with ``self`` when the Stitch M… button is clicked.
    show_close_button:
        Whether to render the close button in the header.
    """

    panel_close_requested = Signal(object)   # emits self
    loaded = Signal(dict)                    # emits metadata dict

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        on_progress: Optional[Callable[[int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_confirm: Optional[Callable[["FilePanel"], None]] = None,
        on_stitch: Optional[Callable[["FilePanel"], None]] = None,
        on_add_file: Optional[Callable[[], None]] = None,
        show_close_button: bool = True,
    ):
        super().__init__(parent)
        self.record: ND2StudiosRecord = ND2StudiosRecord()
        self._filepath: Optional[str] = None
        self._loader: Optional[LoadWorker] = None
        self._meta_dict: Dict[str, Any] = {}
        self._info_rows: List[ChannelInfoRow] = []
        self._sidebar_expanded: bool = True

        self._cb_progress = on_progress
        self._cb_status = on_status
        self._cb_confirm = on_confirm
        self._cb_stitch = on_stitch
        self._cb_add_file = on_add_file
        self._show_close = show_close_button

        # V1.57 — spatial XY export crop (x, y, w, h in full-frame pixels), and
        # the background export worker. The XY crop is stored as a rect (the
        # live viewer keeps browsing the full volume) and is realized only in
        # the exported file, composing with the T/M/Z tile-strip crop.
        self._xy_crop: Optional[Tuple[int, int, int, int]] = None
        self._export_worker = None

        self._build_ui()

    # ── Construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        # setChildrenCollapsible(True) so setSizes([0, …]) can collapse the
        # controls pane to zero both programmatically and via user drag.
        self._inner_splitter = QSplitter(Qt.Horizontal)
        self._inner_splitter.setChildrenCollapsible(True)

        self._ctrl_widget = self._build_controls()
        self._inner_splitter.addWidget(self._ctrl_widget)

        self.viewer = MultiAxisViewer(self, show_tile_preview=True)
        self.viewer.coords_changed.connect(self._on_coords_changed)
        self.viewer.channels_changed.connect(self._on_channels_changed)
        self.viewer.stitch_requested.connect(self._on_stitch_clicked)
        self.viewer.crop_to_selection_requested.connect(self._on_crop_to_selection)
        # V1.57 — spatial XY crop tool (rubber-band drag or click-to-enter).
        self.viewer.crop_rect_selected.connect(self._on_xy_crop_drag)
        self.viewer.canvas.clicked.connect(self._on_xy_canvas_click)
        self._full_volume = None      # original dataset before any crop
        self._full_timestamps = None
        self._crop_worker = None

        # V1.65 — 2D/3D swap. The 2-D MultiAxisViewer and a lazily-built
        # PyVista3DViewer share a QStackedWidget; the header "3D" button flips
        # them. ``self.viewer`` stays the 2-D viewer so every existing caller
        # (PlayAll banner, zoom reset, crop tools) is unaffected.
        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self.viewer)   # index 0 — 2-D
        self.viewer3d = None                       # built on first 3-D toggle
        self._inner_splitter.addWidget(self._view_stack)

        self._inner_splitter.setStretchFactor(0, 0)
        self._inner_splitter.setStretchFactor(1, 1)
        # Initial sizes: give controls their preferred width; viewer gets rest.
        self._inner_splitter.setSizes([_SIDEBAR_EXPANDED_W, 1])
        self._inner_splitter.splitterMoved.connect(
            lambda _pos, _idx: self.fit_viewer()
        )

        root.addWidget(self._inner_splitter, stretch=1)

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("filePanelHeader")
        bar.setFixedHeight(scaled(30))
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(6, 0, 6, 0)
        hl.setSpacing(4)

        # Use plain ASCII so the button text is always visible on Windows.
        self._btn_collapse = QPushButton("<")
        self._btn_collapse.setObjectName("filePanelCollapseBtn")
        self._btn_collapse.setFixedSize(scaled(22), scaled(22))
        self._btn_collapse.setToolTip("Collapse / expand file controls")
        self._btn_collapse.clicked.connect(self._toggle_sidebar)
        hl.addWidget(self._btn_collapse)

        self._lbl_name = QLabel("No file loaded")
        self._lbl_name.setObjectName("filePanelTitle")
        hl.addWidget(self._lbl_name, stretch=1)

        # V1.65 — volumetric 3-D toggle (icon-only; falls back to "3D" text).
        self._btn_3d = icon_button(
            "fa5s.cube", "3D — toggle volumetric view",
            checkable=True, object_name="toggleBtn",
            button_px=22, icon_px=12,
        )
        self._btn_3d.toggled.connect(self._toggle_3d)
        hl.addWidget(self._btn_3d)

        if self._show_close:
            self._btn_close = QPushButton("x")
            self._btn_close.setObjectName("filePanelCloseBtn")
            self._btn_close.setFixedSize(scaled(22), scaled(22))
            self._btn_close.setToolTip("Close this panel")
            self._btn_close.clicked.connect(
                lambda: self.panel_close_requested.emit(self)
            )
            hl.addWidget(self._btn_close)

        return bar

    def _build_controls(self) -> QWidget:
        ctrl = QWidget()
        ctrl.setObjectName("filePanelControls")
        # Do NOT use setFixedWidth — that locks min AND max, preventing the
        # splitter from ever resizing this widget.  Leave minimumWidth at 0
        # so the pane can be collapsed to zero via setSizes().
        ctrl.setMinimumWidth(0)
        sl = QVBoxLayout(ctrl)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(6)

        # File group
        file_group = QGroupBox("File")
        fl = QVBoxLayout(file_group)
        self.btn_browse = QPushButton("Browse ND2 / TIFF...")
        self.btn_browse.setObjectName("primaryBtn")
        self.btn_browse.clicked.connect(self._on_browse)
        fl.addWidget(self.btn_browse)
        self.btn_reconstruct = QPushButton("Reconstruct from multiple files...")
        self.btn_reconstruct.setToolTip(
            "Chain several ND2/TIFF files along T / M / Z / C into a "
            "single virtual volume."
        )
        self.btn_reconstruct.clicked.connect(self._on_reconstruct)
        fl.addWidget(self.btn_reconstruct)
        # V1.44 — "+ Add File" lives here (in the left controls) instead of a
        # page-level toolbar. Opens another file in a side-by-side panel.
        if self._cb_add_file is not None:
            self.btn_add_file = QPushButton("+ Add File")
            self.btn_add_file.setToolTip(
                "Open another ND2 / TIFF file in a new side-by-side panel.")
            self.btn_add_file.clicked.connect(lambda: self._cb_add_file())
            fl.addWidget(self.btn_add_file)
        self.lbl_filepath = QLabel("No file loaded.")
        self.lbl_filepath.setWordWrap(True)
        self.lbl_filepath.setStyleSheet(scale_qss(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        ))
        fl.addWidget(self.lbl_filepath)
        sl.addWidget(file_group)

        # Z mode + frame stride
        proj_group = QGroupBox("Z mode & range")
        pl = QFormLayout(proj_group)
        self.combo_zproj = QComboBox()
        self.combo_zproj.addItems(["max", "mean", "min", "none"])
        self.combo_zproj.setCurrentText("max")
        self.combo_zproj.currentTextChanged.connect(self._on_z_mode_changed)
        pl.addRow("Z mode", self.combo_zproj)
        self.spin_t_stride = QSpinBox()
        self.spin_t_stride.setRange(1, 1000)
        self.spin_t_stride.setValue(1)
        pl.addRow("Frame stride", self.spin_t_stride)
        sl.addWidget(proj_group)

        # Metadata table
        meta_group = QGroupBox("Metadata")
        ml = QVBoxLayout(meta_group)
        self.meta_table = QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.meta_table.horizontalHeader().setStretchLastSection(True)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setMaximumHeight(scaled(200))
        ml.addWidget(self.meta_table)
        sl.addWidget(meta_group)

        # Channel info
        info_group = QGroupBox("Channel info")
        cl = QVBoxLayout(info_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(scaled(110))
        self._info_container = QWidget()
        self._info_layout = QVBoxLayout(self._info_container)
        self._info_layout.setContentsMargins(2, 2, 2, 2)
        self._info_layout.addStretch(1)
        scroll.setWidget(self._info_container)
        cl.addWidget(scroll)
        sl.addWidget(info_group)

        # Confirm + Stitch
        btn_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Import")
        self.btn_confirm.setObjectName("successBtn")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)
        btn_row.addWidget(self.btn_confirm)
        self.btn_stitch = QPushButton("Stitch M...")
        self.btn_stitch.setEnabled(False)
        self.btn_stitch.setToolTip(
            "Stitch multipoint tiles into a single TIFF stack."
        )
        self.btn_stitch.clicked.connect(self._on_stitch_clicked)
        btn_row.addWidget(self.btn_stitch)
        sl.addLayout(btn_row)

        # V1.43 — revert a tile-strip crop back to the full dataset. Hidden
        # until a crop is applied.
        self.btn_revert_crop = QPushButton("Revert to full data")
        self.btn_revert_crop.setObjectName("compactBtn")
        self.btn_revert_crop.setToolTip(
            "Undo the frame crop and restore the full imported dataset.")
        self.btn_revert_crop.clicked.connect(self._on_revert_crop)
        self.btn_revert_crop.setVisible(False)
        sl.addWidget(self.btn_revert_crop)

        sl.addWidget(self._build_export_group())

        sl.addStretch(1)
        return ctrl

    def _build_export_group(self) -> QGroupBox:
        """V1.57 — spatial XY crop tool + one-click export of the cropped data.

        Export honors both the T/M/Z tile-strip crop (already baked into
        ``record._raw_volume``) and the spatial XY rectangle stored here.
        """
        group = QGroupBox("Export")
        gl = QVBoxLayout(group)

        crop_row = QHBoxLayout()
        self.btn_crop_xy = icon_button(
            "fa5s.crop-alt",
            "Crop — draw a rectangle on the image to set the export region",
            text=" Crop XY", checkable=True, object_name="toggleBtn",
        )
        self.btn_crop_xy.toggled.connect(self._on_crop_xy_toggled)
        crop_row.addWidget(self.btn_crop_xy)
        self.btn_reset_xy = icon_button(
            "fa5s.undo", "Reset — clear the XY export crop", text=" Reset",
            object_name="compactBtn",
        )
        self.btn_reset_xy.setEnabled(False)
        self.btn_reset_xy.clicked.connect(self._reset_xy_crop)
        crop_row.addWidget(self.btn_reset_xy)
        gl.addLayout(crop_row)

        self.lbl_xy_crop_status = QLabel("No XY crop")
        self.lbl_xy_crop_status.setStyleSheet(scale_qss(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        ))
        gl.addWidget(self.lbl_xy_crop_status)

        # V1.72 — main export: split the (cropped) volume along M/T/Z/C into
        # separate files (one per split-axis index; kept axes bundled per file).
        self.btn_split_export = icon_button(
            "fa5s.layer-group",
            "Export by axis — split the cropped data along M / T / Z / C into "
            "multiple TIFF / PNG / movie files",
            text=" Export by axis…", object_name="primaryBtn",
        )
        self.btn_split_export.setEnabled(False)
        self.btn_split_export.clicked.connect(self._open_split_export_dialog)
        gl.addWidget(self.btn_split_export)

        # Quick export of the current M/Z view only (single hyperstack / movie /
        # PNG sequence) — kept from V1.57.
        self.btn_export = icon_button(
            "fa5s.download",
            "Quick export — write only the current (cropped) M/Z view to disk",
            text=" Quick export…", object_name="compactBtn",
        )
        self.btn_export.setEnabled(False)
        menu = QMenu(self.btn_export)
        menu.addAction("TIFF hyperstack…", self._export_tiff)
        menu.addAction("Movie (MP4)…", lambda: self._export_movie("mp4"))
        menu.addAction("Movie (GIF)…", lambda: self._export_movie("gif"))
        menu.addAction("Image sequence (PNG)…", self._export_image_sequence)
        self.btn_export.setMenu(menu)
        gl.addWidget(self.btn_export)

        return group

    # ── Sidebar collapse/expand ──────────────────────────────────────

    def _toggle_sidebar(self) -> None:
        """Collapse or expand the controls pane using the splitter directly.

        Driving ``QSplitter.setSizes()`` is the correct, splitter-aware way
        to programmatically collapse a pane.  Property-animating
        ``minimumWidth``/``maximumWidth`` fights the splitter's own layout
        engine and produces no visible change when the widget was built with
        ``setFixedWidth``.
        """
        total = self._inner_splitter.width()
        if self._sidebar_expanded:
            self._inner_splitter.setSizes([0, total])
            self._btn_collapse.setText(">")
            self._sidebar_expanded = False
        else:
            sidebar_w = min(_SIDEBAR_EXPANDED_W, max(80, total - 200))
            self._inner_splitter.setSizes([sidebar_w, total - sidebar_w])
            self._btn_collapse.setText("<")
            self._sidebar_expanded = True
        self.fit_viewer()

    def fit_viewer(self) -> None:
        """Ask the canvas to zoom-to-fit the image in the current viewport."""
        stack = getattr(self, "_view_stack", None)
        viewer3d = getattr(self, "viewer3d", None)
        if (viewer3d is not None and stack is not None
                and stack.currentWidget() is viewer3d):
            viewer3d.reset_camera()
            return
        canvas = getattr(self.viewer, "canvas", None)
        if canvas is not None:
            reset = getattr(canvas, "reset_zoom", None)
            if callable(reset):
                reset()

    # ── V1.65 — 2D / 3D toggle ───────────────────────────────────────
    def _ensure_viewer3d(self) -> PyVista3DViewer:
        """Lazily construct the 3-D viewer and add it to the view stack."""
        if self.viewer3d is None:
            self.viewer3d = PyVista3DViewer(self)
            # Keep the record's position/LUT authoritative regardless of view.
            self.viewer3d.coords_changed.connect(self._on_coords_changed)
            self.viewer3d.channels_changed.connect(self._on_channels_changed)
            self._view_stack.addWidget(self.viewer3d)   # index 1 — 3-D
        return self.viewer3d

    def _toggle_3d(self, enabled: bool) -> None:
        if enabled:
            viewer3d = self._ensure_viewer3d()
            self._view_stack.setCurrentWidget(viewer3d)   # show the canvas first…
            self._feed_viewer3d()                          # …then build + render
        else:
            self._view_stack.setCurrentWidget(self.viewer)
            self.fit_viewer()

    def _feed_viewer3d(self) -> None:
        """Push the current raw volume + channel state into the 3-D viewer."""
        if self.viewer3d is None or self.record is None:
            return
        volume = getattr(self.record, "_raw_volume", None)
        # Mirror the 2-D per-channel color/LUT so the two views agree.
        state = self.viewer.channel_state() or self.record.channel_display
        if volume is not None:
            self.viewer3d.set_volume(
                volume,
                channel_display=state,
                z_mode=self.combo_zproj.currentText(),
                z_index=int(getattr(self.record, "z_view_index", 0)),
                m=int(getattr(self.record, "m_index", 0)), t=0,
                z=int(getattr(self.record, "z_view_index", 0)),
            )
        else:
            self.viewer3d.set_channels(
                self.record._raw_channels or {}, channel_display=state,
            )

    # ── File operations ──────────────────────────────────────────────

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ND2 or TIFF", "",
            "Microscopy files (*.nd2 *.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        self._filepath = path
        self.lbl_filepath.setText(path)
        self._lbl_name.setText(os.path.basename(path))
        self._start_load_worker(filepaths=[path], chain_axis="Z")

    def _on_reconstruct(self) -> None:
        from nd2studios.pages.reconstruct_dialog import ReconstructDialog
        dialog = ReconstructDialog(parent=self)
        if dialog.exec() != ReconstructDialog.Accepted:
            return
        paths = dialog.filepaths()
        axis = dialog.chain_axis()
        mapping = dialog.chain_mapping()
        if len(paths) < 2:
            return
        self._filepath = paths[0]
        first = os.path.basename(paths[0])
        label = f"{len(paths)} files on {axis}: {first} ..."
        self.lbl_filepath.setText(label)
        self._lbl_name.setText(f"{first} (+{len(paths) - 1})")
        self._start_load_worker(
            filepaths=paths, chain_axis=axis, chain_mapping=mapping,
        )

    def _start_load_worker(
        self,
        filepaths: List[str],
        chain_axis: str,
        chain_mapping=None,
    ) -> None:
        self.btn_confirm.setEnabled(False)
        self.btn_stitch.setEnabled(False)
        self._loader = LoadWorker(
            filepaths=filepaths,
            chain_axis=chain_axis,
            chain_mapping=chain_mapping,
            z_projection=self.combo_zproj.currentText(),
            t_stride=int(self.spin_t_stride.value()),
        )
        if self._cb_progress:
            self._loader.progress.connect(self._cb_progress)
        if self._cb_status:
            self._loader.status.connect(self._cb_status)
        self._loader.finished.connect(self._on_loaded)
        self._loader.error.connect(self._on_error)
        self._loader.start()

    def _on_loaded(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        meta = payload.get("metadata", {})
        self._meta_dict = meta
        self._populate_metadata_table(meta)
        self._populate_info_rows(meta)

        rec = self.record
        release_record_arrays(rec)
        rec._raw_channels = payload.get("channels", {})
        rec._raw_volume = payload.get("volume")
        rec._frame_timestamps = payload.get("frame_timestamps_s")
        rec.nd2_metadata = meta
        rec.n_frames = int(meta.get("n_timepoints", 0))
        rec.frame_height = int(meta.get("height", 0))
        rec.frame_width = int(meta.get("width", 0))
        rec.n_multipoints = int(meta.get("n_multipoints", 1))
        rec.n_zslices = int(meta.get("n_zslices", 1))
        rec.pixel_size_um = float(meta.get("pixel_size_um", 1.0))

        volume = payload.get("volume")
        # V1.43 — a fresh import clears any prior crop and records the full
        # dataset/timestamps so "Revert to full data" can restore them.
        self._full_volume = None
        self._full_timestamps = None
        self.btn_revert_crop.setVisible(False)
        # V1.57 — a fresh import clears the spatial XY export crop.
        self._reset_xy_crop()
        self.viewer.set_frame_timestamps(rec._frame_timestamps)
        if volume is not None:
            self.viewer.set_volume(
                volume,
                channel_display=rec.channel_display,
                z_mode=self.combo_zproj.currentText(),
                z_index=rec.z_view_index,
                m=rec.m_index, t=0, z=rec.z_view_index,
                stage_xy_um=list(meta.get("stage_xy_um") or []),
            )
        else:
            self.viewer.set_channels(
                rec._raw_channels or {},
                channel_display=rec.channel_display,
            )

        self.btn_confirm.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_split_export.setEnabled(True)
        n_m = int(meta.get("n_multipoints", 1))
        self.btn_stitch.setEnabled(n_m > 1)

        if self._cb_progress:
            self._cb_progress(0)
        if self._cb_status:
            self._cb_status(
                "File loaded. Scroll M / T / Z, tweak channels, then Confirm."
            )

        self.loaded.emit(meta)

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Load failed", msg)
        if self._cb_progress:
            self._cb_progress(0)
        if self._cb_status:
            self._cb_status("Load failed.")

    # ── Viewer hooks ────────────────────────────────────────────────

    def _on_coords_changed(self, m: int, t: int, z: int) -> None:
        self.record.m_index = int(m)
        self.record.z_view_index = int(z)

    def _on_channels_changed(self) -> None:
        self.record.channel_display = self.viewer.channel_state()

    def _on_z_mode_changed(self, _mode: str) -> None:
        rec = self.record
        if rec._raw_volume is None:
            return
        m, t, z = self.viewer.coords()
        self.viewer.set_volume(
            rec._raw_volume,
            channel_display=rec.channel_display,
            z_mode=self.combo_zproj.currentText(),
            z_index=int(z), m=int(m), t=int(t), z=int(z),
            stage_xy_um=list((rec.nd2_metadata or {}).get("stage_xy_um") or []),
        )

    # ── V1.43 crop-to-selection ──
    def _on_crop_to_selection(self, axis: str, sel) -> None:
        """Crop the dataset to the tile-strip selection (off-thread).

        T selection keeps all M and Z; M/Z selection constrains only that axis.
        The original file is untouched — the crop builds a fresh in-RAM dataset.
        """
        from nd2studios.workers.crop_worker import CropWorker

        rec = self.record
        volume = rec._raw_volume
        indices = sorted(sel)
        if volume is None or len(indices) < 1:
            return
        # Remember the full dataset on the first crop so revert can restore it.
        if self._full_volume is None:
            self._full_volume = volume
            self._full_timestamps = rec._frame_timestamps

        if self._cb_status:
            self._cb_status(f"Cropping to {len(indices)} {axis.upper()} frame(s)…")

        worker = CropWorker(volume, axis, indices, parent=self)
        self._crop_worker = worker
        if self._cb_progress:
            worker.progress.connect(self._cb_progress)
        if self._cb_status:
            worker.status.connect(self._cb_status)
        worker.finished.connect(lambda cropped, ax=axis, idx=indices:
                                self._on_crop_done(cropped, ax, idx))
        worker.error.connect(lambda m: self._on_error(m))
        worker.start()

    def _on_crop_done(self, cropped, axis: str, indices) -> None:
        rec = self.record
        rec._raw_volume = cropped
        # Subset T timestamps to match a T crop so the metadata stays correct.
        if axis == "t" and self._full_timestamps is not None:
            try:
                ts = [self._full_timestamps[i] for i in indices
                      if 0 <= i < len(self._full_timestamps)]
                rec._frame_timestamps = ts
            except Exception:
                pass
        m, t, z = self.viewer.coords()
        self.viewer.set_frame_timestamps(rec._frame_timestamps)
        self.viewer.set_volume(
            cropped,
            channel_display=rec.channel_display,
            z_mode=self.combo_zproj.currentText(),
            z_index=0, m=0, t=0, z=0,
            stage_xy_um=list((rec.nd2_metadata or {}).get("stage_xy_um") or []),
        )
        self.btn_revert_crop.setVisible(True)
        if self._cb_progress:
            self._cb_progress(0)
        if self._cb_status:
            self._cb_status(
                f"Cropped to {cropped.n_multipoints}×{cropped.n_timepoints}×"
                f"{cropped.n_zslices} (M×T×Z). Original file untouched.")

    def _on_revert_crop(self) -> None:
        if self._full_volume is None:
            return
        rec = self.record
        rec._raw_volume = self._full_volume
        rec._frame_timestamps = self._full_timestamps
        self.viewer.set_frame_timestamps(self._full_timestamps)
        self.viewer.set_volume(
            self._full_volume,
            channel_display=rec.channel_display,
            z_mode=self.combo_zproj.currentText(),
            z_index=rec.z_view_index,
            m=0, t=0, z=rec.z_view_index,
            stage_xy_um=list((rec.nd2_metadata or {}).get("stage_xy_um") or []),
        )
        self._full_volume = None
        self._full_timestamps = None
        self.btn_revert_crop.setVisible(False)
        if self._cb_status:
            self._cb_status("Reverted to full dataset.")

    # ── V1.57 spatial XY export crop ──
    def _on_crop_xy_toggled(self, enabled: bool) -> None:
        self.viewer.set_crop_mode(enabled)

    def _on_xy_crop_drag(self, x: int, y: int, w: int, h: int) -> None:
        if not self.btn_crop_xy.isChecked():
            return
        result = self._show_xy_crop_dialog(x, y, w, h)
        if result is not None:
            self._apply_xy_crop(*result)

    def _on_xy_canvas_click(self, iy: float, ix: float) -> None:
        if not self.btn_crop_xy.isChecked():
            return
        result = self._show_xy_crop_dialog(int(ix), int(iy), 0, 0)
        if result is not None:
            self._apply_xy_crop(*result)

    def _frame_hw(self) -> Optional[Tuple[int, int]]:
        """Full-frame (height, width) of the currently loaded data, or None."""
        rec = self.record
        vol = rec._raw_volume
        if vol is not None:
            return int(vol.height), int(vol.width)
        src = rec._raw_channels or {}
        if not src:
            return None
        sample = next(iter(src.values()))
        shape = getattr(sample, "shape", None) or np.asarray(sample).shape
        return int(shape[-2]), int(shape[-1])

    def _show_xy_crop_dialog(
        self, x: int, y: int, w: int, h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Confirm/edit the export crop rectangle. Returns (x, y, w, h) or None."""
        hw = self._frame_hw()
        if hw is None:
            return None
        img_h, img_w = hw

        dlg = QDialog(self)
        dlg.setWindowTitle("Crop export region")
        layout = QVBoxLayout(dlg)
        info = QLabel(
            f"Image: {img_w} × {img_h} px  "
            f"(X = columns from left, Y = rows from top)"
        )
        info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        layout.addWidget(info)

        form = QFormLayout()
        sp_x = QSpinBox(); sp_x.setRange(0, img_w - 1)
        sp_x.setValue(max(0, min(x, img_w - 1)))
        sp_y = QSpinBox(); sp_y.setRange(0, img_h - 1)
        sp_y.setValue(max(0, min(y, img_h - 1)))
        sp_w = QSpinBox(); sp_w.setRange(1, img_w)
        sp_w.setValue(w if w > 0 else max(1, img_w - x))
        sp_h = QSpinBox(); sp_h.setRange(1, img_h)
        sp_h.setValue(h if h > 0 else max(1, img_h - y))
        form.addRow("X (left corner):", sp_x)
        form.addRow("Y (top corner):", sp_y)
        form.addRow("Width (px):", sp_w)
        form.addRow("Height (px):", sp_h)
        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        cx, cy, cw, ch = sp_x.value(), sp_y.value(), sp_w.value(), sp_h.value()
        cw = min(cw, img_w - cx)
        ch = min(ch, img_h - cy)
        return (cx, cy, cw, ch)

    def _apply_xy_crop(self, x: int, y: int, w: int, h: int) -> None:
        self._xy_crop = (x, y, w, h)
        self.record.crop_rect = (x, y, w, h)
        self.btn_crop_xy.setChecked(False)
        self.btn_reset_xy.setEnabled(True)
        self.lbl_xy_crop_status.setText(
            f"Export crop: x={x}, y={y}, {w}×{h} px")

    def _reset_xy_crop(self) -> None:
        self._xy_crop = None
        self.record.crop_rect = None
        if hasattr(self, "btn_crop_xy"):
            self.btn_crop_xy.setChecked(False)
            self.btn_reset_xy.setEnabled(False)
            self.lbl_xy_crop_status.setText("No XY crop")

    # ── V1.57 export from the Import tab ──
    def _export_basename(self) -> str:
        stem = (os.path.splitext(os.path.basename(self._filepath))[0]
                if self._filepath else "export")
        return f"{stem}_crop" if self._xy_crop is not None else stem

    def _export_source_channels(self) -> "OrderedDict[str, Any]":
        """Channels for the current M/Z view, with the XY crop applied.

        The T/M/Z tile-strip crop is already reflected because the cropped
        volume replaced ``record._raw_volume``.
        """
        rec = self.record
        vol = rec._raw_volume
        if vol is not None:
            m, _t, z = self.viewer.coords()
            chans: "OrderedDict[str, Any]" = OrderedDict(
                vol.all_channels_as_lazy(
                    m=int(m), z_mode=self.combo_zproj.currentText(),
                    z_index=int(z),
                )
            )
        else:
            chans = OrderedDict(rec._raw_channels or {})

        if self._xy_crop is not None:
            x, y, w, h = self._xy_crop
            cropped: "OrderedDict[str, Any]" = OrderedDict()
            for name, ch in chans.items():
                crop_fn = getattr(ch, "crop", None)
                if callable(crop_fn):
                    cropped[name] = crop_fn(y, y + h, x, x + w)
                else:
                    cropped[name] = np.asarray(ch)[..., y:y + h, x:x + w]
            chans = cropped
        return chans

    def _colors_enabled_lut(self, names):
        from nd2studios.widgets.image_viewer import CHANNEL_COLORS
        state = self.viewer.channel_state()
        colors: Dict[str, Tuple[int, int, int]] = {}
        enabled: Dict[str, bool] = {}
        lut: Dict[str, Tuple[float, float, float]] = {}
        for name in names:
            cfg = state.get(name, {})
            colors[name] = CHANNEL_COLORS.get(
                cfg.get("color", "gray"), (255, 255, 255))
            enabled[name] = bool(cfg.get("enabled", True))
            if "lut_lo" in cfg and "lut_hi" in cfg:
                lut[name] = (
                    float(cfg["lut_lo"]), float(cfg["lut_hi"]),
                    float(cfg.get("lut_gamma", 1.0)),
                )
        return colors, enabled, lut

    def _frame_timestamps(self):
        ts = self.record._frame_timestamps
        return np.asarray(ts) if ts is not None else None

    def _pixel_size_um(self) -> float:
        return float(self.record.pixel_size_um or 1.0)

    def _export_tiff(self) -> None:
        from nd2studios.workers.export_worker import ExportRequest
        rec = self.record
        if rec._raw_volume is None and not rec._raw_channels:
            QMessageBox.information(self, "Nothing to export", "Load a file first.")
            return

        is_zstack = (
            rec._raw_volume is not None
            and rec.n_zslices > 1
            and self.combo_zproj.currentText() == "none"
        )
        suffix = "_zstack" if is_zstack else "_tiff"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export TIFF", f"{self._export_basename()}{suffix}.tif",
            "TIFF (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return

        if is_zstack:
            vol = rec._raw_volume
            m, _t, _z = self.viewer.coords()
            state = self.viewer.channel_state()
            enabled = {
                name: bool(state.get(name, {}).get("enabled", True))
                for name in vol.channel_names
            }
            req = ExportRequest(
                mode="tiff_zstack", filepath=path, enabled=enabled,
                pixel_size_um=self._pixel_size_um(), bit_depth="passthrough",
                raw_volume=vol, m_index=int(m), crop_rect=self._xy_crop,
            )
        else:
            chans = self._export_source_channels()
            if not chans:
                QMessageBox.information(self, "Nothing to export", "Load a file first.")
                return
            colors, enabled, _lut = self._colors_enabled_lut(chans)
            req = ExportRequest(
                mode="tiff_stack", filepath=path, channels=dict(chans),
                colors=colors, enabled=enabled,
                pixel_size_um=self._pixel_size_um(), bit_depth="passthrough",
            )
        self._run_export(req, "Writing TIFF…")

    def _materialized_channels(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, ch in self._export_source_channels().items():
            m = getattr(ch, "materialize", None)
            out[name] = m() if callable(m) else np.asarray(ch)
        return out

    def _export_movie(self, fmt: str) -> None:
        from nd2studios.backend.exporters.movie_exporter import MovieOptions
        from nd2studios.widgets.export_preview_dialog import ExportPreviewDialog
        from nd2studios.workers.export_worker import ExportRequest

        channels = self._materialized_channels()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Load a file first.")
            return
        colors, enabled, lut = self._colors_enabled_lut(channels)
        ext = ".mp4" if fmt == "mp4" else ".gif"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Movie", f"{self._export_basename()}_movie{ext}",
            f"{fmt.upper()} (*{ext});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext

        opts = MovieOptions(fps=10.0, codec=fmt)
        dlg = ExportPreviewDialog(
            channels=channels, colors=colors, enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(),
            lut_settings=lut, movie_options=opts,
            title="Movie Export — Preview", parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        req = ExportRequest(
            mode="movie", filepath=path, channels=channels, colors=colors,
            enabled=enabled, pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(), movie_options=opts,
            lut_settings=lut, image_adjustments=dlg.adjustments(),
        )
        self._run_export(req, "Rendering movie…")

    def _export_image_sequence(self) -> None:
        from nd2studios.backend.exporters.movie_exporter import MovieOptions
        from nd2studios.widgets.export_preview_dialog import ExportPreviewDialog
        from nd2studios.workers.export_worker import ExportRequest

        channels = self._materialized_channels()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Load a file first.")
            return
        colors, enabled, lut = self._colors_enabled_lut(channels)
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder for image sequence", "")
        if not out_dir:
            return

        opts = MovieOptions(fps=10.0, codec="mp4")
        dlg = ExportPreviewDialog(
            channels=channels, colors=colors, enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(),
            lut_settings=lut, movie_options=opts,
            title="Image Sequence Export — Preview", parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        req = ExportRequest(
            mode="image_sequence", filepath=out_dir, channels=channels,
            colors=colors, enabled=enabled, pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(), movie_options=opts,
            lut_settings=lut, image_adjustments=dlg.adjustments(),
            basename=self._export_basename(),
        )
        self._run_export(req, "Writing image sequence…")

    # ── V1.72 split-by-axis export ──
    def _open_split_export_dialog(self) -> None:
        """Open the Split/Keep matrix dialog and run the multi-file export.

        Splits the current (cropped) volume along M/T/Z/C into separate files.
        Honors the XY crop, the enabled channels, and the per-channel color/LUT
        from the live viewer, so output matches what is on screen.
        """
        from nd2studios.widgets.split_export_dialog import SplitExportDialog
        from nd2studios.workers.export_worker import ExportRequest

        rec = self.record
        vol = rec._raw_volume
        if vol is None:
            QMessageBox.information(
                self, "Volume required",
                "Export by axis needs a volume with M / T / Z axes. Load an "
                "ND2 / TIFF volume first (use Quick export for channel-only data).",
            )
            return

        names = list(vol.channel_names)
        colors, enabled, lut = self._colors_enabled_lut(names)
        z_mode = self.combo_zproj.currentText()

        n_z_raw = int(vol.n_zslices)
        iterate_z = (z_mode == "none" and n_z_raw > 1)
        n_z_eff = n_z_raw if iterate_z else 1
        enabled_channels = [n for n in names if enabled.get(n, True)]
        if not enabled_channels:
            QMessageBox.information(
                self, "No channels enabled",
                "Enable at least one channel in the viewer before exporting.")
            return

        dlg = SplitExportDialog(
            n_m=int(vol.n_multipoints), n_t=int(vol.n_timepoints),
            n_z_eff=n_z_eff, iterate_z=iterate_z,
            n_c_enabled=len(enabled_channels), crop_rect=self._xy_crop,
            default_basename=self._export_basename(), z_mode=z_mode, parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        spec = dlg.spec()
        req = ExportRequest(
            mode="split", filepath=spec.output_dir, split_spec=spec,
            raw_volume=vol, colors=colors, enabled=enabled, lut_settings=lut,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=self._frame_timestamps(),
            crop_rect=self._xy_crop, bit_depth=spec.bit_depth,
            movie_options=dlg.movie_options(),
        )
        self._run_export(req, f"Split export → {spec.output_dir}")

    def _run_export(self, request, label: str) -> None:
        from nd2studios.workers.export_worker import ExportWorker
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.information(self, "Busy", "An export is already running.")
            return
        self._export_worker = ExportWorker(request, parent=self)
        if self._cb_progress:
            self._export_worker.progress.connect(self._cb_progress)
        if self._cb_status:
            self._export_worker.status.connect(self._cb_status)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_error)
        if self._cb_status:
            self._cb_status(label)
        self._export_worker.start()

    def _on_export_done(self, result: Any) -> None:
        if self._cb_progress:
            self._cb_progress(0)
        if self._cb_status:
            self._cb_status(f"Saved: {result}")
        QMessageBox.information(self, "Export complete", f"Wrote:\n{result}")

    def _on_confirm_clicked(self) -> None:
        if self._cb_confirm:
            self._cb_confirm(self)

    def _on_stitch_clicked(self) -> None:
        if self._cb_stitch:
            self._cb_stitch(self)

    # ── Metadata population ──────────────────────────────────────────

    def _populate_metadata_table(self, meta: Dict[str, Any]) -> None:
        rows: List[Tuple[str, str]] = [
            ("File", os.path.basename(meta.get("filepath", ""))),
            (
                "Dimensions",
                f"T={meta.get('n_timepoints')}, Z={meta.get('n_zslices')}, "
                f"C={meta.get('n_channels')}, P={meta.get('n_multipoints')}",
            ),
            ("Frame size", f"{meta.get('width')} x {meta.get('height')} px"),
            ("Dtype", str(meta.get("dtype", ""))),
            ("Pixel size", f"{meta.get('pixel_size_um', 1.0):.4f} um"),
            ("Z step", f"{meta.get('z_step_um', 1.0):.3f} um"),
        ]
        objm = meta.get("objective_magnification")
        objna = meta.get("objective_na")
        objname = meta.get("objective_name", "")
        if any([objname, objm, objna]):
            obj_text = objname or ""
            if objm is not None:
                obj_text += f"  {objm:g}x"
            if objna is not None:
                obj_text += f"  NA {objna:g}"
            rows.append(("Objective", obj_text.strip()))
        if meta.get("camera_name"):
            rows.append(("Camera", meta["camera_name"]))
        if meta.get("microscope_name"):
            rows.append(("Microscope", meta["microscope_name"]))
        if meta.get("binning_x") is not None:
            rows.append((
                "Binning",
                f"{meta['binning_x']}x{meta.get('binning_y', meta['binning_x'])}",
            ))
        ts = meta.get("frame_timestamps_s") or []
        if ts:
            rows.append((
                "Acq. span",
                f"{ts[-1] - ts[0]:.2f} s ({len(ts)} frames)",
            ))
            if len(ts) > 1:
                dt = (ts[-1] - ts[0]) / (len(ts) - 1)
                rows.append(("Mean dt", f"{dt:.3f} s"))

        self.meta_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.meta_table.setItem(i, 0, QTableWidgetItem(k))
            self.meta_table.setItem(i, 1, QTableWidgetItem(str(v)))

    def _populate_info_rows(self, meta: Dict[str, Any]) -> None:
        for row in self._info_rows:
            row.setParent(None)
            row.deleteLater()
        self._info_rows.clear()

        names: List[str] = list(
            meta.get("channel_names")
            or [f"Ch{i}" for i in range(int(meta.get("n_channels", 1)))]
        )
        exposures = meta.get("channel_exposure_ms") or []
        emissions = meta.get("channel_emission_nm") or []
        excitations = meta.get("channel_excitation_nm") or []
        for i, name in enumerate(names):
            row = ChannelInfoRow(
                idx=i, name=name,
                exposure_ms=(exposures[i] if i < len(exposures) else None),
                emission_nm=(emissions[i] if i < len(emissions) else None),
                excitation_nm=(excitations[i] if i < len(excitations) else None),
            )
            self._info_layout.insertWidget(self._info_layout.count() - 1, row)
            self._info_rows.append(row)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath

    def z_mode(self) -> str:
        return self.combo_zproj.currentText()

    def t_stride(self) -> int:
        return int(self.spin_t_stride.value())
