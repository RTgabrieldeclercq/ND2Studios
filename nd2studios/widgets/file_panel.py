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
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
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
        lbl.setMinimumWidth(110)
        layout.addWidget(lbl)
        bits: List[str] = []
        if exposure_ms is not None:
            bits.append(f"{exposure_ms:.0f} ms")
        if excitation_nm is not None:
            bits.append(f"ex {excitation_nm:.0f}")
        if emission_nm is not None:
            bits.append(f"em {emission_nm:.0f}")
        info = QLabel("  ".join(bits) or "—")
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
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
        self._show_close = show_close_button

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
        self._full_volume = None      # original dataset before any crop
        self._full_timestamps = None
        self._crop_worker = None
        self._inner_splitter.addWidget(self.viewer)

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
        bar.setFixedHeight(30)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(6, 0, 6, 0)
        hl.setSpacing(4)

        # Use plain ASCII so the button text is always visible on Windows.
        self._btn_collapse = QPushButton("<")
        self._btn_collapse.setObjectName("filePanelCollapseBtn")
        self._btn_collapse.setFixedSize(22, 22)
        self._btn_collapse.setToolTip("Collapse / expand file controls")
        self._btn_collapse.clicked.connect(self._toggle_sidebar)
        hl.addWidget(self._btn_collapse)

        self._lbl_name = QLabel("No file loaded")
        self._lbl_name.setObjectName("filePanelTitle")
        hl.addWidget(self._lbl_name, stretch=1)

        if self._show_close:
            self._btn_close = QPushButton("x")
            self._btn_close.setObjectName("filePanelCloseBtn")
            self._btn_close.setFixedSize(22, 22)
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
        self.lbl_filepath = QLabel("No file loaded.")
        self.lbl_filepath.setWordWrap(True)
        self.lbl_filepath.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
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
        self.meta_table.setMaximumHeight(200)
        ml.addWidget(self.meta_table)
        sl.addWidget(meta_group)

        # Channel info
        info_group = QGroupBox("Channel info")
        cl = QVBoxLayout(info_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(110)
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

        sl.addStretch(1)
        return ctrl

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
        canvas = getattr(self.viewer, "canvas", None)
        if canvas is not None:
            reset = getattr(canvas, "reset_zoom", None)
            if callable(reset):
                reset()

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
