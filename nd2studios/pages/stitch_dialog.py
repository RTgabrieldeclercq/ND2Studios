"""
``StitchDialog`` — opened from the Import page to stitch multipoint tiles into a
pyramidal OME-TIFF using the V1.54 regime-aware pipeline.

Layout
------

    ┌──────────────────────────────────────────────┐
    │  Tile selection (click / drag to pick M)     │
    │  [Select all] [Select none]   N/M · WxH · MB │
    │  Stitching: regime / engine / blend / align  │
    │             illumination                     │
    │  Channels:  [☑DAPI ☑GFP ☐TRITC]              │
    │  Z mode:    [max ▾]                          │
    │  Output:    […/out.ome.tif]  [Browse]        │
    │             [Cancel]   [Export]              │
    └──────────────────────────────────────────────┘

The tile-selection preview uses ``compute_tile_layout`` (coordinate placement),
which is regime-agnostic for a thumbnail — registration only nudges tiles by a
fraction of a tile, so the preview still shows the true footprint.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from nd2studios.backend.stitch import StitchConfig, compute_tile_layout
from nd2studios.backend.stitch.config import (
    BLEND_AVERAGE, BLEND_FEATHER, BLEND_MAX, BLEND_NONE,
    ENGINE_ASHLAR, ENGINE_AUTO, ENGINE_COORDINATE, ENGINE_M2STITCH, ENGINE_PHASE,
    ILLUM_BASIC, ILLUM_BUILTIN, ILLUM_NONE,
    REGIME_AUTO, REGIME_OVERLAP, REGIME_ZERO,
)
from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.core.settings import Settings
from nd2studios.widgets.tile_layout import TileLayoutWidget
from nd2studios.workers.stitch_worker import StitchRequest, StitchWorker


_REGIME_ITEMS = [("Auto-detect", REGIME_AUTO),
                 ("Overlap (register)", REGIME_OVERLAP),
                 ("Zero-overlap (coordinates)", REGIME_ZERO)]
_ENGINE_ITEMS = [("Auto", ENGINE_AUTO),
                 ("Coordinate placement", ENGINE_COORDINATE),
                 ("Phase correlation (built-in)", ENGINE_PHASE),
                 ("m2stitch (grid)", ENGINE_M2STITCH),
                 ("ashlar", ENGINE_ASHLAR)]
_BLEND_ITEMS = [("Feather", BLEND_FEATHER), ("Average", BLEND_AVERAGE),
                ("Max", BLEND_MAX), ("None (overwrite)", BLEND_NONE)]
_ILLUM_ITEMS = [("None", ILLUM_NONE), ("Built-in flat-field", ILLUM_BUILTIN),
                ("BaSiC", ILLUM_BASIC)]


def _combo(items: List[Tuple[str, str]]) -> QComboBox:
    combo = QComboBox()
    for label, value in items:
        combo.addItem(label, value)
    return combo


class StitchDialog(QDialog):
    """Configure and launch a multipoint stitch → pyramidal OME-TIFF export."""

    def __init__(self,
                 volume: LazyND2Volume,
                 stage_xy_um: List[Tuple[float, float]],
                 channel_display: Optional[Dict[str, Dict[str, str]]] = None,
                 main_window=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Stitch M positions")
        self.setMinimumSize(680, 760)

        self.volume = volume
        self.stage_xy_um = list(stage_xy_um or [])
        self.main_window = main_window
        self._channel_display = channel_display or {}
        self._channel_checkboxes: List[QCheckBox] = []
        self._worker: Optional[StitchWorker] = None
        self._output_path: str = ""
        self._selected: Set[int] = set(range(volume.n_multipoints))

        self._build_ui()
        self._refresh_summary()

    # ── UI ──
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Interactive tile selection.
        layout_group = QGroupBox("Tile selection")
        lg_layout = QVBoxLayout(layout_group)
        self.tile_widget = TileLayoutWidget(
            mode="select", show_expand_button=True, minimum_size=(620, 260))
        self.tile_widget.set_tile_layout(
            stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height, tile_w=self.volume.width,
            m_indices=list(range(self.volume.n_multipoints)),
            current_m=0, selected_indices=self._selected)
        self.tile_widget.selection_changed.connect(self._on_selection_changed)
        self.tile_widget.expand_requested.connect(self._open_expanded_dialog)
        lg_layout.addWidget(self.tile_widget, stretch=1)

        bulk_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all_tiles(True))
        bulk_row.addWidget(btn_all)
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all_tiles(False))
        bulk_row.addWidget(btn_none)
        bulk_row.addStretch(1)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        bulk_row.addWidget(self.lbl_summary)
        lg_layout.addLayout(bulk_row)
        layout.addWidget(layout_group)

        # Stitching config.
        stitch_group = QGroupBox("Stitching")
        sform = QFormLayout(stitch_group)
        self.combo_regime = _combo(_REGIME_ITEMS)
        self.combo_engine = _combo(_ENGINE_ITEMS)
        self.combo_blend = _combo(_BLEND_ITEMS)
        self.combo_align = QComboBox()
        for c, name in enumerate(self.volume.channel_names):
            self.combo_align.addItem(name, c)
        self.combo_illum = _combo(_ILLUM_ITEMS)
        sform.addRow("Regime", self.combo_regime)
        sform.addRow("Engine", self.combo_engine)
        sform.addRow("Blend", self.combo_blend)
        sform.addRow("Align channel", self.combo_align)
        sform.addRow("Illumination", self.combo_illum)
        self.lbl_stitch_hint = QLabel(
            "Auto detects overlap from stage coordinates: overlapping tiles are "
            "registered, non-overlapping tiles placed by coordinates.")
        self.lbl_stitch_hint.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.lbl_stitch_hint.setWordWrap(True)
        sform.addRow(self.lbl_stitch_hint)
        layout.addWidget(stitch_group)

        # Channels (which to include; OME-TIFF preserves them all).
        ch_group = QGroupBox("Channels")
        ch_outer = QVBoxLayout(ch_group)
        row = QHBoxLayout()
        for name in self.volume.channel_names:
            cb = QCheckBox(name)
            cb.setChecked(self._channel_display.get(name, {}).get("enabled", True))
            cb.stateChanged.connect(lambda *_: self._refresh_summary())
            row.addWidget(cb)
            self._channel_checkboxes.append(cb)
        row.addStretch(1)
        ch_outer.addLayout(row)
        layout.addWidget(ch_group)

        # Z handling.
        z_group = QGroupBox("Z handling")
        z_outer = QVBoxLayout(z_group)
        z_form = QFormLayout()
        self.combo_z = QComboBox()
        self.combo_z.addItems(["max", "mean", "min", "none"])
        z_form.addRow("Z mode", self.combo_z)
        z_outer.addLayout(z_form)
        self.lbl_z_hint = QLabel("")
        self.lbl_z_hint.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.lbl_z_hint.setWordWrap(True)
        z_outer.addWidget(self.lbl_z_hint)
        layout.addWidget(z_group)
        self.combo_z.currentTextChanged.connect(self._on_z_mode_changed)
        self._on_z_mode_changed(self.combo_z.currentText())

        # Output.
        out_group = QGroupBox("Output (pyramidal OME-TIFF)")
        out_form = QHBoxLayout(out_group)
        self.lbl_out = QLabel("(no path chosen)")
        self.lbl_out.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        out_form.addWidget(self.lbl_out, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        out_form.addWidget(btn_browse)
        layout.addWidget(out_group)

        # Buttons.
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply)
        self._btn_export = bb.button(QDialogButtonBox.StandardButton.Apply)
        self._btn_export.setText("Export")
        self._btn_export.setObjectName("primaryBtn")
        bb.rejected.connect(self.reject)
        self._btn_export.clicked.connect(self._on_export)
        layout.addWidget(bb)

    # ── Selection ──
    def _on_selection_changed(self, indices: Set[int]) -> None:
        self._selected = set(indices)
        self._refresh_summary()

    def _set_all_tiles(self, value: bool) -> None:
        new_set = set(range(self.volume.n_multipoints)) if value else set()
        self.tile_widget.set_selected_indices(new_set)
        self._selected = new_set
        self._refresh_summary()

    def _open_expanded_dialog(self) -> None:
        from nd2studios.widgets.tile_layout import TileLayoutDialog
        dlg = TileLayoutDialog(
            mode="select", stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height, tile_w=self.volume.width,
            m_indices=list(range(self.volume.n_multipoints)),
            current_m=0, selected_indices=self._selected, parent=self)
        dlg.selection_changed.connect(self._sync_from_dialog)
        dlg.exec()

    def _sync_from_dialog(self, indices: Set[int]) -> None:
        self._selected = set(indices)
        self.tile_widget.set_selected_indices(self._selected)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n_total = self.volume.n_multipoints
        n_sel = len(self._selected)
        layout = compute_tile_layout(
            stage_xy_um=self.stage_xy_um, pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height, tile_w=self.volume.width,
            m_indices=sorted(self._selected)) if self._selected else None
        if layout is not None:
            n_ch = sum(1 for cb in self._channel_checkboxes if cb.isChecked()) or 1
            n_z_out = (self.volume.n_zslices
                       if self.combo_z.currentText() == "none" and self.volume.n_zslices > 1
                       else 1)
            mb = (layout.canvas_h * layout.canvas_w * n_ch
                  * self.volume.dtype.itemsize
                  * self.volume.n_timepoints * n_z_out) / 1_048_576
            z_note = f" · {n_z_out} Z" if n_z_out > 1 else ""
            self.lbl_summary.setText(
                f"{n_sel} / {n_total} tiles · {layout.canvas_w}×{layout.canvas_h} px"
                f"{z_note} · ~{mb:.0f} MB")
        else:
            self.lbl_summary.setText(f"0 / {n_total} tiles selected")

    def _on_z_mode_changed(self, mode: str) -> None:
        if self.volume.n_zslices <= 1:
            self.lbl_z_hint.setText("Single-Z file — Z mode has no effect.")
        elif mode == "none":
            self.lbl_z_hint.setText(
                f"All {self.volume.n_zslices} Z planes preserved (TZCYX).")
        else:
            self.lbl_z_hint.setText(
                f"{self.volume.n_zslices} Z planes collapsed via {mode}-projection.")
        self._refresh_summary()

    # ── Config ──
    def _selected_channels(self) -> List[int]:
        return [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]

    def _build_config(self) -> StitchConfig:
        return StitchConfig(
            regime=self.combo_regime.currentData(),
            engine=self.combo_engine.currentData(),
            blend=self.combo_blend.currentData(),
            align_channel=int(self.combo_align.currentData() or 0),
            illumination_correction=self.combo_illum.currentData(),
            z_mode=self.combo_z.currentText(),
            z_index=0,
            pixel_size_um=float(self.volume.pixel_size_um),
        )

    # ── Output / Export ──
    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Stitched OME-TIFF", "",
            "OME-TIFF (*.ome.tif *.ome.tiff);;TIFF (*.tif *.tiff);;All files (*)")
        if path:
            if not path.lower().endswith((".tif", ".tiff")):
                path += ".ome.tif"
            self._output_path = path
            self.lbl_out.setText(path)

    def _on_export(self) -> None:
        m_indices = sorted(self._selected)
        if not m_indices:
            QMessageBox.information(self, "Pick tiles",
                                    "Click at least one tile to include it.")
            return
        channel_indices = self._selected_channels()
        if not channel_indices:
            QMessageBox.information(self, "Pick channels", "Enable at least one channel.")
            return
        if not self._output_path:
            self._browse()
            if not self._output_path:
                return
        # Guard against a second click spawning a concurrent worker writing to
        # the same output path.
        if self._worker is not None and self._worker.isRunning():
            return
        self._btn_export.setEnabled(False)

        request = StitchRequest(
            volume=self.volume,
            stage_xy_um=self.stage_xy_um,
            m_indices=m_indices,
            channel_indices=channel_indices,
            config=self._build_config(),
            filepath=self._output_path,
        )
        self._worker = StitchWorker(request)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        if self.main_window is not None:
            self.main_window.set_status_text("Stitching…")
        self._worker.start()

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_done(self, result: str) -> None:
        res = getattr(self._worker, "result", None)
        if self.main_window is not None:
            self.main_window.set_progress(0)
            self.main_window.set_status_text(f"Stitched: {result}")
        detail = f"Wrote:\n{result}"
        if res is not None:
            detail += (f"\n\nRegime: {res.regime}   Engine: {res.engine}"
                       f"\nCanvas: {res.canvas_w}×{res.canvas_h} px   "
                       f"Tiles: {res.n_tiles}")
            if res.qc.get("png"):
                detail += f"\nQC: {os.path.basename(res.qc['png'])}"
        QMessageBox.information(self, "Stitch complete", detail)
        if self._worker is not None:
            self._worker.wait(5000)
        self.accept()

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Stitch failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
        if self._worker is not None:
            self._worker.wait(5000)
        self._btn_export.setEnabled(True)
