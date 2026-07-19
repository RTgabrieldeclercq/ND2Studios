"""
Split-by-axis export dialog (V1.72).

Presents a per-axis **Split / Keep** matrix for M / T / Z / C plus independent
output-format checkboxes (TIFF hyperstack · PNG sequence · Movie), and a live
file-count summary. On accept it hands back a
:class:`~nd2studios.backend.exporters.split_exporter.SplitExportSpec` (via
:meth:`spec`) and a :class:`MovieOptions` (via :meth:`movie_options`) for the
export worker.

Layout follows the ND2Studios widget conventions: PySide6 only, hand-coded,
``scaled()`` on every fixed size / spacing / margin, ``objectName`` for theming,
and ``setAutoDefault(False)`` on tool buttons so Enter maps to OK.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from nd2studios.backend.exporters.movie_exporter import MovieOptions
from nd2studios.backend.exporters.split_exporter import (
    SplitExportSpec, plan_split_export,
)
from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import scaled, scale_qss


class SplitExportDialog(QDialog):
    """Configure a split-by-axis export.

    Parameters
    ----------
    n_m, n_t : int — multipoint / timepoint counts of the (cropped) volume.
    n_z_eff : int — effective Z count after projection (1 when projected).
    iterate_z : bool — True only when Z is browsable (z_mode 'none' and n_z > 1).
    n_c_enabled : int — number of enabled channels.
    crop_rect : (x, y, w, h) or None — the active XY export crop, for display.
    default_basename : str — prefilled output base name.
    z_mode : str — the panel's current Z mode (stored into the spec).
    """

    def __init__(
        self,
        *,
        n_m: int,
        n_t: int,
        n_z_eff: int,
        iterate_z: bool,
        n_c_enabled: int,
        crop_rect: Optional[Tuple[int, int, int, int]],
        default_basename: str,
        z_mode: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export by axis (split M / T / Z / C)")
        self._n_m = int(n_m)
        self._n_t = int(n_t)
        self._n_z = int(n_z_eff)
        self._iterate_z = bool(iterate_z)
        self._n_c = int(n_c_enabled)
        self._z_mode = z_mode
        self._out_dir = ""

        self._build_ui(crop_rect, default_basename)
        self._update_summary()
        self._update_ok()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(
        self, crop_rect: Optional[Tuple[int, int, int, int]],
        default_basename: str,
    ) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(scaled(12), scaled(10), scaled(12), scaled(10))
        root.setSpacing(scaled(8))

        info = QLabel(
            "Each axis: <b>Split</b> writes one file per index; <b>Keep</b> "
            "bundles every index inside each file. Split axes together define "
            "how many files are written."
        )
        info.setWordWrap(True)
        info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        root.addWidget(info)

        crop_note = "Full frame (no XY crop)"
        if crop_rect:
            x, y, w, h = crop_rect
            crop_note = f"XY crop: x={x}, y={y}, {w}×{h} px"
        dims = QLabel(
            f"Volume: M={self._n_m} · T={self._n_t} · "
            f"Z={self._n_z}{'' if self._iterate_z else ' (projected)'} · "
            f"C={self._n_c} enabled    |    {crop_note}"
        )
        dims.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        root.addWidget(dims)

        root.addWidget(self._build_axis_group())
        root.addWidget(self._build_format_group())
        root.addWidget(self._build_lut_group())
        root.addWidget(self._build_output_group(default_basename))

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setObjectName("splitExportSummary")
        self.lbl_summary.setStyleSheet(scale_qss(
            f"color: {Settings.ACCENT_CYAN}; font: 9pt;"
        ))
        root.addWidget(self.lbl_summary)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

    def _build_axis_group(self) -> QGroupBox:
        group = QGroupBox("Axes")
        grid = QGridLayout(group)
        grid.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        grid.setSpacing(scaled(6))

        for col, text in enumerate(("Axis", "Split (one file each)",
                                    "Keep (bundle in each file)", "")):
            hdr = QLabel(text)
            hdr.setStyleSheet(scale_qss(
                f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
            grid.addWidget(hdr, 0, col)

        # Per-axis: (attr, label, size, splittable, split_default, note)
        self._axis_groups: Dict[str, QButtonGroup] = {}
        self._split_radios: Dict[str, QRadioButton] = {}

        rows = [
            ("m", "M (positions)", self._n_m, self._n_m > 1, self._n_m > 1, ""),
            ("t", "T (timepoints)", self._n_t, self._n_t > 1, False, ""),
            ("z", "Z (slices)", self._n_z, self._iterate_z, False,
             "" if self._iterate_z else "projected → 1 plane"),
            ("c", "C (channels)", self._n_c, self._n_c > 1, False, ""),
        ]
        for r, (attr, label, size, splittable, split_default, note) in enumerate(
                rows, start=1):
            lbl = QLabel(f"{label} — {size}")
            grid.addWidget(lbl, r, 0)

            rb_split = QRadioButton()
            rb_keep = QRadioButton()
            bg = QButtonGroup(self)
            bg.setExclusive(True)
            bg.addButton(rb_split, 1)
            bg.addButton(rb_keep, 0)
            grid.addWidget(rb_split, r, 1, alignment=Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(rb_keep, r, 2, alignment=Qt.AlignmentFlag.AlignCenter)

            if not splittable:
                rb_split.setEnabled(False)
                rb_keep.setChecked(True)
            else:
                (rb_split if split_default else rb_keep).setChecked(True)

            if note:
                note_lbl = QLabel(note)
                note_lbl.setStyleSheet(scale_qss(
                    f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
                grid.addWidget(note_lbl, r, 3)

            bg.buttonToggled.connect(lambda *_: self._update_summary())
            self._axis_groups[attr] = bg
            self._split_radios[attr] = rb_split

        return group

    def _build_format_group(self) -> QGroupBox:
        group = QGroupBox("Output formats (one run can write several)")
        v = QVBoxLayout(group)
        v.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        v.setSpacing(scaled(4))

        # TIFF row + bit depth.
        tiff_row = QHBoxLayout()
        self.cb_tiff = QCheckBox("TIFF hyperstack")
        self.cb_tiff.setChecked(True)
        self.cb_tiff.toggled.connect(self._on_format_toggled)
        tiff_row.addWidget(self.cb_tiff)
        tiff_row.addWidget(QLabel("bit depth:"))
        self.combo_bitdepth = QComboBox()
        self.combo_bitdepth.addItems(["passthrough", "uint16", "uint8"])
        tiff_row.addWidget(self.combo_bitdepth)
        tiff_row.addStretch(1)
        v.addLayout(tiff_row)

        # PNG row.
        self.cb_png = QCheckBox("PNG image sequence (composited RGB, LUT applied)")
        self.cb_png.toggled.connect(self._on_format_toggled)
        v.addWidget(self.cb_png)

        # Movie row + format + fps.
        movie_row = QHBoxLayout()
        self.cb_movie = QCheckBox("Movie")
        self.cb_movie.toggled.connect(self._on_format_toggled)
        movie_row.addWidget(self.cb_movie)
        self.combo_movie_fmt = QComboBox()
        self.combo_movie_fmt.addItems(["mp4", "gif"])
        movie_row.addWidget(self.combo_movie_fmt)
        movie_row.addWidget(QLabel("fps:"))
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(0.5, 60.0)
        self.spin_fps.setSingleStep(0.5)
        self.spin_fps.setValue(10.0)
        movie_row.addWidget(self.spin_fps)
        movie_row.addStretch(1)
        v.addLayout(movie_row)

        return group

    def _build_lut_group(self) -> QGroupBox:
        group = QGroupBox("Contrast / LUT")
        v = QVBoxLayout(group)
        v.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        v.setSpacing(scaled(2))

        self._lut_group = QButtonGroup(self)
        self._lut_group.setExclusive(True)
        self._lut_ids = {0: "auto", 1: "manual", 2: "full"}
        labels = {
            0: "Auto-scale each channel (0.5–99.5 percentile)",
            1: "Keep manual (current viewer LUTs)",
            2: "Full range (0 … dtype max, no stretch)",
        }
        for i in (0, 1, 2):
            rb = QRadioButton(labels[i])
            self._lut_group.addButton(rb, i)
            v.addWidget(rb)
        self._lut_group.button(1).setChecked(True)   # default: manual

        note = QLabel(
            "Applies to PNG / movie, and to TIFF only when bit depth is "
            "uint8 / uint16 (a passthrough TIFF always stores raw data)."
        )
        note.setWordWrap(True)
        note.setStyleSheet(scale_qss(
            f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        v.addWidget(note)
        return group

    def _build_output_group(self, default_basename: str) -> QGroupBox:
        group = QGroupBox("Output")
        grid = QGridLayout(group)
        grid.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        grid.setSpacing(scaled(6))

        grid.addWidget(QLabel("Base name:"), 0, 0)
        self.le_basename = QLineEdit(default_basename)
        grid.addWidget(self.le_basename, 0, 1, 1, 2)

        grid.addWidget(QLabel("Folder:"), 1, 0)
        self.le_folder = QLineEdit()
        self.le_folder.setReadOnly(True)
        self.le_folder.setPlaceholderText("Choose an output folder…")
        self.le_folder.textChanged.connect(lambda *_: self._update_ok())
        grid.addWidget(self.le_folder, 1, 1)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setObjectName("compactBtn")
        self.btn_browse.setAutoDefault(False)
        self.btn_browse.clicked.connect(self._choose_folder)
        grid.addWidget(self.btn_browse, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", "")
        if path:
            self._out_dir = path
            self.le_folder.setText(path)

    def _on_format_toggled(self, *_args) -> None:
        self._update_summary()
        self._update_ok()

    def _flags(self) -> Dict[str, bool]:
        return {
            "split_m": self._axis_groups["m"].checkedId() == 1,
            "split_t": self._axis_groups["t"].checkedId() == 1,
            "split_z": self._axis_groups["z"].checkedId() == 1,
            "split_c": self._axis_groups["c"].checkedId() == 1,
            "write_tiff": self.cb_tiff.isChecked(),
            "write_png": self.cb_png.isChecked(),
            "write_movie": self.cb_movie.isChecked(),
        }

    def _update_summary(self) -> None:
        plan = plan_split_export(
            self._n_m, self._n_t, self._n_z, self._n_c, self._flags())
        self.lbl_summary.setText(plan["description"])

    def _update_ok(self) -> None:
        f = self._flags()
        any_format = f["write_tiff"] or f["write_png"] or f["write_movie"]
        ok = any_format and bool(self._out_dir)
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    # ── Results (valid after accept) ─────────────────────────────────────────
    def spec(self) -> SplitExportSpec:
        f = self._flags()
        basename = self.le_basename.text().strip() or "export"
        return SplitExportSpec(
            output_dir=self._out_dir,
            basename=basename,
            split_m=f["split_m"], split_t=f["split_t"],
            split_z=f["split_z"], split_c=f["split_c"],
            write_tiff=f["write_tiff"], write_png=f["write_png"],
            write_movie=f["write_movie"],
            z_mode=self._z_mode,
            bit_depth=self.combo_bitdepth.currentText(),
            lut_mode=self._lut_ids.get(self._lut_group.checkedId(), "manual"),
        )

    def movie_options(self) -> MovieOptions:
        return MovieOptions(
            fps=float(self.spin_fps.value()),
            codec=self.combo_movie_fmt.currentText(),
        )
