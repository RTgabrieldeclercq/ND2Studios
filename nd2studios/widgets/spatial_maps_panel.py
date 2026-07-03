"""In-viewer Spatial Maps panel — a faithful port of Cell-Tracker's spatial page.

``SpatialMapsPanel`` reproduces Cell-Tracker's Eulerian spatial-field view
(density, mean area, intensity, fold-change, self-fold, speed, velocity,
divergence, curl — plus interpolation of *any* measurement column) with the same
controls and the same matplotlib rendering (background image, cell-mask mode,
cell borders, quiver, scale bar, auto / global colour scaling). It is embedded in
the Pipelines image viewer as the "Spatial Maps" tab.

Two sidebar layouts share one set of control widgets (reparented on switch):

* **compact** (docked viewer) — a top toolbar of section dropdown buttons;
  opening one reveals that section's controls as a lateral row above the canvas.
* **full** (viewer maximized) — Cell-Tracker's vertical sidebar of stacked
  collapsible sections.

Configurations save/load as named **templates** via
:mod:`nd2studios.backend.spatial_templates`; a Spatial Maps node carries template
names and pre-loads them through :meth:`set_node_templates`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea,
    QSizePolicy, QInputDialog, QFileDialog, QMessageBox, QMenu, QButtonGroup,
    QDialog, QListWidget, QListWidgetItem, QAbstractItemView, QFrame,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.image_viewer import frame_to_uint8
from nd2studios.widgets.scale_bar import draw_scale_bar
from nd2studios.widgets.frame_strip import FrameStrip
from nd2studios.widgets.icon_button import icon_button, bind_toggle_icon
from nd2studios.backend.celltracker.fields import (
    compute_spatial_fields, FIELD_OPTIONS, _binned_mean_field, cell_footprint,
)
from nd2studios.backend.celltracker.metrics import compute_self_fold_change
from nd2studios.backend import spatial_templates

CMAPS_SEQ = ["viridis", "plasma", "inferno", "magma", "hot", "YlOrRd", "bone"]
CMAPS_DIV = ["coolwarm", "bwr", "seismic", "PiYG", "PRGn"]
ALL_CMAPS = CMAPS_SEQ + CMAPS_DIV

# Section order shown both in the compact top-bar and the full vertical sidebar.
_SECTION_ORDER = ("source", "field", "grid", "scale", "display", "scalebar")
_SECTION_TITLES = {
    "source": "Data Source", "field": "Field", "grid": "Grid",
    "scale": "Scale", "display": "Display", "scalebar": "Scale Bar",
}


class _CollapsibleSection(QWidget):
    """A clickable header that collapses/expands a content widget (full sidebar)."""

    def __init__(self, title: str, parent=None, collapsed=False):
        super().__init__(parent)
        self._collapsed = collapsed
        self._title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._btn = QPushButton(f"{'▶' if collapsed else '▼'}  {title}")
        self._btn.setStyleSheet(
            f"text-align: left; padding: 4px 8px; font: bold 9pt; "
            f"color: {Settings.ACCENT_CYAN}; background: {Settings.BG_TERTIARY}; "
            f"border: none; border-radius: 4px;")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)
        self._holder = QWidget()
        self._holder_lay = QVBoxLayout(self._holder)
        self._holder_lay.setContentsMargins(4, 4, 4, 4)
        self._holder_lay.setSpacing(3)
        lay.addWidget(self._holder)
        if collapsed:
            self._holder.setVisible(False)

    def set_content(self, w: QWidget) -> None:
        self._holder_lay.addWidget(w)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._holder.setVisible(not self._collapsed)
        self._btn.setText(f"{'▶' if self._collapsed else '▼'}  {self._title}")


class SpatialMapsPanel(QWidget):
    """Embeddable Spatial Maps view (compute + render + templates)."""

    m_change_requested = Signal(int)   # user picked a different multipoint
    status_message = Signal(str)       # surfaces hints to the page status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        # ── data state (set via set_data) ──
        self._tracked_df = None
        self._label_stack: Optional[np.ndarray] = None
        self._channels: Optional[Dict[str, np.ndarray]] = None
        self._raw_channels: Optional[Dict[str, np.ndarray]] = None
        self._field_shape: Tuple[int, int] = (0, 0)
        self._n_frames = 1
        self._m = 0
        self._current_fields: Dict[str, np.ndarray] = {}
        self._cell_mask: Optional[np.ndarray] = None
        self._t = 0
        # ── ui state ──
        self._compact = True
        self._active_compact_key: Optional[str] = None
        self._node_template_names: List[str] = []
        self._applying = False  # guard recompute storms during apply_config
        self._sections: Dict[str, QWidget] = {}
        # Per-frame field cache (keyed by (t, compute-signature)) so scrubbing
        # back/forth and playback don't recompute the griddata interpolation; the
        # heavy field computation runs once per frame per parameter set.
        self._field_cache: Dict[tuple, tuple] = {}
        self._pending_t: Optional[int] = None

        # Zoom / pan (matplotlib data-limit) state for the map canvas. ``_view``
        # is the current (x0, x1, y0, y1) data rect, or None for the full extent;
        # it persists across redraws / frame changes so zoom isn't lost.
        self._ax = None
        self._view: Optional[Tuple[float, float, float, float]] = None
        self._pan_mode = False
        self._pan_px0: Optional[tuple] = None
        self._pan_bb: Optional[tuple] = None
        self._pan_view0: Optional[tuple] = None

        # Frame scrubbing debounce — a fast drag emits many steps; coalesce them
        # so only the final frame is computed + drawn (the spatial field + the
        # matplotlib redraw are expensive).
        self._frame_debounce = QTimer(self)
        self._frame_debounce.setSingleShot(True)
        self._frame_debounce.setInterval(10)
        self._frame_debounce.timeout.connect(self._render_pending_frame)
        # Playback timer (▶ / ⏸ button).
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_controls()
        self._build_ui()
        self.set_compact(True)

    # ════════════════════════════════════════════════════════════════════
    # Control widgets (built once; reparented between the two layouts)
    # ════════════════════════════════════════════════════════════════════
    def _build_controls(self) -> None:
        # -- Data Source --
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)
        self.combo_source = QComboBox()
        self.combo_source.addItems(["Full field", "Cell mask"])
        self.combo_source.currentTextChanged.connect(self._on_compute_param_changed)
        self.combo_img_source = QComboBox()
        self.combo_img_source.addItems(["Processed", "Raw"])
        self.combo_img_source.currentTextChanged.connect(self._on_compute_param_changed)
        self.combo_int_ch = QComboBox()
        self.combo_int_ch.currentTextChanged.connect(self._on_compute_param_changed)
        self.combo_m = QComboBox()
        self.combo_m.setVisible(False)
        self.combo_m.currentIndexChanged.connect(self._on_m_changed)
        for w in (QLabel("Src:"), self.combo_source, self.combo_img_source,
                  QLabel("Int:"), self.combo_int_ch, self.combo_m):
            bl.addWidget(w)
        self._sections["source"] = body

        # -- Field --
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        self.combo_field = QComboBox()
        for key, label, _tip in FIELD_OPTIONS:
            self.combo_field.addItem(label, key)
        self._n_builtin_fields = self.combo_field.count()
        self.combo_field.currentIndexChanged.connect(self._on_field_changed)
        bl.addWidget(QLabel("Field:"))
        bl.addWidget(self.combo_field, stretch=1)
        self._sections["field"] = body

        # -- Grid --
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(5, 100)
        self.spin_grid.setValue(20)
        self.spin_grid.setSuffix("px")
        self.spin_grid.valueChanged.connect(self._on_compute_param_changed)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0, 10)
        self.spin_sigma.setValue(2.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.valueChanged.connect(self._on_compute_param_changed)
        for w in (QLabel("Step:"), self.spin_grid, QLabel("σ:"), self.spin_sigma):
            bl.addWidget(w)
        self._sections["grid"] = body

        # -- Scale --
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        self.cb_auto_scale = QCheckBox("Auto/frame")
        self.cb_auto_scale.stateChanged.connect(self._draw_frame)
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e6, 1e6)
        self.spin_vmin.setDecimals(3)
        self.spin_vmin.setPrefix("Min:")
        self.spin_vmin.valueChanged.connect(self._draw_frame)
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e6, 1e6)
        self.spin_vmax.setDecimals(3)
        self.spin_vmax.setValue(1.0)
        self.spin_vmax.setPrefix("Max:")
        self.spin_vmax.valueChanged.connect(self._draw_frame)
        self.btn_calc_scale = QPushButton("Global Scale")
        self.btn_calc_scale.clicked.connect(self._compute_global_scale)
        for w in (self.cb_auto_scale, self.spin_vmin, self.spin_vmax,
                  self.btn_calc_scale):
            bl.addWidget(w)
        self._sections["scale"] = body

        # -- Display --
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        self.cb_overlay = QCheckBox("Image")
        self.cb_overlay.setChecked(True)
        self.cb_overlay.stateChanged.connect(self._draw_frame)
        self.cb_cell_borders = QCheckBox("Borders")
        self.cb_cell_borders.stateChanged.connect(self._draw_frame)
        self.combo_border_color = QComboBox()
        self.combo_border_color.addItems(
            ["white", "red", "green", "cyan", "yellow", "magenta"])
        self.combo_border_color.currentTextChanged.connect(self._draw_frame)
        self.combo_bg_ch = QComboBox()
        self.combo_bg_ch.currentTextChanged.connect(self._draw_frame)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(ALL_CMAPS)
        self.combo_cmap.currentTextChanged.connect(self._draw_frame)
        self.spin_opacity = QDoubleSpinBox()
        self.spin_opacity.setRange(0.1, 1.0)
        self.spin_opacity.setValue(0.6)
        self.spin_opacity.setSingleStep(0.1)
        self.spin_opacity.setPrefix("α:")
        self.spin_opacity.valueChanged.connect(self._draw_frame)
        for w in (self.cb_overlay, self.cb_cell_borders,
                  QLabel("Brd:"), self.combo_border_color,
                  QLabel("BG:"), self.combo_bg_ch,
                  QLabel("Cmap:"), self.combo_cmap, self.spin_opacity):
            bl.addWidget(w)
        self._sections["display"] = body

        # -- Scale Bar --
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        self.cb_scalebar = QCheckBox("Bar")
        self.cb_scalebar.stateChanged.connect(self._draw_frame)
        self.spin_px_um = QDoubleSpinBox()
        self.spin_px_um.setRange(0.001, 100)
        self.spin_px_um.setValue(1.0)
        self.spin_px_um.setDecimals(3)
        self.spin_px_um.setPrefix("px=")
        self.spin_px_um.setSuffix("µm")
        self.spin_px_um.valueChanged.connect(self._draw_frame)
        self.spin_bar_um = QDoubleSpinBox()
        self.spin_bar_um.setRange(1, 10000)
        self.spin_bar_um.setValue(100)
        self.spin_bar_um.setSuffix("µm")
        self.spin_bar_um.valueChanged.connect(self._draw_frame)
        self.spin_bar_thick = QSpinBox()
        self.spin_bar_thick.setRange(1, 30)
        self.spin_bar_thick.setValue(5)
        self.spin_bar_thick.setPrefix("Thk:")
        self.spin_bar_thick.valueChanged.connect(self._draw_frame)
        self.spin_bar_font = QSpinBox()
        self.spin_bar_font.setRange(4, 30)
        self.spin_bar_font.setValue(10)
        self.spin_bar_font.setPrefix("Fnt:")
        self.spin_bar_font.valueChanged.connect(self._draw_frame)
        self.combo_bar_loc = QComboBox()
        self.combo_bar_loc.addItems(
            ["bottom-right", "bottom-left", "top-right", "top-left"])
        self.combo_bar_loc.currentTextChanged.connect(self._draw_frame)
        self.combo_bar_color = QComboBox()
        self.combo_bar_color.addItems(["white", "black", "yellow", "cyan", "red"])
        self.combo_bar_color.currentTextChanged.connect(self._draw_frame)
        self.spin_bar_bg = QDoubleSpinBox()
        self.spin_bar_bg.setRange(0, 1)
        self.spin_bar_bg.setValue(0.5)
        self.spin_bar_bg.setSingleStep(0.1)
        self.spin_bar_bg.setPrefix("BG:")
        self.spin_bar_bg.valueChanged.connect(self._draw_frame)
        self.combo_bar_text_pos = QComboBox()
        self.combo_bar_text_pos.addItems(
            ["auto", "above", "below", "left", "right", "center", "none"])
        self.combo_bar_text_pos.currentTextChanged.connect(self._draw_frame)
        self.spin_bar_text_off = QDoubleSpinBox()
        self.spin_bar_text_off.setRange(0.0, 20.0)
        self.spin_bar_text_off.setValue(0.5)
        self.spin_bar_text_off.setSingleStep(0.1)
        self.spin_bar_text_off.setDecimals(2)
        self.spin_bar_text_off.setPrefix("Gap:")
        self.spin_bar_text_off.valueChanged.connect(self._draw_frame)
        for w in (self.cb_scalebar, self.spin_px_um, self.spin_bar_um,
                  self.spin_bar_thick, self.spin_bar_font, self.combo_bar_loc,
                  self.combo_bar_color, self.spin_bar_bg,
                  QLabel("Txt:"), self.combo_bar_text_pos, self.spin_bar_text_off):
            bl.addWidget(w)
        self._sections["scalebar"] = body

    # ════════════════════════════════════════════════════════════════════
    # Static UI (canvas, template/save toolbar, frame slider, two sidebars)
    # ════════════════════════════════════════════════════════════════════
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        # Compact top bar: section dropdown buttons + a single lateral flyout.
        self._compact_top = QWidget()
        ct = QVBoxLayout(self._compact_top)
        ct.setContentsMargins(0, 0, 0, 0)
        ct.setSpacing(2)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(3)
        self._compact_btn_group = QButtonGroup(self)
        self._compact_btn_group.setExclusive(True)
        self._compact_buttons: Dict[str, QToolButton] = {}
        for key in _SECTION_ORDER:
            b = QToolButton()
            b.setText(_SECTION_TITLES[key] + " ▾")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, k=key: self._on_compact_btn(k))
            self._compact_btn_group.addButton(b)
            self._compact_buttons[key] = b
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        ct.addLayout(btn_row)
        self._flyout = QWidget()
        self._flyout_lay = QHBoxLayout(self._flyout)
        self._flyout_lay.setContentsMargins(2, 2, 2, 2)
        self._flyout_lay.setSpacing(4)
        self._flyout.setVisible(False)
        ct.addWidget(self._flyout)
        outer.addWidget(self._compact_top)

        # Middle: [full sidebar] | [toolbar + canvas + stats + frame slider]
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(4)

        self._full_sidebar = QScrollArea()
        self._full_sidebar.setWidgetResizable(True)
        self._full_sidebar.setFixedWidth(270)
        self._full_sidebar.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_inner = QWidget()
        self._side_lay = QVBoxLayout(side_inner)
        self._side_lay.setContentsMargins(2, 2, 2, 2)
        self._side_lay.setSpacing(4)
        self._collapsibles: Dict[str, _CollapsibleSection] = {}
        for key in _SECTION_ORDER:
            sec = _CollapsibleSection(
                _SECTION_TITLES[key],
                collapsed=key in ("grid", "scale", "scalebar"))
            self._collapsibles[key] = sec
            self._side_lay.addWidget(sec)
        self._side_lay.addStretch(1)
        self._full_sidebar.setWidget(side_inner)
        mid.addWidget(self._full_sidebar)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(3)
        right.addLayout(self._build_template_toolbar())
        right.addLayout(self._build_zoom_toolbar())
        self.canvas = MplCanvas(self, width=8, height=6)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        # Zoom (scroll) + pan (drag) on the map, like the image viewer.
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        right.addWidget(self.canvas, stretch=1)
        self.lbl_stats = QLabel("—")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        right.addWidget(self.lbl_stats)
        right.addLayout(self._build_frame_row())
        mid.addLayout(right, stretch=1)
        outer.addLayout(mid, stretch=1)

    def _build_template_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(QLabel("Template:"))
        self.combo_template = QComboBox()
        self.combo_template.setMinimumWidth(160)
        self.combo_template.activated.connect(self._on_template_activated)
        row.addWidget(self.combo_template)
        btn_save_tpl = QPushButton("Save template…")
        btn_save_tpl.clicked.connect(self._on_save_template)
        row.addWidget(btn_save_tpl)
        btn_load_file = QPushButton("Load file…")
        btn_load_file.setToolTip("Import a template JSON file into the library.")
        btn_load_file.clicked.connect(self._on_load_file)
        row.addWidget(btn_load_file)
        row.addStretch(1)
        self.btn_save_img = QToolButton()
        self.btn_save_img.setText("Save image ▾")
        self.btn_save_img.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.btn_save_img)
        menu.addAction("Current frame…", self._save_current_frame)
        menu.addAction("All frames…", self._save_all_frames)
        self.btn_save_img.setMenu(menu)
        row.addWidget(self.btn_save_img)
        self._refresh_template_combo()
        return row

    def _build_frame_row(self) -> QHBoxLayout:
        """The ND2Studios T-axis control set: NIS-Elements ``FrameStrip`` +
        play/pause + FPS (mirrors ``MultiAxisViewer._make_axis_row`` for T)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel("T:")
        lbl.setFixedWidth(18)
        row.addWidget(lbl)
        self._frame_strip = FrameStrip("T")
        self._frame_strip.current_changed.connect(self._on_strip_current)
        row.addWidget(self._frame_strip, stretch=1)
        self._frame_label = QLabel("1/1")
        self._frame_label.setFixedWidth(56)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        row.addWidget(self._frame_label)
        self._play_btn = icon_button(
            "fa5s.play", "Play / pause", checkable=True,
            object_name="playBtn", button_px=26, icon_px=12)
        bind_toggle_icon(self._play_btn, "fa5s.play", "fa5s.pause",
                         color_checked=Settings.ACCENT_GREEN)
        self._play_btn.toggled.connect(self._on_play_toggled)
        row.addWidget(self._play_btn)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.1, 60.0)
        self._fps_spin.setValue(5.0)
        self._fps_spin.setSingleStep(0.5)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setFixedWidth(72)
        self._fps_spin.setToolTip("Playback speed")
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        row.addWidget(self._fps_spin)
        return row

    # ── frame navigation (FrameStrip + playback, debounced) ─────────────────
    def _on_strip_current(self, t: int) -> None:
        """Strip step (drag / arrow / playback). Debounced: a fast drag coalesces
        to a single compute + redraw of the final frame."""
        self._pending_t = int(t)
        self._frame_label.setText(f"{int(t) + 1}/{max(1, self._n_frames)}")
        self._frame_debounce.start()

    def _render_pending_frame(self) -> None:
        if self._pending_t is None:
            return
        self._t = self._pending_t
        self._compute_frame(self._t)
        self._draw_frame()

    def _on_play_toggled(self, on: bool) -> None:
        if on and self._n_frames > 1:
            self._play_timer.start(int(1000 / max(0.1, self._fps_spin.value())))
        else:
            self._play_timer.stop()

    def _on_fps_changed(self, *_a) -> None:
        if self._play_btn.isChecked():
            self._play_timer.start(int(1000 / max(0.1, self._fps_spin.value())))

    def _advance_frame(self) -> None:
        nxt = (self._t + 1) % max(1, self._n_frames)
        self._frame_strip.set_current(nxt, emit=True)

    # ── zoom / pan (data-limit) ─────────────────────────────────────────────
    def _build_zoom_toolbar(self) -> QHBoxLayout:
        """Home / + / - / Pan controls — same set as the image viewer's
        ``ZoomToolbar`` (here driving matplotlib axis limits)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        BTN_W, BTN_H = 56, 24
        btn_home = QPushButton("Home")
        btn_home.setObjectName("compactBtn")
        btn_home.setToolTip("Reset view (fit map to window)")
        btn_home.setFixedSize(BTN_W, BTN_H)
        btn_home.clicked.connect(self._on_zoom_home)
        row.addWidget(btn_home)
        btn_in = QPushButton("+")
        btn_in.setObjectName("compactBtn")
        btn_in.setToolTip("Zoom in")
        btn_in.setFixedSize(BTN_H, BTN_H)
        btn_in.clicked.connect(lambda: self._zoom_about(0.8))
        row.addWidget(btn_in)
        btn_out = QPushButton("-")
        btn_out.setObjectName("compactBtn")
        btn_out.setToolTip("Zoom out")
        btn_out.setFixedSize(BTN_H, BTN_H)
        btn_out.clicked.connect(lambda: self._zoom_about(1.25))
        row.addWidget(btn_out)
        self._btn_pan = QPushButton("Pan")
        self._btn_pan.setObjectName("compactBtn")
        self._btn_pan.setToolTip(
            "Toggle pan. When on, left-click and drag to move the zoomed map.")
        self._btn_pan.setCheckable(True)
        self._btn_pan.setFixedSize(BTN_W, BTN_H)
        self._btn_pan.toggled.connect(self._set_pan_mode)
        row.addWidget(self._btn_pan)
        self._lbl_zoom = QLabel("100%")
        self._lbl_zoom.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self._lbl_zoom.setMinimumWidth(48)
        row.addWidget(self._lbl_zoom)
        row.addStretch(1)
        return row

    def _full_rect(self) -> Optional[Tuple[float, float, float, float]]:
        H, W = self._field_shape
        if H <= 0 or W <= 0:
            return None
        return (0.0, float(W), 0.0, float(H))

    def _current_rect(self) -> Optional[Tuple[float, float, float, float]]:
        return self._view if self._view is not None else self._full_rect()

    def _apply_view(self, ax) -> None:
        rect = self._current_rect()
        if rect is None:
            return
        x0, x1, y0, y1 = rect
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # inverted (image convention: y increases downward)

    @staticmethod
    def _clamp_axis(a0, a1, lo, hi):
        width = a1 - a0
        if width >= (hi - lo):
            return lo, hi
        if a0 < lo:
            a0, a1 = lo, lo + width
        if a1 > hi:
            a1, a0 = hi, hi - width
        return max(a0, lo), min(a1, hi)

    def _update_zoom_label(self) -> None:
        full = self._full_rect()
        if full is None or not hasattr(self, "_lbl_zoom"):
            return
        rect = self._current_rect()
        pct = int(round((full[1] - full[0]) / max(1e-6, rect[1] - rect[0]) * 100))
        self._lbl_zoom.setText(f"{pct}%")

    def _zoom_about(self, factor: float, center=None) -> None:
        full = self._full_rect()
        if full is None:
            return
        x0, x1, y0, y1 = self._current_rect()
        w, h = x1 - x0, y1 - y0
        cx = center[0] if center and center[0] is not None else (x0 + x1) / 2
        cy = center[1] if center and center[1] is not None else (y0 + y1) / 2
        fw, fh = full[1] - full[0], full[3] - full[2]
        nw = min(max(w * factor, 4.0), fw)
        nh = min(max(h * factor, 4.0), fh)
        rx = (cx - x0) / w if w else 0.5
        ry = (cy - y0) / h if h else 0.5
        nx0, nx1 = self._clamp_axis(cx - rx * nw, cx - rx * nw + nw, full[0], full[1])
        ny0, ny1 = self._clamp_axis(cy - ry * nh, cy - ry * nh + nh, full[2], full[3])
        if (nx1 - nx0) >= fw - 1e-6 and (ny1 - ny0) >= fh - 1e-6:
            self._view = None
        else:
            self._view = (nx0, nx1, ny0, ny1)
        self._draw_frame()

    def _on_zoom_home(self) -> None:
        self._view = None
        self._draw_frame()

    def _set_pan_mode(self, on: bool) -> None:
        self._pan_mode = bool(on)
        self.canvas.setCursor(
            Qt.CursorShape.OpenHandCursor if on else Qt.CursorShape.ArrowCursor)

    def _on_scroll(self, event) -> None:
        if self._ax is None or event.inaxes is not self._ax:
            return
        self._zoom_about(0.8 if event.button == "up" else 1.25,
                         center=(event.xdata, event.ydata))

    def _on_canvas_press(self, event) -> None:
        if (not self._pan_mode or event.button != 1 or self._ax is None
                or event.inaxes is not self._ax):
            return
        bb = self._ax.get_window_extent()
        self._pan_px0 = (event.x, event.y)
        self._pan_bb = (bb.width, bb.height)
        self._pan_view0 = self._current_rect()
        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_canvas_motion(self, event) -> None:
        if self._pan_view0 is None or self._ax is None or event.x is None:
            return
        x0, x1, y0, y1 = self._pan_view0
        bw, bh = self._pan_bb
        if bw <= 0 or bh <= 0:
            return
        ddx = ((event.x - self._pan_px0[0]) / bw) * (x1 - x0)
        ddy = -((event.y - self._pan_px0[1]) / bh) * (y1 - y0)
        full = self._full_rect()
        nx0, nx1 = self._clamp_axis(x0 - ddx, x1 - ddx, full[0], full[1])
        ny0, ny1 = self._clamp_axis(y0 - ddy, y1 - ddy, full[2], full[3])
        self._view = (nx0, nx1, ny0, ny1)
        self._ax.set_xlim(nx0, nx1)
        self._ax.set_ylim(ny1, ny0)
        self.canvas.draw_idle()
        self._update_zoom_label()

    def _on_canvas_release(self, event) -> None:
        if self._pan_view0 is not None:
            self._pan_view0 = None
            self.canvas.setCursor(
                Qt.CursorShape.OpenHandCursor if self._pan_mode
                else Qt.CursorShape.ArrowCursor)
            self._draw_frame()  # reposition the scale bar in the new corner

    # ── layout switching ──────────────────────────────────────────────────
    def _detach_sections(self) -> None:
        for body in self._sections.values():
            body.setParent(None)

    def set_compact(self, compact: bool) -> None:
        """Switch between the compact top-bar (docked) and full vertical sidebar
        (maximized) layouts. Same control widgets, reparented either way."""
        self._compact = bool(compact)
        self._detach_sections()
        if self._compact:
            self._full_sidebar.setVisible(False)
            self._compact_top.setVisible(True)
            self._active_compact_key = None
            self._flyout.setVisible(False)
            for b in self._compact_buttons.values():
                b.setChecked(False)
        else:
            self._compact_top.setVisible(False)
            self._full_sidebar.setVisible(True)
            for key in _SECTION_ORDER:
                self._collapsibles[key].set_content(self._sections[key])
                self._sections[key].setVisible(True)

    def _on_compact_btn(self, key: str) -> None:
        # Toggle the flyout for the chosen section (clicking the active one hides).
        if self._active_compact_key == key:
            self._active_compact_key = None
            self._compact_buttons[key].setChecked(False)
            self._flyout.setVisible(False)
            self._sections[key].setParent(None)
            return
        # Remove whatever body is currently mounted.
        while self._flyout_lay.count():
            it = self._flyout_lay.takeAt(0)
            if it.widget() is not None:
                it.widget().setParent(None)
        self._active_compact_key = key
        body = self._sections[key]
        self._flyout_lay.addWidget(body)
        self._flyout_lay.addStretch(1)
        body.setVisible(True)
        self._flyout.setVisible(True)

    # ════════════════════════════════════════════════════════════════════
    # Data feeding
    # ════════════════════════════════════════════════════════════════════
    def set_data(self, tracked_df, label_stack, channels, raw_channels,
                 field_shape, pixel_size_um, n_frames, m=0, n_multipoints=1) -> None:
        """Populate the panel for one multipoint ``m``.

        ``tracked_df`` is a Cell-Tracker-style DataFrame (see
        ``celltracker_bridge.build_tracked_df``); ``label_stack`` is ``(T,H,W)`` or
        None; ``channels`` / ``raw_channels`` are ``{name: (T,H,W)}`` (raw may be
        None); ``field_shape`` is ``(H, W)``.
        """
        self._tracked_df = tracked_df
        self._label_stack = label_stack
        self._channels = channels or {}
        self._raw_channels = raw_channels
        self._field_shape = tuple(field_shape) if field_shape else (0, 0)
        self._n_frames = max(1, int(n_frames or 1))
        self._m = int(m)
        self._field_cache.clear()  # new data invalidates every cached frame
        self._view = None          # new dataset → reset zoom to full extent

        # Field dropdown: built-in fields + any extra numeric measurement columns.
        self._refresh_field_columns()

        # Intensity-channel combo: channels + any "{ch}_mean" df columns.
        names = set(self._channels.keys())
        if tracked_df is not None:
            for c in list(getattr(tracked_df, "columns", [])):
                if isinstance(c, str) and c.endswith("_mean"):
                    names.add(c[:-len("_mean")])
        self.combo_int_ch.blockSignals(True)
        self.combo_int_ch.clear()
        self.combo_int_ch.addItems(sorted(names))
        self.combo_int_ch.blockSignals(False)

        self.combo_bg_ch.blockSignals(True)
        self.combo_bg_ch.clear()
        self.combo_bg_ch.addItems(list(self._channels.keys()))
        self.combo_bg_ch.blockSignals(False)

        # Raw / image-source controls only make sense with a raw stack present.
        has_raw = bool(self._raw_channels)
        self.combo_img_source.setEnabled(has_raw)
        if not has_raw and self.combo_img_source.currentText() == "Raw":
            self.combo_img_source.setCurrentText("Processed")

        # Multipoint selector (only when more than one M exists).
        self.combo_m.blockSignals(True)
        self.combo_m.clear()
        if n_multipoints and n_multipoints > 1:
            self.combo_m.addItems([f"M{i + 1}" for i in range(int(n_multipoints))])
            self.combo_m.setCurrentIndex(min(self._m, int(n_multipoints) - 1))
            self.combo_m.setVisible(True)
        else:
            self.combo_m.setVisible(False)
        self.combo_m.blockSignals(False)

        if pixel_size_um and pixel_size_um > 0:
            self.spin_px_um.blockSignals(True)
            self.spin_px_um.setValue(float(pixel_size_um))
            self.spin_px_um.blockSignals(False)

        # Land on a frame that actually has measured objects. In *preview* the
        # rows are scoped to the selected T-planes (often not frame 0), so a naive
        # _t = 0 would find nothing and render blank.
        frames_present: list = []
        df = self._tracked_df
        if df is not None and len(df) and "frame" in getattr(df, "columns", []):
            try:
                frames_present = sorted({int(v) for v in df["frame"].tolist()})
            except (TypeError, ValueError):
                frames_present = []
        if frames_present and self._t not in frames_present:
            self._t = frames_present[0]
        self._t = min(max(0, self._t), max(0, self._n_frames - 1))
        self._frame_strip.set_count(self._n_frames)
        self._frame_strip.set_current(self._t, emit=False)
        self._frame_label.setText(f"{self._t + 1}/{max(1, self._n_frames)}")

        # Fast open: compute just the current frame and fit the colour scale to it
        # (a single griddata pass) rather than scanning every frame. The user can
        # press "Global Scale" for a fixed cross-frame range.
        if frames_present:
            self._compute_frame(self._t)
            self._autofit_scale_from_current()
            self._draw_frame()
        else:
            self._recompute_and_draw()

    def _refresh_field_columns(self) -> None:
        """Append numeric measurement columns (beyond the built-ins) to the Field
        combo as binned ``col:<name>`` entries (Gaussian-weighted local mean)."""
        self.combo_field.blockSignals(True)
        try:
            while self.combo_field.count() > self._n_builtin_fields:
                self.combo_field.removeItem(self.combo_field.count() - 1)
            df = self._tracked_df
            if df is None:
                return
            builtin_cols = {"frame", "label", "track_id", "centroid_y",
                            "centroid_x", "area"}
            for c in list(getattr(df, "columns", [])):
                if not isinstance(c, str) or c in builtin_cols:
                    continue
                if c.endswith("_mean"):
                    continue  # surfaced through the Intensity field instead
                try:
                    if not np.issubdtype(df[c].dtype, np.number):
                        continue
                except (TypeError, ValueError):
                    continue
                self.combo_field.addItem(f"⟐ {c}", f"col:{c}")
        finally:
            self.combo_field.blockSignals(False)

    def _on_m_changed(self, idx: int) -> None:
        if idx >= 0 and idx != self._m:
            self.m_change_requested.emit(idx)

    # ════════════════════════════════════════════════════════════════════
    # Computation  (ported from Cell-Tracker spatial_page)
    # ════════════════════════════════════════════════════════════════════
    def _get_img(self, ch, t):
        use_raw = self.combo_img_source.currentText() == "Raw"
        if use_raw and self._raw_channels and ch in self._raw_channels:
            d = self._raw_channels[ch]
            return d[t] if t < d.shape[0] else None
        if self._channels and ch in self._channels:
            d = self._channels[ch]
            return d[t] if t < d.shape[0] else None
        return None

    def _on_field_changed(self, *a):
        if self._applying:
            return
        if self._tracked_df is not None and len(self._tracked_df):
            # Fit the scale to the current frame (fast). "Global Scale" still does
            # the full cross-frame pass on demand.
            self._compute_frame(self._t)
            self._autofit_scale_from_current()
            self._draw_frame()
        else:
            self._recompute_and_draw()

    def _autofit_scale_from_current(self) -> None:
        """Set vmin/vmax from the current frame's field (one-frame auto-fit)."""
        fk = self.combo_field.currentData()
        arr = self._current_fields.get(fk)
        if arr is None:
            return
        v = arr[~np.isnan(arr)]
        if not len(v):
            return
        if fk in ("divergence", "curl"):
            m = max(abs(v.min()), abs(v.max()))
            lo, hi = -m, m
        elif fk in ("fold_change", "self_fold"):
            d = max(abs(v.min() - 1), abs(v.max() - 1))
            lo, hi = 1 - d, 1 + d
        else:
            lo, hi = float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 98))
        self.spin_vmin.blockSignals(True)
        self.spin_vmax.blockSignals(True)
        self.spin_vmin.setValue(lo)
        self.spin_vmax.setValue(hi)
        self.spin_vmin.blockSignals(False)
        self.spin_vmax.blockSignals(False)

    def _recompute_and_draw(self, *a):
        if self._applying:
            return
        self._compute_frame(self._t)
        self._draw_frame()

    def _on_compute_param_changed(self, *a):
        """A parameter that changes the computed field (grid / sigma / channel /
        source) — drop the per-frame cache, then recompute the current frame."""
        if self._applying:
            return
        self._field_cache.clear()
        self._recompute_and_draw()

    def _on_frame_changed(self, t):
        """Immediate (non-debounced) compute + draw — used programmatically (e.g.
        exporting every frame)."""
        self._t = int(t)
        self._frame_label.setText(f"{self._t + 1}/{max(1, self._n_frames)}")
        self._compute_frame(self._t)
        self._draw_frame()

    def _compute_frame(self, t):
        df = self._tracked_df
        if (df is None or self._field_shape == (0, 0) or len(df) == 0
                or "frame" not in getattr(df, "columns", [])):
            self._current_fields = {}
            self._cell_mask = None
            return
        ch = self.combo_int_ch.currentText()
        int_col = f"{ch}_mean" if ch else None
        use_mask = "mask" in self.combo_source.currentText().lower()

        # Base spatial fields depend only on (frame, grid, sigma, channel, source)
        # — not on the selected display field — so cache them per that signature.
        sig = (int(self.spin_grid.value()), float(self.spin_sigma.value()),
               int_col, use_mask, self.combo_img_source.currentText())
        key = (int(t), sig)
        cached = self._field_cache.get(key)
        if cached is not None:
            self._current_fields = dict(cached[0])
            self._cell_mask = cached[1]
        else:
            self._cell_mask = None
            if (use_mask and self._label_stack is not None
                    and t < self._label_stack.shape[0]):
                self._cell_mask = self._label_stack[t] > 0
            fields = compute_spatial_fields(
                self._tracked_df, frame=t, field_shape=self._field_shape,
                grid_step=self.spin_grid.value(), sigma=self.spin_sigma.value(),
                intensity_col=int_col)
            # Ensure a grid exists even when the frame had no objects (needed by
            # the pixel-field / column-interp paths and by pcolormesh).
            if "grid_y" not in fields:
                H, W = self._field_shape
                gs = self.spin_grid.value()
                gx = np.arange(0, W, gs).astype(float)
                gy = np.arange(0, H, gs).astype(float)
                grid_x, grid_y = np.meshgrid(gx, gy)
                fields["grid_x"] = grid_x
                fields["grid_y"] = grid_y
            img = self._get_img(ch, t) if ch else None
            if img is not None:
                fields.update(self._pixel_fields(img, self._cell_mask))
            self._current_fields = fields
            self._field_cache[key] = (dict(fields), self._cell_mask)

        # Selected-field-dependent layers (computed on top of the cached base).
        fk = self.combo_field.currentData()
        if (fk == "self_fold" and int_col and int_col in self._tracked_df.columns
                and "self_fold" not in self._current_fields):
            self._compute_self_fold_field(t, int_col)
        elif (isinstance(fk, str) and fk.startswith("col:")
                and fk not in self._current_fields):
            self._interp_column(t, fk[len("col:"):])

    def _pixel_fields(self, img, mask):
        from scipy.ndimage import gaussian_filter
        H, W = img.shape
        gs = self.spin_grid.value()
        sigma = self.spin_sigma.value()
        gy, gx = np.arange(0, H, gs), np.arange(0, W, gs)
        ny, nx = len(gy), len(gx)
        ig = np.zeros((ny, nx))
        cg = np.zeros((ny, nx))
        f = img.astype(np.float64)
        for i in range(ny):
            for j in range(nx):
                y0, y1 = gy[i], min(gy[i] + gs, H)
                x0, x1 = gx[j], min(gx[j] + gs, W)
                p = f[y0:y1, x0:x1]
                if mask is not None:
                    mm = mask[y0:y1, x0:x1]
                    if mm.any():
                        ig[i, j] = p[mm].mean()
                        cg[i, j] = mm.sum()
                    else:
                        ig[i, j] = np.nan
                else:
                    ig[i, j] = p.mean()
                    cg[i, j] = p.size
        v = ~np.isnan(ig)
        ig[~v] = 0
        ism = gaussian_filter(ig, sigma=sigma)
        csm = gaussian_filter(cg.astype(float), sigma=sigma)
        with np.errstate(divide='ignore', invalid='ignore'):
            ism = np.where(csm > 0, ism, np.nan)
        result = {"intensity": ism}
        gm = float(f[mask].mean()) if mask is not None and mask.any() else float(f.mean())
        if gm > 0:
            result["fold_change"] = ism / gm
        return result

    def _bin_value_field(self, t: int, col: str, fill: float = np.nan):
        """Bin a per-cell value column onto the grid as a Gaussian-weighted local
        mean, masked to the cell footprint — the same method (and masking) the
        built-in fields use (see
        :func:`nd2studios.backend.celltracker.fields._binned_mean_field`). Returns
        the grid array, or ``None`` when the frame has no usable samples. Shares the
        grid already built by :meth:`_compute_frame`."""
        grid_y = self._current_fields.get("grid_y")
        if grid_y is None:
            return None
        fdf = self._tracked_df[self._tracked_df["frame"] == t]
        if fdf.empty or col not in fdf.columns:
            return None
        vals = fdf[col].values.astype(float)
        if not np.isfinite(vals).any():
            return None
        cy = fdf["centroid_y"].values
        cx = fdf["centroid_x"].values
        gs = int(self.spin_grid.value())
        footprint = cell_footprint(cy, cx, grid_y.shape, gs)
        return _binned_mean_field(
            cy, cx, vals, grid_y.shape, gs,
            float(self.spin_sigma.value()), fill=fill, mask=footprint)

    def _compute_self_fold_field(self, t, int_col):
        self._tracked_df = compute_self_fold_change(self._tracked_df, int_col)
        arr = self._bin_value_field(t, "_self_fold")
        if arr is not None:
            self._current_fields["self_fold"] = arr

    def _interp_column(self, t, col):
        """Bin an arbitrary numeric measurement column onto the grid as a
        Gaussian-weighted local mean (covers the former 'Interpolated Spatial
        Maps' node)."""
        arr = self._bin_value_field(t, col)
        if arr is not None:
            self._current_fields[f"col:{col}"] = arr

    def _compute_global_scale(self):
        if self._tracked_df is None or not len(self._tracked_df):
            self._recompute_and_draw()
            return
        from PySide6.QtWidgets import QApplication
        self.btn_calc_scale.setEnabled(False)
        self.btn_calc_scale.setText("...")
        QApplication.processEvents()
        fk = self.combo_field.currentData()
        vals = []
        for t in range(self._n_frames):
            self._compute_frame(t)
            if fk in self._current_fields:
                a = self._current_fields[fk]
                v = a[~np.isnan(a)]
                if len(v):
                    vals.extend([v.min(), v.max()])
            if t % 10 == 0:
                QApplication.processEvents()
        if vals:
            lo, hi = min(vals), max(vals)
            if fk in ("divergence", "curl"):
                m = max(abs(lo), abs(hi))
                lo, hi = -m, m
            elif fk in ("fold_change", "self_fold"):
                d = max(abs(lo - 1), abs(hi - 1))
                lo, hi = 1 - d, 1 + d
            self.spin_vmin.blockSignals(True)
            self.spin_vmax.blockSignals(True)
            self.spin_vmin.setValue(lo)
            self.spin_vmax.setValue(hi)
            self.spin_vmin.blockSignals(False)
            self.spin_vmax.blockSignals(False)
            self.cb_auto_scale.blockSignals(True)
            self.cb_auto_scale.setChecked(False)
            self.cb_auto_scale.blockSignals(False)
        self.btn_calc_scale.setEnabled(True)
        self.btn_calc_scale.setText("Global Scale")
        self._compute_frame(self._t)
        self._draw_frame()

    # ════════════════════════════════════════════════════════════════════
    # Drawing  (ported from Cell-Tracker spatial_page)
    # ════════════════════════════════════════════════════════════════════
    def _draw_frame(self, *a):
        if self._applying:
            return
        if not self._current_fields:
            self._ax = None
            self.canvas.fig.clear()
            self.canvas.draw_idle()
            self.lbl_stats.setText(
                "No measured objects on this frame / multipoint.")
            return
        fk = self.combo_field.currentData()
        if fk not in self._current_fields:
            self._ax = None
            self.lbl_stats.setText(f"'{fk}' unavailable on this frame")
            self.canvas.fig.clear()
            self.canvas.draw_idle()
            return

        field = self._current_fields[fk]
        gy = self._current_fields.get("grid_y")
        gx = self._current_fields.get("grid_x")
        if gy is None:
            return

        t = self._t
        cm = self.combo_cmap.currentText()
        try:
            matplotlib.colormaps[cm]
        except (KeyError, ValueError):
            cm = "viridis"
        alpha = self.spin_opacity.value()

        if self.cb_auto_scale.isChecked():
            v = field[~np.isnan(field)]
            if len(v):
                if fk in ("divergence", "curl"):
                    m = max(abs(v.min()), abs(v.max()))
                    vmin, vmax = -m, m
                elif fk in ("fold_change", "self_fold"):
                    d = max(abs(v.min() - 1), abs(v.max() - 1))
                    vmin, vmax = 1 - d, 1 + d
                else:
                    vmin, vmax = float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 98))
            else:
                vmin, vmax = 0, 1
        else:
            vmin, vmax = self.spin_vmin.value(), self.spin_vmax.value()

        self.canvas.fig.clear()
        # Fixed axes layout: the map fills the left, the colour (intensity) bar is
        # pinned to a fixed strip on the figure's right edge so it stays attached
        # to the viewer's right border even when the map is zoomed in.
        ax = self.canvas.fig.add_axes([0.07, 0.07, 0.80, 0.86])
        ax.set_facecolor("black")
        self._ax = ax

        H, W = self._field_shape
        use_mask = self._cell_mask is not None

        if self.cb_overlay.isChecked() and self._channels:
            bgch = self.combo_bg_ch.currentText()
            src = self._channels
            if self.combo_img_source.currentText() == "Raw" and self._raw_channels:
                src = self._raw_channels
            if bgch in src and t < src[bgch].shape[0]:
                bg = frame_to_uint8(src[bgch][t])
                if use_mask:
                    bg = bg * self._cell_mask.astype(np.uint8)
                ax.imshow(bg, cmap="gray", aspect="equal", extent=[0, W, H, 0])

        if vmin >= vmax:
            vmax = vmin + 0.001

        if use_mask:
            from scipy.ndimage import zoom as ndi_zoom
            scale_y = H / field.shape[0]
            scale_x = W / field.shape[1]
            field_full = ndi_zoom(field, (scale_y, scale_x), order=1)
            field_full = field_full[:H, :W]
            field_masked = np.where(self._cell_mask, field_full, np.nan)
            im = ax.imshow(field_masked, cmap=cm, alpha=alpha, vmin=vmin, vmax=vmax,
                           aspect="equal", extent=[0, W, H, 0], interpolation="bilinear")
        else:
            im = ax.pcolormesh(gx, gy, field, cmap=cm, alpha=alpha,
                               vmin=vmin, vmax=vmax, shading="auto")

        cax = self.canvas.fig.add_axes([0.89, 0.07, 0.025, 0.86])
        self.canvas.fig.colorbar(im, cax=cax)
        cax.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
        cax.yaxis.label.set_color(Settings.FG_SECONDARY)

        if (self.cb_cell_borders.isChecked() and self._label_stack is not None
                and t < self._label_stack.shape[0]):
            from skimage.segmentation import find_boundaries
            by, bx = np.where(find_boundaries(self._label_stack[t], mode="outer"))
            if len(by) > 30000:
                s = len(by) // 30000
                by, bx = by[::s], bx[::s]
            ax.scatter(bx, by, s=0.1, c=self.combo_border_color.currentText(),
                       alpha=0.5, marker=".")

        if fk in ("speed", "velocity_y", "velocity_x", "divergence", "curl"):
            vy = self._current_fields.get("velocity_y")
            vx = self._current_fields.get("velocity_x")
            if vy is not None and vx is not None:
                s = max(1, min(vy.shape[0], vy.shape[1]) // 15)
                ax.quiver(gx[::s, ::s], gy[::s, ::s], vx[::s, ::s], vy[::s, ::s],
                          color="white", alpha=0.7, scale_units="xy",
                          angles="xy", width=0.003)

        label = next((l for k, l, _ in FIELD_OPTIONS if k == fk), None)
        if label is None:
            label = fk[len("col:"):] if isinstance(fk, str) and fk.startswith("col:") else str(fk)
        ax.set_title(f"{label} — Frame {t}", fontsize=10, color=Settings.FG_PRIMARY)
        ax.set_aspect("equal")
        ax.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
        # Apply the current zoom/pan view (full extent when not zoomed). Drawn
        # before the scale bar so the bar lands in the visible corner.
        self._apply_view(ax)

        if self.cb_scalebar.isChecked():
            bar_color = self.combo_bar_color.currentText()
            draw_scale_bar(
                ax,
                pixel_size_um=self.spin_px_um.value(),
                bar_length_um=self.spin_bar_um.value(),
                location=self.combo_bar_loc.currentText(),
                bar_color=bar_color, text_color=bar_color,
                font_size=self.spin_bar_font.value(),
                bar_thickness=self.spin_bar_thick.value(),
                bg_alpha=self.spin_bar_bg.value(),
                text_position=self.combo_bar_text_pos.currentText(),
                text_offset=self.spin_bar_text_off.value(),
            )

        self.canvas.draw_idle()
        self._update_zoom_label()

        v = field[~np.isnan(field)]
        if len(v):
            self.lbl_stats.setText(
                f"[{vmin:.3f}, {vmax:.3f}]  μ={v.mean():.3f}  σ={v.std():.3f}")

    # ════════════════════════════════════════════════════════════════════
    # Config + templates
    # ════════════════════════════════════════════════════════════════════
    def get_config(self) -> Dict[str, Any]:
        """Serialize every sidebar control to a JSON-ready template dict."""
        return {
            "version": 1,
            "field": self.combo_field.currentData(),
            "intensity_channel": self.combo_int_ch.currentText(),
            "source": self.combo_source.currentText(),
            "image_source": self.combo_img_source.currentText(),
            "grid_step": self.spin_grid.value(),
            "sigma": self.spin_sigma.value(),
            "auto_scale": self.cb_auto_scale.isChecked(),
            "vmin": self.spin_vmin.value(),
            "vmax": self.spin_vmax.value(),
            "overlay": self.cb_overlay.isChecked(),
            "borders": self.cb_cell_borders.isChecked(),
            "border_color": self.combo_border_color.currentText(),
            "bg_channel": self.combo_bg_ch.currentText(),
            "colormap": self.combo_cmap.currentText(),
            "opacity": self.spin_opacity.value(),
            "scalebar": self.cb_scalebar.isChecked(),
            "px_um": self.spin_px_um.value(),
            "bar_um": self.spin_bar_um.value(),
            "bar_thick": self.spin_bar_thick.value(),
            "bar_font": self.spin_bar_font.value(),
            "bar_loc": self.combo_bar_loc.currentText(),
            "bar_color": self.combo_bar_color.currentText(),
            "bar_bg": self.spin_bar_bg.value(),
            "bar_text_pos": self.combo_bar_text_pos.currentText(),
            "bar_text_off": self.spin_bar_text_off.value(),
        }

    def apply_config(self, cfg: Dict[str, Any]) -> None:
        """Restore sidebar controls from a template dict, then recompute + draw.

        Channel-valued fields are only applied when present in the current data;
        a single recompute runs at the end (signals are suppressed meanwhile)."""
        if not cfg:
            return
        self._applying = True
        try:
            def _combo_text(combo, key):
                if key in cfg and combo.findText(str(cfg[key])) >= 0:
                    combo.setCurrentText(str(cfg[key]))

            def _spin(spin, key):
                if key in cfg:
                    try:
                        spin.setValue(type(spin.value())(cfg[key]))
                    except (TypeError, ValueError):
                        pass

            def _check(cb, key):
                if key in cfg:
                    cb.setChecked(bool(cfg[key]))

            if "field" in cfg:
                idx = self.combo_field.findData(cfg["field"])
                if idx >= 0:
                    self.combo_field.setCurrentIndex(idx)
            _combo_text(self.combo_int_ch, "intensity_channel")
            _combo_text(self.combo_source, "source")
            _combo_text(self.combo_img_source, "image_source")
            _spin(self.spin_grid, "grid_step")
            _spin(self.spin_sigma, "sigma")
            _check(self.cb_auto_scale, "auto_scale")
            _spin(self.spin_vmin, "vmin")
            _spin(self.spin_vmax, "vmax")
            _check(self.cb_overlay, "overlay")
            _check(self.cb_cell_borders, "borders")
            _combo_text(self.combo_border_color, "border_color")
            _combo_text(self.combo_bg_ch, "bg_channel")
            _combo_text(self.combo_cmap, "colormap")
            _spin(self.spin_opacity, "opacity")
            _check(self.cb_scalebar, "scalebar")
            _spin(self.spin_px_um, "px_um")
            _spin(self.spin_bar_um, "bar_um")
            _spin(self.spin_bar_thick, "bar_thick")
            _spin(self.spin_bar_font, "bar_font")
            _combo_text(self.combo_bar_loc, "bar_loc")
            _combo_text(self.combo_bar_color, "bar_color")
            _spin(self.spin_bar_bg, "bar_bg")
            _combo_text(self.combo_bar_text_pos, "bar_text_pos")
            _spin(self.spin_bar_text_off, "bar_text_off")
        finally:
            self._applying = False
        self._recompute_and_draw()

    def set_node_templates(self, names: List[str]) -> None:
        """Pre-load the node's templates: apply the first, expose all as presets.

        Names missing from the local library are reported via ``status_message``
        (the user can import their JSON via "Load file…")."""
        self._node_template_names = list(names or [])
        self._refresh_template_combo()
        missing = [n for n in self._node_template_names
                   if n not in spatial_templates.list_templates()]
        applied = None
        for n in self._node_template_names:
            try:
                cfg = spatial_templates.load_template(n)
            except KeyError:
                continue
            self.apply_config(cfg)
            applied = n
            break
        if applied is not None:
            idx = self.combo_template.findText(applied)
            if idx >= 0:
                self.combo_template.blockSignals(True)
                self.combo_template.setCurrentIndex(idx)
                self.combo_template.blockSignals(False)
        if missing:
            self.status_message.emit(
                "Spatial map template(s) not in this machine's library: "
                + ", ".join(missing) + " — use 'Load file…' to import.")

    def _refresh_template_combo(self) -> None:
        self.combo_template.blockSignals(True)
        self.combo_template.clear()
        self.combo_template.addItem("— select template —", "")
        lib = spatial_templates.list_templates()
        # Node-attached names first (even if not yet in the local library).
        for n in self._node_template_names:
            self.combo_template.addItem(n, n)
        for n in lib:
            if n not in self._node_template_names:
                self.combo_template.addItem(n, n)
        self.combo_template.blockSignals(False)

    def _on_template_activated(self, idx: int) -> None:
        name = self.combo_template.itemData(idx)
        if not name:
            return
        try:
            cfg = spatial_templates.load_template(name)
        except KeyError:
            self.status_message.emit(
                f"Template '{name}' is not in the local library — import its JSON.")
            return
        self.apply_config(cfg)

    def _on_save_template(self) -> None:
        name, ok = QInputDialog.getText(self, "Save spatial-map template",
                                        "Template name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        existing = spatial_templates.list_templates()
        if name in existing:
            if QMessageBox.question(
                    self, "Overwrite template",
                    f"A template named '{name}' already exists. Overwrite it?"
            ) != QMessageBox.StandardButton.Yes:
                return
        try:
            spatial_templates.save_template(name, self.get_config())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save template", f"Could not save: {exc}")
            return
        self._refresh_template_combo()
        idx = self.combo_template.findText(name)
        if idx >= 0:
            self.combo_template.blockSignals(True)
            self.combo_template.setCurrentIndex(idx)
            self.combo_template.blockSignals(False)
        self.status_message.emit(f"Saved spatial-map template '{name}'.")

    def _on_load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import spatial-map template(s)", "",
            "Spatial-map templates (*.json);;All files (*)")
        if not path:
            return
        try:
            imported = spatial_templates.import_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load template", f"Could not import: {exc}")
            return
        self._refresh_template_combo()
        if imported:
            idx = self.combo_template.findText(imported[0])
            if idx >= 0:
                self.combo_template.setCurrentIndex(idx)
                self._on_template_activated(idx)
        self.status_message.emit(
            f"Imported {len(imported)} template(s): {', '.join(imported)}.")

    # ── image export (in-tab Save) ──────────────────────────────────────────
    def _save_current_frame(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save current map", f"spatial_map_T{self._t:04d}.png",
            "PNG image (*.png);;TIFF (raw field) (*.tif *.tiff)")
        if not path:
            return
        try:
            if path.lower().endswith((".tif", ".tiff")):
                fk = self.combo_field.currentData()
                arr = self._current_fields.get(fk)
                if arr is None:
                    QMessageBox.warning(self, "Save map", "No field to save.")
                    return
                import tifffile
                tifffile.imwrite(path, np.asarray(arr, dtype=np.float32))
            else:
                self.canvas.fig.savefig(path, dpi=150, facecolor="white")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save map", f"Could not save: {exc}")
            return
        self.status_message.emit(f"Saved map to {path}.")

    def _save_all_frames(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(self, "Save all frames to folder")
        if not out_dir:
            return
        import os
        fk = self.combo_field.currentData()
        label = str(fk).replace("col:", "")
        saved = 0
        keep = self._t
        try:
            for t in range(self._n_frames):
                self._on_frame_changed(t)
                self.canvas.fig.savefig(
                    os.path.join(out_dir, f"{label}_M{self._m + 1:02d}_T{t:04d}.png"),
                    dpi=150, facecolor="white")
                saved += 1
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save frames", f"Stopped after {saved}: {exc}")
        finally:
            self._frame_strip.set_current(keep, emit=False)
            self._on_frame_changed(keep)
        self.status_message.emit(f"Saved {saved} frame(s) to {out_dir}.")


class SpatialTemplatePicker(QDialog):
    """Edit which saved templates a Spatial Maps node carries.

    Left: the global template library. Right: the node's ordered template list.
    Supports add / remove / reorder and importing an external JSON file. The
    accepted value is the list of template names (``get_names``)."""

    def __init__(self, names: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial map templates")
        self.resize(460, 360)
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Templates load into the Spatial Maps tab when this node runs. "
            "Save templates from the tab's 'Save template…' button."))

        cols = QHBoxLayout()
        # Library
        lib_box = QVBoxLayout()
        lib_box.addWidget(QLabel("Library"))
        self.list_lib = QListWidget()
        self.list_lib.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        lib_box.addWidget(self.list_lib)
        btn_import = QPushButton("Import JSON…")
        btn_import.clicked.connect(self._import)
        lib_box.addWidget(btn_import)
        cols.addLayout(lib_box)

        # Add / remove
        mid = QVBoxLayout()
        mid.addStretch(1)
        btn_add = QPushButton("→ Add")
        btn_add.clicked.connect(self._add)
        btn_rm = QPushButton("Remove ←")
        btn_rm.clicked.connect(self._remove)
        btn_up = QPushButton("Up")
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_dn = QPushButton("Down")
        btn_dn.clicked.connect(lambda: self._move(1))
        for b in (btn_add, btn_rm, btn_up, btn_dn):
            mid.addWidget(b)
        mid.addStretch(1)
        cols.addLayout(mid)

        # Node's selected templates
        sel_box = QVBoxLayout()
        sel_box.addWidget(QLabel("This node"))
        self.list_sel = QListWidget()
        self.list_sel.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        sel_box.addWidget(self.list_sel)
        cols.addLayout(sel_box)
        root.addLayout(cols)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        root.addLayout(btns)

        for n in names or []:
            self.list_sel.addItem(QListWidgetItem(n))
        self._reload_library()

    def _reload_library(self) -> None:
        self.list_lib.clear()
        for n in spatial_templates.list_templates():
            self.list_lib.addItem(QListWidgetItem(n))

    def _selected_names(self) -> List[str]:
        return [self.list_sel.item(i).text() for i in range(self.list_sel.count())]

    def _add(self) -> None:
        have = set(self._selected_names())
        for it in self.list_lib.selectedItems():
            if it.text() not in have:
                self.list_sel.addItem(QListWidgetItem(it.text()))
                have.add(it.text())

    def _remove(self) -> None:
        for it in self.list_sel.selectedItems():
            self.list_sel.takeItem(self.list_sel.row(it))

    def _move(self, delta: int) -> None:
        row = self.list_sel.currentRow()
        if row < 0:
            return
        new = row + delta
        if not (0 <= new < self.list_sel.count()):
            return
        it = self.list_sel.takeItem(row)
        self.list_sel.insertItem(new, it)
        self.list_sel.setCurrentRow(new)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import spatial-map template(s)", "",
            "Spatial-map templates (*.json);;All files (*)")
        if not path:
            return
        try:
            spatial_templates.import_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import", f"Could not import: {exc}")
            return
        self._reload_library()

    def get_names(self) -> List[str]:
        return self._selected_names()
