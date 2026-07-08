"""In-viewer DVC (ALDVC) result panel — a playable field-series viewer.

``DVCPanel`` is the "DVC" tab in the Pipelines image-viewer stack (sibling of
:class:`~nd2studios.widgets.serialtrack_panel.SerialTrackPanel` and
:class:`~nd2studios.widgets.spatial_maps_panel.SpatialMapsPanel`). A DVC node
computes a dense displacement + strain field for **every frame** of the timelapse
(cumulative: frame vs the fixed reference; incremental: frame vs the previous),
and this panel plays through them like any other ND2Studios viewer:

* **Frame transport** — a ``FrameStrip`` + play/pause + fps scrub the field series.
* **Scalar field** — displacement magnitude / component, strain component,
  effective strain, divergence, curl, det(F), or von-Mises stress, as a
  heatmap / filled contour / line contour over the (per-frame) backdrop.
* **Quiver** — the gridded displacement vectors.
* **Histogram** — the grid-node displacement-magnitude distribution.

It has the **colour-scale controls of Cell-Tracker's Spatial Maps** (auto /
manual Min-Max / a "Global Scale" button that fits the range over the *whole
series* so the scale holds across frames) and the **image-viewer zoom controls**
(Home / + / − / Pan / zoom-%, scroll-to-zoom, drag-to-pan) driving the matplotlib
axis limits.

Each frame's field is turned into a SerialTrack
:class:`~nd2studios.backend.serialtrack_analysis.FieldBundle` so every derived
quantity comes from :func:`serialtrack_analysis.scalar_field` — no DVC-specific
field maths. A Z-slice slider scrubs 3D volumes within a frame.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QPushButton, QSizePolicy,
)

from nd2studios.core.settings import Settings
from nd2studios.core.dvc_registry import DVCResult
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.frame_strip import FrameStrip
from nd2studios.widgets.icon_button import scaled
from nd2studios.widgets.image_viewer import frame_to_uint8
from nd2studios.widgets.scale_bar import draw_scale_bar
from nd2studios.backend import serialtrack_analysis as sta
from nd2studios.backend.serialtrack.fields import DisplacementField, StrainField

CMAPS_SEQ = ["viridis", "plasma", "inferno", "magma", "hot", "turbo"]
CMAPS_DIV = ["coolwarm", "bwr", "seismic", "PiYG", "PRGn", "RdBu_r"]
ALL_CMAPS = CMAPS_SEQ + CMAPS_DIV

_VIEWS = [
    ("field", "Scalar field"),
    ("quiver", "Quiver"),
    ("heatmap_quiver", "Heatmap + Quiver"),
    ("histogram", "Displacement histogram"),
]

# (key, label, divergent) — z-components appended for 3D in _field_defs().
_FIELDS_2D = [
    ("disp_mag", "Displacement |u|", False),
    ("u_x", "Displacement uₓ", True),
    ("u_y", "Displacement u_y", True),
    ("e_xx", "Strain εₓₓ", True),
    ("e_yy", "Strain ε_yy", True),
    ("e_xy", "Strain εₓ_y", True),
    ("eff_strain", "Effective strain", False),
    ("div", "Divergence (dilatation)", True),
    ("curl", "Curl (vorticity)", True),
    ("detF", "Jacobian det(F)", False),
    ("von_mises", "Von Mises stress", False),
    ("s_xx", "Stress σₓₓ", True),
    ("s_yy", "Stress σ_yy", True),
    ("s_xy", "Stress σₓ_y", True),
]
_FIELDS_3D_EXTRA = [
    ("u_z", "Displacement u_z", True),
    ("e_zz", "Strain ε_zz", True),
    ("e_yz", "Strain ε_yz", True),
    ("e_xz", "Strain εₓ_z", True),
]
_STRESS_KEYS = {"von_mises", "s_xx", "s_yy", "s_xy", "s_yz", "s_xz", "s_zz"}


def field_bundle_from_result(result: DVCResult) -> sta.FieldBundle:
    """Build a SerialTrack :class:`FieldBundle` from a :class:`DVCResult`.

    Recomputes the strain tensor from the gridded displacement (via
    ``DisplacementField.gradient``) so every ``serialtrack_analysis.scalar_field``
    view works, letting the user switch fields without re-running DVC.
    """
    ndim = int(result.dim)
    coords = np.asarray(result.grid_coords)                 # (*grid, ndim)
    disp = np.asarray(result.displacement_field)            # (*grid, ndim)
    grids = tuple(coords[..., d] for d in range(ndim))
    components = np.moveaxis(disp, -1, 0)                    # (ndim, *grid)
    ps = (np.asarray(result.voxel_size_um, dtype=float)
          if result.voxel_size_um and len(result.voxel_size_um) == ndim
          else np.ones(ndim))
    dfield = DisplacementField(grids=grids, components=components,
                               pixel_steps=ps, time_step=1.0)
    F = dfield.gradient()                                    # (ndim,ndim,*grid)
    eps = 0.5 * (F + F.transpose(1, 0, *range(2, 2 + ndim)))
    sfield = StrainField(grids=grids, F_tensor=F, eps_tensor=eps, pixel_steps=ps)
    flat = coords.reshape(-1, ndim)
    return sta.FieldBundle(
        t=0, coords=flat, disp=components.reshape(ndim, -1).T,
        disp_field=dfield, strain_field=sfield, ndim=ndim, time_step=1.0)


class DVCPanel(QWidget):
    """Embeddable, playable DVC displacement/strain field-series viewer."""

    m_change_requested = Signal(int)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ── field series (set via set_data) ──
        self._series: Dict[int, DVCResult] = {}       # frame t → result
        self._backgrounds: Dict[int, np.ndarray] = {}  # frame t → (H,W) backdrop
        self._n_frames = 1
        self._t = 0                                    # current frame (0-based)
        self._pixel_size: Optional[float] = None
        self._field_shape: Tuple[int, int] = (0, 0)
        self._m = 0
        self._n_multipoints = 1
        self._increments: Dict[int, DVCResult] = {}    # raw per-step increments
        self._incr_mode = False                        # show increment vs cumulative
        self._fb_cache: Dict[tuple, sta.FieldBundle] = {}   # keyed (incr_mode, t)
        self._applying = False
        # ── zoom / pan state (drives matplotlib axis limits) ──
        self._ax = None
        self._view: Optional[Tuple[float, float, float, float]] = None
        self._pan_mode = False
        self._pan_px0 = (0.0, 0.0)
        self._pan_bb = (1.0, 1.0)
        self._pan_view0: Optional[Tuple[float, float, float, float]] = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(scaled(4), scaled(4), scaled(4), scaled(4))
        root.setSpacing(scaled(4))

        row = QHBoxLayout()
        row.setSpacing(scaled(6))
        self.cmb_view = self._combo([lbl for _k, lbl in _VIEWS], self._on_view_changed)
        self.cmb_field = self._combo([], self._on_field_changed)
        self.cmb_render = self._combo(["Heatmap", "Filled contour", "Line contour"],
                                      self._render)
        self.cmb_cmap = self._combo(ALL_CMAPS, self._render)
        self.cmb_disp = self._combo(["Cumulative", "Increment"], self._on_disp_changed)
        self.cmb_disp.setToolTip(
            "Incremental mode only: show the accumulated Cumulative field (ALDVC's "
            "reported output) or the raw per-step Increment.")
        self.spn_arrows = QSpinBox()
        self.spn_arrows.setRange(4, 64)
        self.spn_arrows.setValue(20)
        self.spn_arrows.valueChanged.connect(self._render)
        self.chk_bg = QCheckBox("Frame")
        self.chk_bg.setChecked(True)
        self.chk_bg.toggled.connect(self._render)
        self.chk_scalebar = QCheckBox("Scale bar")
        self.chk_scalebar.toggled.connect(self._render)
        self.spn_E = QDoubleSpinBox()
        self.spn_E.setRange(0.0, 1e9)
        self.spn_E.setValue(1000.0)
        self.spn_E.setPrefix("E=")
        self.spn_E.valueChanged.connect(self._render)
        self.spn_nu = QDoubleSpinBox()
        self.spn_nu.setRange(0.0, 0.49)
        self.spn_nu.setSingleStep(0.01)
        self.spn_nu.setValue(0.3)
        self.spn_nu.setPrefix("ν=")
        self.spn_nu.valueChanged.connect(self._render)
        for w, lbl in ((self.cmb_view, "View:"), (self.cmb_field, "Field:"),
                       (self.cmb_render, "Render:"), (self.cmb_cmap, "Cmap:"),
                       (self.spn_arrows, "Arrows:")):
            row.addWidget(QLabel(lbl))
            row.addWidget(w)
        row.addWidget(QLabel("Show:"))
        row.addWidget(self.cmb_disp)
        row.addWidget(self.chk_bg)
        row.addWidget(self.chk_scalebar)
        row.addWidget(self.spn_E)
        row.addWidget(self.spn_nu)
        row.addStretch(1)
        root.addLayout(row)

        # ── Colour-scale controls (same model as Cell-Tracker Spatial Maps) ──
        srow = QHBoxLayout()
        srow.setSpacing(scaled(6))
        self.chk_auto = QCheckBox("Auto")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "Auto-fit the colour range to the current frame (2–98th percentile). "
            "Uncheck for the manual Min/Max, or press Global Scale.")
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        self.spn_vmin = QDoubleSpinBox()
        self.spn_vmin.setRange(-1e9, 1e9)
        self.spn_vmin.setDecimals(4)
        self.spn_vmin.setPrefix("Min:")
        self.spn_vmin.valueChanged.connect(self._on_range_edited)
        self.spn_vmax = QDoubleSpinBox()
        self.spn_vmax.setRange(-1e9, 1e9)
        self.spn_vmax.setDecimals(4)
        self.spn_vmax.setValue(1.0)
        self.spn_vmax.setPrefix("Max:")
        self.spn_vmax.valueChanged.connect(self._on_range_edited)
        self.btn_global = QPushButton("Global Scale")
        self.btn_global.setObjectName("compactBtn")
        self.btn_global.setToolTip(
            "Fit the colour range to the selected quantity over the WHOLE series "
            "(every frame + all Z), so the scale is fixed while playing / scrubbing.")
        self.btn_global.clicked.connect(self._compute_global_scale)
        for w in (self.spn_vmin, self.spn_vmax):
            w.setFixedWidth(scaled(96))
        srow.addWidget(QLabel("Scale:"))
        srow.addWidget(self.chk_auto)
        srow.addWidget(self.spn_vmin)
        srow.addWidget(self.spn_vmax)
        srow.addWidget(self.btn_global)
        srow.addStretch(1)
        root.addLayout(srow)
        self._scale_widgets = [self.chk_auto, self.spn_vmin, self.spn_vmax,
                               self.btn_global]

        self.canvas = MplCanvas(self, width=7, height=5)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        root.addWidget(self.canvas, stretch=1)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        # ── Zoom toolbar (same button set as the image viewer's ZoomToolbar) +
        #    Z-slice scrubber (3D within a frame) + info line. ──
        zrow = QHBoxLayout()
        zrow.setSpacing(scaled(4))
        self.btn_home = self._zoom_btn("Home", "Reset view (fit field to window)",
                                       self._on_zoom_home, wide=True)
        self.btn_zoom_in = self._zoom_btn("+", "Zoom in",
                                          lambda: self._zoom_about(0.8))
        self.btn_zoom_out = self._zoom_btn("−", "Zoom out",
                                           lambda: self._zoom_about(1.25))
        self.btn_pan = self._zoom_btn("Pan", "Toggle pan — left-drag to move the "
                                      "zoomed field.", None, wide=True, checkable=True)
        self.btn_pan.toggled.connect(self._set_pan_mode)
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.lbl_zoom.setMinimumWidth(scaled(44))
        zrow.addWidget(self.btn_home)
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_pan)
        zrow.addWidget(self.lbl_zoom)
        zrow.addSpacing(scaled(10))
        self.lbl_z = QLabel("Z:")
        self.sld_z = QSlider(Qt.Orientation.Horizontal)
        self.sld_z.setRange(0, 0)
        self.sld_z.valueChanged.connect(self._on_z_changed)
        self.lbl_zval = QLabel("–")
        self.lbl_zval.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        zrow.addWidget(self.lbl_z)
        zrow.addWidget(self.sld_z, stretch=1)
        zrow.addWidget(self.lbl_zval)
        zrow.addWidget(self.lbl_info, stretch=2)
        root.addLayout(zrow)

        # ── Frame transport (play through the field series, like the viewer) ──
        trow = QHBoxLayout()
        trow.setSpacing(scaled(6))
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("compactBtn")
        self.btn_play.setFixedWidth(scaled(32))
        self.btn_play.setToolTip("Play / pause the field series through frames.")
        self.btn_play.clicked.connect(self._toggle_play)
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 60)
        self.spn_fps.setValue(6)
        self.spn_fps.setSuffix(" fps")
        self.lbl_frame = QLabel("T 0/0")
        self.lbl_frame.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.strip = FrameStrip()
        self.strip.current_changed.connect(self._on_frame_changed)
        trow.addWidget(self.btn_play)
        trow.addWidget(self.spn_fps)
        trow.addWidget(self.lbl_frame)
        trow.addWidget(self.strip, stretch=1)
        root.addLayout(trow)

        self._sync_control_visibility()

    def _combo(self, items: List[str], slot) -> QComboBox:
        cb = QComboBox()
        cb.addItems(items)
        cb.currentIndexChanged.connect(slot)
        return cb

    def _zoom_btn(self, text: str, tip: str, slot, *, wide: bool = False,
                  checkable: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("compactBtn")
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFixedSize(scaled(56) if wide else scaled(26), scaled(26))
        if slot is not None:
            b.clicked.connect(slot)
        return b

    # ── public API ──────────────────────────────────────────────────────
    def set_data(
        self, series: Dict[int, DVCResult],
        backgrounds: Optional[Dict[int, np.ndarray]] = None,
        frames: Optional[List[int]] = None,
        pixel_size: Optional[float] = None,
        field_shape: Optional[Tuple[int, int]] = None,
        *, m: int = 0, n_multipoints: int = 1, n_frames_total: Optional[int] = None,
        increments: Optional[Dict[int, DVCResult]] = None,
    ) -> None:
        """Feed one multipoint's DVC field **series** (``{frame: DVCResult}``) +
        per-frame backdrops. ``increments`` (incremental mode only) is the raw
        per-step field the "Show: Cumulative/Increment" toggle switches to.
        ``n_frames_total`` sizes the transport strip (frames without a result —
        e.g. the reference — are shown as empty)."""
        self._applying = True
        self._play_timer.stop()
        self.btn_play.setText("▶")
        self._series = {int(t): r for t, r in (series or {}).items()}
        self._increments = {int(t): r for t, r in (increments or {}).items()}
        self._incr_mode = False
        self.cmb_disp.blockSignals(True)
        self.cmb_disp.setCurrentIndex(0)
        self.cmb_disp.blockSignals(False)
        self._backgrounds = {int(t): np.asarray(b)
                             for t, b in (backgrounds or {}).items()}
        self._m = int(m)
        self._n_multipoints = int(max(1, n_multipoints))
        self._pixel_size = pixel_size
        self._fb_cache.clear()
        self._view = None
        frames_present = sorted(self._series)
        if n_frames_total is not None and n_frames_total > 0:
            self._n_frames = int(n_frames_total)
        else:
            self._n_frames = (max(frames_present) + 1) if frames_present else 1
        if field_shape is not None:
            self._field_shape = tuple(int(v) for v in field_shape)

        # Field combo for this series' dimensionality (from any result).
        self.cmb_field.blockSignals(True)
        self.cmb_field.clear()
        self.cmb_field.addItems([lbl for _k, lbl, _d in self._field_defs()])
        self.cmb_field.blockSignals(False)

        # Start on the first frame that actually has a field.
        self._t = frames_present[0] if frames_present else 0
        self.strip.set_count(self._n_frames)
        self.strip.set_current(self._t, emit=False)
        self._update_frame_label()
        self._sync_zslider()
        self._applying = False
        self._autofit_scale()
        self._sync_control_visibility()
        self._render()

    def set_compact(self, compact: bool) -> None:
        return

    def set_node_templates(self, names: List[str]) -> None:
        return

    # ── current frame accessors ─────────────────────────────────────────
    def _active_series(self) -> Dict[int, DVCResult]:
        """The series the toggle selects — raw increments in increment mode
        (incremental only), else the primary (cumulative) series."""
        if self._incr_mode and self._increments:
            return self._increments
        return self._series

    def _result(self) -> Optional[DVCResult]:
        return self._active_series().get(int(self._t))

    def _fb(self) -> Optional[sta.FieldBundle]:
        t = int(self._t)
        series = self._active_series()
        if t not in series:
            return None
        key = (bool(self._incr_mode), t)
        if key not in self._fb_cache:
            try:
                self._fb_cache[key] = field_bundle_from_result(series[t])
            except Exception as exc:  # noqa: BLE001
                self.status_message.emit(f"DVC field error: {exc}")
                return None
        return self._fb_cache.get(key)

    def _bg(self) -> Optional[np.ndarray]:
        return self._backgrounds.get(int(self._t))

    # ── frame transport ─────────────────────────────────────────────────
    def _update_frame_label(self) -> None:
        self.lbl_frame.setText(f"T {int(self._t) + 1}/{self._n_frames}")

    def _on_frame_changed(self, t: int) -> None:
        self._t = int(t)
        self._update_frame_label()
        self._sync_zslider()
        if self.chk_auto.isChecked():
            self._autofit_scale()
        self._sync_control_visibility()
        self._render()

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶")
        else:
            self._play_timer.start(int(1000 / max(1, self.spn_fps.value())))
            self.btn_play.setText("⏸")

    def _advance_frame(self) -> None:
        nxt = (int(self._t) + 1) % max(1, self._n_frames)
        self.strip.set_current(nxt, emit=True)

    def _sync_zslider(self) -> None:
        gz = self._grid_nz()
        self.sld_z.blockSignals(True)
        self.sld_z.setRange(0, max(0, gz - 1))
        if gz and self.sld_z.value() >= gz:
            self.sld_z.setValue(gz // 2)
        self.sld_z.blockSignals(False)

    # ── helpers ─────────────────────────────────────────────────────────
    def _ndim(self) -> int:
        r = self._result()
        if r is not None:
            return int(r.dim)
        for r in self._series.values():          # any result defines the dim
            return int(r.dim)
        return 2

    def _field_defs(self):
        defs = list(_FIELDS_2D)
        if self._ndim() == 3:
            defs = defs + _FIELDS_3D_EXTRA
        return defs

    def _field_key(self) -> str:
        defs = self._field_defs()
        i = self.cmb_field.currentIndex()
        return defs[i][0] if 0 <= i < len(defs) else "disp_mag"

    def _divergent(self) -> set:
        return {k for k, _l, d in self._field_defs() if d}

    def _view_key(self) -> str:
        return _VIEWS[self.cmb_view.currentIndex()][0]

    def _grid_nz(self) -> int:
        r = self._result()
        if r is None or r.dim != 3:
            return 0
        return int(r.grid_coords.shape[0])

    def _on_view_changed(self, *_):
        self._sync_control_visibility()
        self._render()

    def _on_field_changed(self, *_):
        if self._applying:
            return
        self._sync_control_visibility()
        self._autofit_scale()
        self._render()

    def _on_z_changed(self, *_):
        if not self._applying:
            self._render()

    def _on_disp_changed(self, *_):
        if self._applying:
            return
        self._incr_mode = (self.cmb_disp.currentIndex() == 1)
        self._sync_zslider()
        if self.chk_auto.isChecked():
            self._autofit_scale()
        self._sync_control_visibility()
        self._render()

    def _on_auto_toggled(self, *_):
        if self._applying:
            return
        if self.chk_auto.isChecked():
            self._autofit_scale()
        self._render()

    def _on_range_edited(self, *_):
        if self._applying:
            return
        if self.chk_auto.isChecked():
            self.chk_auto.blockSignals(True)
            self.chk_auto.setChecked(False)
            self.chk_auto.blockSignals(False)
        self._render()

    def _sync_control_visibility(self) -> None:
        view = self._view_key()
        is_field = view in ("field", "heatmap_quiver")
        is_quiver = view in ("quiver", "heatmap_quiver")
        self.cmb_field.setVisible(is_field)
        self.cmb_render.setVisible(view == "field")
        self.cmb_cmap.setVisible(view in ("field", "heatmap_quiver", "quiver"))
        self.spn_arrows.setVisible(is_quiver)
        show_bg = view in ("field", "quiver", "heatmap_quiver")
        self.chk_bg.setVisible(show_bg)
        self.chk_scalebar.setVisible(show_bg)
        is_stress = is_field and self._field_key() in _STRESS_KEYS
        self.spn_E.setVisible(is_stress)
        self.spn_nu.setVisible(is_stress)
        for w in self._scale_widgets:
            w.setVisible(is_field)
        self.cmb_disp.setVisible(bool(self._increments))   # incremental mode only
        show_z = self._grid_nz() > 1 and view != "histogram"
        self.lbl_z.setVisible(show_z)
        self.sld_z.setVisible(show_z)
        self.lbl_zval.setVisible(show_z)

    def _stress_kw(self) -> Dict[str, float]:
        return {"youngs_modulus": float(self.spn_E.value()),
                "poisson_ratio": float(self.spn_nu.value())}

    def _zslice(self, arr: np.ndarray) -> np.ndarray:
        if self._ndim() == 3 and arr.ndim >= 3:
            zi = int(min(max(0, self.sld_z.value()), arr.shape[0] - 1))
            return arr[zi]
        return arr

    def _xy_grids(self, fb: sta.FieldBundle):
        if self._ndim() == 3:
            gy, gx = fb.grids[1], fb.grids[2]
            return self._zslice(gy), self._zslice(gx)
        return fb.grids[0], fb.grids[1]

    # ── colour scale (Cell-Tracker model) ───────────────────────────────
    def _field_data(self, fb: Optional[sta.FieldBundle]) -> Optional[np.ndarray]:
        if fb is None:
            return None
        try:
            return np.asarray(
                sta.scalar_field(fb, self._field_key(), **self._stress_kw()),
                dtype=float)
        except Exception:  # noqa: BLE001
            return None

    def _set_range(self, lo: float, hi: float) -> None:
        for spn, v in ((self.spn_vmin, lo), (self.spn_vmax, hi)):
            spn.blockSignals(True)
            spn.setValue(float(v))
            spn.blockSignals(False)

    def _symmetric(self, lo: float, hi: float) -> Tuple[float, float]:
        if self._field_key() in self._divergent():
            m = max(abs(lo), abs(hi)) or 1.0
            return -m, m
        return lo, hi

    def _autofit_scale(self) -> None:
        if not self.chk_auto.isChecked():
            return
        data = self._field_data(self._fb())
        if data is None:
            return
        v = data[np.isfinite(data)]
        if v.size == 0:
            return
        lo, hi = self._symmetric(float(np.percentile(v, 2)),
                                 float(np.percentile(v, 98)))
        if hi <= lo:
            hi = lo + 1.0
        self._set_range(lo, hi)

    def _compute_global_scale(self) -> None:
        """Fit Min/Max over the selected quantity across the WHOLE (active) series
        (every frame, all Z) — Cell-Tracker's cross-frame Global Scale."""
        series = self._active_series()
        lo, hi = np.inf, -np.inf
        for t in sorted(series):
            try:
                fb = field_bundle_from_result(series[t])
                data = self._field_data(fb)
            except Exception:  # noqa: BLE001
                continue
            if data is None:
                continue
            v = data[np.isfinite(data)]
            if v.size:
                lo = min(lo, float(v.min()))
                hi = max(hi, float(v.max()))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        lo, hi = self._symmetric(lo, hi)
        if hi <= lo:
            hi = lo + 1.0
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(False)
        self.chk_auto.blockSignals(False)
        self._set_range(lo, hi)
        self._render()

    def _clim(self, data: np.ndarray) -> Tuple[float, float]:
        if self.chk_auto.isChecked():
            v = data[np.isfinite(data)]
            if v.size == 0:
                return 0.0, 1.0
            lo, hi = self._symmetric(float(np.percentile(v, 2)),
                                     float(np.percentile(v, 98)))
        else:
            lo, hi = float(self.spn_vmin.value()), float(self.spn_vmax.value())
        if hi <= lo:
            hi = lo + 1e-9
        return lo, hi

    # ── zoom / pan (drives matplotlib axis limits; mirrors SpatialMapsPanel) ──
    def _full_rect(self) -> Optional[Tuple[float, float, float, float]]:
        H, W = self._field_shape
        if H <= 0 or W <= 0:
            fb = self._fb()
            if fb is not None:
                gy, gx = fb.grids[-2], fb.grids[-1]
                return (float(gx.min()), float(gx.max()),
                        float(gy.min()), float(gy.max()))
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
        ax.set_ylim(y1, y0)
        self._update_zoom_label()

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
        rect = self._current_rect()
        if full is None or rect is None:
            return
        pct = int(round((full[1] - full[0]) / max(1e-6, rect[1] - rect[0]) * 100))
        self.lbl_zoom.setText(f"{pct}%")

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
        self._render()

    def _on_zoom_home(self) -> None:
        self._view = None
        self._render()

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
        if full is None:
            return
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
            self._render()

    # ── rendering ───────────────────────────────────────────────────────
    def _render(self, *_):
        if self._applying:
            return
        self.canvas.clear()
        ax = self.canvas.add_subplot(111)
        self._ax = ax
        if not self._series:
            ax.text(0.5, 0.5, "No DVC result.\nRun a DVC node first.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw()
            return
        self._update_info()
        fb = self._fb()
        if fb is None:                       # frame with no computed field
            self._draw_background(ax)
            ax.text(0.5, 0.5,
                    f"No DVC field for frame T{int(self._t) + 1}.\n"
                    "(reference frame, or not computed)",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            self._apply_view(ax)
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            return
        try:
            view = self._view_key()
            if view == "field":
                self._render_scalar(ax, fb, quiver=False)
            elif view == "heatmap_quiver":
                self._render_scalar(ax, fb, quiver=True)
            elif view == "quiver":
                self._render_quiver(ax, fb)
            elif view == "histogram":
                self._render_histogram(ax, fb)
        except Exception as exc:  # noqa: BLE001 — never crash the GUI
            ax.clear()
            ax.text(0.5, 0.5, f"Render error:\n{exc}", ha="center", va="center",
                    color=Settings.ACCENT_RED, transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _update_info(self) -> None:
        r = self._result()
        if r is None:
            self.lbl_zval.setText("–")
            self.lbl_info.setText("")
            return
        g = "×".join(str(s) for s in r.diagnostics.get("grid_shape", []))
        z = self.sld_z.value() + 1 if self._grid_nz() else 0
        self.lbl_zval.setText(f"{z}/{self._grid_nz()}" if self._grid_nz() else "–")
        self.lbl_info.setText(
            f"{r.method} · {r.dim}D · grid {g} · ADMM {r.iterations} "
            f"(conv={r.converged}) · β={r.beta:.3g} · "
            f"medZNCC={r.diagnostics.get('median_zncc', 0):.3f}")

    def _draw_background(self, ax) -> None:
        H, W = self._field_shape if self._field_shape != (0, 0) else (0, 0)
        bg = self._bg()
        if self.chk_bg.isChecked() and bg is not None:
            try:
                img = frame_to_uint8(np.asarray(bg))
                H, W = img.shape[-2], img.shape[-1]
                ax.imshow(img, cmap="gray", extent=[0, W, H, 0], zorder=0,
                          aspect="equal")
            except Exception:  # noqa: BLE001
                pass
        if H and W:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
        if self.chk_scalebar.isChecked() and self._pixel_size:
            draw_scale_bar(ax, pixel_size_um=self._pixel_size)

    def _render_scalar(self, ax, fb, quiver: bool) -> None:
        key = self._field_key()
        data = self._zslice(np.asarray(
            sta.scalar_field(fb, key, **self._stress_kw()), dtype=float))
        gy, gx = self._xy_grids(fb)
        cmap = self.cmb_cmap.currentText()
        divergent = key in self._divergent()
        if divergent and cmap in CMAPS_SEQ:
            cmap = "coolwarm"
        vmin, vmax = self._clim(data)
        self._draw_background(ax)
        render = self.cmb_render.currentText()
        if render == "Filled contour":
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(gx, gy, np.clip(data, vmin, vmax), levels=levels,
                             cmap=cmap, alpha=0.85, zorder=1, extend="both")
        elif render == "Line contour":
            levels = np.linspace(vmin, vmax, 12)
            im = ax.contour(gx, gy, data, levels=levels, cmap=cmap, zorder=1)
        else:
            extent = [float(gx.min()), float(gx.max()),
                      float(gy.max()), float(gy.min())]
            im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper",
                           vmin=vmin, vmax=vmax, alpha=0.85, zorder=1,
                           aspect="equal")
        cbar = self.canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
        label = dict((k, l) for k, l, _d in self._field_defs()).get(key, key)
        ax.set_title(f"{label} · T{int(self._t) + 1}", color=Settings.FG_PRIMARY,
                     fontsize=9)
        if quiver:
            self._quiver_on(ax, fb)
        self._apply_view(ax)

    def _render_quiver(self, ax, fb) -> None:
        self._draw_background(ax)
        self._quiver_on(ax, fb)
        ax.set_title(f"Displacement field · T{int(self._t) + 1}",
                     color=Settings.FG_PRIMARY, fontsize=9)
        self._apply_view(ax)

    def _quiver_on(self, ax, fb) -> None:
        comp = fb.disp_field.components   # (ndim, *grid)
        if self._ndim() == 3:
            uy, ux = self._zslice(comp[1]), self._zslice(comp[2])
        else:
            uy, ux = comp[0], comp[1]
        gy, gx = self._xy_grids(fb)
        target = int(self.spn_arrows.value())
        step = max(1, max(gy.shape) // target)
        sl = (slice(None, None, step), slice(None, None, step))
        X, Y, U, V = gx[sl], gy[sl], ux[sl], uy[sl]
        mag = np.sqrt(U ** 2 + V ** 2)
        cmap = self.cmb_cmap.currentText()
        if cmap in CMAPS_DIV:
            cmap = "viridis"
        ax.quiver(X, Y, U, V, mag, cmap=cmap, angles="xy",
                  scale_units="xy", scale=1.0, width=0.003, zorder=4)

    def _render_histogram(self, ax, fb) -> None:
        mag = np.asarray(sta.displacement_magnitude(fb), dtype=float).ravel()
        mag = mag[np.isfinite(mag)]
        if mag.size == 0:
            ax.text(0.5, 0.5, "No displacement data.", ha="center", va="center",
                    color=Settings.FG_SECONDARY, transform=ax.transAxes)
            ax.set_axis_off()
            return
        if self._pixel_size:
            mag = mag * float(self._pixel_size)
            unit = "µm"
        else:
            unit = "vox"
        ax.hist(mag, bins=40, color=Settings.ACCENT_CYAN, alpha=0.85)
        ax.axvline(float(np.mean(mag)), color=Settings.ACCENT_PINK, lw=1.2,
                   label=f"mean {np.mean(mag):.2f} {unit}")
        ax.axvline(float(np.median(mag)), color=Settings.ACCENT_YELLOW, lw=1.0,
                   ls="--", label=f"median {np.median(mag):.2f} {unit}")
        ax.set_xlabel(f"Displacement magnitude ({unit})", fontsize=8)
        ax.set_ylabel("Grid nodes", fontsize=8)
        ax.set_title(f"Displacement distribution · T{int(self._t) + 1} · "
                     f"n={mag.size}", color=Settings.FG_PRIMARY, fontsize=9)
        ax.legend(fontsize=7, facecolor=Settings.BG_TERTIARY,
                  edgecolor=Settings.BORDER_COLOR, labelcolor=Settings.FG_SECONDARY)
