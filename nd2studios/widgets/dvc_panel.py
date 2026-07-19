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
field maths. A **grid-Z** slider steps the correlation grid's Z-planes within a
frame — these are the subset-center planes the DVC field is solved on
(``Gz ≈ (Z − subset)/spacing + 1``), *not* the raw image Z-stack, so a deep
stack yields only a handful of grid planes (lower the subset spacing for finer
Z sampling).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QPushButton, QSizePolicy, QStackedWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.core.dvc_registry import DVCResult
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.frame_strip import FrameStrip
from nd2studios.widgets.icon_button import scaled, scale_qss
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
    ("object3d", "3D Object"),
    ("unwrap", "2D Unwrap"),
]

# Scalars the 3-D object / unwrap views can colour the surface by (label, key) —
# ``disp_mag`` / ``u_*`` / strains / ``qfactor`` match ``dvc_field`` outputs;
# ``u_perp`` / ``u_par`` are the V1.68 surface normal/tangential decomposition
# (Stout et al. 2016 Fig 4C/D). Unavailable ones fall back to displacement
# magnitude at build.
_OBJECT_SCALARS = [
    ("Displacement |u|", "disp_mag"),
    ("Normal u⊥", "u_perp"), ("Tangential u∥", "u_par"),
    ("uₓ", "u_x"), ("u_y", "u_y"), ("u_z", "u_z"),
    ("Strain εₓₓ", "e_xx"), ("Strain ε_yy", "e_yy"), ("Strain ε_zz", "e_zz"),
    ("Strain εₓ_y", "e_xy"), ("Strain εₓ_z", "e_xz"), ("Strain ε_yz", "e_yz"),
    ("Effective strain", "eff_strain"), ("Q-factor", "qfactor"),
]
# 3-D-only object scalars (dropped from the picker for a 2-D DIC result).
_OBJECT_SCALARS_3D_ONLY = {"u_z", "e_zz", "e_xz", "e_yz"}
# Signed object scalars → divergent colormap centred at 0 (Stout et al. Fig 4C).
_DIVERGENT_OBJECT_SCALARS = {"u_perp", "u_x", "u_y", "u_z",
                             "e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"}

# Context-channel overlay render modes (surrounding factors around the object).
_CONTEXT_MODES = [("Off", "none"), ("Cloud (volume)", "volume"),
                  ("Cloud (MIP)", "mip"), ("Shell (iso)", "iso")]
# 2-D unwrap cartographic projections (Stout et al. Fig 4E–G).
_UNWRAP_PROJECTIONS = [("Mollweide (equal-area)", "mollweide"),
                       ("Equirectangular", "equirectangular")]


class _SurfaceFieldWorker(QThread):
    """Off-thread build of the DVC-on-object **surface** (V1.68, Stout et al. 2016).

    ``dvc_object_surface`` runs marching cubes + Taubin smoothing + a
    ``RegularGridInterpolator`` sample of the displacement onto the surface
    vertices + the MDM tensor solve — too slow for the GUI thread (it would freeze
    view-switch / scrub / playback). This builds the :class:`SurfaceField`
    off-thread and hands it back via :attr:`done`; every VTK/render call stays on
    the GUI thread in the slot. ``gen`` lets the panel drop a result a newer
    dataset has superseded.
    """

    done = Signal(object, object, int)   # (cache_key, SurfaceField|None, generation)

    def __init__(self, result, mask, mask_vsz, scalar, key, gen,
                 smooth_iterations=10, with_interior=False, parent=None):
        super().__init__(parent)
        self._result = result
        self._mask = mask
        self._mvs = mask_vsz
        self._scalar = scalar
        self._key = key
        self._gen = int(gen)
        self._smooth = int(smooth_iterations)
        self._with_interior = bool(with_interior)

    def run(self) -> None:
        sf = None
        try:
            from nd2studios.backend.viz3d.overlays import dvc_object_surface
            keys = [self._scalar, "disp_mag", "u_perp", "u_par"]
            sf = dvc_object_surface(
                self._result, self._mask, self._mvs, scalar_keys=keys,
                smooth_iterations=self._smooth, with_interior=self._with_interior,
                with_metrics=True)
        except Exception:  # noqa: BLE001 — surfaced as a failed build in the slot
            sf = None
        self.done.emit(self._key, sf, self._gen)


class _MultiSurfaceFieldWorker(QThread):
    """Off-thread build of the **"All granules" composite** surface (V1.74).

    Builds one :class:`SurfaceField` per granule (``dvc_object_surface`` with
    ``with_metrics=False`` — a composite MDM is ill-defined and skipping it keeps
    the build cheap) from each granule's own field + cropped mask, then concatenates
    them into a single :class:`SurfaceField` via
    :func:`~nd2studios.backend.viz3d.overlays.merge_surface_fields`. Emits the merged
    field on the same :attr:`done` signal as :class:`_SurfaceFieldWorker`, so the
    panel's ``_on_surface_field_done`` slot handles both uniformly. ``items`` is a
    list of ``(DVCResult, mask, offset_um)`` triples (one per granule at this
    frame); ``offset_um`` is the granule's crop-origin translation (world
    ``(x, y, z)`` µm) so components land at their true relative positions.
    """

    done = Signal(object, object, int)   # (cache_key, SurfaceField|None, generation)

    def __init__(self, items, mask_vsz, scalar, key, gen,
                 smooth_iterations=10, parent=None):
        super().__init__(parent)
        self._items = list(items)
        self._mvs = mask_vsz
        self._scalar = scalar
        self._key = key
        self._gen = int(gen)
        self._smooth = int(smooth_iterations)

    def run(self) -> None:
        merged = None
        try:
            from nd2studios.backend.viz3d.overlays import (
                dvc_object_surface, merge_surface_fields,
            )
            keys = [self._scalar, "disp_mag", "u_perp", "u_par"]
            fields, offsets = [], []
            for result, mask, offset_um in self._items:
                sf = dvc_object_surface(
                    result, mask, self._mvs, scalar_keys=keys,
                    smooth_iterations=self._smooth, with_interior=False,
                    with_metrics=False)
                if sf is not None and not sf.is_empty:
                    fields.append(sf)
                    offsets.append(offset_um)
            merged = merge_surface_fields(fields, offsets) if fields else None
            if merged is not None and merged.is_empty:
                merged = None
        except Exception:  # noqa: BLE001 — surfaced as a failed build in the slot
            merged = None
        self.done.emit(self._key, merged, self._gen)

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
    # V1.79: the manual "Export" button was removed — DVC export is now baked into the
    # pipeline's Output node (wire DVC → Output; it auto-saves on Run).

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
        # V1.74 — per-granule (per-object) surface DVC. ``_objects`` is
        # ``{oid: {"series","bg","increment","mask","n_voxels"}}`` for the current M
        # (from a Granule Volume Mask → DVC edge on "Objects" scope); ``_active_oid``
        # is the selected granule; ``_show_all_objects`` selects the composite view.
        self._objects: Dict[int, Dict[str, Any]] = {}
        self._active_oid: Optional[int] = None
        self._show_all_objects = False
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
        # V1.74 — per-granule (per-object) selector: shown only when a Granule Volume
        # Mask → DVC edge ran on "Objects" scope (each granule has its own field).
        # "All granules" composites every granule surface into one 3-D scene.
        self.cmb_object = QComboBox()
        self.cmb_object.currentIndexChanged.connect(self._on_object_changed)
        self.cmb_object.setToolTip(
            "Which granule / object to show. Each granule got its own DVC solve "
            "(Objects scope). 'All granules' composites every granule surface into "
            "one 3-D scene (per-granule MDM / 2-D unwrap need a single granule).")
        self.cmb_field = self._combo([], self._on_field_changed)
        self.cmb_render = self._combo(["Heatmap", "Filled contour", "Line contour"],
                                      self._render)
        self.cmb_cmap = self._combo(ALL_CMAPS, self._render)
        # 3-D object view: which DVC scalar colours the object surface.
        self.cmb_obj = QComboBox()
        for label, key in _OBJECT_SCALARS:
            self.cmb_obj.addItem(label, key)
        self.cmb_obj.currentIndexChanged.connect(self._on_obj_scalar_changed)
        self.cmb_obj.setToolTip("Which DVC quantity colours the 3-D object surface "
                                "(u⊥ = outward/inward, u∥ = tangential).")
        # V1.68 — surrounding-channel context overlay (the "factors acting on the
        # object" rendered around it after DVC compiles) + its render mode/opacity.
        self.cmb_ctx = QComboBox()
        self.cmb_ctx.addItem("Off", -1)
        self.cmb_ctx.currentIndexChanged.connect(self._on_context_changed)
        self.cmb_ctx.setToolTip("Surrounding channel drawn as a translucent cloud/"
                                "shell around the object (a different channel).")
        self.cmb_ctx_mode = QComboBox()
        for label, key in _CONTEXT_MODES:
            self.cmb_ctx_mode.addItem(label, key)
        self.cmb_ctx_mode.currentIndexChanged.connect(self._on_context_changed)
        self.cmb_ctx_mode.setToolTip("How the surrounding channel is rendered.")
        self.spn_ctx_op = QDoubleSpinBox()
        self.spn_ctx_op.setRange(0.0, 1.0)
        self.spn_ctx_op.setSingleStep(0.05)
        self.spn_ctx_op.setValue(0.25)
        self.spn_ctx_op.setPrefix("α=")
        self.spn_ctx_op.valueChanged.connect(self._on_context_changed)
        self.spn_ctx_op.setToolTip("Surrounding-channel opacity.")
        # V1.68 — 2-D unwrap cartographic projection (Mollweide / equirectangular).
        self.cmb_proj = QComboBox()
        for label, key in _UNWRAP_PROJECTIONS:
            self.cmb_proj.addItem(label, key)
        self.cmb_proj.currentIndexChanged.connect(self._render)
        self.cmb_proj.setToolTip("2-D unwrap projection of the object surface.")
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
        for w, lbl in ((self.cmb_view, "View:"), (self.cmb_object, "Object:"),
                       (self.cmb_field, "Field:"),
                       (self.cmb_obj, "Colour:"),
                       (self.cmb_render, "Render:"), (self.cmb_cmap, "Cmap:"),
                       (self.spn_arrows, "Arrows:"),
                       (self.cmb_proj, "Projection:"),
                       (self.cmb_ctx, "Context:"), (self.cmb_ctx_mode, "Mode:"),
                       (self.spn_ctx_op, "")):
            self._ctx_labels = getattr(self, "_ctx_labels", {})
            lblw = QLabel(lbl)
            self._ctx_labels[w] = lblw
            row.addWidget(lblw)
            row.addWidget(w)
        row.addWidget(QLabel("Show:"))
        row.addWidget(self.cmb_disp)
        row.addWidget(self.chk_bg)
        row.addWidget(self.chk_scalebar)
        row.addWidget(self.spn_E)
        row.addWidget(self.spn_nu)
        row.addStretch(1)
        root.addLayout(row)

        # V1.68 — Mean Deformation Metrics readout (Stout et al. 2016): a compact
        # one-line summary of ⟨F⟩ / ⟨J⟩ / ⟨λᵢ⟩ / ⟨θ⟩ / ⟨Θ⟩ for the object surface.
        mrow = QHBoxLayout()
        mrow.setSpacing(scaled(6))
        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_metrics.setStyleSheet(scale_qss(
            f"QLabel {{ color: {Settings.FG_SECONDARY}; font-size: 8pt; }}"))
        self.lbl_metrics.setWordWrap(True)
        mrow.addWidget(self.lbl_metrics, stretch=1)
        root.addLayout(mrow)

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
        # The 2-D matplotlib canvas and a lazily-built PyVista 3-D viewer share one
        # stacked area; the "3D Object" view swaps to the 3-D viewer in place.
        self._canvas_stack = QStackedWidget()
        self._canvas_stack.addWidget(self.canvas)     # index 0 — 2-D field views
        self._view3d = None                            # lazily built (index 1)
        root.addWidget(self._canvas_stack, stretch=1)
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
        self.lbl_zoom.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        self.lbl_zoom.setMinimumWidth(scaled(44))
        zrow.addWidget(self.btn_home)
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_pan)
        zrow.addWidget(self.lbl_zoom)
        zrow.addSpacing(scaled(10))
        # NB: this slider steps the DVC *correlation grid's* Z-planes (subset
        # centers), NOT the raw image Z-stack. The field is only solved at grid
        # nodes, so Gz ≈ (Z − subset)/spacing + 1 — a 100-slice stack yields only
        # a handful of grid planes. Labelled/tooltipped so it isn't mistaken for
        # a raw-Z scrubber (see module docstring).
        _z_tip = (
            "DVC correlation-grid Z-plane (subset centers) — not the raw image "
            "Z-stack.\nThe field is solved only at grid nodes, so the count is "
            "Gz ≈ (Z − subset)/spacing + 1: a deep stack (e.g. 100 slices) gives "
            "only a handful of grid planes.\nLower the node's 'Subset spacing' "
            "for finer Z sampling."
        )
        self.lbl_z = QLabel("Grid Z:")
        self.lbl_z.setToolTip(_z_tip)
        self.sld_z = QSlider(Qt.Orientation.Horizontal)
        self.sld_z.setRange(0, 0)
        self.sld_z.setToolTip(_z_tip)
        self.sld_z.valueChanged.connect(self._on_z_changed)
        self.lbl_zval = QLabel("–")
        self.lbl_zval.setToolTip(_z_tip)
        self.lbl_zval.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
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
        self.lbl_frame.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
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
        masks: Optional[Dict[int, np.ndarray]] = None,
        mask_voxel_size: Optional[Tuple[float, float, float]] = None,
        context_provider: Optional[Any] = None,
        context_channels: Optional[List[str]] = None,
        frame_times_s: Optional[List[float]] = None,
        surface_smooth_iterations: int = 10,
        objects: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        """Feed one multipoint's DVC field **series** (``{frame: DVCResult}``) +
        per-frame backdrops. ``increments`` (incremental mode only) is the raw
        per-step field the "Show: Cumulative/Increment" toggle switches to.
        ``n_frames_total`` sizes the transport strip (frames without a result —
        e.g. the reference — are shown as empty). ``masks`` is the per-frame 3-D
        object mask (``{t: (Z,H,W) bool}``, from a 3D Mask Drawing node) the "3D
        Object" / "2D Unwrap" views render the field onto; ``mask_voxel_size`` is
        the raw ``(dz, dy, dx)`` µm spacing used to align the mask with the DVC
        grid.

        V1.68 (Stout et al. 2016): ``context_provider(m, t, channel)->(Z,H,W)``
        lazily fetches a **surrounding** channel for the 3-D context overlay;
        ``context_channels`` are its selectable channel names; ``frame_times_s``
        are per-frame timestamps driving the cumulative rotation ``⟨Θ⟩``;
        ``surface_smooth_iterations`` is the mask node's Taubin smoothing count.

        V1.74: ``objects`` (per-object DVC — a Granule Volume Mask → DVC edge on
        "Objects" scope) is ``{object_id: {"series": {t: DVCResult}, "bg": {t:
        (H,W)}, "increment": {t: DVCResult}, "mask": {t: (Z,H,W) bool},
        "n_voxels": int}}`` for this M. It drives the granule selector + the "All
        granules" composite; the positional ``series``/``masks`` above stay the
        **largest** object (the default single-surface view). ``None`` ⇒ no selector
        (the legacy whole-frame / single-mask path, unchanged)."""
        self._applying = True
        self._masks = {int(t): np.asarray(mv) for t, mv in (masks or {}).items()}
        self._mask_voxel_size = tuple(mask_voxel_size) if mask_voxel_size else None
        self._context_provider = context_provider
        self._frame_times = list(frame_times_s) if frame_times_s else None
        self._surface_smooth = int(surface_smooth_iterations)
        self._obj_sf_cache = {}
        self._obj_failed_keys = set()   # surface builds that failed (no retry loop)
        self._theta_series = {}        # {t: ⟨θ⟩ deg} accumulated for ⟨Θ⟩
        # V1.74 — per-granule (per-object) bundles for the selector / "All" composite.
        self._objects = {int(o): b for o, b in (objects or {}).items()}
        self._active_oid = None
        self._show_all_objects = False
        # Bump the surface generation so any in-flight surface worker for the
        # previous dataset/M is dropped (its result won't match) instead of
        # poisoning the fresh cache.
        self._obj_gen = getattr(self, "_obj_gen", 0) + 1
        self._obj_pending = None
        # Populate the surrounding-channel picker (preserving the prior choice).
        self.cmb_ctx.blockSignals(True)
        keep_ctx = self.cmb_ctx.currentText()
        self.cmb_ctx.clear()
        self.cmb_ctx.addItem("Off", -1)
        for ci, cname in enumerate(context_channels or []):
            self.cmb_ctx.addItem(str(cname), ci)
        ci_keep = self.cmb_ctx.findText(keep_ctx)
        if ci_keep >= 0:
            self.cmb_ctx.setCurrentIndex(ci_keep)
        self.cmb_ctx.blockSignals(False)
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

        # Object-scalar combo, dimension-gated like the field combo: a 2-D DIC
        # result has no z-components, so drop them (else the picker would offer
        # scalars the object view can't colour by).
        is3d = self._ndim() == 3
        self.cmb_obj.blockSignals(True)
        keep = self.cmb_obj.currentData()
        self.cmb_obj.clear()
        for label, key in _OBJECT_SCALARS:
            if not is3d and key in _OBJECT_SCALARS_3D_ONLY:
                continue
            self.cmb_obj.addItem(label, key)
        ri = self.cmb_obj.findData(keep)
        if ri >= 0:
            self.cmb_obj.setCurrentIndex(ri)
        self.cmb_obj.blockSignals(False)

        # V1.74 — per-granule selector. Entries: "All granules (N)" then one per
        # granule sorted by voxel count desc; default to the LARGEST (so the initial
        # view matches the pre-selector single-surface behavior). Signals blocked —
        # the positional series/masks above are already the largest object.
        self._populate_object_combo()

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

    # ── per-granule (per-object) selector (V1.74) ───────────────────────────────
    def _objects_sorted(self) -> List[int]:
        """Object ids present for this M, largest (most voxels) first."""
        return sorted(self._objects,
                      key=lambda o: -int(self._objects[o].get("n_voxels", 0)))

    def _populate_object_combo(self) -> None:
        """Fill the granule selector from ``self._objects`` and default to the
        largest granule (matching the pre-selector single-surface behavior). Does
        NOT emit ``_on_object_changed`` — the positional series/masks fed to
        :meth:`set_data` are already the largest object's."""
        self.cmb_object.blockSignals(True)
        self.cmb_object.clear()
        oids = self._objects_sorted()
        if oids:
            self.cmb_object.addItem(f"All granules ({len(oids)})", "__all__")
            for oid in oids:
                vox = int(self._objects[oid].get("n_voxels", 0))
                vlbl = f"{vox / 1000:.0f}k" if vox >= 1000 else str(vox)
                self.cmb_object.addItem(f"Granule {oid} · {vlbl} vox", int(oid))
            self.cmb_object.setCurrentIndex(1)   # index 0 = "All"; 1 = largest
            self._active_oid = int(oids[0])
        self._show_all_objects = False
        self.cmb_object.blockSignals(False)

    def _apply_active_object(self, oid: int) -> None:
        """Swap the panel's active field series / backdrops / masks to granule
        ``oid``'s bundle and reset the per-frame + surface caches."""
        b = self._objects.get(int(oid)) or {}
        self._series = {int(t): r for t, r in (b.get("series") or {}).items()}
        self._increments = {int(t): r for t, r in (b.get("increment") or {}).items()}
        self._backgrounds = {int(t): np.asarray(v)
                             for t, v in (b.get("bg") or {}).items()}
        self._masks = {int(t): np.asarray(v)
                       for t, v in (b.get("mask") or {}).items()}
        self._active_oid = int(oid)
        self._fb_cache.clear()
        self._reset_surface_caches()

    def _reset_surface_caches(self) -> None:
        """Drop cached surfaces / failures / in-flight builds (object switched)."""
        self._obj_sf_cache = {}
        self._obj_failed_keys = set()
        self._theta_series = {}
        self._obj_pending = None
        self._obj_gen = getattr(self, "_obj_gen", 0) + 1

    def _on_object_changed(self, *_):
        if self._applying or not self._objects:
            return
        data = self.cmb_object.currentData()
        self._incr_mode = False
        self.cmb_disp.blockSignals(True)
        self.cmb_disp.setCurrentIndex(0)
        self.cmb_disp.blockSignals(False)
        if data == "__all__":
            # Composite: 2-D views still show the LARGEST granule (composited 2-D
            # fields are ill-defined); the 3-D Object view merges every granule.
            oids = self._objects_sorted()
            if oids:
                self._apply_active_object(int(oids[0]))
            self._show_all_objects = True
        else:
            self._show_all_objects = False
            try:
                self._apply_active_object(int(data))
            except (TypeError, ValueError):
                return
        self._view = None
        self._sync_zslider()
        if self.chk_auto.isChecked():
            self._autofit_scale()
        self._sync_control_visibility()
        self._render()

    # ── 3-D object surface + 2-D unwrap (DVC field on a drawn mask; V1.68) ──────
    def _obj_scalar_key(self) -> str:
        d = self.cmb_obj.currentData()
        return str(d) if d else "disp_mag"

    def _on_obj_scalar_changed(self, *_):
        if self._applying:
            return
        if self._view_key() in ("object3d", "unwrap"):
            self._render()

    def _on_context_changed(self, *_):
        if self._applying:
            return
        if self._view_key() == "object3d":
            # Context is a viewer-only overlay — re-apply without rebuilding.
            self._apply_context_channel()

    def _ensure_view3d(self):
        """Lazily build the embedded PyVista viewer (index 1 of the canvas stack)."""
        if getattr(self, "_view3d", None) is None:
            from nd2studios.widgets.viewer3d import PyVista3DViewer
            self._view3d = PyVista3DViewer(self)
            self._canvas_stack.addWidget(self._view3d)
        return self._view3d

    def _surface_key(self):
        """Cache key for the built surface:
        ``(object, incr_mode, t, scalar, smooth_iters)``. ``object`` is the active
        granule id, ``"__all__"`` for the composite, or ``None`` (no per-object
        data) — so single-object and all-granule builds key the same cache without
        colliding."""
        obj = ("__all__" if getattr(self, "_show_all_objects", False)
               else getattr(self, "_active_oid", None))
        return (obj, bool(self._incr_mode), int(self._t), self._obj_scalar_key(),
                int(getattr(self, "_surface_smooth", 10)))

    def _need_surface_field(self):
        """Cached :class:`SurfaceField` for the current frame, or launch a build
        off-thread (coalesced) and return None (the done slot re-renders). In "All
        granules" mode this is the MERGED composite across every granule."""
        if getattr(self, "_show_all_objects", False) and self._objects:
            return self._need_all_surface_field()
        r = self._result()
        mask = (getattr(self, "_masks", {}) or {}).get(int(self._t))
        if r is None or mask is None:
            return None
        key = self._surface_key()
        sf = getattr(self, "_obj_sf_cache", {}).get(key)
        if sf is not None:
            return sf
        # A build that already failed for this exact key is NOT retried — otherwise
        # a deterministic failure (missing dep / degenerate mask) would spin an
        # unbounded rebuild loop from the done-slot's re-render.
        if key in getattr(self, "_obj_failed_keys", set()):
            return None
        self._obj_pending = {"kind": "single", "result": r, "mask": mask,
                             "scalar": self._obj_scalar_key(), "key": key,
                             "gen": self._obj_gen}
        w = getattr(self, "_obj_worker", None)
        if w is not None and w.isRunning():
            self.status_message.emit("3D Object: building surface… (queued)")
            return None
        self._launch_obj_build()
        return None

    def _need_all_surface_field(self):
        """Cached MERGED surface across every granule for the current frame (the
        "All granules" composite), or launch an off-thread multi-build (coalesced)
        and return None. Each granule's own field + cropped mask contribute one
        component (:func:`backend.viz3d.overlays.merge_surface_fields`)."""
        key = self._surface_key()
        sf = getattr(self, "_obj_sf_cache", {}).get(key)
        if sf is not None:
            return sf
        if key in getattr(self, "_obj_failed_keys", set()):
            return None
        t = int(self._t)
        incr = bool(self._incr_mode)
        mvs = getattr(self, "_mask_voxel_size", None) or (1.0, 1.0, 1.0)
        dz, dy, dx = float(mvs[0]), float(mvs[1]), float(mvs[2])
        items = []
        for oid in self._objects_sorted():
            b = self._objects.get(oid) or {}
            series = (b.get("increment") if incr else b.get("series")) or {}
            r = series.get(t)
            mask = (b.get("mask") or {}).get(t)
            if r is not None and mask is not None:
                # Crop origin (z0, y0, x0) voxels → world (x, y, z) µm offset, so each
                # granule's crop-local surface lands at its true relative position.
                z0, y0, x0 = (int(v) for v in b.get("origin", (0, 0, 0)))
                offset_um = (x0 * dx, y0 * dy, z0 * dz)
                items.append((r, np.asarray(mask), offset_um))
        if not items:
            return None
        self._obj_pending = {"kind": "multi", "items": items,
                             "scalar": self._obj_scalar_key(), "key": key,
                             "gen": self._obj_gen}
        w = getattr(self, "_obj_worker", None)
        if w is not None and w.isRunning():
            self.status_message.emit(
                "3D Object: building all-granule surface… (queued)")
            return None
        self._launch_obj_build()
        return None

    def _render_object3d(self) -> None:
        """Render the DVC field on the drawn object as a smoothed **surface mesh**
        (V1.68, Stout et al. 2016): the boundary ``∂V`` coloured by the selected
        scalar (u⊥ divergent, magnitudes sequential), the surrounding channel as an
        optional context cloud, and the MDM suite in the readout. The viewer's
        Volume / MIP / Slices / Iso buttons style the optional interior."""
        view = self._ensure_view3d()
        self._canvas_stack.setCurrentWidget(view)
        self._update_info()
        # V1.74 — "All granules" composites every granule surface at this frame; it
        # does NOT gate on any single object's field/mask (a granule may lack a field
        # on the reference frame while others have one).
        if getattr(self, "_show_all_objects", False) and self._objects:
            sf = self._need_surface_field()      # merged composite
            if sf is not None:
                self._show_object_sf(self._surface_key(), sf)
                self._update_metrics_label_all(sf)
            elif self._surface_key() in getattr(self, "_obj_failed_keys", set()):
                view.set_overlay(None)
                view.set_context_channel(None)
                self._update_metrics_label(None)
                self.status_message.emit(
                    "3D Object: all-granule surface build failed.")
            else:
                self.status_message.emit(
                    "3D Object: building all-granule surface…")
            return
        r = self._result()
        if r is None:
            view.set_overlay(None)
            view.set_context_channel(None)
            self._update_metrics_label(None)
            self.status_message.emit(
                f"3D Object: no DVC field on frame T{int(self._t) + 1}.")
            return
        if (getattr(self, "_masks", {}) or {}).get(int(self._t)) is None:
            view.set_overlay(None)
            view.set_context_channel(None)
            self._update_metrics_label(None)
            self.status_message.emit(
                "3D Object: no mask on this frame — add a '3D Mask Drawing' node "
                "upstream of DVC and Run, or scrub to a masked frame.")
            return
        sf = self._need_surface_field()
        if sf is not None:
            self._show_object_sf(self._surface_key(), sf)
        elif self._surface_key() in getattr(self, "_obj_failed_keys", set()):
            view.set_overlay(None)
            self._update_metrics_label(None)
            self.status_message.emit("3D Object: surface build failed.")

    def _render_unwrap(self, ax) -> None:
        """Draw the 2-D cartographic unwrap (Mollweide / equirectangular) of the
        object surface on the matplotlib canvas (Stout et al. Fig 4E–G)."""
        if getattr(self, "_show_all_objects", False) and self._objects:
            # A single-pole unwrap of many disjoint granules is ill-defined.
            ax.text(0.5, 0.5, "2-D unwrap needs a single granule.\nPick one in the "
                    "Object selector.", ha="center", va="center",
                    color=Settings.FG_SECONDARY, transform=ax.transAxes)
            ax.set_axis_off()
            return
        r = self._result()
        if r is None:
            ax.text(0.5, 0.5, f"No DVC field on frame T{int(self._t) + 1}.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        if (getattr(self, "_masks", {}) or {}).get(int(self._t)) is None:
            ax.text(0.5, 0.5, "No object mask on this frame.\nAdd a '3D Mask "
                    "Drawing' node upstream of DVC and Run.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        sf = self._need_surface_field()
        if sf is None:
            failed = self._surface_key() in getattr(self, "_obj_failed_keys", set())
            msg = ("2D Unwrap: surface build failed." if failed
                   else "Building object surface…")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color=Settings.FG_SECONDARY, transform=ax.transAxes)
            ax.set_axis_off()
            return
        self._draw_unwrap(ax, sf)

    def _draw_unwrap(self, ax, sf) -> None:
        import numpy as _np
        from nd2studios.backend.viz3d.overlays import unwrap_surface
        proj = self.cmb_proj.currentData() or "mollweide"
        requested = self._obj_scalar_key()
        eff = (requested if (sf.scalars and requested in sf.scalars)
               else sf.default_scalar)
        um = unwrap_surface(sf, eff, projection=proj, width=360, height=180)
        if um.is_empty:
            ax.text(0.5, 0.5, "Unwrap unavailable for this surface.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        divergent = eff in _DIVERGENT_OBJECT_SCALARS
        cmap = self.cmb_cmap.currentText() or ("coolwarm" if divergent else "viridis")
        vals = um.values
        finite = vals[_np.isfinite(vals)]
        if divergent and finite.size:
            a = float(_np.nanpercentile(_np.abs(finite), 98)) or 1e-6
            vmin, vmax = -a, a
        elif finite.size:
            vmin = float(_np.nanpercentile(finite, 2))
            vmax = float(_np.nanpercentile(finite, 98))
            if vmax <= vmin:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       aspect="equal", interpolation="nearest")
        try:
            self.canvas.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        except Exception:  # noqa: BLE001
            pass
        # Tangential u∥ streamlines (Fig 4F/G): vec is (east=+x, north=+y up), and
        # the image rows increase downward → flip the north component.
        try:
            spd = _np.hypot(um.vec_u, um.vec_v)
            if _np.any(spd > 0):
                H, W = vals.shape
                ax.streamplot(_np.arange(W), _np.arange(H), um.vec_u, -um.vec_v,
                              density=1.0, color="white", linewidth=0.5,
                              arrowsize=0.6)
        except Exception:  # noqa: BLE001
            pass
        ax.set_title(f"2D unwrap ({proj}) · {eff}", color=Settings.FG_PRIMARY,
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    def _launch_obj_build(self) -> None:
        pend = getattr(self, "_obj_pending", None)
        if pend is None:
            return
        self._obj_pending = None
        smooth = int(getattr(self, "_surface_smooth", 10))
        if pend.get("kind") == "multi":
            self.status_message.emit(
                "3D Object: building all-granule surfaces (marching cubes)…")
            w = _MultiSurfaceFieldWorker(
                pend["items"], self._mask_voxel_size, pend["scalar"],
                pend["key"], pend["gen"], smooth_iterations=smooth, parent=self)
        else:
            self.status_message.emit(
                "3D Object: building the deformation surface (marching cubes + MDM)…")
            w = _SurfaceFieldWorker(
                pend["result"], pend["mask"], self._mask_voxel_size,
                pend["scalar"], pend["key"], pend["gen"],
                smooth_iterations=smooth, with_interior=False, parent=self)
        w.done.connect(self._on_surface_field_done)
        self._obj_worker = w
        w.start()

    def _on_surface_field_done(self, key, sf, gen: int) -> None:
        if int(gen) != int(getattr(self, "_obj_gen", 0)):
            # A newer dataset/M superseded this build — drop its result, but still
            # launch any current-gen build the coalescer queued behind this worker
            # (else it is orphaned: no worker running, pending never launched).
            if getattr(self, "_obj_pending", None) is not None:
                self._launch_obj_build()
            return
        if sf is not None:
            self._obj_sf_cache[key] = sf
            mdm = getattr(sf, "mdm", None)
            if mdm is not None:
                self._theta_series[int(key[2])] = float(mdm.theta_deg)  # key[2]=t
        else:
            # Record the failure so neither view retries this key (no rebuild loop).
            self._obj_failed_keys = getattr(self, "_obj_failed_keys", set())
            self._obj_failed_keys.add(key)
        cur = self._surface_key()
        if key == cur:
            view = self._view_key()
            if view == "object3d":
                if sf is not None:
                    self._show_object_sf(key, sf)
                    if getattr(self, "_show_all_objects", False):
                        self._update_metrics_label_all(sf)
                else:
                    self._update_metrics_label(None)
                    self.status_message.emit("3D Object: surface build failed.")
            elif view == "unwrap":
                # Safe to re-render: a failed key is now in _obj_failed_keys, so
                # _render_unwrap shows the failure instead of relaunching.
                self._render()
        # Run the latest coalesced request, if the user moved on while we built.
        if getattr(self, "_obj_pending", None) is not None:
            self._launch_obj_build()

    def _show_object_sf(self, key, sf) -> None:
        """Feed a built :class:`SurfaceField` to the embedded viewer + context +
        metrics, colouring by the scalar actually present (a missing scalar falls
        back to displacement magnitude, matching the status label)."""
        view = self._ensure_view3d()
        requested = key[3]                       # key = (obj, incr, t, scalar, smooth)
        eff = (requested if (sf.scalars and requested in sf.scalars)
               else sf.default_scalar)
        view.set_overlay(sf, scalar=eff)
        self._apply_context_channel()
        self._update_metrics_label(sf)
        note = "" if eff == requested else f" ({requested} unavailable → {eff})"
        self.status_message.emit(
            f"3D Object surface · T{int(self._t) + 1} · {eff}{note} · "
            f"{sf.n_vertices} verts — drag to rotate.")

    def _apply_context_channel(self) -> None:
        """Push the surrounding-channel context volume (if any) to the viewer."""
        view = getattr(self, "_view3d", None)
        if view is None:
            return
        prov = getattr(self, "_context_provider", None)
        ci = self.cmb_ctx.currentData()
        mode = self.cmb_ctx_mode.currentData() or "none"
        if prov is None or ci is None or int(ci) < 0 or mode == "none":
            view.set_context_channel(None)
            return
        try:
            vol = prov(int(self._m), int(self._t), int(ci))
        except Exception:  # noqa: BLE001
            vol = None
        if vol is None:
            view.set_context_channel(None)
            return
        view.set_context_channel(
            vol, voxel_size_um=getattr(self, "_mask_voxel_size", None),
            mode=mode, opacity=float(self.spn_ctx_op.value()))

    def set_context_default(self, channel_name: Optional[str],
                            mode_value: Optional[str]) -> None:
        """Default the surrounding-channel overlay to a specific channel + render mode
        (V1.77 — a Prism's converged view-only channel, e.g. green-in-shell).

        Applied only when the user has **not** already picked a context channel (the
        selector is on "Off"), so it auto-shows the overlay on first populate but never
        overrides a manual choice. No-op if the channel isn't in the current list."""
        if not channel_name:
            return
        cur = self.cmb_ctx.currentData()   # None (empty) / -1 (Off) / 0..N (a channel)
        if cur is not None and int(cur) >= 0:  # user already chose a channel (incl. 0)
            return
        idx = self.cmb_ctx.findText(str(channel_name))
        if idx < 0:
            return
        self.cmb_ctx.blockSignals(True)
        self.cmb_ctx.setCurrentIndex(idx)
        self.cmb_ctx.blockSignals(False)
        if mode_value:
            mi = self.cmb_ctx_mode.findData(mode_value)
            if mi >= 0:
                self.cmb_ctx_mode.blockSignals(True)
                self.cmb_ctx_mode.setCurrentIndex(mi)
                self.cmb_ctx_mode.blockSignals(False)
        self._apply_context_channel()

    def _update_metrics_label(self, sf) -> None:
        """Update the MDM readout for the current object surface (Stout et al.)."""
        if sf is None or getattr(sf, "mdm", None) is None:
            self.lbl_metrics.setText("")
            return
        from nd2studios.backend.viz3d.mdm import cumulative_rotation
        m = sf.mdm
        lam = m.stretches
        ts = sorted(getattr(self, "_theta_series", {}))
        thetas = [self._theta_series[t] for t in ts]
        times = None
        ft = getattr(self, "_frame_times", None)
        if ft is not None:
            try:
                times = [float(ft[t]) for t in ts]
            except Exception:  # noqa: BLE001
                times = None
        theta_cum = cumulative_rotation(thetas, times)
        self.lbl_metrics.setText(
            f"MDM (Stout et al. 2016) · ⟨J⟩ = {m.J:.3f} (vol. ratio) · "
            f"⟨λ₁,λ₂,λ₃⟩ = [{lam[0]:.3f}, {lam[1]:.3f}, {lam[2]:.3f}] "
            f"(principal stretches) · ⟨θ⟩ = {m.theta_deg:.2f}° (mean rotation) · "
            f"⟨Θ⟩ = {theta_cum:.2f}° (cumulative over {len(ts)} frame(s))")

    def _update_metrics_label_all(self, sf) -> None:
        """MDM readout for the "All granules" composite (V1.74). Per-object MDM is
        not aggregated (a composite ⟨F⟩ is ill-defined) — show the granule count and
        prompt to pick one granule for its metrics."""
        n = len(self._objects)
        verts = int(getattr(sf, "n_vertices", 0)) if sf is not None else 0
        self.lbl_metrics.setText(
            f"All granules · {n} granule(s) · {verts} surface verts · "
            "select a single granule for its Mean Deformation Metrics (Stout 2016).")

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
        is_obj = view == "object3d"
        is_unwrap = view == "unwrap"
        is_surface = is_obj or is_unwrap        # both consume the object surface
        has_objects = bool(getattr(self, "_objects", {}))
        self.cmb_object.setVisible(has_objects)  # V1.74 per-granule selector
        self.cmb_obj.setVisible(is_surface)     # colour scalar for object / unwrap
        self.cmb_field.setVisible(is_field)
        self.cmb_render.setVisible(view == "field")
        self.cmb_cmap.setVisible(view in ("field", "heatmap_quiver", "quiver",
                                          "unwrap"))
        self.spn_arrows.setVisible(is_quiver)
        # V1.68 surface controls: context cloud (3-D only) + unwrap projection.
        self.cmb_proj.setVisible(is_unwrap)
        for w in (self.cmb_ctx, self.cmb_ctx_mode, self.spn_ctx_op):
            w.setVisible(is_obj)
        self.lbl_metrics.setVisible(is_surface)
        # Each control's label mirrors its widget's *logical* visibility (NOT
        # w.isVisible(), which is False whenever the panel/ancestor is hidden and
        # would then leave every label permanently hidden on re-show).
        _label_flags = {
            self.cmb_view: True,
            self.cmb_object: has_objects,
            self.cmb_field: is_field,
            self.cmb_obj: is_surface,
            self.cmb_render: view == "field",
            self.cmb_cmap: view in ("field", "heatmap_quiver", "quiver", "unwrap"),
            self.spn_arrows: is_quiver,
            self.cmb_proj: is_unwrap,
            self.cmb_ctx: is_obj,
            self.cmb_ctx_mode: is_obj,
            self.spn_ctx_op: is_obj,
        }
        for w, lblw in getattr(self, "_ctx_labels", {}).items():
            lblw.setVisible(bool(_label_flags.get(w, True)))
        show_bg = view in ("field", "quiver", "heatmap_quiver")
        self.chk_bg.setVisible(show_bg)
        self.chk_scalebar.setVisible(show_bg)
        is_stress = is_field and self._field_key() in _STRESS_KEYS
        self.spn_E.setVisible(is_stress)
        self.spn_nu.setVisible(is_stress)
        for w in self._scale_widgets:
            w.setVisible(is_field)
        self.cmb_disp.setVisible(bool(self._increments))   # incremental mode only
        show_z = self._grid_nz() > 1 and view not in ("histogram", "object3d",
                                                      "unwrap")
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
        # In the "3D Object" view the Home button resets the embedded PyVista camera
        # (fit + isometric) — the 2-D matplotlib axis-limit reset below does not apply
        # to the 3-D scene. Every other view resets the 2-D zoom rectangle.
        if (self._view_key() == "object3d"
                and getattr(self, "_view3d", None) is not None):
            try:
                self._view3d.reset_camera()
            except Exception:  # noqa: BLE001
                pass
            return
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
        if self._view_key() == "object3d":
            self._render_object3d()
            return
        self._canvas_stack.setCurrentWidget(self.canvas)
        self.canvas.clear()
        ax = self.canvas.add_subplot(111)
        self._ax = ax
        if not self._series:
            ax.text(0.5, 0.5, "No DVC result.\nRun a DVC node first.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.safe_draw()
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
            self.canvas.safe_tight_layout()
            self.canvas.safe_draw()
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
            elif view == "unwrap":
                self._render_unwrap(ax)
        except Exception as exc:  # noqa: BLE001 — never crash the GUI
            ax.clear()
            ax.text(0.5, 0.5, f"Render error:\n{exc}", ha="center", va="center",
                    color=Settings.ACCENT_RED, transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        self.canvas.safe_tight_layout()
        self.canvas.safe_draw()

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
