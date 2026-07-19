"""Interactive per-Z editor for the 3D Mask Drawing node (``special:mask3d``).

``Mask3DEditorDialog`` embeds the reusable :class:`~nd2studios.widgets.image_viewer.ImageCanvas`
and a Z scrubber so the user can draw / edit an object mask on each plane of the
wired channel's raw ``(Z, H, W)`` volume. It supports the node's three creation
modes:

* **Manual (per-plane)** — draw a rect / ellipse / polygon on each Z plane.
* **Propagate across Z** — draw a few planes; the volume is filled by copying or
  signed-distance interpolation between them (previewed live via *Show 3D preview*).
* **Threshold seed + edit** — seed each plane's outline from an intensity threshold
  (0 → Otsu), then hand-correct (draw more, or *Erase* to subtract).

Drawn shapes are kept as vector geometry (``{t: {z_key: [shape, …]}}`` with
``z_key`` an int Z index or ``"all"`` — the :mod:`manual_mask` schema) and returned
to the node via :meth:`result_shapes`; the actual ``(Z, H, W)`` rasterization
happens on Run (see :mod:`nd2studios.backend.analysis.mask3d`). No new drawing
primitives — the canvas' ``set_draw_mode`` / ``shape_drawn`` and the ``manual_mask``
rasterization helpers are reused.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import icon_button, scaled, scale_qss
from nd2studios.widgets.image_viewer import ImageCanvas, frame_to_uint8
from nd2studios.backend.analysis import mask3d

# Modes / propagate — string values mirror the node's ParamSpec choices exactly.
MODE_MANUAL = "Manual (per-plane)"
MODE_PROPAGATE = "Propagate across Z"
MODE_THRESHOLD = "Threshold seed + edit"
PROP_COPY = "Copy to all Z"
PROP_INTERP = "Interpolate between planes"

_PROP_KEY = {PROP_COPY: mask3d.PROPAGATE_COPY, PROP_INTERP: mask3d.PROPAGATE_INTERPOLATE}
_OVERLAY_RGB = np.array([239, 83, 80], dtype=np.float32)  # tint for masked pixels


class Mask3DEditorDialog(QDialog):
    """Modal per-Z mask editor over the wired channel's processed ``(Z, H, W)``
    volume.

    ``get_volume(channel, t)`` returns the volume actually drawn over — the same
    registered + cropped frame the DVC node correlates (so a drawn mask aligns with
    the DVC field). ``channels`` are the channels wired into the node; when more than
    one is wired the user picks which to display via a dropdown (the drawn mask is an
    object mask and is the same regardless of which channel is shown)."""

    def __init__(
        self,
        get_volume: Callable[[str, int], np.ndarray],
        n_z: int,
        n_t: int,
        cur_t: int,
        shapes_by_t: Optional[Dict[Any, Dict[Any, List[Dict[str, Any]]]]] = None,
        mode: str = MODE_PROPAGATE,
        propagate: str = PROP_INTERP,
        pixel_size: Optional[float] = None,
        channels: Optional[List[str]] = None,
        channel: Optional[str] = None,
        processed: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw 3D mask")
        self._get_volume_cb = get_volume
        self._channels = [str(c) for c in (channels or [])]
        if channel in self._channels:
            self._channel = channel
        else:
            self._channel = self._channels[0] if self._channels else None
        self._processed = bool(processed)
        self._n_z = max(1, int(n_z))
        self._n_t = max(1, int(n_t))
        self._t = int(min(max(0, cur_t), self._n_t - 1))
        self._z = self._n_z // 2
        self._opacity = 0.45
        self._erase = False
        self._vol_cache: Dict[tuple, np.ndarray] = {}   # (channel, t) → (Z,H,W)
        # Working copy of the drawn shapes, keyed by int t → {z_key: [shapes]}.
        self._shapes: Dict[int, Dict[Any, List[Dict[str, Any]]]] = {}
        for t_key, by_z in (shapes_by_t or {}).items():
            try:
                ti = int(t_key)
            except (TypeError, ValueError):
                continue
            if isinstance(by_z, dict):  # {z_key: [shapes]}; skip anything malformed
                self._shapes[ti] = copy.deepcopy(by_z)
        self._mode = mode if mode in (MODE_MANUAL, MODE_PROPAGATE, MODE_THRESHOLD) \
            else MODE_PROPAGATE
        self._propagate = propagate if propagate in (PROP_COPY, PROP_INTERP) \
            else PROP_INTERP
        self._build_ui()
        self._sync_mode_visibility()
        self._reload()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.resize(scaled(960), scaled(700))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(scaled(8), scaled(8), scaled(8), scaled(8))
        outer.setSpacing(scaled(6))

        info = QLabel(
            "Draw the object on each Z plane. Pick a tool, then click-drag on the "
            "image. 'Erase' subtracts. Scrub Z below; scroll to zoom, Pan to move.")
        info.setWordWrap(True)
        info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        outer.addWidget(info)

        body = QHBoxLayout()
        body.setSpacing(scaled(8))

        # Left: canvas + Z/T scrubbers.
        left = QVBoxLayout()
        left.setSpacing(scaled(4))
        self.canvas = ImageCanvas(self)
        self.canvas.shape_drawn.connect(self._on_shape_drawn)
        left.addWidget(self.canvas, stretch=1)
        left.addLayout(self._build_z_row())
        if self._n_t > 1:
            left.addLayout(self._build_t_row())
        body.addLayout(left, stretch=1)

        # Right: controls.
        body.addWidget(self._build_controls())
        outer.addLayout(body, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        for b in bb.buttons():
            b.setAutoDefault(False)
        outer.addWidget(bb)

    def _build_z_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(scaled(6))
        lbl = QLabel("Z:")
        lbl.setFixedWidth(scaled(18))
        row.addWidget(lbl)
        self.sld_z = QSlider(Qt.Orientation.Horizontal)
        self.sld_z.setRange(0, self._n_z - 1)
        self.sld_z.setValue(self._z)
        self.sld_z.valueChanged.connect(self._on_z_changed)
        row.addWidget(self.sld_z, stretch=1)
        self.lbl_z = QLabel(f"{self._z + 1}/{self._n_z}")
        self.lbl_z.setFixedWidth(scaled(56))
        self.lbl_z.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_z.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        row.addWidget(self.lbl_z)
        return row

    def _build_t_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(scaled(6))
        lbl = QLabel("T:")
        lbl.setFixedWidth(scaled(18))
        row.addWidget(lbl)
        self.sld_t = QSlider(Qt.Orientation.Horizontal)
        self.sld_t.setRange(0, self._n_t - 1)
        self.sld_t.setValue(self._t)
        self.sld_t.valueChanged.connect(self._on_t_changed)
        row.addWidget(self.sld_t, stretch=1)
        self.lbl_t = QLabel(f"{self._t + 1}/{self._n_t}")
        self.lbl_t.setFixedWidth(scaled(56))
        self.lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_t.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        row.addWidget(self.lbl_t)
        return row

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(scaled(250))
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled(6))

        # — Source (which wired channel to display / draw over) —
        gs = QGroupBox("Source")
        gsl = QVBoxLayout(gs)
        gsl.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        gsl.setSpacing(scaled(4))
        crow = QHBoxLayout()
        crow.setSpacing(scaled(4))
        crow.addWidget(QLabel("Channel:"))
        self.cmb_channel = QComboBox()
        self.cmb_channel.addItems(self._channels or ["(channel)"])
        if self._channel:
            self.cmb_channel.setCurrentText(self._channel)
        # Only interactive when more than one channel is wired in; otherwise it just
        # shows which channel the node is drawing over.
        self.cmb_channel.setEnabled(len(self._channels) > 1)
        self.cmb_channel.setToolTip(
            "Channel to display / draw over. The drawn object mask is the same "
            "regardless of channel; this only changes the background image.")
        self.cmb_channel.currentTextChanged.connect(self._on_channel_changed)
        crow.addWidget(self.cmb_channel, 1)
        gsl.addLayout(crow)
        if self._processed:
            note = QLabel("Showing the registered (drift-corrected) image — matches "
                          "what DVC uses.")
            note.setWordWrap(True)
            note.setStyleSheet(scale_qss(
                f"color: {Settings.ACCENT_CYAN}; font: 8pt;"))
            gsl.addWidget(note)
        lay.addWidget(gs)

        # — Mode —
        gm = QGroupBox("Mode")
        gml = QVBoxLayout(gm)
        gml.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        gml.setSpacing(scaled(4))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([MODE_MANUAL, MODE_PROPAGATE, MODE_THRESHOLD])
        self.cmb_mode.setCurrentText(self._mode)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        gml.addWidget(self.cmb_mode)
        self.cmb_prop = QComboBox()
        self.cmb_prop.addItems([PROP_COPY, PROP_INTERP])
        self.cmb_prop.setCurrentText(self._propagate)
        self.cmb_prop.currentTextChanged.connect(self._on_prop_changed)
        gml.addWidget(self.cmb_prop)
        self.btn_propagate = QPushButton("Propagate now")
        self.btn_propagate.setObjectName("compactBtn")
        self.btn_propagate.setAutoDefault(False)
        self.btn_propagate.setToolTip(
            "Fill the Z planes between (and across) the ones you drew — copying or "
            "interpolating the outline per the choice above — as editable shapes you "
            "can then refine. Re-run after drawing more planes to update.")
        self.btn_propagate.clicked.connect(self._propagate_now)
        gml.addWidget(self.btn_propagate)
        lay.addWidget(gm)

        # — Tools —
        gt = QGroupBox("Tools")
        gtl = QVBoxLayout(gt)
        gtl.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        gtl.setSpacing(scaled(4))
        tool_row = QHBoxLayout()
        tool_row.setSpacing(scaled(3))
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self.btn_rect = self._tool_btn("fa5s.vector-square", "Rectangle — drag a box", "rect")
        self.btn_ellipse = self._tool_btn("fa5s.circle", "Ellipse — drag a box", "ellipse")
        self.btn_poly = self._tool_btn("fa5s.draw-polygon", "Polygon — drag a freehand outline", "polygon")
        self.btn_pan = self._tool_btn("fa5s.arrows-alt", "Pan — drag to move the zoomed image", "pan")
        for b in (self.btn_rect, self.btn_ellipse, self.btn_poly, self.btn_pan):
            tool_row.addWidget(b)
        tool_row.addStretch(1)
        gtl.addLayout(tool_row)
        self.btn_erase = icon_button(
            "fa5s.eraser", "Erase — the next shapes you draw subtract from the mask",
            text=" Erase", checkable=True, object_name="compactBtn", icon_px=12)
        self.btn_erase.setAutoDefault(False)
        self.btn_erase.toggled.connect(self._on_erase_toggled)
        gtl.addWidget(self.btn_erase)
        clr_row = QHBoxLayout()
        clr_row.setSpacing(scaled(3))
        self.btn_clear_z = QPushButton("Clear plane")
        self.btn_clear_z.setObjectName("compactBtn")
        self.btn_clear_z.setAutoDefault(False)
        self.btn_clear_z.setToolTip("Remove every shape drawn on the current Z plane.")
        self.btn_clear_z.clicked.connect(self._clear_plane)
        self.btn_clear_all = QPushButton("Clear all")
        self.btn_clear_all.setObjectName("dangerBtn")
        self.btn_clear_all.setAutoDefault(False)
        self.btn_clear_all.setToolTip("Remove every shape on every Z plane of this frame.")
        self.btn_clear_all.clicked.connect(self._clear_all)
        clr_row.addWidget(self.btn_clear_z)
        clr_row.addWidget(self.btn_clear_all)
        gtl.addLayout(clr_row)
        # Connect BEFORE checking the initial tool so the toggle actually applies
        # the draw mode to the canvas (a check emitted before connect is lost, and
        # _on_tool_toggled is the only place set_draw_mode is called).
        self._tool_group.buttonToggled.connect(self._on_tool_toggled)
        self.btn_poly.setChecked(True)
        lay.addWidget(gt)

        # — Threshold seed —
        self.grp_thresh = QGroupBox("Threshold seed")
        gthl = QVBoxLayout(self.grp_thresh)
        gthl.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        gthl.setSpacing(scaled(4))
        hint = QLabel("Draw a rough mask around the object first — seeding refines "
                      "the outline within that area.")
        hint.setWordWrap(True)
        hint.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        gthl.addWidget(hint)
        trow = QHBoxLayout()
        trow.setSpacing(scaled(4))
        trow.addWidget(QLabel("Level:"))
        self.spn_thresh = QDoubleSpinBox()
        self.spn_thresh.setRange(0.0, 1e9)
        self.spn_thresh.setDecimals(1)
        self.spn_thresh.setValue(0.0)
        self.spn_thresh.setToolTip("Intensity threshold. 0 = automatic (Otsu) per plane.")
        self.spn_thresh.setFixedWidth(scaled(90))
        trow.addWidget(self.spn_thresh)
        trow.addStretch(1)
        gthl.addLayout(trow)
        srow = QHBoxLayout()
        srow.setSpacing(scaled(3))
        self.btn_seed_z = QPushButton("Seed this Z")
        self.btn_seed_z.setObjectName("compactBtn")
        self.btn_seed_z.setAutoDefault(False)
        self.btn_seed_z.setToolTip(
            "Refine the current Z plane's outline from the threshold, within the "
            "drawn object area (replaces this plane).")
        self.btn_seed_z.clicked.connect(lambda: self._seed_threshold(all_z=False))
        self.btn_seed_all = QPushButton("Seed all Z")
        self.btn_seed_all.setObjectName("compactBtn")
        self.btn_seed_all.setAutoDefault(False)
        self.btn_seed_all.setToolTip(
            "Seed an outline on every Z plane that has no shapes yet, within the "
            "manually drawn object area (keeps hand-drawn planes).")
        self.btn_seed_all.clicked.connect(lambda: self._seed_threshold(all_z=True))
        srow.addWidget(self.btn_seed_z)
        srow.addWidget(self.btn_seed_all)
        gthl.addLayout(srow)
        lay.addWidget(self.grp_thresh)

        # — Display —
        gd = QGroupBox("Display")
        gdl = QVBoxLayout(gd)
        gdl.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        gdl.setSpacing(scaled(4))
        self.chk_preview = QCheckBox("Show 3D preview")
        self.chk_preview.setToolTip(
            "Overlay the built 3D mask (after propagation) at this Z, instead of "
            "just the raw shapes drawn on this plane.")
        self.chk_preview.toggled.connect(lambda _c: self._reload())
        gdl.addWidget(self.chk_preview)
        orow = QHBoxLayout()
        orow.setSpacing(scaled(4))
        orow.addWidget(QLabel("Opacity:"))
        self.spn_opacity = QDoubleSpinBox()
        self.spn_opacity.setRange(0.1, 1.0)
        self.spn_opacity.setSingleStep(0.05)
        self.spn_opacity.setValue(self._opacity)
        self.spn_opacity.setFixedWidth(scaled(70))
        self.spn_opacity.valueChanged.connect(self._on_opacity_changed)
        orow.addWidget(self.spn_opacity)
        orow.addStretch(1)
        gdl.addLayout(orow)
        lay.addWidget(gd)

        lay.addStretch(1)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        lay.addWidget(self.lbl_status)
        return panel

    def _tool_btn(self, icon: str, tip: str, tool: str) -> QPushButton:
        b = icon_button(icon, tip, checkable=True, object_name="pipelineToolBtn",
                        icon_px=13, button_px=30)
        b.setAutoDefault(False)
        b.setProperty("tool", tool)
        self._tool_group.addButton(b)
        return b

    # ── data access ──────────────────────────────────────────────────────
    def _get_volume(self, t: int) -> Optional[np.ndarray]:
        key = (self._channel, int(t))
        if key not in self._vol_cache:
            try:
                vol = np.asarray(self._get_volume_cb(self._channel, int(t)))
            except Exception as exc:  # noqa: BLE001 — surface, never crash the dialog
                self.lbl_status.setText(f"Volume read failed: {exc}")
                return None
            if vol.ndim == 2:
                vol = vol[None, ...]
            self._vol_cache[key] = vol
        return self._vol_cache.get(key)

    def _on_channel_changed(self, text: str) -> None:
        if text in self._channels and text != self._channel:
            self._channel = text
            self._reload()

    def _plane_shape(self) -> tuple:
        vol = self._get_volume(self._t)
        if vol is None or vol.ndim != 3:
            return (0, 0)
        return (int(vol.shape[1]), int(vol.shape[2]))

    def _shapes_for_t(self, t: int, create: bool = False) -> Dict[Any, List[Dict[str, Any]]]:
        t = int(t)
        by_z = self._shapes.get(t)
        if by_z is None:
            if not create:
                return {}
            by_z = {}
            self._shapes[t] = by_z
        # Return the stored dict itself (even when empty) — not ``by_z or {}``,
        # which would hand back a throwaway {} for a freshly-created empty slot and
        # silently drop appends.
        return by_z

    # ── rendering ────────────────────────────────────────────────────────
    def _plane_mask_2d(self, H: int, W: int) -> Optional[np.ndarray]:
        by_z = self._shapes_for_t(self._t)
        if self.chk_preview.isChecked() and by_z:
            vol = mask3d.build_mask_volume(
                by_z, self._n_z, H, W, self._propagate_key())
            zi = int(min(max(0, self._z), vol.shape[0] - 1))
            return vol[zi]
        # Raw: 'all' shapes + this plane's shapes.
        shapes = list(by_z.get("all", [])) + list(by_z.get(str(self._z), [])) \
            + list(by_z.get(self._z, []))
        if not shapes:
            return None
        return mask3d.rasterize_plane(shapes, H, W)

    def _reload(self) -> None:
        vol = self._get_volume(self._t)
        if vol is None or vol.ndim != 3:
            self.canvas.setText("No volume.")
            return
        Z, H, W = vol.shape
        self._z = int(min(max(0, self._z), Z - 1))
        plane = np.asarray(vol[self._z])
        gray = frame_to_uint8(plane)
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8, copy=True)
        mask = self._plane_mask_2d(H, W)
        if mask is not None and mask.any():
            a = float(self._opacity)
            sel = rgb[mask].astype(np.float32)
            rgb[mask] = ((1.0 - a) * sel + a * _OVERLAY_RGB).astype(np.uint8)
        self.canvas.set_image(np.ascontiguousarray(rgb))
        self._update_status()

    def _update_status(self) -> None:
        by_z = self._shapes_for_t(self._t)
        drawn = sum(len(v) for k, v in by_z.items())
        planes = sum(1 for k in by_z if k not in ("all", "ALL"))
        self.lbl_status.setText(
            f"T{self._t + 1} · Z{self._z + 1}/{self._n_z} · "
            f"{drawn} shape(s) on {planes} plane(s)"
            + (" + all-Z" if ("all" in by_z or "ALL" in by_z) else ""))

    # ── event handlers ───────────────────────────────────────────────────
    def _on_shape_drawn(self, mode: str, verts) -> None:
        pts = [[float(v[0]), float(v[1])] for v in (verts or [])]
        if (mode in ("rect", "ellipse") and len(pts) != 2) or \
                (mode == "polygon" and len(pts) < 3):
            return
        shape = {"type": str(mode), "vertices": pts,
                 "op": "sub" if self._erase else "add"}
        by_z = self._shapes_for_t(self._t, create=True)
        by_z.setdefault(str(self._z), []).append(shape)
        self._reload()

    def _on_tool_toggled(self, button, checked: bool) -> None:
        if not checked:
            return
        tool = button.property("tool")
        if tool == "pan":
            self.canvas.set_draw_mode(None)
            self.canvas.set_pan_mode(True)
        else:
            self.canvas.set_pan_mode(False)
            self.canvas.set_draw_mode(str(tool))

    def _on_erase_toggled(self, on: bool) -> None:
        self._erase = bool(on)

    def _on_z_changed(self, z: int) -> None:
        self._z = int(z)
        self.lbl_z.setText(f"{self._z + 1}/{self._n_z}")
        self._reload()

    def _on_t_changed(self, t: int) -> None:
        self._t = int(t)
        self.lbl_t.setText(f"{self._t + 1}/{self._n_t}")
        self._reload()

    def _on_mode_changed(self, text: str) -> None:
        self._mode = text
        self._sync_mode_visibility()
        self._reload()

    def _on_prop_changed(self, text: str) -> None:
        self._propagate = text
        self._reload()

    def _on_opacity_changed(self, v: float) -> None:
        self._opacity = float(v)
        self._reload()

    def _sync_mode_visibility(self) -> None:
        is_prop = self._mode == MODE_PROPAGATE
        self.cmb_prop.setVisible(is_prop)
        self.btn_propagate.setVisible(is_prop)
        self.grp_thresh.setVisible(self._mode == MODE_THRESHOLD)

    def _propagate_key(self) -> str:
        if self._mode == MODE_MANUAL:
            return mask3d.PROPAGATE_NONE
        if self._mode == MODE_THRESHOLD:
            # Threshold seeds every plane it can, so no gap-filling by default.
            return mask3d.PROPAGATE_NONE
        return _PROP_KEY.get(self._propagate, mask3d.PROPAGATE_INTERPOLATE)

    def _propagate_now(self) -> None:
        """Materialize the Z-propagation into editable shapes: fill the planes
        between (Interpolate) or across (Copy) the drawn ones with polygons the user
        can then refine. Re-running after drawing more planes recomputes the fill
        (previously auto-propagated planes are dropped first) so it tracks new edits;
        hand-drawn / seeded planes are always kept."""
        vol = self._get_volume(self._t)
        if vol is None or vol.ndim != 3:
            return
        Z, H, W = vol.shape
        by_z = self._shapes_for_t(self._t, create=True)
        # Strip shapes from a PREVIOUS propagate (tagged) so re-propagating reflects
        # the current drawings; hand-drawn / seeded shapes (untagged) are kept, and a
        # plane left empty by the strip is removed so it can be re-filled.
        for k in list(by_z.keys()):
            kept = [s for s in (by_z.get(k) or []) if s.get("src") != "propagate"]
            if kept:
                by_z[k] = kept
            else:
                del by_z[k]
        drawn = [k for k in by_z if k not in ("all", "ALL") and by_z.get(k)]
        if not drawn:
            self._reload()
            self.lbl_status.setText(   # set AFTER reload (_update_status overwrites)
                "Draw the object on at least one Z plane first, then Propagate.")
            return
        prop_key = _PROP_KEY.get(self._propagate, mask3d.PROPAGATE_INTERPOLATE)
        vol_mask = mask3d.build_mask_volume(by_z, Z, H, W, prop_key)
        # max_regions=0 → trace EVERY shape on each plane, not just the largest, so
        # all drawn shapes (multiple disjoint freeforms) propagate together.
        filled = mask3d.volume_to_shapes(vol_mask, max_regions=0)
        n = 0
        for z, shapes in filled.items():
            key = str(int(z))
            if not by_z.get(key):        # keep hand-drawn / seeded planes
                for s in shapes:
                    s["src"] = "propagate"
                by_z[key] = shapes
                n += 1
        self._reload()
        self.lbl_status.setText(         # after reload — _update_status runs in it
            f"Propagated to {n} plane(s) ({self._propagate.lower()}) — "
            "edit any plane to refine, or Propagate again.")

    def _seed_roi(self, Z: int, H: int, W: int) -> Optional[np.ndarray]:
        """A ``(Z,H,W)`` ROI that constrains threshold-seeding to the manually drawn
        object's footprint, so the seed refines the object *inside* the drawn area
        rather than snapping to the brightest blob anywhere in the plane.

        The footprint is the union of every drawn plane (extruded through Z), lightly
        dilated so a slightly-tight hand drawing still captures the object's edge.
        Returns ``None`` when nothing is drawn yet (→ whole-plane seeding)."""
        by_z = self._shapes_for_t(self._t)
        if not by_z:
            return None
        manual = mask3d.build_mask_volume(by_z, Z, H, W, mask3d.PROPAGATE_COPY)
        if not manual.any():
            return None
        foot = manual[0]                       # COPY → every plane is the footprint
        try:
            from scipy.ndimage import binary_dilation
            from skimage.morphology import disk
            rad = int(np.clip(round(0.012 * max(H, W)), 3, 15))
            foot = binary_dilation(foot, structure=disk(rad))
        except Exception:  # noqa: BLE001 — dilation is a nicety, not required
            pass
        return np.repeat(foot[None, ...], Z, axis=0)

    def _seed_threshold(self, all_z: bool) -> None:
        vol = self._get_volume(self._t)
        if vol is None or vol.ndim != 3:
            return
        Z, H, W = vol.shape
        thr = float(self.spn_thresh.value())
        roi = self._seed_roi(Z, H, W)
        by_z = self._shapes_for_t(self._t, create=True)
        if all_z:
            seeded = mask3d.threshold_seed_shapes(vol, threshold=thr, roi=roi)
            n = 0
            for z, shapes in seeded.items():
                key = str(int(z))
                if not by_z.get(key):  # keep hand-drawn planes; only fill empties
                    by_z[key] = shapes
                    n += 1
            kept = len(seeded) - n
            scope = "within the drawn area" if roi is not None else "(whole plane)"
            msg = f"Seeded {n} empty plane(s) from threshold {scope}."
            if kept:
                msg += f" Kept {kept} plane(s) with existing shapes."
            if roi is None:
                msg += " Draw a rough mask first to constrain the seed."
        else:
            roi_z = roi[self._z:self._z + 1] if roi is not None else None
            seeded = mask3d.threshold_seed_shapes(
                vol[self._z:self._z + 1], threshold=thr, roi=roi_z)
            shapes = seeded.get(0, [])
            if shapes:
                by_z[str(self._z)] = shapes
                msg = f"Seeded Z{self._z + 1}."
            else:
                msg = (f"No object above threshold on Z{self._z + 1}"
                       + (" within the drawn area." if roi is not None else "."))
        # Set the status AFTER reload — _reload() → _update_status() would overwrite it.
        self._reload()
        self.lbl_status.setText(msg)

    def _clear_plane(self) -> None:
        by_z = self._shapes_for_t(self._t)
        by_z.pop(str(self._z), None)
        by_z.pop(self._z, None)
        self._reload()

    def _clear_all(self) -> None:
        self._shapes.pop(self._t, None)
        self._reload()

    # ── results ──────────────────────────────────────────────────────────
    def result_shapes(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """The drawn shapes as ``{str(t): {z_key: [shape, …]}}`` (JSON-ready)."""
        out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for t, by_z in self._shapes.items():
            clean: Dict[str, List[Dict[str, Any]]] = {}
            for z_key, shapes in by_z.items():
                if not shapes:
                    continue
                key = "all" if z_key in ("all", "ALL") else str(int(z_key))
                clean[key] = shapes
            if clean:
                out[str(int(t))] = clean
        return out

    def result_mode(self) -> str:
        return self._mode

    def result_propagate(self) -> str:
        return self._propagate
