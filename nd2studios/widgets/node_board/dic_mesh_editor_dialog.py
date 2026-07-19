"""Interactive ROI / mesh-region editor for the DIC mesh nodes.

``DICMeshEditorDialog`` reproduces pyALDIC's ROI toolbar *flow* on top of the app's
reusable :class:`~nd2studios.widgets.image_viewer.ImageCanvas`: **Add / Cut** mode
combined with **Rectangle / Circle / Polygon** shape tools and a freehand **Brush**,
plus **Invert / Clear / Undo**. Drawn actions are kept as an ordered list of
serializable vector shapes (:mod:`nd2studios.backend.dic.roi`) so the region
round-trips through the pipeline file and is rasterized on Run.

Two modes:

* ``"roi"`` — define the AL-DIC mesh *domain* (used by the DIC Mesh Region node).
  A green tint shows the region and a mesh-grid dot preview (pitch = the node's
  preview step) shows where the finite-element nodes will land.
* ``"refine"`` — paint the adaptive-refinement *brush* region (DIC Mesh Refinement
  node); an orange tint, brush selected by default.

The region is shared across timepoints (a frame-1 ROI, like pyALDIC); the T
scrubber only changes the background the region is drawn over. No new drawing
primitives — the canvas' ``set_draw_mode`` / ``shape_drawn`` are reused.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.backend.dic import roi as dic_roi
from nd2studios.widgets.icon_button import icon_button, scaled, scale_qss
from nd2studios.widgets.image_viewer import ImageCanvas, frame_to_uint8

MODE_ROI = "roi"
MODE_REFINE = "refine"

_TINT_ROI = np.array([76, 175, 80], dtype=np.float32)      # green
_TINT_REFINE = np.array([255, 152, 0], dtype=np.float32)   # orange
_MESH_DOT_RGB = np.array([0, 229, 255], dtype=np.uint8)    # cyan mesh nodes


class DICMeshEditorDialog(QDialog):
    """Modal ROI / brush editor over the wired channel's 2D frame."""

    def __init__(
        self,
        get_frame: Callable[[int], np.ndarray],
        n_t: int,
        cur_t: int,
        shapes: Optional[List[Dict[str, Any]]] = None,
        mode: str = MODE_ROI,
        mesh_step: int = 16,
        brush_radius: int = 16,
        pixel_size: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._mode = MODE_REFINE if mode == MODE_REFINE else MODE_ROI
        self.setWindowTitle("Draw mesh region" if self._mode == MODE_ROI
                            else "Draw refinement brush")
        self._get_frame_cb = get_frame
        self._n_t = max(1, int(n_t))
        self._t = int(min(max(0, cur_t), self._n_t - 1))
        self._mesh_step = max(2, int(mesh_step or 16))
        self._brush_radius = max(1, int(brush_radius or 16))
        self._opacity = 0.40
        self._op = dic_roi.OP_ADD          # "add" | "cut"
        self._tool = "brush" if self._mode == MODE_REFINE else "rect"
        self._show_mesh = self._mode == MODE_ROI
        self._frame_cache: Dict[int, np.ndarray] = {}
        # Working copy of the ordered shape actions.
        self._shapes: List[Dict[str, Any]] = copy.deepcopy(list(shapes or []))
        self._tint = _TINT_ROI if self._mode == MODE_ROI else _TINT_REFINE
        self._build_ui()
        self._apply_tool()
        self._reload()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.resize(scaled(920), scaled(680))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(scaled(8), scaled(8), scaled(8), scaled(8))
        outer.setSpacing(scaled(6))

        msg = ("Draw the mesh domain the DIC solver correlates inside. Pick Add or "
               "Cut, choose a shape, then click-drag on the image."
               if self._mode == MODE_ROI else
               "Paint the region where the adaptive mesh should refine. Pick a shape "
               "or the brush, then click-drag on the image.")
        info = QLabel(msg)
        info.setWordWrap(True)
        info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        outer.addWidget(info)

        body = QHBoxLayout()
        body.setSpacing(scaled(8))

        # Left: canvas + T scrubber.
        left = QVBoxLayout()
        left.setSpacing(scaled(4))
        self.canvas = ImageCanvas(self)
        self.canvas.shape_drawn.connect(self._on_shape_drawn)
        left.addWidget(self.canvas, stretch=1)

        trow = QHBoxLayout()
        trow.setSpacing(scaled(4))
        trow.addWidget(QLabel("T"))
        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.setMinimum(0)
        self.t_slider.setMaximum(self._n_t - 1)
        self.t_slider.setValue(self._t)
        self.t_slider.valueChanged.connect(self._on_t_changed)
        self.t_slider.setEnabled(self._n_t > 1)
        trow.addWidget(self.t_slider, stretch=1)
        self.lbl_t = QLabel(f"{self._t + 1}/{self._n_t}")
        trow.addWidget(self.lbl_t)
        left.addLayout(trow)
        body.addLayout(left, stretch=1)

        # Right: toolbar controls.
        right = QVBoxLayout()
        right.setSpacing(scaled(6))

        # Add / Cut mode.
        mode_box = QGroupBox("Mode")
        mb = QHBoxLayout(mode_box)
        mb.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        mb.setSpacing(scaled(4))
        self._add_btn = icon_button("fa5s.plus", "Add — draw to include the region",
                                    text=" Add", object_name="pipelineToolBtn",
                                    checkable=True, icon_px=13)
        self._cut_btn = icon_button("fa5s.minus", "Cut — draw to remove the region",
                                    text=" Cut", object_name="pipelineToolBtn",
                                    checkable=True, icon_px=13)
        self._add_btn.setChecked(True)
        op_group = QButtonGroup(self)
        op_group.setExclusive(True)
        op_group.addButton(self._add_btn)
        op_group.addButton(self._cut_btn)
        self._add_btn.clicked.connect(lambda: self._set_op(dic_roi.OP_ADD))
        self._cut_btn.clicked.connect(lambda: self._set_op(dic_roi.OP_CUT))
        for b in (self._add_btn, self._cut_btn):
            b.setAutoDefault(False)
            mb.addWidget(b)
        right.addWidget(mode_box)

        # Shape tools.
        tools_box = QGroupBox("Tools")
        tg = QVBoxLayout(tools_box)
        tg.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        tg.setSpacing(scaled(4))
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_btns: Dict[str, Any] = {}
        for tool, icon, tip in (
            ("rect", "fa5s.vector-square", "Rectangle — drag a box"),
            ("ellipse", "fa5s.circle", "Circle / ellipse — drag a bounding box"),
            ("polygon", "fa5s.draw-polygon", "Polygon — click-drag a freehand outline"),
            ("brush", "fa5s.paint-brush", "Brush — paint a freehand band"),
            ("pan", "fa5s.hand-paper", "Pan — move / zoom without drawing"),
        ):
            b = icon_button(icon, tip, object_name="pipelineToolBtn",
                            checkable=True, icon_px=14, button_px=34)
            b.setAutoDefault(False)
            b.setProperty("tool", tool)
            b.clicked.connect(lambda _c=False, _t=tool: self._set_tool(_t))
            self._tool_group.addButton(b)
            self._tool_btns[tool] = b
            tg.addWidget(b)
        self._tool_btns[self._tool].setChecked(True)
        # Brush radius.
        br = QHBoxLayout()
        br.setSpacing(scaled(4))
        br.addWidget(QLabel("Brush px"))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(2, 500)
        self._brush_spin.setValue(self._brush_radius)
        self._brush_spin.setFixedSize(scaled(64), scaled(26))
        self._brush_spin.valueChanged.connect(self._on_brush_radius)
        br.addWidget(self._brush_spin)
        br.addStretch(1)
        tg.addLayout(br)
        right.addWidget(tools_box)

        # Edit actions.
        edit_box = QGroupBox("Edit")
        eb = QVBoxLayout(edit_box)
        eb.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(6))
        eb.setSpacing(scaled(4))
        self._undo_btn = icon_button("fa5s.undo", "Undo — remove the last action",
                                     text=" Undo", object_name="pipelineToolBtn",
                                     icon_px=13)
        self._invert_btn = icon_button("fa5s.exchange-alt",
                                       "Invert — flip inside / outside",
                                       text=" Invert", object_name="pipelineToolBtn",
                                       icon_px=13)
        self._clear_btn = icon_button("fa5s.trash", "Clear — remove the whole region",
                                      text=" Clear", object_name="pipelineToolBtn",
                                      icon_px=13)
        self._undo_btn.clicked.connect(self._on_undo)
        self._invert_btn.clicked.connect(self._on_invert)
        self._clear_btn.clicked.connect(self._on_clear)
        for b in (self._undo_btn, self._invert_btn, self._clear_btn):
            b.setAutoDefault(False)
            eb.addWidget(b)
        right.addWidget(edit_box)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(
            scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)
        right.addStretch(1)
        body.addLayout(right)
        outer.addLayout(body, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ── frame + overlay compositing ─────────────────────────────────────────
    def _frame(self, t: int) -> Optional[np.ndarray]:
        if t not in self._frame_cache:
            try:
                fr = self._get_frame_cb(int(t))
            except Exception:  # noqa: BLE001
                fr = None
            if fr is not None:
                fr = np.asarray(fr)
                if fr.ndim == 3:           # collapse a Z / channel axis to 2D
                    fr = fr.max(axis=0)
            self._frame_cache[int(t)] = fr
        return self._frame_cache.get(int(t))

    def _reload(self) -> None:
        frame = self._frame(self._t)
        if frame is None or frame.ndim != 2:
            self.canvas.setText("No frame.")
            return
        H, W = frame.shape
        gray = frame_to_uint8(frame)
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8, copy=True)
        mask = dic_roi.build_roi_mask(self._shapes, H, W)
        if mask.any():
            a = float(self._opacity)
            sel = rgb[mask].astype(np.float32)
            rgb[mask] = ((1.0 - a) * sel + a * self._tint).astype(np.uint8)
            if self._show_mesh:
                self._stamp_mesh(rgb, mask)
        self.canvas.set_image(np.ascontiguousarray(rgb))
        self._update_status(mask)

    def _stamp_mesh(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        """Overlay mesh-node dots at the preview pitch inside the ROI."""
        H, W = mask.shape
        step = max(2, int(self._mesh_step))
        ys = np.arange(step // 2, H, step)
        xs = np.arange(step // 2, W, step)
        for y in ys:
            for x in xs:
                if mask[y, x]:
                    y0, y1 = max(0, y - 1), min(H, y + 2)
                    x0, x1 = max(0, x - 1), min(W, x + 2)
                    rgb[y0:y1, x0:x1] = _MESH_DOT_RGB

    def _update_status(self, mask: np.ndarray) -> None:
        n = int(len(self._shapes))
        px = int(mask.sum())
        nodes = ""
        if self._show_mesh and px:
            step = max(2, int(self._mesh_step))
            H, W = mask.shape
            ys = np.arange(step // 2, H, step)
            xs = np.arange(step // 2, W, step)
            cnt = int(mask[np.ix_(ys, xs)].sum()) if ys.size and xs.size else 0
            nodes = f" · ~{cnt} mesh node(s)"
        self.lbl_status.setText(f"{n} action(s) · {px} px in region{nodes}")

    # ── tool wiring ─────────────────────────────────────────────────────────
    def _apply_tool(self) -> None:
        if self._tool == "pan":
            self.canvas.set_draw_mode(None)
            self.canvas.set_pan_mode(True)
        else:
            self.canvas.set_pan_mode(False)
            # Brush reuses the freehand polygon draw mode; recorded as a brush shape.
            draw_mode = "polygon" if self._tool == "brush" else self._tool
            self.canvas.set_draw_mode(draw_mode)

    def _set_tool(self, tool: str) -> None:
        self._tool = tool
        self._apply_tool()

    def _set_op(self, op: str) -> None:
        self._op = op

    def _on_brush_radius(self, v: int) -> None:
        self._brush_radius = int(v)

    def _on_shape_drawn(self, mode: str, verts) -> None:
        pts = [[float(v[0]), float(v[1])] for v in (verts or [])]
        if self._tool == "brush":
            if len(pts) < 1:
                return
            shape = {"type": "brush", "vertices": pts,
                     "radius": int(self._brush_radius), "op": self._op}
        elif mode in ("rect", "ellipse"):
            if len(pts) != 2:
                return
            shape = {"type": str(mode), "vertices": pts, "op": self._op}
        elif mode == "polygon":
            if len(pts) < 3:
                return
            shape = {"type": "polygon", "vertices": pts, "op": self._op}
        else:
            return
        self._shapes.append(shape)
        self._reload()

    def _on_undo(self) -> None:
        if self._shapes:
            self._shapes.pop()
            self._reload()

    def _on_invert(self) -> None:
        self._shapes.append({"type": "invert"})
        self._reload()

    def _on_clear(self) -> None:
        self._shapes.append({"type": "clear"})
        self._reload()

    def _on_t_changed(self, t: int) -> None:
        self._t = int(t)
        self.lbl_t.setText(f"{self._t + 1}/{self._n_t}")
        self._reload()

    # ── results ───────────────────────────────────────────────────────────
    def result_shapes(self) -> List[Dict[str, Any]]:
        """The ordered shape-action list (compacted: drop leading no-ops)."""
        return copy.deepcopy(self._shapes)
