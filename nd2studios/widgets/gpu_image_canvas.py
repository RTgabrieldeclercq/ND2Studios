"""
``GpuImageCanvas`` — pyqtgraph-backed image canvas with per-channel
layers, GPU LUT/levels, and tool parity with the legacy
:class:`nd2studios.widgets.image_viewer.ImageCanvas`.

V1.36 Phase 4. Hot-path display widget for the multi-axis viewer.

Design notes
------------
- One :class:`pyqtgraph.ImageItem` per channel, stacked in the same
  :class:`pyqtgraph.ViewBox`. Composition mode is
  ``CompositionMode_Plus`` so Qt sums the channels additively — the
  exact replacement for the legacy CPU-side ``apply_lut → multiply
  → sum → clip`` loop in ``MultiAxisViewer._compose_current_frame``.
- A separate "composite" :class:`ImageItem` is reserved for callers
  that already produce an ``(H, W, 3) uint8`` array (export preview,
  the post-process hook, etc.). Per-channel layers and the composite
  layer are mutually visible: pushing a composite hides the channels,
  pushing per-channel updates hides the composite.
- Tools (crop / draw / vertex-edit / click-to-report-pixel) are
  reimplemented on top of pyqtgraph's scene mouse events and a
  custom :class:`_ToolOverlayItem` that paints in scene/image-pixel
  coordinates. The active tool disables :class:`ViewBox` pan/zoom so
  drags don't fight.
- Pan/zoom is delegated to pyqtgraph's :class:`ViewBox`. The
  :class:`ZoomToolbar` calls into ``reset_zoom`` / ``zoom_in`` /
  ``zoom_out`` / ``set_pan_mode`` — same signature as the legacy
  canvas, so the toolbar code does not change.

Signals match the legacy canvas exactly so consumers (analysis page,
multi-axis viewer, etc.) need no edits.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
# Force pyqtgraph to bind to PySide6 even if PyQt6 also happens to be
# installed on the same interpreter. pyqtgraph's auto-detect order
# picks the first binding it imports; without this hint a stray PyQt6
# install on the McGhee Lab box would route ``pg.ImageItem`` through
# PyQt6 while the rest of the app uses PySide6 — and ``QVBoxLayout``
# would reject ``pg.GraphicsLayoutWidget`` as a foreign widget.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
# Pre-import PySide6 before pyqtgraph for the same reason — the env
# var is the belt; this is the suspenders.
from PySide6.QtCore import QPointF, QRectF, Qt, Signal  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QBrush, QColor, QPainter, QPen, QPolygonF,
)
from PySide6.QtWidgets import (  # noqa: E402
    QGraphicsItem, QSizePolicy, QVBoxLayout, QWidget,
)
import pyqtgraph as pg  # noqa: E402

from nd2studios.core.settings import Settings


# Same default cycle the chip strip uses — overridable per channel via
# ``set_channel_color``. Kept here so the canvas can be exercised
# standalone in tests / profiling without the chip strip wired up.
_DEFAULT_CHANNEL_COLORS = [
    (0, 255, 0),     # green
    (255, 0, 0),     # red
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (255, 255, 0),   # yellow
    (0, 100, 255),   # blue
    (255, 165, 0),   # orange
]


def _build_color_lut(rgb: Tuple[int, int, int], n: int = 256) -> np.ndarray:
    """Build a 256x4 RGBA LUT from black to ``rgb``.

    The LUT is what pyqtgraph applies after normalizing the image to
    the channel's ``(lo, hi)`` levels. Alpha is fully opaque; the
    additive composition mode handles inter-channel blending.
    """
    r, g, b = rgb
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lut = np.empty((n, 4), dtype=np.uint8)
    lut[:, 0] = (t * r).astype(np.uint8)
    lut[:, 1] = (t * g).astype(np.uint8)
    lut[:, 2] = (t * b).astype(np.uint8)
    lut[:, 3] = 255
    return lut


class _ChannelLayer:
    """One channel = one :class:`pg.ImageItem` + LUT + levels.

    Cheap value object — does not own a parent widget. Attach
    ``self.item`` to a :class:`pg.ViewBox`.
    """

    def __init__(self, name: str, rgb: Tuple[int, int, int]):
        self.name = name
        self.rgb = rgb
        self.item = pg.ImageItem(axisOrder="row-major")
        self.item.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.item.setLookupTable(_build_color_lut(rgb))
        self._levels: Optional[Tuple[float, float]] = None
        self._visible: bool = True
        self.item.setVisible(True)

    def set_image(self, plane: np.ndarray) -> None:
        # ``autoLevels=False`` preserves user-set contrast across slider
        # scrubs; we manage levels explicitly via ``set_levels``.
        self.item.setImage(plane, autoLevels=False, autoDownsample=True)
        if self._levels is None:
            lo = float(plane.min())
            hi = float(plane.max())
            if hi <= lo:
                hi = lo + 1.0
            self.set_levels((lo, hi))

    def set_levels(self, levels: Tuple[float, float]) -> None:
        lo, hi = float(levels[0]), float(levels[1])
        if hi <= lo:
            hi = lo + 1.0
        self._levels = (lo, hi)
        self.item.setLevels((lo, hi))

    def get_levels(self) -> Optional[Tuple[float, float]]:
        return self._levels

    def set_color(self, rgb: Tuple[int, int, int]) -> None:
        if rgb == self.rgb:
            return
        self.rgb = rgb
        self.item.setLookupTable(_build_color_lut(rgb))

    def set_visible(self, on: bool) -> None:
        self._visible = bool(on)
        self.item.setVisible(self._visible)


class _ToolOverlayItem(QGraphicsItem):
    """Foreground :class:`QGraphicsItem` that paints crop / draw / edit
    overlays in scene (= image-pixel) coordinates.

    Attached to the :class:`ViewBox` so it tracks pan/zoom for free.
    The canvas owns all interactive state; this item only paints.
    """

    def __init__(self, canvas: "GpuImageCanvas"):
        super().__init__()
        self._canvas = canvas
        # Always on top of the channel + composite layers.
        self.setZValue(1000)
        # Don't accept mouse events here — the canvas listens on the
        # scene's signals and routes by tool mode.
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        w = max(1, self._canvas._img_w)
        h = max(1, self._canvas._img_h)
        return QRectF(0, 0, w, h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        c = self._canvas
        # Crop selection (yellow dashed).
        if c._crop_drag_start is not None and c._crop_drag_end is not None:
            (sy, sx), (ey, ex) = c._crop_drag_start, c._crop_drag_end
            rect = QRectF(min(sx, ex), min(sy, ey), abs(ex - sx), abs(ey - sy))
            pen = QPen(QColor(255, 220, 0), 0.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 220, 0, 40)))
            painter.drawRect(rect)

        # In-progress draw preview (cyan dashed).
        if c._draw_mode is not None and c._draw_dragging:
            pen = QPen(QColor(0, 220, 255), 0.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 220, 255, 40)))
            if (c._draw_mode in ("rect", "ellipse")
                    and c._draw_drag_start is not None
                    and c._draw_drag_end is not None):
                (sy, sx), (ey, ex) = c._draw_drag_start, c._draw_drag_end
                rect = QRectF(min(sx, ex), min(sy, ey),
                              abs(ex - sx), abs(ey - sy))
                if c._draw_mode == "rect":
                    painter.drawRect(rect)
                else:
                    painter.drawEllipse(rect)
            elif c._draw_mode == "polygon" and len(c._poly_vertices) >= 2:
                poly = QPolygonF()
                for vy, vx in c._poly_vertices:
                    poly.append(QPointF(vx, vy))
                painter.drawPolyline(poly)

        # Vertex-edit handles (cyan outline + white dots).
        if c._edit_vertices:
            outline = QPolygonF()
            for iy, ix in c._edit_vertices:
                outline.append(QPointF(ix, iy))
            pen = QPen(QColor(0, 220, 255), 0.0, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(outline)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 140, 200), 0.0))
            # Handle radius in scene coords scales with zoom; use a
            # cosmetic stroke and a small fixed pixel radius via
            # device transform inverse.
            handle_r = 4.0 / max(c._scene_scale(), 0.01)
            for i, (iy, ix) in enumerate(c._edit_vertices):
                r = handle_r * (1.4 if i == c._edit_drag_idx else 1.0)
                painter.drawEllipse(QPointF(ix, iy), r, r)


class GpuImageCanvas(QWidget):
    """pyqtgraph-backed image canvas — drop-in for the legacy
    :class:`ImageCanvas` on the multi-axis viewer."""

    # ── Signals (mirror legacy ImageCanvas) ───────────────────────
    clicked = Signal(float, float)              # iy, ix (image pixels)
    zoom_changed = Signal(float)                # zoom multiplier (1.0 = fit)
    pan_mode_changed = Signal(bool)             # True if pan tool is active
    crop_rect_selected = Signal(int, int, int, int)  # x, y, w, h
    shape_drawn = Signal(str, list)             # (mode, [(iy, ix), ...])
    vertex_moved = Signal(int, float, float)    # vertex index, iy, ix
    edit_committed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: {Settings.BG_SECONDARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(Settings.BG_SECONDARY)
        layout.addWidget(self._glw)

        self._viewbox = self._glw.addViewBox(lockAspect=True)
        self._viewbox.setBackgroundColor(Settings.BG_SECONDARY)
        self._viewbox.setMouseEnabled(x=True, y=True)
        self._viewbox.invertY(True)  # image convention: (0, 0) top-left
        # ``RectMode`` left-drag would draw a zoom rectangle; we want
        # click-to-report-pixel by default. Switch to ``PanMode`` only
        # when the pan tool is on.
        self._viewbox.setMouseMode(pg.ViewBox.PanMode)
        # But we want left-click for tool/pixel; default pyqtgraph
        # uses left for pan in PanMode. The cleanest cut: leave
        # PanMode on (right-button + scroll do their normal thing),
        # and gate left-button via our scene click handler — when
        # pan_mode is False we ``setMouseEnabled(False, False)``-style
        # block left pan by intercepting press events.
        self._viewbox.setMenuEnabled(False)

        # ── Channel & composite layers ────────────────────────────
        self._channels: List[_ChannelLayer] = []
        self._composite_item = pg.ImageItem(axisOrder="row-major")
        # Composite path is the *fallback*; per-channel is default.
        self._composite_item.setVisible(False)
        self._viewbox.addItem(self._composite_item)

        # ── Tool overlay (paints on top of everything) ────────────
        self._overlay_item = _ToolOverlayItem(self)
        self._viewbox.addItem(self._overlay_item)

        # ── State mirroring the legacy ImageCanvas ────────────────
        self._img_w: int = 0
        self._img_h: int = 0
        self._mode: str = "none"   # "none" | "channels" | "composite"
        self._overlay_fn: Optional[Callable] = None  # painter callback
        # Zoom is reported relative to a fit-to-window baseline. We
        # track it as a fractional multiplier so the legacy toolbar
        # readout ("100%", "130%", ...) behaves the same as before.
        self._zoom: float = 1.0

        # Tool state.
        self.pan_mode: bool = False
        self._crop_mode: bool = False
        self._crop_drag_start: Optional[Tuple[float, float]] = None
        self._crop_drag_end: Optional[Tuple[float, float]] = None
        self._crop_dragging: bool = False

        self._draw_mode: Optional[str] = None
        self._draw_drag_start: Optional[Tuple[float, float]] = None
        self._draw_drag_end: Optional[Tuple[float, float]] = None
        self._draw_dragging: bool = False
        self._poly_vertices: List[Tuple[float, float]] = []

        self._edit_vertices: Optional[List[Tuple[float, float]]] = None
        self._edit_drag_idx: Optional[int] = None

        # ── Scene event wiring ────────────────────────────────────
        scene = self._viewbox.scene()
        scene.sigMouseClicked.connect(self._on_scene_clicked)
        scene.sigMouseMoved.connect(self._on_scene_moved)
        # ViewBox doesn't emit mouseReleased through the scene
        # signal; we install an event filter on the GraphicsView to
        # catch press/release at the QGraphicsView level.
        self._glw.viewport().installEventFilter(self)

        # Apply a sane default left-button policy: no pan unless the
        # user toggles pan mode. PyQtGraph PanMode drives left-pan;
        # we suppress it by toggling ``setMouseEnabled`` while the
        # tool is engaged.
        self._apply_pan_policy()

    # ── Channel API (used by the GPU render path) ─────────────────
    def configure_channels(
        self,
        names: List[str],
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        """(Re)create channel layers. Called once per file open."""
        for layer in self._channels:
            self._viewbox.removeItem(layer.item)
        self._channels.clear()
        for i, name in enumerate(names):
            rgb = (colors[i] if colors is not None and i < len(colors)
                   else _DEFAULT_CHANNEL_COLORS[i % len(_DEFAULT_CHANNEL_COLORS)])
            layer = _ChannelLayer(name, rgb)
            self._channels.append(layer)
            self._viewbox.addItem(layer.item)
        # Keep the overlay item on top after re-adding layers.
        self._overlay_item.setZValue(1000)

    def update_channel(self, c: int, plane: np.ndarray) -> None:
        """Push a 2D plane to channel ``c``. Engages per-channel mode."""
        if not (0 <= c < len(self._channels)):
            return
        if plane.ndim != 2:
            return
        if self._mode != "channels":
            self._composite_item.setVisible(False)
            # Restore each layer's intended Qt visibility from its
            # ``_visible`` flag — ``set_image`` hid them on the way
            # into composite mode and didn't touch their flags.
            for layer in self._channels:
                layer.item.setVisible(layer._visible)
            self._mode = "channels"
        h, w = plane.shape
        if (h, w) != (self._img_h, self._img_w):
            self._img_h, self._img_w = h, w
            # Force the overlay to recompute its bounding rect.
            self._overlay_item.prepareGeometryChange()
        self._channels[c].set_image(plane)

    def set_channel_visible(self, c: int, on: bool) -> None:
        if 0 <= c < len(self._channels):
            self._channels[c].set_visible(on)

    def set_channel_levels(self, c: int, levels: Tuple[float, float]) -> None:
        if 0 <= c < len(self._channels):
            self._channels[c].set_levels(levels)

    def set_channel_color(self, c: int, rgb: Tuple[int, int, int]) -> None:
        if 0 <= c < len(self._channels):
            self._channels[c].set_color(rgb)

    def n_channels(self) -> int:
        return len(self._channels)

    # ── Backward-compatible RGB / grayscale entry points ─────────
    def set_image(self, rgb_array: np.ndarray) -> None:
        """Show a pre-composed ``(H, W, 3) uint8`` array via the
        composite fallback layer. Disables per-channel layers so the
        two paths don't double-up."""
        if rgb_array is None:
            return
        if self._mode != "composite":
            for layer in self._channels:
                layer.item.setVisible(False)
            self._composite_item.setVisible(True)
            self._mode = "composite"
        a = np.ascontiguousarray(rgb_array)
        if a.ndim == 3 and a.shape[2] in (3, 4):
            h, w = a.shape[:2]
        elif a.ndim == 2:
            h, w = a.shape
        else:
            return
        if (h, w) != (self._img_h, self._img_w):
            self._img_h, self._img_w = h, w
            self._overlay_item.prepareGeometryChange()
        # ``levels=(0, 255)`` ensures the LUT-less RGB path is
        # passed through 1:1 — pyqtgraph would otherwise normalize.
        self._composite_item.setImage(
            a, autoLevels=False, levels=(0, 255), autoDownsample=True,
        )

    def set_grayscale(self, gray_uint8: np.ndarray) -> None:
        """Composite-fallback path with an explicit grayscale array."""
        if gray_uint8 is None:
            return
        if self._mode != "composite":
            for layer in self._channels:
                layer.item.setVisible(False)
            self._composite_item.setVisible(True)
            self._mode = "composite"
        a = np.ascontiguousarray(gray_uint8)
        h, w = a.shape[:2]
        if (h, w) != (self._img_h, self._img_w):
            self._img_h, self._img_w = h, w
            self._overlay_item.prepareGeometryChange()
        self._composite_item.setImage(
            a, autoLevels=False, levels=(0, 255), autoDownsample=True,
        )

    def restore_channel_mode(self) -> None:
        """Show per-channel layers again after a composite-mode push.

        Called by the viewer when it knows the next render will be
        per-channel (e.g. the post-process hook was just cleared).
        """
        if self._mode == "channels":
            return
        self._composite_item.setVisible(False)
        for layer in self._channels:
            layer.item.setVisible(layer._visible)
        self._mode = "channels"

    # ── Overlay callback (legacy parity) ──────────────────────────
    def set_overlay(self, fn: Optional[Callable]) -> None:
        """Legacy ``set_overlay`` hook. Kept as a no-op for the GPU
        canvas — the legacy hook signature is
        ``fn(painter, scale, ox, oy, pw, ph)`` which assumes a
        QPainter rendering into a QLabel and widget-space offsets.
        Consumers that need scene-space drawing should use a custom
        :class:`QGraphicsItem`. The hook is recorded for future use
        but does not paint.
        """
        self._overlay_fn = fn

    # ── Zoom toolbar API ──────────────────────────────────────────
    def _emit_zoom(self) -> None:
        """Estimate a fit-to-window-relative zoom multiplier and emit
        :attr:`zoom_changed`. Used by :class:`ZoomToolbar` to render
        the percentage readout."""
        if self._img_w <= 0 or self._img_h <= 0:
            self._zoom = 1.0
            self.zoom_changed.emit(self._zoom)
            return
        vb_rect = self._viewbox.viewRect()
        if vb_rect.width() <= 0 or vb_rect.height() <= 0:
            return
        # ``fit`` view rect: the entire image is shown. Comparing
        # current view width to the image width gives us the inverse
        # of the zoom multiplier (smaller view = more zoomed in).
        fit_w_ratio = self._img_w / max(vb_rect.width(), 1e-6)
        fit_h_ratio = self._img_h / max(vb_rect.height(), 1e-6)
        # The aspect lock keeps these in sync but the ViewBox can
        # report tiny floating drift — average them.
        self._zoom = float((fit_w_ratio + fit_h_ratio) * 0.5)
        self.zoom_changed.emit(self._zoom)

    def reset_zoom(self) -> None:
        self._viewbox.autoRange()
        self._zoom = 1.0
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        # Scale by < 1 shrinks the visible rect → zooms in.
        self._viewbox.scaleBy((1 / 1.3, 1 / 1.3))
        self._emit_zoom()

    def zoom_out(self) -> None:
        self._viewbox.scaleBy((1.3, 1.3))
        self._emit_zoom()

    def set_pan_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.pan_mode:
            return
        self.pan_mode = enabled
        self._apply_pan_policy()
        self.setCursor(Qt.CursorShape.OpenHandCursor if enabled
                       else Qt.CursorShape.ArrowCursor)
        self.pan_mode_changed.emit(enabled)

    def _apply_pan_policy(self) -> None:
        """Engage pyqtgraph's pan only when our pan_mode is on.

        Outside pan mode we want left-click to be a tool action
        (crop drag / draw stroke / vertex grab / pixel-click), not a
        view pan. We achieve that by disabling :meth:`ViewBox.setMouseEnabled`
        on left-button. Middle-button pan still works via the scene
        default.
        """
        # ``ViewBox.setMouseEnabled`` toggles wheel-zoom + drag-pan.
        # When pan_mode is off, we still want wheel zoom (toolbar
        # buttons cover it) but we don't want left-drag pan; the
        # cleanest way is to set the mouse mode and rely on scene
        # event filtering. We leave both axes enabled and intercept.
        # The actual left-drag suppression happens in
        # :meth:`eventFilter` below.
        pass

    # ── Tool mode setters (legacy parity) ─────────────────────────
    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = bool(enabled)
        self._crop_drag_start = None
        self._crop_drag_end = None
        self._crop_dragging = False
        if enabled:
            self.set_pan_mode(False)
            self._clear_draw_state()
            self._draw_mode = None
            self._edit_vertices = None
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._overlay_item.update()

    def set_draw_mode(self, mode: Optional[str]) -> None:
        if mode not in (None, "rect", "ellipse", "polygon"):
            return
        self._clear_draw_state()
        self._draw_mode = mode
        if mode is not None:
            self.set_pan_mode(False)
            self._crop_mode = False
            self._crop_drag_start = None
            self._crop_drag_end = None
            self._crop_dragging = False
            self._edit_vertices = None
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._overlay_item.update()

    def set_edit_vertices(
        self, vertices: Optional[List[Tuple[float, float]]],
    ) -> None:
        if vertices is None:
            self._edit_vertices = None
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._overlay_item.update()
            return
        self._edit_vertices = [(float(y), float(x)) for y, x in vertices]
        self._edit_drag_idx = None
        self.set_pan_mode(False)
        self._crop_mode = False
        self._crop_drag_start = None
        self._crop_drag_end = None
        self._crop_dragging = False
        self._draw_mode = None
        self._clear_draw_state()
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._overlay_item.update()

    def _clear_draw_state(self) -> None:
        self._draw_drag_start = None
        self._draw_drag_end = None
        self._draw_dragging = False
        self._poly_vertices = []

    # ── Coordinate transforms (legacy parity) ─────────────────────
    def _scene_to_image(self, scene_pos: QPointF) -> Tuple[float, float]:
        view_pt = self._viewbox.mapSceneToView(scene_pos)
        return float(view_pt.y()), float(view_pt.x())

    def widget_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        # Translate widget → GraphicsView viewport → scene → image.
        view_pos = self._glw.viewport().mapFrom(self, QPointF(wx, wy).toPoint())
        scene_pos = self._glw.mapToScene(view_pos)
        return self._scene_to_image(scene_pos)

    def image_to_widget(self, iy: float, ix: float) -> Tuple[float, float]:
        scene_pos = self._viewbox.mapFromView(QPointF(ix, iy))
        view_pos = self._glw.mapFromScene(scene_pos)
        widget_pos = self._glw.viewport().mapTo(self, view_pos)
        return float(widget_pos.x()), float(widget_pos.y())

    def _scene_scale(self) -> float:
        """Pixels-per-image-unit at the current zoom (used by the
        overlay to draw fixed-screen-size handles)."""
        vb_rect = self._viewbox.viewRect()
        gv_rect = self._glw.viewport().rect()
        if vb_rect.width() <= 0:
            return 1.0
        return gv_rect.width() / vb_rect.width()

    # ── Scene event handlers (tool dispatch) ──────────────────────
    def _hit_test_vertex(self, scene_pos: QPointF) -> Optional[int]:
        if not self._edit_vertices:
            return None
        # 10-pixel hit radius in widget space; convert to image-pixel
        # radius via the current scene scale.
        scale = max(self._scene_scale(), 0.01)
        r_img = 10.0 / scale
        r_img2 = r_img * r_img
        iy, ix = self._scene_to_image(scene_pos)
        best = None
        best_d2 = r_img2
        for i, (vy, vx) in enumerate(self._edit_vertices):
            d2 = (vy - iy) ** 2 + (vx - ix) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = i
        return best

    def _on_scene_clicked(self, ev) -> None:
        """pyqtgraph emits a single click signal even for drag-starts.

        We use it only for taps (mouse-press + release at near-same
        point). True drags are handled in :meth:`eventFilter`.
        """
        # We rely on the eventFilter for press/move/release; just
        # consume so pyqtgraph doesn't fall through to context menus.
        ev.accept()

    def _on_scene_moved(self, scene_pos) -> None:
        # Currently unused — drag updates go through the eventFilter
        # for correct release-vs-move ordering. Keeping the hookup
        # so future phases (status-bar pixel readout) can fill in.
        return

    # ── Press / move / release via viewport event filter ──────────
    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        et = event.type()
        if et == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if self._handle_left_press(event):
                    return True
        elif et == QEvent.Type.MouseMove:
            if self._handle_mouse_move(event):
                return True
        elif et == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                if self._handle_left_release(event):
                    return True
        return super().eventFilter(obj, event)

    def _widget_event_to_image(self, event) -> Tuple[float, float]:
        # event.position() (Qt6) / event.pos() (Qt5) — PySide6 gives both.
        try:
            pos = event.position()
            wx, wy = pos.x(), pos.y()
        except AttributeError:
            wx, wy = event.pos().x(), event.pos().y()
        view_pos = QPointF(wx, wy)
        scene_pos = self._glw.mapToScene(view_pos.toPoint())
        return self._scene_to_image(scene_pos)

    def _handle_left_press(self, event) -> bool:
        # Vertex edit takes precedence — handle drag of an existing handle.
        if self._edit_vertices is not None:
            try:
                pos = event.position()
                wx, wy = pos.x(), pos.y()
            except AttributeError:
                wx, wy = event.pos().x(), event.pos().y()
            scene_pos = self._glw.mapToScene(QPointF(wx, wy).toPoint())
            idx = self._hit_test_vertex(scene_pos)
            if idx is not None:
                self._edit_drag_idx = idx
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._overlay_item.update()
                return True
            # Click in empty space inside edit mode = no-op (matches legacy).
            return True

        if self._draw_mode is not None:
            iy, ix = self._widget_event_to_image(event)
            if self._draw_mode in ("rect", "ellipse"):
                self._draw_drag_start = (iy, ix)
                self._draw_drag_end = (iy, ix)
                self._draw_dragging = True
            else:  # polygon
                self._poly_vertices = [(iy, ix)]
                self._draw_dragging = True
            self._overlay_item.update()
            return True

        if self._crop_mode:
            iy, ix = self._widget_event_to_image(event)
            self._crop_drag_start = (iy, ix)
            self._crop_drag_end = (iy, ix)
            self._crop_dragging = True
            return True

        if self.pan_mode:
            # Let pyqtgraph handle pan natively in PanMode.
            return False

        # Default left-button: emit pixel click and suppress
        # pyqtgraph's left-drag pan so the ViewBox doesn't pan when
        # the user is just trying to click.
        iy, ix = self._widget_event_to_image(event)
        if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
            self.clicked.emit(iy, ix)
        # Returning True swallows the press so the ViewBox does not
        # start a pan. Wheel/scroll/middle-drag still work since
        # we only intercept the left button.
        return True

    def _handle_mouse_move(self, event) -> bool:
        if (self._edit_vertices is not None
                and self._edit_drag_idx is not None):
            iy, ix = self._widget_event_to_image(event)
            iy = max(0.0, min(float(self._img_h - 1), iy))
            ix = max(0.0, min(float(self._img_w - 1), ix))
            self._edit_vertices[self._edit_drag_idx] = (iy, ix)
            self.vertex_moved.emit(self._edit_drag_idx, iy, ix)
            self._overlay_item.update()
            return True

        if self._draw_mode is not None and self._draw_dragging:
            iy, ix = self._widget_event_to_image(event)
            if self._draw_mode in ("rect", "ellipse"):
                self._draw_drag_end = (iy, ix)
            else:  # polygon
                if self._poly_vertices:
                    py, px = self._poly_vertices[-1]
                    if (iy - py) ** 2 + (ix - px) ** 2 >= 4.0:
                        self._poly_vertices.append((iy, ix))
                else:
                    self._poly_vertices.append((iy, ix))
            self._overlay_item.update()
            return True

        if self._crop_mode and self._crop_dragging:
            iy, ix = self._widget_event_to_image(event)
            self._crop_drag_end = (iy, ix)
            self._overlay_item.update()
            return True

        return False

    def _handle_left_release(self, event) -> bool:
        if self._edit_drag_idx is not None:
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.edit_committed.emit()
            self._overlay_item.update()
            return True

        if self._draw_dragging and self._draw_mode is not None:
            mode = self._draw_mode
            vertices: List[Tuple[float, float]] = []
            if mode in ("rect", "ellipse"):
                if (self._draw_drag_start is not None
                        and self._draw_drag_end is not None):
                    sy, sx = self._draw_drag_start
                    ey, ex = self._draw_drag_end
                    if abs(ex - sx) > 3 or abs(ey - sy) > 3:
                        vertices = [(sy, sx), (ey, ex)]
            else:  # polygon
                if len(self._poly_vertices) >= 3:
                    vertices = list(self._poly_vertices)
            self._clear_draw_state()
            self._overlay_item.update()
            if vertices:
                self.shape_drawn.emit(mode, vertices)
            return True

        if self._crop_dragging:
            self._crop_dragging = False
            if self._crop_drag_start and self._crop_drag_end:
                sy, sx = self._crop_drag_start
                ey, ex = self._crop_drag_end
                dx, dy = abs(ex - sx), abs(ey - sy)
                if dx > 3 or dy > 3:
                    x = max(0, int(min(sx, ex)))
                    y = max(0, int(min(sy, ey)))
                    w = max(1, int(dx))
                    h = max(1, int(dy))
                    w = min(w, self._img_w - x)
                    h = min(h, self._img_h - y)
                    self._crop_drag_start = None
                    self._crop_drag_end = None
                    self._overlay_item.update()
                    self.crop_rect_selected.emit(x, y, w, h)
                else:
                    iy, ix = self._crop_drag_start
                    self._crop_drag_start = None
                    self._crop_drag_end = None
                    self._overlay_item.update()
                    if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
                        self.clicked.emit(iy, ix)
            return True

        return False
