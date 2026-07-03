"""Cubic-Bézier wire between two ports (V1.45).

Colored by the source port's :class:`PortType`. The scene calls
:meth:`set_endpoints` whenever either endpoint moves (node drag) or while
rubber-banding a new connection. Because connection anchors live on the
top/bottom edges of a node (V1.45.1), wires flow **vertically** top→bottom.

Routing (V1.45.3): when the two nodes line up the wire is a straight drop; when
they are offset laterally it becomes an **orthogonal elbow** — drop out of the
output, run horizontally across the gap *between* the two nodes, then drop into
the input — with the two bends drawn as **rounded corners**. Routing through the
inter-node gap keeps the wire off the node bodies (it never overlaps them).

A committed wire is interactive: hovering thickens it and shows a tooltip, and
**double-clicking it disconnects** the edge (the scene's ``disconnect_edge``).
The hit area is widened with a :class:`QPainterPathStroker` so the thin curve
is easy to grab. The transient ``"__temp__"`` drag wire stays inert.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsPathItem

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import scaled

_TEMP_ID = "__temp__"


class EdgeItem(QGraphicsPathItem):
    HIT_WIDTH = 12  # base (96-DPI) px of grab tolerance around the curve

    def __init__(self, edge_id: str, color_hex: str) -> None:
        super().__init__()
        self.edge_id = edge_id
        self._color = QColor(color_hex)
        self._hover = False
        self._highlight = False
        self.setZValue(-1)
        if self._interactive():
            self.setAcceptHoverEvents(True)
            self.setToolTip("Double-click to disconnect")
        self._apply_pen()
        self._p1 = QPointF()
        self._p2 = QPointF()

    # ── appearance ────────────────────────────────────────────────────────
    def _interactive(self) -> bool:
        return self.edge_id != _TEMP_ID

    def _apply_pen(self) -> None:
        if self._highlight:
            pen = QPen(QColor(Settings.ACCENT_GOLD))
            pen.setWidthF(scaled(3.0))
        else:
            pen = QPen(self._color)
            pen.setWidthF(scaled(3.0) if self._hover else scaled(2.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.setZValue(0 if self._highlight else -1)

    def set_color(self, color_hex: str) -> None:
        self._color = QColor(color_hex)
        self._apply_pen()

    def set_highlighted(self, on: bool) -> None:
        """Gold the wire when it lies on the previewed node's connected set."""
        if self._highlight == on:
            return
        self._highlight = on
        self._apply_pen()

    def set_endpoints(self, p1: QPointF, p2: QPointF) -> None:
        self._p1 = p1
        self._p2 = p2
        self._rebuild()

    def _rebuild(self) -> None:
        x1, y1 = self._p1.x(), self._p1.y()
        x2, y2 = self._p2.x(), self._p2.y()
        if abs(x2 - x1) < 0.5:
            # Anchors line up vertically — a straight drop needs no corners.
            path = QPainterPath(self._p1)
            path.lineTo(self._p2)
            self.setPath(path)
            return
        # Orthogonal elbow routed through the gap between the two nodes: down
        # out of the output, across at the midpoint, down into the input. The
        # two bends are rounded; running across the inter-node gap keeps the
        # wire from overlapping either node's body.
        mid_y = (y1 + y2) / 2.0
        pts = [
            QPointF(x1, y1), QPointF(x1, mid_y),
            QPointF(x2, mid_y), QPointF(x2, y2),
        ]
        self.setPath(self._rounded_polyline(pts, scaled(11)))

    @staticmethod
    def _rounded_polyline(pts, radius: float) -> QPainterPath:
        """Polyline through ``pts`` with each interior corner rounded by up to
        ``radius`` (clamped to half the shorter adjacent segment)."""
        path = QPainterPath(pts[0])
        for i in range(1, len(pts) - 1):
            prev, cur, nxt = pts[i - 1], pts[i], pts[i + 1]
            v1, v2 = prev - cur, nxt - cur
            l1 = math.hypot(v1.x(), v1.y())
            l2 = math.hypot(v2.x(), v2.y())
            r = min(radius, l1 / 2.0, l2 / 2.0)
            if r <= 0.5 or l1 == 0.0 or l2 == 0.0:
                path.lineTo(cur)
                continue
            path.lineTo(cur + v1 / l1 * r)   # stop short of the corner…
            path.quadTo(cur, cur + v2 / l2 * r)  # …round through it
        path.lineTo(pts[-1])
        return path

    def shape(self):  # noqa: N802 (Qt naming)
        """Widen the clickable region around the thin curve for easy grabbing."""
        stroker = QPainterPathStroker()
        stroker.setWidth(scaled(self.HIT_WIDTH))
        return stroker.createStroke(self.path())

    # ── interaction ───────────────────────────────────────────────────────
    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self._apply_pen()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self._apply_pen()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Accept the press so the view treats the wire as the target (no
        # rubber-band) and the follow-up double-click is delivered here.
        if self._interactive() and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._interactive() and event.button() == Qt.MouseButton.LeftButton:
            scene = self.scene()
            fn = getattr(scene, "disconnect_edge", None)
            if callable(fn):
                fn(self.edge_id)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)
