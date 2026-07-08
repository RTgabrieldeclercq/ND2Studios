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
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsPathItem

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import scaled

_TEMP_ID = "__temp__"

# V1.49: loop / iteration back-edge — a distinct amber wire routed down the left
# gutter, from a node's bottom output up to the top input of the same or an
# upstream node. Drawn solid with a downward arrowhead into the entry.
_LOOP_COLOR = "#F5A623"


class EdgeItem(QGraphicsPathItem):
    HIT_WIDTH = 12  # base (96-DPI) px of grab tolerance around the curve

    def __init__(self, edge_id: str, color_hex: str) -> None:
        super().__init__()
        self.edge_id = edge_id
        self._color = QColor(color_hex)
        self._hover = False
        self._highlight = False
        # V1.48: channel-colored strands drawn parallel to a structural wire, one
        # per channel flowing through it (propagation overlay). A pure channel
        # wire instead colors the base pen (``_is_channel``) and draws no strands.
        self._strand_colors: list = []
        self._is_channel = False
        self._is_loop = False  # V1.49 loop back-edge
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
        elif self._is_loop:
            # Loop back-edge: distinct amber, a touch thicker so it reads apart
            # from the structural wires it runs beside.
            pen = QPen(QColor(_LOOP_COLOR))
            pen.setWidthF(scaled(3.0) if self._hover else scaled(2.4))
        else:
            pen = QPen(self._color)
            pen.setWidthF(scaled(3.0) if self._hover else scaled(2.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        # A pure channel wire reads as a thinner dashed channel-colored line.
        if self._is_channel and not self._highlight and not self._is_loop:
            pen.setWidthF(scaled(2.4) if self._hover else scaled(1.8))
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setZValue(
            0 if self._highlight
            else (0.5 if self._is_loop
                  else (-0.5 if self._is_channel else -1)))

    def set_loop_edge(self, on: bool) -> None:
        """Mark this as a loop / iteration back-edge (V1.49) so it routes down the
        left gutter with a distinct amber style + arrowhead."""
        if self._is_loop == bool(on):
            return
        self.prepareGeometryChange()  # routing + arrow change the bounds
        self._is_loop = bool(on)
        if self._interactive():
            self.setToolTip("Loop connector — double-click to configure; "
                            "cut with the scissors tool to remove")
        self._apply_pen()
        self._rebuild()

    def set_color(self, color_hex: str) -> None:
        self._color = QColor(color_hex)
        self._apply_pen()

    def set_channel_edge(self, on: bool) -> None:
        """Mark this as a pure channel wire (channel source → rainbow port), so it
        renders as a thinner dashed channel-colored line, distinct from structural
        wires (V1.48)."""
        if self._is_channel == bool(on):
            return
        self._is_channel = bool(on)
        self._apply_pen()

    def set_channel_strands(self, colors: list) -> None:
        """Set the per-channel colors drawn parallel to this (structural) wire —
        the propagation overlay (V1.48). Pass ``[]`` to clear."""
        colors = list(colors or [])
        if colors == self._strand_colors:
            return
        self.prepareGeometryChange()  # bounds grow with the offset strands
        self._strand_colors = colors
        self.update()

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
        if self._is_loop:
            self._rebuild_loop(x1, y1, x2, y2)
            return
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

    def _rebuild_loop(self, x1, y1, x2, y2) -> None:
        """Route the loop back-edge down the left gutter: a short stub down out of
        the bottom output, left to a gutter column, up past the target, then right
        and down into the top input (``p2``)."""
        stub = scaled(16)
        gutter = min(x1, x2) - scaled(52)
        approach = y2 - scaled(16)  # come at the top input from just above it
        pts = [
            QPointF(x1, y1),
            QPointF(x1, y1 + stub),
            QPointF(gutter, y1 + stub),
            QPointF(gutter, approach),
            QPointF(x2, approach),
            QPointF(x2, y2),
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

    def boundingRect(self):  # noqa: N802 (Qt naming)
        r = super().boundingRect()
        if self._strand_colors:
            pad = scaled(2.6) * len(self._strand_colors) + scaled(3)
            r = r.adjusted(-pad, -pad, pad, pad)
        if self._is_loop:
            r = r.adjusted(0, -scaled(8), 0, scaled(8))  # room for the arrowhead
        return r

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802
        # Base wire (structural type color, or dashed channel color) first…
        super().paint(painter, option, widget)
        # A loop back-edge draws a downward arrowhead into the top input (p2).
        if self._is_loop:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            tip = self._p2
            a = scaled(5)
            head = QPainterPath(QPointF(tip.x(), tip.y()))
            head.lineTo(tip.x() - a, tip.y() - a * 1.7)
            head.lineTo(tip.x() + a, tip.y() - a * 1.7)
            head.closeSubpath()
            col = QColor(Settings.ACCENT_GOLD) if self._highlight else QColor(_LOOP_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(col)
            painter.drawPath(head)
            return
        # …then the per-channel propagation strands, parallel copies offset
        # sideways so several channels read as distinct colored threads.
        if not self._strand_colors:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = self.path()
        n = len(self._strand_colors)
        spread = scaled(2.6)
        for i, hexc in enumerate(self._strand_colors):
            off = (i - (n - 1) / 2.0) * spread
            pen = QPen(QColor(hexc))
            pen.setWidthF(scaled(1.6))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.save()
            painter.translate(off, 0.0)
            painter.drawPath(path)
            painter.restore()

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
            # A loop edge is configured (not disconnected) on double-click; it is
            # removed with the scissors tool instead.
            if self._is_loop:
                fn = getattr(scene, "edit_loop_edge", None)
                if callable(fn):
                    fn(self.edge_id)
                    event.accept()
                    return
            fn = getattr(scene, "disconnect_edge", None)
            if callable(fn):
                fn(self.edge_id)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)
