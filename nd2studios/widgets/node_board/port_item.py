"""Typed connection point for the node board (V1.45).

A :class:`PortItem` is a child of its :class:`NodeItem`; its local origin
``(0, 0)`` is the exact connection point a wire snaps to. Color encodes the
:class:`PortType` (spec §6.2). The scene drives connection drags — ports only
provide hover feedback and a brief reject flash.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsObject

from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.model import Port, PortType
from nd2studios.widgets.icon_button import scaled

PORT_COLORS = {
    PortType.IMAGE: Settings.FG_PRIMARY,    # channel dict
    PortType.BINARY: Settings.ACCENT_ORANGE,  # label masks
    PortType.DATA: Settings.ACCENT_PURPLE,   # measurement rows
    PortType.VALUE: Settings.ACCENT_CYAN,    # scalar
    PortType.ANY: Settings.FG_SECONDARY,     # wildcard (logic / special nodes)
}


def port_color(port_type: PortType) -> str:
    return PORT_COLORS.get(port_type, Settings.FG_PRIMARY)


class PortItem(QGraphicsObject):
    """A single typed port. ``node_item`` is the owning :class:`NodeItem`."""

    def __init__(self, port: Port, node_item: "QGraphicsObject") -> None:
        super().__init__(node_item)
        self.port = port
        self.node_item = node_item
        self._r = scaled(6)
        self._hover = False
        self._reject = False
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setToolTip(f"{port.name} · {port.type.value}")
        # Connections are started by dragging from OUTPUT ports only (handled in
        # NodeScene.mousePressEvent). INPUT ports therefore have no press role —
        # make them mouse-transparent so a press on a top anchor falls through to
        # the node body and drags the whole node. They're still valid drop
        # targets on release (NodeScene._port_at uses items(), not button hits).
        if port.is_input:
            self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ── geometry ──────────────────────────────────────────────────────────
    def boundingRect(self) -> QRectF:
        pad = self._r + scaled(3)
        return QRectF(-pad, -pad, 2 * pad, 2 * pad)

    def center_scene(self) -> QPointF:
        return self.mapToScene(QPointF(0.0, 0.0))

    def color_hex(self) -> str:
        return port_color(self.port.type)

    # ── paint ─────────────────────────────────────────────────────────────
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        col = QColor(self.color_hex())
        painter.setBrush(QBrush(col))
        ring = QColor(Settings.ACCENT_RED) if self._reject else QColor(Settings.BG_PRIMARY)
        pen = QPen(ring)
        pen.setWidthF(scaled(2) if (self._hover or self._reject) else scaled(1.5))
        painter.setPen(pen)
        r = self._r + (scaled(1.5) if self._hover else 0)
        painter.drawEllipse(QPointF(0.0, 0.0), r, r)

    # ── feedback ──────────────────────────────────────────────────────────
    def flash_reject(self) -> None:
        self._reject = True
        self.update()
        QTimer.singleShot(220, self._clear_reject)

    def _clear_reject(self) -> None:
        self._reject = False
        self.update()

    def hoverEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)
