"""Typed connection point for the node board (V1.45).

A :class:`PortItem` is a child of its :class:`NodeItem`; its local origin
``(0, 0)`` is the exact connection point a wire snaps to. Color encodes the
:class:`PortType` (spec §6.2). The scene drives connection drags — ports only
provide hover feedback and a brief reject flash.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
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
    PortType.CHANNEL: Settings.FG_SECONDARY,  # rainbow channel-flow port (V1.48)
}

# Rainbow stops for CHANNEL ports (V1.48) — a channel-agnostic rainbow port that
# accepts any channel wire. A channel *source* port is instead painted its own
# channel color (via ``set_display_color``).
_RAINBOW_STOPS = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759",
                  "#00c7be", "#0a84ff", "#bf5af2"]


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
        # V1.48: a channel-source port paints its own channel color (set here);
        # a rainbow (CHANNEL) port with no display color paints the rainbow.
        self._display_color: str = ""
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        tip = (f"{port.name} · {port.type.value}"
               if port.type is not PortType.CHANNEL
               else "Channel port — wire channels here (rainbow)")
        self.setToolTip(tip)
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
        return self._display_color or port_color(self.port.type)

    def set_display_color(self, color_hex: str) -> None:
        """Override the port fill (channel-source ports carry a channel color)."""
        if color_hex != self._display_color:
            self._display_color = color_hex or ""
            self.update()

    def _is_rainbow(self) -> bool:
        """A rainbow (channel-agnostic) port: CHANNEL type with no channel color."""
        return self.port.type is PortType.CHANNEL and not self._display_color

    # ── paint ─────────────────────────────────────────────────────────────
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self._r + (scaled(1.5) if self._hover else 0)
        if self._is_rainbow():
            grad = QLinearGradient(-r, 0, r, 0)
            n = len(_RAINBOW_STOPS)
            for i, hexc in enumerate(_RAINBOW_STOPS):
                grad.setColorAt(i / (n - 1), QColor(hexc))
            painter.setBrush(QBrush(grad))
        else:
            painter.setBrush(QBrush(QColor(self.color_hex())))
        ring = QColor(Settings.ACCENT_RED) if self._reject else QColor(Settings.BG_PRIMARY)
        pen = QPen(ring)
        pen.setWidthF(scaled(2) if (self._hover or self._reject) else scaled(1.5))
        painter.setPen(pen)
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
