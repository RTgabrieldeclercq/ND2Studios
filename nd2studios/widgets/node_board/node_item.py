"""Rounded node with a stage-accent lip and hover/selected corner buttons.

A :class:`NodeItem` renders one model :class:`Node`. Its **left** edge carries a
thin colored **lip** in the current sub-tab accent (the node belongs to exactly
one stage). Connection anchors sit on the **top** (inputs) and **bottom**
(outputs) edges so wires flow vertically top→bottom — the left edge is reserved
for the lip. On hover — and permanently while selected — three corner buttons
appear top-right: **Disconnect / Duplicate / Delete**, each with a tooltip
(spec §4.5). The buttons call back to the owning scene by duck-typed method
name so this module doesn't import the scene (avoids a cycle).
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsObject, QGraphicsProxyWidget, QLineEdit,
)

from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.model import Node, NodeCategory, NodeRole, PortType, ShapeKind
from nd2studios.widgets.icon_button import make_icon, scaled, scale_qss, scaled_pt
from nd2studios.widgets.node_board.port_item import PortItem


class _NameLineEdit(QLineEdit):
    """``QLineEdit`` for in-place node renaming that also reports Escape."""

    escaped = Signal()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.key() == Qt.Key.Key_Escape:
            self.escaped.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _CornerButton(QGraphicsObject):
    """Small icon hit-area shown on hover/selection, with its own tooltip."""

    def __init__(
        self,
        icon_name: str,
        tooltip: str,
        callback: Callable[[], None],
        parent: QGraphicsObject,
        color: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._size = scaled(18)
        self._callback = callback
        self._hover = False
        self.setToolTip(tooltip)
        self.setAcceptHoverEvents(True)
        self.setZValue(4)
        icon_px = scaled(12)
        icon = make_icon(icon_name, color or Settings.FG_SECONDARY)
        self._pix = icon.pixmap(QSize(icon_px, icon_px))

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._size, self._size)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._hover:
            painter.setBrush(QColor(Settings.BG_HOVER))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.boundingRect(), scaled(4), scaled(4))
        if not self._pix.isNull():
            x = (self._size - self._pix.width()) / 2
            y = (self._size - self._pix.height()) / 2
            painter.drawPixmap(int(x), int(y), self._pix)
        elif self.toolTip():
            # qtawesome missing — show first letter so the button isn't blank.
            painter.setPen(QColor(Settings.FG_SECONDARY))
            painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter,
                             self.toolTip()[:1])

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self._callback()
            return
        super().mousePressEvent(event)


class NodeItem(QGraphicsObject):
    """Visual + interactive representation of one model :class:`Node`."""

    BASE_WIDTH = 172
    # V1.77 Prism (ShapeKind.GEM) silhouette proportions — shared by _body_path
    # (outline), _build_ports (implicit, via the general branch) and _paint_gem
    # (facet vertices), so the facets align exactly with the outline.
    _GEM_TABLE_INSET = 0.26   # flat top/bottom half-inset (fraction of width)
    _GEM_SHOULDER = 0.26      # girdle shoulder height (fraction of height)

    def __init__(self, node: Node, accent_hex: str) -> None:
        super().__init__()
        self.node = node
        self._accent = accent_hex
        # V1.45 merge: a node is drawn as a rounded rect, an upright triangle
        # (if-else), or a hexagon (special). The shape drives geometry, ports
        # and paint; the accent is the node's *category* color.
        self._shape = getattr(node, "shape_kind", ShapeKind.RECT) or ShapeKind.RECT
        # V1.48: channel-source nodes render as small pills (compact width).
        self._w = (scaled(112) if self._shape is ShapeKind.PILL
                   else scaled(self.BASE_WIDTH))
        self._lip_w = scaled(5)          # vertical accent strip on the LEFT edge
        self._title_h = scaled(26)
        self._body_pad = scaled(16)      # extra body height below the title row
        self._tri_h = scaled(94)         # taller body so a triangle reads as one
        self._radius = scaled(8)
        self._hover = False
        self._highlight = False          # on the previewed node's connected set
        self._previewed = False          # the node whose result is in the viewer
        self._run_state = ""             # "" | "shaded" | "current" | "done" (Run)

        # In-place name editing (INPUT / OUTPUT nodes): double-click to edit.
        self._editor: Optional[QGraphicsProxyWidget] = None
        self._editor_widget: Optional[_NameLineEdit] = None
        self._edit_cancelled = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self.setPos(QPointF(float(node.pos[0]), float(node.pos[1])))

        self.port_items: dict = {}
        self._buttons: List[_CornerButton] = []
        self._build_ports()
        self._build_buttons()
        self._update_button_visibility()

    # ── geometry ──────────────────────────────────────────────────────────
    def _height(self) -> float:
        # Ports live on the top/bottom edges (not stacked vertically), so the
        # body height is fixed. Triangles are taller so the silhouette reads;
        # channel pills are short.
        if self._shape is ShapeKind.TRIANGLE:
            return self._tri_h
        if self._shape is ShapeKind.GEM:
            # A gem reads as a tall faceted crystal; grow with rainbow inputs (on the
            # vertical mid-sides) so several channel wires fit like a rect process.
            n_ci = sum(1 for p in self.node.inputs if p.type is PortType.CHANNEL)
            return self._tri_h + scaled(13) * max(0, n_ci - 1)
        if self._shape is ShapeKind.PILL:
            # Grow with the number of rainbow inputs so several channels fit down
            # the left edge (channel-source pills have none → the base height).
            n_ci = sum(1 for p in self.node.inputs if p.type is PortType.CHANNEL)
            return scaled(26) + scaled(14) * max(0, n_ci)
        n_ci = sum(1 for p in self.node.inputs if p.type is PortType.CHANNEL)
        base = self._title_h + self._body_pad
        # Rect process nodes stack their rainbow inputs down the left edge; grow
        # so they don't crowd (base fits ~1 rainbow port).
        return base + scaled(13) * max(0, n_ci - 1)

    def _is_shape(self) -> bool:
        """True for the non-rectangular silhouettes (triangle / hexagon / gem)."""
        return self._shape in (ShapeKind.TRIANGLE, ShapeKind.HEXAGON, ShapeKind.GEM)

    def _port_x(self, i: int, n: int) -> float:
        """Even horizontal spread for ``n`` ports along a top/bottom edge."""
        return self._w * (i + 1) / (n + 1)

    def _port_y(self, i: int, n: int) -> float:
        """Even vertical spread for ``n`` ports along a left/right edge (V1.48
        rainbow channel ports)."""
        return self._height() * (i + 1) / (n + 1)

    def _body_path(self) -> QPainterPath:
        """The node silhouette in item coordinates (origin at top-left)."""
        w, h = self._w, self._height()
        p = QPainterPath()
        if self._shape is ShapeKind.TRIANGLE:
            # Upright triangle: apex top-center (the input), base across the
            # bottom (true output bottom-left, false output bottom-right).
            p.moveTo(w / 2.0, 0.0)
            p.lineTo(w, h)
            p.lineTo(0.0, h)
            p.closeSubpath()
        elif self._shape is ShapeKind.HEXAGON:
            inset = w * 0.22
            p.moveTo(inset, 0.0)
            p.lineTo(w - inset, 0.0)
            p.lineTo(w, h / 2.0)
            p.lineTo(w - inset, h)
            p.lineTo(inset, h)
            p.lineTo(0.0, h / 2.0)
            p.closeSubpath()
        elif self._shape is ShapeKind.GEM:
            # V1.77 Prism — an upright faceted crystal (barrel octagon): a narrow flat
            # table on top, angled shoulders out to the widest girdle at the vertical
            # mid-sides, then angled shoulders in to a flat culet on the bottom. The
            # flat top/bottom carry the structural in/out ports; the vertical mid-sides
            # carry the rainbow channel ports. Facets + gloss are painted in _paint_gem.
            tx = w * self._GEM_TABLE_INSET
            sh = h * self._GEM_SHOULDER
            p.moveTo(tx, 0.0)
            p.lineTo(w - tx, 0.0)
            p.lineTo(w, sh)
            p.lineTo(w, h - sh)
            p.lineTo(w - tx, h)
            p.lineTo(tx, h)
            p.lineTo(0.0, h - sh)
            p.lineTo(0.0, sh)
            p.closeSubpath()
        elif self._shape is ShapeKind.PILL:
            p.addRoundedRect(QRectF(0, 0, w, h), h / 2.0, h / 2.0)
        else:
            p.addRoundedRect(QRectF(0, 0, w, h), self._radius, self._radius)
        return p

    def boundingRect(self) -> QRectF:
        # Pad by the widest border pen (the 3px "previewed" gold outline) so a
        # repaint after the border shrinks fully clears the old outline — an
        # update() only invalidates boundingRect, and a thick pen strokes ~1.5px
        # outside the body path. Without the pad a gold ghost ring lingers when
        # the previewed node changes.
        m = scaled(3)
        return QRectF(-m, -m, self._w + 2 * m, self._height() + 2 * m)

    def shape(self) -> QPainterPath:  # noqa: N802 (Qt naming)
        # Keep mouse hit-testing on the visible body, not the padded bounds.
        return self._body_path()

    def set_accent(self, accent_hex: str) -> None:
        self._accent = accent_hex
        self.update()

    def set_run_state(self, state: str) -> None:
        """Run visualization: ``"shaded"`` (pending / un-taken branch — dimmed),
        ``"current"`` (executing — gold outline), ``"cached"`` (V1.53 frozen
        upstream — dimmed with a white border, won't re-run), ``"done"`` / ``""``
        (normal). Independent of the preview highlight."""
        state = state or ""
        if state == self._run_state:
            return
        self._run_state = state
        self.setZValue(2 if state == "current" else 1)
        self.update()

    def set_highlighted(self, on: bool, *, previewed: bool = False) -> None:
        """Golden outline state. ``previewed`` is the single node whose result is
        shown in the viewer (drawn thicker); the rest of its connected set just
        gets the golden border."""
        if self._highlight == on and self._previewed == previewed:
            return
        self._highlight = on
        self._previewed = previewed
        self.setZValue(2 if previewed else (1.5 if on else 1))
        self.update()

    # ── children ──────────────────────────────────────────────────────────
    def _build_ports(self) -> None:
        h = self._height()
        if self._shape is ShapeKind.TRIANGLE:
            # Single input at the apex; outputs at the two base corners so wires
            # leave true (bottom-left) and false (bottom-right) distinctly.
            if self.node.inputs:
                port = self.node.inputs[0]
                it = PortItem(port, self)
                it.setPos(self._w / 2.0, 0.0)
                self.port_items[port.id] = it
            n_out = len(self.node.outputs)
            corner_xs = ([self._w * 0.16, self._w * 0.84] if n_out == 2
                         else [self._port_x(i, n_out) for i in range(n_out)])
            for i, port in enumerate(self.node.outputs):
                it = PortItem(port, self)
                it.setPos(corner_xs[i] if i < len(corner_xs)
                          else self._port_x(i, n_out), h)
                self.port_items[port.id] = it
            return
        # V1.48: structural (non-channel) ports flow vertically — inputs on the
        # TOP edge, outputs on the BOTTOM. Rainbow CHANNEL ports flow
        # horizontally — inputs on the LEFT edge, outputs on the RIGHT — so
        # channel flow reads orthogonal to pipeline flow.
        struct_in = [p for p in self.node.inputs if p.type is not PortType.CHANNEL]
        struct_out = [p for p in self.node.outputs if p.type is not PortType.CHANNEL]
        chan_in = [p for p in self.node.inputs if p.type is PortType.CHANNEL]
        chan_out = [p for p in self.node.outputs if p.type is PortType.CHANNEL]
        n_in = len(struct_in)
        for i, port in enumerate(struct_in):
            it = PortItem(port, self)
            it.setPos(self._port_x(i, n_in), 0.0)
            self.port_items[port.id] = it
        n_out = len(struct_out)
        for i, port in enumerate(struct_out):
            it = PortItem(port, self)
            it.setPos(self._port_x(i, n_out), h)
            self.port_items[port.id] = it
        n_ci = len(chan_in)
        for i, port in enumerate(chan_in):
            it = PortItem(port, self)
            it.setPos(0.0, self._port_y(i, n_ci))
            self.port_items[port.id] = it
        n_co = len(chan_out)
        for i, port in enumerate(chan_out):
            it = PortItem(port, self)
            it.setPos(self._w, self._port_y(i, n_co))
            self.port_items[port.id] = it

    def rebuild_ports(self) -> None:
        """Recreate the port items from the (mutated) model — used when rainbow
        channel ports are spawned / trimmed (V1.48). The node's height may change
        with the rainbow-input count, so bracket it with ``prepareGeometryChange``.
        The caller (scene) re-indexes ports and refreshes touching edges."""
        self.prepareGeometryChange()
        scene = self.scene()
        for it in list(self.port_items.values()):
            if scene is not None:
                scene.removeItem(it)
            else:
                it.setParentItem(None)
        self.port_items.clear()
        self._build_ports()
        self.update()

    def _build_buttons(self) -> None:
        # V1.48: channel-source pills are auto-managed by the page — no corner
        # controls (disconnect/duplicate/delete would desync them).
        if getattr(self.node, "category", None) is NodeCategory.CHANNEL:
            return
        specs = [
            ("fa5s.unlink", "Disconnect", "disconnect_node", Settings.ACCENT_CYAN),
            ("fa5s.clone", "Duplicate", "duplicate_node", Settings.FG_SECONDARY),
            ("fa5s.trash", "Delete", "delete_node", Settings.ACCENT_RED),
        ]
        size = scaled(18)
        gap = scaled(3)
        x = self._w - scaled(6)
        y = scaled(5)
        for icon, tip, method, color in specs:
            x -= size
            btn = _CornerButton(
                icon, tip,
                callback=lambda m=method: self._invoke_scene(m),
                parent=self, color=color,
            )
            btn.setPos(x, y)
            self._buttons.append(btn)
            x -= gap

    def _invoke_scene(self, method_name: str) -> None:
        scene = self.scene()
        fn = getattr(scene, method_name, None)
        if callable(fn):
            fn(self.node.id)

    def _update_button_visibility(self) -> None:
        visible = self._hover or self.isSelected()
        for btn in self._buttons:
            btn.setVisible(visible)

    # ── paint ─────────────────────────────────────────────────────────────
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._shape is ShapeKind.PILL:
            self._paint_pill(painter)
            return
        selected = self.isSelected()
        # Dim for a disabled node, a Run "shaded" node (pending / un-taken), or a
        # V1.53 "cached" node (frozen upstream of a checkpoint — won't re-run).
        cached = self._run_state == "cached"
        dim = (not self.node.enabled) or (self._run_state == "shaded") or cached
        current = self._run_state == "current"
        h = self._height()
        body_path = self._body_path()

        if self._shape is ShapeKind.GEM:
            # V1.77 Prism — a faceted 2.5D crystal (facets + gloss + seams).
            self._paint_gem(painter, body_path, dim, h)
        elif self._is_shape():
            # Triangle / hexagon: a dark body tinted by the category color so an
            # if-else reads purple and a special node reads orange (Dismiss red).
            painter.fillPath(body_path, QColor(Settings.BG_TERTIARY))
            tint = QColor(self._accent)
            tint.setAlpha(28 if dim else 55)
            painter.fillPath(body_path, tint)
        else:
            # Rect: left lip in the category accent + neutral body.
            lip_path = QPainterPath()
            lip_path.addRoundedRect(
                QRectF(0, 0, self._lip_w + self._radius, h),
                self._radius, self._radius,
            )
            accent = QColor(self._accent)
            if dim:
                accent.setAlpha(120)
            painter.fillPath(lip_path, accent)
            body_rect = QRectF(self._lip_w, 0, self._w - self._lip_w, h)
            bp = QPainterPath()
            bp.addRoundedRect(body_rect, self._radius, self._radius)
            body_col = QColor(Settings.BG_TERTIARY)
            if dim:
                body_col.setAlpha(150)
            painter.fillPath(bp, body_col)

        # Border priority: previewed / Run-current (thick gold) > highlighted
        # (gold) > selected (accent) > shape category color > rect idle border.
        if self._previewed or current:
            border = QPen(QColor(Settings.ACCENT_GOLD))
            border.setWidthF(scaled(3))
        elif cached:
            # Frozen upstream: dashed white outline signals "cached, won't re-run".
            border = QPen(QColor(Settings.ACCENT_WHITE))
            border.setWidthF(scaled(1.6))
            border.setStyle(Qt.PenStyle.DashLine)
        elif self._highlight:
            border = QPen(QColor(Settings.ACCENT_GOLD))
            border.setWidthF(scaled(2))
        elif selected:
            border = QPen(QColor(self._accent))
            border.setWidthF(scaled(2))
        elif self._is_shape():
            c = QColor(self._accent)
            if dim:
                c.setAlpha(120)
            border = QPen(c)
            border.setWidthF(scaled(2))
        else:
            border = QPen(QColor(Settings.BORDER_COLOR))
            border.setWidthF(scaled(1))
        painter.setPen(border)
        painter.drawPath(body_path)

        # Title — hidden while the inline name editor is open. Shape nodes center
        # the label; the triangle places it in the wider lower half.
        if self._editor is not None:
            return
        font = QFont()
        font.setPointSizeF(scaled_pt(max(7.5, 9.0)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(Settings.FG_PRIMARY if not dim else Settings.FG_SECONDARY))
        if self._shape is ShapeKind.TRIANGLE:
            title_rect = QRectF(self._w * 0.14, h * 0.52,
                                self._w * 0.72, h * 0.44)
            align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        elif self._shape is ShapeKind.HEXAGON:
            title_rect = QRectF(self._w * 0.22, 0, self._w * 0.56, h)
            align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        elif self._shape is ShapeKind.GEM:
            # Center the label over the girdle band where the gem is widest.
            title_rect = QRectF(self._w * 0.16, 0, self._w * 0.68, h)
            align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        else:
            title_rect = QRectF(self._lip_w + scaled(10), 0,
                                self._w - self._lip_w - scaled(74), h)
            align = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        metrics = QFontMetrics(font)
        text = metrics.elidedText(self.node.title, Qt.TextElideMode.ElideRight,
                                  int(title_rect.width()))
        painter.drawText(title_rect, align, text)

        # Port labels (small, subdued) for action nodes with named ports.
        if self.node.role is NodeRole.ACTION:
            return

    def _paint_pill(self, painter: QPainter) -> None:
        """Compact channel-source capsule, tinted its channel color (V1.48)."""
        h = self._height()
        body = self._body_path()
        dim = not self.node.enabled
        accent = QColor(self._accent)
        fill = QColor(accent)
        fill.setAlpha(40 if dim else 78)
        painter.fillPath(body, fill)
        if self._previewed or self._highlight:
            pen = QPen(QColor(Settings.ACCENT_GOLD))
            pen.setWidthF(scaled(2))
        else:
            pen = QPen(accent)
            pen.setWidthF(scaled(1.6) if not self.isSelected() else scaled(2.2))
        painter.setPen(pen)
        painter.drawPath(body)
        font = QFont()
        font.setPointSizeF(scaled_pt(8.5))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(Settings.FG_PRIMARY if not dim else Settings.FG_SECONDARY))
        # Leave room on the right for the channel output port dot.
        text_rect = QRectF(scaled(10), 0, self._w - scaled(26), h)
        metrics = QFontMetrics(font)
        text = metrics.elidedText(self.node.title, Qt.TextElideMode.ElideRight,
                                  int(text_rect.width()))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter
                         | Qt.AlignmentFlag.AlignLeft, text)

    def _paint_gem(self, painter: QPainter, body_path: QPainterPath,
                   dim: bool, h: float) -> None:
        """Paint the Prism node as a faceted 2.5D crystal (V1.77).

        A dark base, then eight accent facets meeting at the center — brighter on the
        upper-right (light from that corner), darker on the lower-left — a diagonal
        gloss sheen, and thin facet seams. Vertices reuse the ``_body_path`` GEM outline
        proportions so facets align exactly with the silhouette. Honors ``dim`` (Run
        shading / disabled) exactly as the triangle/hexagon tint does."""
        w = self._w
        painter.fillPath(body_path, QColor(Settings.BG_TERTIARY))
        accent = QColor(self._accent)
        tx = w * self._GEM_TABLE_INSET
        sh = h * self._GEM_SHOULDER
        # Outline vertices …
        TL, TR = QPointF(tx, 0.0), QPointF(w - tx, 0.0)
        RU, RL = QPointF(w, sh), QPointF(w, h - sh)
        BR, BL = QPointF(w - tx, h), QPointF(tx, h)
        LL, LU = QPointF(0.0, h - sh), QPointF(0.0, sh)
        # … and the internal anchors (table/culet mids, girdle mids, center).
        TM, BM = QPointF(w / 2.0, 0.0), QPointF(w / 2.0, h)
        LM, RM = QPointF(0.0, h / 2.0), QPointF(w, h / 2.0)
        C = QPointF(w / 2.0, h / 2.0)
        base_alpha = 70 if dim else 165

        def facet(pts, shade: int) -> None:
            c = (QColor(accent).lighter(shade) if shade >= 100
                 else QColor(accent).darker(200 - shade))
            c.setAlpha(base_alpha)
            fp = QPainterPath(pts[0])
            for q in pts[1:]:
                fp.lineTo(q)
            fp.closeSubpath()
            painter.fillPath(fp, c)

        # Light from the upper-right: right/top facets brighter, lower-left darkest.
        facet([TL, TM, C, LU], 108)   # crown left
        facet([TM, TR, RU, C], 162)   # crown right (brightest)
        facet([LU, C, LM], 88)        # upper-left
        facet([RU, RM, C], 138)       # upper-right
        facet([LM, C, LL], 80)        # lower-left (darkest)
        facet([RM, RL, C], 118)       # lower-right
        facet([LL, BL, BM, C], 96)    # pavilion left
        facet([C, BM, BR, RL], 126)   # pavilion right

        # Diagonal gloss sheen (top-left highlight → bottom-right shadow).
        g = QLinearGradient(0.0, 0.0, w, h)
        g.setColorAt(0.0, QColor(255, 255, 255, 12 if dim else 30))
        g.setColorAt(0.5, QColor(255, 255, 255, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, 14 if dim else 34))
        painter.fillPath(body_path, g)

        # Facet seams — the center vertical, the girdle, and the four girdle diagonals.
        seam = QPen(QColor(Settings.BG_PRIMARY))
        seam.setWidthF(scaled(1.0))
        painter.setPen(seam)
        for a, b in ((TM, BM), (LM, RM), (LU, C), (RU, C), (LL, C), (RL, C)):
            painter.drawLine(a, b)

    # ── interaction ───────────────────────────────────────────────────────
    def itemChange(self, change, value):  # noqa: N802 (Qt naming)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node.pos = (self.pos().x(), self.pos().y())
            scene = self.scene()
            fn = getattr(scene, "_on_node_geometry_changed", None)
            if callable(fn):
                fn(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_button_visibility()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self._update_button_visibility()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self._update_button_visibility()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # V1.61: renaming moved to double-click (see mouseDoubleClickEvent); a
        # plain press just selects / drags the node.
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        # V1.61 gesture policy: double-clicking an INPUT / OUTPUT node opens the
        # inline rename editor (Enter / click-away commits, Esc cancels) instead
        # of promoting it to the previewed node. ACTION nodes keep promoting to
        # the previewed node; channel pills are inert.
        if self._is_renamable():
            self._begin_name_edit()
            event.accept()
            return
        # Channel-source pills are not previewable — swallow the double-click.
        if getattr(self.node, "category", None) is NodeCategory.CHANNEL:
            event.accept()
            return
        scene = self.scene()
        fn = getattr(scene, "_emit_double_click", None)
        if callable(fn):
            fn(self.node.id)
        super().mouseDoubleClickEvent(event)

    # ── in-place name editing (INPUT / OUTPUT nodes) ────────────────────────
    def _is_renamable(self) -> bool:
        # V1.61: input nodes (one per loaded file) are renamable too, not just
        # output/bridge nodes.
        return self.node.role in (NodeRole.INPUT, NodeRole.OUTPUT)

    def _name_hit(self, pos: QPointF) -> bool:
        """True if ``pos`` (item coords) is on the node's name area (the body
        right of the lip)."""
        return (self._lip_w <= pos.x() <= self._w
                and 0.0 <= pos.y() <= self._height())

    def _name_edit_rect(self) -> QRectF:
        eh = scaled(20)
        x = self._lip_w + scaled(6)
        return QRectF(x, (self._height() - eh) / 2,
                      self._w - self._lip_w - scaled(12), eh)

    def _begin_name_edit(self) -> None:
        scene = self.scene()
        if self._editor is not None or scene is None:
            return
        self._edit_cancelled = False
        edit = _NameLineEdit(self.node.title)
        edit.setObjectName("nodeNameEdit")
        edit.setStyleSheet(scale_qss(
            f"background:{Settings.BG_SECONDARY}; color:{Settings.FG_PRIMARY};"
            f"border:1px solid {Settings.ACCENT_GOLD}; border-radius:4px;"
            "padding:1px 4px;"
            f"selection-background-color:{Settings.BG_HOVER};"
        ))
        edit.selectAll()
        edit.editingFinished.connect(self._commit_name_edit)
        edit.escaped.connect(self._cancel_name_edit)

        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(edit)
        proxy.setZValue(5)
        proxy.setGeometry(self._name_edit_rect())
        self._editor = proxy
        self._editor_widget = edit
        self.update()  # hide the painted title under the editor
        edit.setFocus(Qt.FocusReason.MouseFocusReason)

    def _commit_name_edit(self) -> None:
        if self._editor is None or self._edit_cancelled:
            return
        new_title = (self._editor_widget.text() if self._editor_widget else "").strip()
        self._end_name_edit()
        if new_title and new_title != self.node.title:
            scene = self.scene()
            fn = getattr(scene, "rename_node", None)
            if callable(fn):
                fn(self.node.id, new_title)

    def _cancel_name_edit(self) -> None:
        self._edit_cancelled = True
        self._end_name_edit()

    def _end_name_edit(self) -> None:
        proxy = self._editor
        self._editor = None
        self._editor_widget = None
        if proxy is not None:
            scene = self.scene()
            if scene is not None:
                scene.removeItem(proxy)
            proxy.deleteLater()
        self.update()
