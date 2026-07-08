"""``QGraphicsScene`` driving one stage's node graph (V1.45).

One :class:`NodeScene` wraps one :class:`GraphSlice`. It builds the node/edge
items, mediates type-checked drag-to-connect (DAG enforced, one wire per input,
fan-out on outputs), and exposes the Disconnect / Duplicate / Delete operations
the node corner buttons call. Structural and parameter changes emit
:attr:`graph_changed` so the page can mark the preview dirty.

Switching sub-tabs swaps the scene in the single ``QGraphicsView`` (the page
owns one scene per :class:`Stage`).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsScene, QMenu

from nd2studios.widgets.icon_button import scaled

from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.executor import edge_channels
from nd2studios.pipeline_graph.loop import default_loop_config
from nd2studios.pipeline_graph.model import (
    Edge, GraphSlice, LOOP_KIND, Node, NodeCategory, NodeRole, Port, PortType,
    can_connect, can_connect_loop, clone_node, new_id, would_create_cycle,
)
from nd2studios.pipeline_graph.registry_adapter import (
    RAINBOW_IN_NAME, RAINBOW_OUT_NAME, SPECIAL_DISMISS_OP_KEY, NodeSpec,
    build_node, channel_name_for_op_key, spec_takes_channels,
)
from nd2studios.widgets.node_board.edge_item import EdgeItem
from nd2studios.widgets.node_board.node_item import NodeItem
from nd2studios.widgets.node_board.port_item import PortItem


# Per-category node color for the merged Analysis scene (V1.45). The scene's
# own ``accent`` is the per-sub-tab fallback (e.g. Processing cyan).
_CATEGORY_COLOR = {
    NodeCategory.PROCESSING: Settings.ACCENT_CYAN,
    NodeCategory.ANALYSIS: Settings.ACCENT_PINK,
    NodeCategory.RESULTS: Settings.ACCENT_GREEN,
    NodeCategory.LOGIC: Settings.ACCENT_PURPLE,
    NodeCategory.SPECIAL: Settings.ACCENT_ORANGE,
    NodeCategory.CHECKPOINT: Settings.ACCENT_WHITE,  # V1.53 freeze/cache node
    # Channel pills get their per-channel color at refresh; this is the fallback.
    NodeCategory.CHANNEL: Settings.FG_SECONDARY,
}


class NodeScene(QGraphicsScene):
    selection_changed = Signal(str)     # node_id, or "" when not exactly one
    node_double_clicked = Signal(str)   # node_id
    graph_changed = Signal()
    node_created = Signal(str)          # node_id
    node_renamed = Signal(str, str)     # node_id, new title
    loop_edge_edit_requested = Signal(str)  # V1.49: loop edge_id to configure

    def __init__(
        self,
        graph_slice: GraphSlice,
        accent_hex: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.slice = graph_slice
        self.accent = accent_hex
        self.action_specs: List[NodeSpec] = []
        self.output_spec: Optional[NodeSpec] = None
        self.allow_add = True
        # Called with a freshly-created OUTPUT Node so the page can register a
        # bridge + auto-name it. Optional.
        self.on_output_created: Optional[Callable[[Node], None]] = None

        self._node_items: Dict[str, NodeItem] = {}
        self._edge_items: Dict[str, EdgeItem] = {}
        self._port_index: Dict[str, PortItem] = {}
        self._drag_from: Optional[PortItem] = None
        self._temp_edge: Optional[EdgeItem] = None
        # V1.48 scissors: when armed, a left drag draws a cut stroke and any
        # crossed wire (structural OR channel) is removed; a click cuts the wire
        # under the cursor.
        self._cut_mode = False
        self._cut_start: Optional[QPointF] = None
        self._cut_item = None
        # V1.49 loop mode: when armed, a drag from a node's bottom output to the
        # top input of the same or an upstream node creates a *loop* back-edge.
        self._loop_mode = False
        # V1.48 channel-flow context (pushed by the page): live channel names +
        # their display colors, used to color channel wires / propagation strands
        # / channel-source pills.
        self._all_names: List[str] = []
        self._channel_colors: Dict[str, str] = {}

        self.setBackgroundBrush(QColor(Settings.BG_PRIMARY))
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.selectionChanged.connect(self._emit_selection)
        self._rebuild_from_slice()

    # ── (re)build from model ──────────────────────────────────────────────
    def _rebuild_from_slice(self) -> None:
        self.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._port_index.clear()
        for node in self.slice.nodes.values():
            self._add_node_item(node)
        for edge in self.slice.edges.values():
            self._add_edge_item(edge)

    def _color_for_node(self, node: Node) -> str:
        """Per-category node color; Dismiss is red; falls back to scene accent."""
        if node.op_key == SPECIAL_DISMISS_OP_KEY:
            return Settings.ACCENT_RED
        return _CATEGORY_COLOR.get(getattr(node, "category", None), self.accent)

    def _add_node_item(self, node: Node) -> NodeItem:
        item = NodeItem(node, self._color_for_node(node))
        self.addItem(item)
        self._node_items[node.id] = item
        for pid, pit in item.port_items.items():
            self._port_index[pid] = pit
        return item

    def _add_edge_item(self, edge: Edge) -> EdgeItem:
        src = self._port_index.get(edge.src_port)
        color = src.color_hex() if src is not None else Settings.FG_PRIMARY
        item = EdgeItem(edge.id, color)
        if edge.kind == LOOP_KIND:  # V1.49: amber left-gutter routing + arrow
            item.set_loop_edge(True)
        self.addItem(item)
        self._edge_items[edge.id] = item
        self._refresh_edge_item(item, edge)
        return item

    def _refresh_edge_item(self, item: EdgeItem, edge: Edge) -> None:
        src = self._port_index.get(edge.src_port)
        dst = self._port_index.get(edge.dst_port)
        if src is not None and dst is not None:
            item.set_endpoints(src.center_scene(), dst.center_scene())

    # ── public node ops (called by node corner buttons / page) ────────────
    def add_node_from_spec(self, spec: NodeSpec, pos: tuple) -> Node:
        node = build_node(spec, pos=pos)
        self.slice.add_node(node)
        self._add_node_item(node)
        if node.role is NodeRole.OUTPUT and self.on_output_created is not None:
            self.on_output_created(node)
        self.node_created.emit(node.id)
        self.graph_changed.emit()
        return node

    def delete_node(self, node_id: str) -> None:
        touching = [
            e.id for e in self.slice.edges.values()
            if e.src_node == node_id or e.dst_node == node_id
        ]
        for eid in touching:
            self._drop_edge_item(eid)
        self.slice.remove_node(node_id)
        item = self._node_items.pop(node_id, None)
        if item is not None:
            for port in item.node.inputs + item.node.outputs:
                self._port_index.pop(port.id, None)
            self.removeItem(item)
        self.graph_changed.emit()

    def disconnect_node(self, node_id: str) -> None:
        touching = [
            e.id for e in self.slice.edges.values()
            if e.src_node == node_id or e.dst_node == node_id
        ]
        for eid in touching:
            self._remove_edge(eid)
        if touching:
            self.graph_changed.emit()

    def duplicate_node(self, node_id: str) -> None:
        src = self.slice.nodes.get(node_id)
        if src is None:
            return
        offset = (src.pos[0] + 28, src.pos[1] + 28)
        node = clone_node(src, offset)
        self.slice.add_node(node)
        self._add_node_item(node)
        if node.role is NodeRole.OUTPUT and self.on_output_created is not None:
            self.on_output_created(node)
        self.node_created.emit(node.id)
        self.graph_changed.emit()

    def rename_node(self, node_id: str, title: str) -> None:
        node = self.slice.nodes.get(node_id)
        if node is None:
            return
        node.title = title
        item = self._node_items.get(node_id)
        if item is not None:
            item.update()
        # A title change doesn't alter the recipe/topology, so emit a dedicated
        # signal (not graph_changed) — no preview recompute is needed; the page
        # uses it to keep the node's bridge name in sync.
        self.node_renamed.emit(node_id, title)

    # ── loop connector (V1.49) ────────────────────────────────────────────
    def set_loop_mode(self, on: bool) -> None:
        """Arm / disarm loop-wire mode. While armed, a drag from a bottom output
        to a top input (same node or an upstream node) creates a loop back-edge."""
        self._loop_mode = bool(on)

    def edit_loop_edge(self, edge_id: str) -> None:
        """Request the page open the Loop Settings dialog for this loop edge."""
        edge = self.slice.edges.get(edge_id)
        if edge is not None and edge.kind == LOOP_KIND:
            self.loop_edge_edit_requested.emit(edge_id)

    def _has_loop_edge(self, src_node: str, dst_node: str) -> bool:
        return any(e.kind == LOOP_KIND and e.src_node == src_node
                   and e.dst_node == dst_node for e in self.slice.edges.values())

    def _try_connect_loop(self, out_item: PortItem, in_item: PortItem) -> None:
        """Create a loop back-edge (out=bottom, in=top). Unlike a structural wire
        this permits a self-loop and skips the cycle check (loops are intentionally
        cyclic); it coexists with the input's structural feed."""
        out_node = out_item.node_item.node
        in_node = in_item.node_item.node
        if not can_connect_loop(out_item.port, in_item.port):
            in_item.flash_reject()
            return
        if self._has_loop_edge(out_node.id, in_node.id):
            in_item.flash_reject()  # already looped this pair
            return
        edge = Edge(
            id=new_id("loop"),
            src_node=out_node.id, src_port=out_item.port.id,
            dst_node=in_node.id, dst_port=in_item.port.id,
            kind=LOOP_KIND, params=default_loop_config(),
        )
        self.slice.add_edge(edge)
        self._add_edge_item(edge)
        self.graph_changed.emit()
        # Open the settings dialog straight away so the loop is configured on
        # creation (double-clicking it later reopens the same dialog).
        self.loop_edge_edit_requested.emit(edge.id)

    # ── edge helpers ──────────────────────────────────────────────────────
    def disconnect_edge(self, edge_id: str) -> None:
        """Remove a single wire (the EdgeItem double-click handler calls this)."""
        if edge_id not in self._edge_items and edge_id not in self.slice.edges:
            return
        self._remove_edge(edge_id)
        self.graph_changed.emit()

    def _remove_edge(self, edge_id: str) -> None:
        edge = self.slice.edges.get(edge_id)
        dst_node = edge.dst_node if edge is not None else None
        self.slice.remove_edge(edge_id)
        self._drop_edge_item(edge_id)
        # V1.48: freeing a rainbow input may leave a surplus free port — trim it.
        if dst_node is not None:
            node = self.slice.nodes.get(dst_node)
            if node is not None and any(p.type is PortType.CHANNEL
                                        for p in node.inputs):
                self._sync_rainbow_ports(dst_node)

    def _drop_edge_item(self, edge_id: str) -> None:
        item = self._edge_items.pop(edge_id, None)
        if item is not None:
            self.removeItem(item)

    def _on_node_geometry_changed(self, node_item: NodeItem) -> None:
        nid = node_item.node.id
        for eid, item in self._edge_items.items():
            edge = self.slice.edges.get(eid)
            if edge and (edge.src_node == nid or edge.dst_node == nid):
                self._refresh_edge_item(item, edge)

    # ── channel-flow layer (V1.48) ────────────────────────────────────────
    def set_channel_context(self, all_names: List[str],
                            channel_colors: Dict[str, str]) -> None:
        """Push the live channel names + display colors, then recolor channel
        wires / propagation strands / channel-source pills."""
        self._all_names = list(all_names or [])
        self._channel_colors = dict(channel_colors or {})
        self.ensure_rainbow_ports()
        self.refresh_channel_visuals()

    def ensure_rainbow_ports(self) -> None:
        """Guarantee every process node has rainbow channel ports + exactly one
        free rainbow input (e.g. after loading a saved graph). Migrates process
        nodes saved before V1.48 (no rainbow ports) by adding the initial pair."""
        for nid, node in list(self.slice.nodes.items()):
            has_rainbow = any(p.type is PortType.CHANNEL for p in node.inputs)
            if not has_rainbow and spec_takes_channels(
                    node.op_key, node.role, [p.type for p in node.inputs]):
                node.inputs.append(
                    Port(new_id("p"), RAINBOW_IN_NAME, PortType.CHANNEL, True))
                node.outputs.append(
                    Port(new_id("p"), RAINBOW_OUT_NAME, PortType.CHANNEL, False))
                self._rebuild_node_ports(nid)
            self._sync_rainbow_ports(nid)

    def _sync_rainbow_ports(self, node_id: str) -> None:
        """Keep exactly one FREE (unwired) rainbow input port on a process node:
        add one when the last free port gets wired; trim surplus free ones on
        disconnect. Rebuilds the node's ports when the set changes (V1.48)."""
        node = self.slice.nodes.get(node_id)
        if node is None:
            return
        rainbow = [p for p in node.inputs if p.type is PortType.CHANNEL]
        if not rainbow:
            return
        wired_ids = {e.dst_port for e in self.slice.edges.values()}
        free = [p for p in rainbow if p.id not in wired_ids]
        changed = False
        if not free:
            node.inputs.append(
                Port(new_id("p"), RAINBOW_IN_NAME, PortType.CHANNEL, True))
            changed = True
        else:
            for victim in free[1:]:  # keep exactly one free
                node.inputs.remove(victim)
                changed = True
        if changed:
            self._rebuild_node_ports(node_id)

    def _rebuild_node_ports(self, node_id: str) -> None:
        item = self._node_items.get(node_id)
        if item is None:
            return
        for pid in list(item.port_items.keys()):
            self._port_index.pop(pid, None)
        item.rebuild_ports()
        for pid, pit in item.port_items.items():
            self._port_index[pid] = pit
        self._on_node_geometry_changed(item)
        self.refresh_channel_visuals()

    def refresh_channel_visuals(self) -> None:
        """Recolor channel wires (channel-source color / rainbow), draw the
        per-channel propagation strands on structural wires, and tint the
        channel-source pills + their output ports (V1.48)."""
        ec = edge_channels(self.slice, self._all_names) if self._all_names else {}
        for eid, item in self._edge_items.items():
            edge = self.slice.edges.get(eid)
            if edge is None:
                continue
            src_port = self.slice.find_port(edge.src_port)
            is_chan = src_port is not None and src_port.type is PortType.CHANNEL
            chans = ec.get(eid, [])
            colors = [self._channel_colors.get(c) or Settings.FG_SECONDARY
                      for c in chans]
            item.set_channel_edge(is_chan)
            if is_chan and len(colors) == 1:
                item.set_color(colors[0])
                item.set_channel_strands([])
            else:
                item.set_channel_strands(colors)
        for nid, item in self._node_items.items():
            node = self.slice.nodes.get(nid)
            if node is None or getattr(node, "category", None) is not NodeCategory.CHANNEL:
                continue
            nm = channel_name_for_op_key(node.op_key)
            col = self._channel_colors.get(nm) if nm else None
            item.set_accent(col or Settings.FG_SECONDARY)
            for p in node.outputs:
                pit = self._port_index.get(p.id)
                if pit is not None:
                    pit.set_display_color(col or "")  # "" → rainbow (the All node)

    # ── selection / double-click ──────────────────────────────────────────
    def _emit_selection(self) -> None:
        nodes = [it for it in self.selectedItems() if isinstance(it, NodeItem)]
        self.selection_changed.emit(nodes[0].node.id if len(nodes) == 1 else "")

    def _emit_double_click(self, node_id: str) -> None:
        self.node_double_clicked.emit(node_id)

    def node_item(self, node_id: str) -> Optional[NodeItem]:
        return self._node_items.get(node_id)

    # ── preview highlight (golden) ────────────────────────────────────────
    def set_highlight(self, node_ids, edge_ids, preview_id: str = "") -> None:
        """Paint the previewed node + its connected nodes/wires gold.

        ``node_ids`` / ``edge_ids`` are the connected set; ``preview_id`` is the
        single node whose result is in the viewer (drawn with a thicker outline).
        Pass empty collections to clear.
        """
        node_ids = set(node_ids or ())
        edge_ids = set(edge_ids or ())
        for nid, item in self._node_items.items():
            item.set_highlighted(nid in node_ids, previewed=(nid == preview_id))
        for eid, item in self._edge_items.items():
            item.set_highlighted(eid in edge_ids)

    # ── Run state (shaded / current / done) ───────────────────────────────
    def set_run_states(self, states: Dict[str, str]) -> None:
        """Paint Run progress: ``states`` maps node_id → ``"shaded"`` /
        ``"current"`` / ``"done"``; any node absent is reset to normal."""
        states = states or {}
        for nid, item in self._node_items.items():
            item.set_run_state(states.get(nid, ""))

    def clear_run_states(self) -> None:
        """Drop all Run shading — back to editor mode (all nodes normal)."""
        for item in self._node_items.values():
            item.set_run_state("")

    # ── connection drag ───────────────────────────────────────────────────
    def _port_at(self, scene_pos: QPointF) -> Optional[PortItem]:
        for it in self.items(scene_pos):
            if isinstance(it, PortItem):
                return it
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._cut_mode and event.button() == Qt.MouseButton.LeftButton:
            self._begin_cut(event.scenePos())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            port = self._port_at(event.scenePos())
            if port is not None and not port.port.is_input:
                self._begin_connection(port)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._cut_mode and self._cut_start is not None:
            if self._cut_item is not None:
                self._cut_item.setLine(
                    self._cut_start.x(), self._cut_start.y(),
                    event.scenePos().x(), event.scenePos().y())
            event.accept()
            return
        if self._drag_from is not None and self._temp_edge is not None:
            self._temp_edge.set_endpoints(
                self._drag_from.center_scene(), event.scenePos()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._cut_mode and self._cut_start is not None:
            self._perform_cut(self._cut_start, event.scenePos())
            self._end_cut()
            event.accept()
            return
        if self._drag_from is not None:
            target = self._port_at(event.scenePos())
            self._finish_connection(target)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── scissors / cut mode (V1.48) ────────────────────────────────────────
    def set_cut_mode(self, on: bool) -> None:
        """Arm / disarm the wire cutter. While armed, a left drag cuts every
        crossed wire and a click cuts the wire under the cursor."""
        self._cut_mode = bool(on)
        if not on:
            self._end_cut()

    def _begin_cut(self, pos: QPointF) -> None:
        self._cut_start = pos
        pen = QPen(QColor(Settings.ACCENT_RED))
        pen.setWidthF(scaled(1.6))
        pen.setStyle(Qt.PenStyle.DashLine)
        self._cut_item = self.addLine(pos.x(), pos.y(), pos.x(), pos.y(), pen)
        self._cut_item.setZValue(50)

    def _end_cut(self) -> None:
        if self._cut_item is not None:
            self.removeItem(self._cut_item)
            self._cut_item = None
        self._cut_start = None

    def _perform_cut(self, a: QPointF, b: QPointF) -> None:
        """Remove every wire crossed by the a→b stroke (or, for a click, the wire
        under the cursor). Works for structural and channel wires alike."""
        moved = (abs(a.x() - b.x()) + abs(a.y() - b.y())) > scaled(3)
        to_cut: List[str] = []
        if moved:
            cut = QPainterPath(a)
            cut.lineTo(b)
            for eid, item in self._edge_items.items():
                try:
                    if item.shape().intersects(cut):
                        to_cut.append(eid)
                except Exception:  # noqa: BLE001
                    continue
        else:
            for it in self.items(a):
                if isinstance(it, EdgeItem):
                    to_cut.append(it.edge_id)
        changed = False
        for eid in to_cut:
            if eid in self._edge_items or eid in self.slice.edges:
                self._remove_edge(eid)
                changed = True
        if changed:
            self.graph_changed.emit()

    def _begin_connection(self, port_item: PortItem) -> None:
        self._drag_from = port_item
        self._temp_edge = EdgeItem("__temp__", port_item.color_hex())
        if self._loop_mode:  # V1.49: preview the loop routing while dragging
            self._temp_edge.set_loop_edge(True)
        self.addItem(self._temp_edge)
        self._temp_edge.set_endpoints(
            port_item.center_scene(), port_item.center_scene()
        )

    def _finish_connection(self, target: Optional[PortItem]) -> None:
        src = self._drag_from
        if self._temp_edge is not None:
            self.removeItem(self._temp_edge)
            self._temp_edge = None
        self._drag_from = None
        if src is None or target is None or target is src:
            return
        if not target.port.is_input:
            target.flash_reject()
            return
        self._try_connect(src, target)

    def _try_connect(self, out_item: PortItem, in_item: PortItem) -> None:
        if self._loop_mode:  # V1.49: loop-wire drag creates a back-edge instead
            self._try_connect_loop(out_item, in_item)
            return
        out_node = out_item.node_item.node
        in_node = in_item.node_item.node
        if out_node.id == in_node.id:
            return
        if not can_connect(out_item.port, in_item.port):
            in_item.flash_reject()
            return
        if would_create_cycle(self.slice, out_node.id, in_node.id):
            in_item.flash_reject()
            return
        # One wire per input port — drop any existing wire into this input.
        existing = self.slice.edge_into_port(in_node.id, in_item.port.id)
        if existing is not None:
            self._remove_edge(existing.id)
        edge = Edge(
            id=new_id("e"),
            src_node=out_node.id, src_port=out_item.port.id,
            dst_node=in_node.id, dst_port=in_item.port.id,
        )
        self.slice.add_edge(edge)
        self._add_edge_item(edge)
        # V1.48: wiring a channel into a rainbow input spawns a fresh free rainbow
        # port for the next channel (so there's always one free to grab).
        if in_item.port.type is PortType.CHANNEL:
            self._sync_rainbow_ports(in_node.id)
        self.graph_changed.emit()

    # ── context menu (add nodes) ──────────────────────────────────────────
    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if not self.allow_add or self._port_at(event.scenePos()) is not None:
            super().contextMenuEvent(event)
            return
        # Don't show the add-menu over an existing node.
        for it in self.items(event.scenePos()):
            if isinstance(it, NodeItem):
                super().contextMenuEvent(event)
                return

        parent = self.views()[0] if self.views() else None
        menu = QMenu(parent)
        pos = event.scenePos()
        scene_pos = (pos.x(), pos.y())

        if self.action_specs:
            ops_menu = menu.addMenu("Add operation")
            for spec in self.action_specs:
                act = ops_menu.addAction(spec.title)
                act.setToolTip(spec.description)
                act.triggered.connect(
                    lambda _checked=False, s=spec, p=scene_pos:
                    self.add_node_from_spec(s, p)
                )
        if self.output_spec is not None:
            menu.addSeparator()
            out_act = menu.addAction("Add output node")
            out_act.triggered.connect(
                lambda _checked=False, p=scene_pos:
                self.add_node_from_spec(self.output_spec, p)
            )
        if menu.isEmpty():
            return
        menu.exec(event.screenPos())
