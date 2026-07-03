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
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsScene, QMenu

from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.model import (
    Edge, GraphSlice, Node, NodeCategory, NodeRole, can_connect, clone_node,
    new_id, would_create_cycle,
)
from nd2studios.pipeline_graph.registry_adapter import (
    SPECIAL_DISMISS_OP_KEY, NodeSpec, build_node,
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
}


class NodeScene(QGraphicsScene):
    selection_changed = Signal(str)     # node_id, or "" when not exactly one
    node_double_clicked = Signal(str)   # node_id
    graph_changed = Signal()
    node_created = Signal(str)          # node_id
    node_renamed = Signal(str, str)     # node_id, new title

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

    # ── edge helpers ──────────────────────────────────────────────────────
    def disconnect_edge(self, edge_id: str) -> None:
        """Remove a single wire (the EdgeItem double-click handler calls this)."""
        if edge_id not in self._edge_items and edge_id not in self.slice.edges:
            return
        self._remove_edge(edge_id)
        self.graph_changed.emit()

    def _remove_edge(self, edge_id: str) -> None:
        self.slice.remove_edge(edge_id)
        self._drop_edge_item(edge_id)

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
        if event.button() == Qt.MouseButton.LeftButton:
            port = self._port_at(event.scenePos())
            if port is not None and not port.port.is_input:
                self._begin_connection(port)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None and self._temp_edge is not None:
            self._temp_edge.set_endpoints(
                self._drag_from.center_scene(), event.scenePos()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None:
            target = self._port_at(event.scenePos())
            self._finish_connection(target)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _begin_connection(self, port_item: PortItem) -> None:
        self._drag_from = port_item
        self._temp_edge = EdgeItem("__temp__", port_item.color_hex())
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
