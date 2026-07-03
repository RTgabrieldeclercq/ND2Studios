"""Qt-free node-graph data model for the Pipelines tab (V1.45).

This module is the source of truth for the GA3-style pipeline graph. It is
**pure**: no PySide6, no app singletons — only dataclasses, enums, and small
graph algorithms. That keeps it headless-testable and lets the executor and
serializer reuse it without dragging the GUI in (mirrors the backend-purity
rule from CLAUDE.md, extended to the graph core).

A :class:`PipelineDoc` holds three :class:`GraphSlice` objects — one per
:class:`Stage` (Processing / Analysis / Results) — plus the :class:`Bridge`
records that connect a producer stage's output to the next stage's input. Each
slice is a directed acyclic graph of :class:`Node` connected by :class:`Edge`.

Conventions (CLAUDE.md): ``snake_case`` funcs, ``CamelCase`` classes,
``from __future__ import annotations``, type hints on public functions.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── enums ──────────────────────────────────────────────────────────────────

class PortType(Enum):
    """The four GA3 data flavors. An input port only accepts its own type."""

    IMAGE = "image"    # Dict[str, np.ndarray] (T, H, W) per channel
    BINARY = "binary"  # Dict[str, np.ndarray] (T, H, W) int32 label masks
    DATA = "data"      # List[Dict[str, Any]] measurement rows
    VALUE = "value"    # scalar / small param
    ANY = "any"        # wildcard — logic/special nodes accept & pass any payload


class Stage(Enum):
    PROCESSING = "processing"  # Recipe-page enhancement plugins
    ANALYSIS = "analysis"      # AnalysisPipeline registry
    RESULTS = "results"        # results_engine outputs


class NodeRole(Enum):
    INPUT = "input"    # source (loaded channels / upstream bridge)
    ACTION = "action"  # one operation (plugin / pipeline / results op)
    OUTPUT = "output"  # sink (a bridge to the next stage)


class NodeCategory(Enum):
    """What a node *is*, for coloring + Add-dialog grouping (V1.45 merge).

    In the merged Analysis sub-tab a single scene holds nodes of several
    categories at once, each painted its own color (analysis = pink,
    results = green, logic = purple, special = orange). Independent of
    :class:`Stage` (which still drives execution semantics).
    """

    PROCESSING = "processing"
    ANALYSIS = "analysis"
    RESULTS = "results"
    LOGIC = "logic"      # if-else branch nodes
    SPECIAL = "special"  # interactive / action nodes (validate, export, …)


class ShapeKind(Enum):
    """The body silhouette a node is drawn with."""

    RECT = "rect"          # default rounded rectangle
    TRIANGLE = "triangle"  # upright triangle (if-else: 1 in top, 2 out bottom)
    HEXAGON = "hexagon"    # special action nodes


_STAGE_CATEGORY = {
    Stage.PROCESSING: NodeCategory.PROCESSING,
    Stage.ANALYSIS: NodeCategory.ANALYSIS,
    Stage.RESULTS: NodeCategory.RESULTS,
}


def category_for_stage(stage: Stage) -> "NodeCategory":
    """Default category for a node, derived from its stage."""
    return _STAGE_CATEGORY.get(stage, NodeCategory.ANALYSIS)


def new_id(prefix: str = "n") -> str:
    """Short unique id, e.g. ``"node-1a2b3c4d"``."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ── dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Port:
    id: str
    name: str
    type: PortType
    is_input: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "is_input": self.is_input,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Port":
        return cls(
            id=d["id"],
            name=d["name"],
            type=PortType(d["type"]),
            is_input=bool(d["is_input"]),
        )


@dataclass
class Node:
    id: str
    stage: Stage
    role: NodeRole
    op_key: str                       # registry key (see registry_adapter)
    title: str
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    pos: Tuple[float, float] = (0.0, 0.0)
    enabled: bool = True
    bridge_id: Optional[str] = None   # set on INPUT / OUTPUT nodes
    category: Optional[NodeCategory] = None  # color/grouping; defaults from stage
    shape_kind: ShapeKind = ShapeKind.RECT

    def __post_init__(self) -> None:
        if self.category is None:
            self.category = category_for_stage(self.stage)

    def input_port(self) -> Optional[Port]:
        """First input port, or ``None`` for source nodes."""
        return self.inputs[0] if self.inputs else None

    def output_port(self) -> Optional[Port]:
        """First output port, or ``None`` for sink nodes."""
        return self.outputs[0] if self.outputs else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage.value,
            "role": self.role.value,
            "op_key": self.op_key,
            "title": self.title,
            "params": dict(self.params),
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "enabled": bool(self.enabled),
            "bridge_id": self.bridge_id,
            "category": self.category.value if self.category else None,
            "shape_kind": self.shape_kind.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        pos = d.get("pos", [0.0, 0.0])
        cat = d.get("category")
        return cls(
            id=d["id"],
            stage=Stage(d["stage"]),
            role=NodeRole(d["role"]),
            op_key=d["op_key"],
            title=d["title"],
            params=dict(d.get("params", {})),
            inputs=[Port.from_dict(p) for p in d.get("inputs", [])],
            outputs=[Port.from_dict(p) for p in d.get("outputs", [])],
            pos=(float(pos[0]), float(pos[1])),
            enabled=bool(d.get("enabled", True)),
            bridge_id=d.get("bridge_id"),
            category=NodeCategory(cat) if cat else None,  # else from stage
            shape_kind=ShapeKind(d.get("shape_kind", "rect")),
        )


@dataclass
class Edge:
    id: str
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "src_node": self.src_node,
            "src_port": self.src_port,
            "dst_node": self.dst_node,
            "dst_port": self.dst_port,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        return cls(
            id=d["id"],
            src_node=d["src_node"],
            src_port=d["src_port"],
            dst_node=d["dst_node"],
            dst_port=d["dst_port"],
        )


@dataclass
class Bridge:
    """A handle shared by a producer stage's output and the next stage's input.

    Renaming either end updates :attr:`name` everywhere (and, for Results
    bridges, the Export source label). Default names auto-increment per stage
    (``Processing #1``, …) and are user-renamable.
    """

    id: str
    producer_stage: Stage
    consumer_stage: Optional[Stage]
    name: str
    payload_type: PortType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "producer_stage": self.producer_stage.value,
            "consumer_stage": (
                self.consumer_stage.value if self.consumer_stage else None
            ),
            "name": self.name,
            "payload_type": self.payload_type.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Bridge":
        cons = d.get("consumer_stage")
        return cls(
            id=d["id"],
            producer_stage=Stage(d["producer_stage"]),
            consumer_stage=Stage(cons) if cons else None,
            name=d["name"],
            payload_type=PortType(d["payload_type"]),
        )


@dataclass
class GraphSlice:
    """The node graph for a single :class:`Stage`."""

    stage: Stage
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)

    # ── mutation ─────────────────────────────────────────────────────────
    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and every edge touching it."""
        self.nodes.pop(node_id, None)
        for eid in [
            e.id for e in self.edges.values()
            if e.src_node == node_id or e.dst_node == node_id
        ]:
            self.edges.pop(eid, None)

    def add_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        return edge

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)

    def remove_edges_for_node(self, node_id: str) -> None:
        for eid in [
            e.id for e in self.edges.values()
            if e.src_node == node_id or e.dst_node == node_id
        ]:
            self.edges.pop(eid, None)

    # ── queries ──────────────────────────────────────────────────────────
    def incoming(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges.values() if e.dst_node == node_id]

    def outgoing(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges.values() if e.src_node == node_id]

    def edge_into_port(self, node_id: str, port_id: str) -> Optional[Edge]:
        for e in self.edges.values():
            if e.dst_node == node_id and e.dst_port == port_id:
                return e
        return None

    def port_owner(self, port_id: str) -> Optional[Node]:
        for node in self.nodes.values():
            if any(p.id == port_id for p in node.inputs + node.outputs):
                return node
        return None

    def find_port(self, port_id: str) -> Optional[Port]:
        for node in self.nodes.values():
            for p in node.inputs + node.outputs:
                if p.id == port_id:
                    return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphSlice":
        sl = cls(stage=Stage(d["stage"]))
        for nd in d.get("nodes", []):
            node = Node.from_dict(nd)
            sl.nodes[node.id] = node
        for ed in d.get("edges", []):
            edge = Edge.from_dict(ed)
            sl.edges[edge.id] = edge
        return sl


@dataclass
class PipelineDoc:
    processing: GraphSlice
    analysis: GraphSlice
    results: GraphSlice
    bridges: Dict[str, Bridge] = field(default_factory=dict)
    schema_version: int = 3  # V3 (V1.45): Track Objects node (was Validate)

    @classmethod
    def empty(cls) -> "PipelineDoc":
        return cls(
            processing=GraphSlice(Stage.PROCESSING),
            analysis=GraphSlice(Stage.ANALYSIS),
            results=GraphSlice(Stage.RESULTS),
        )

    def slice_for(self, stage: Stage) -> GraphSlice:
        return {
            Stage.PROCESSING: self.processing,
            Stage.ANALYSIS: self.analysis,
            Stage.RESULTS: self.results,
        }[stage]

    @property
    def merged(self) -> GraphSlice:
        """The merged Analysis+Results slice (V1.45 merge).

        Analysis and Results now share one editor scene; results nodes live in
        the analysis slice (keeping ``stage=RESULTS`` so they color green). This
        is an alias of :attr:`analysis`; the ``results`` slice is retained empty
        for back-compat and schema migration.
        """
        return self.analysis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "processing": self.processing.to_dict(),
            "analysis": self.analysis.to_dict(),
            "results": self.results.to_dict(),
            "bridges": [b.to_dict() for b in self.bridges.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineDoc":
        doc = cls(
            processing=GraphSlice.from_dict(d["processing"]),
            analysis=GraphSlice.from_dict(d["analysis"]),
            results=GraphSlice.from_dict(d["results"]),
            schema_version=int(d.get("schema_version", 1)),
        )
        for bd in d.get("bridges", []):
            br = Bridge.from_dict(bd)
            doc.bridges[br.id] = br
        return doc


# ── connection rules ───────────────────────────────────────────────────────

def can_connect(src: Port, dst: Port) -> bool:
    """True if a wire from output port ``src`` to input port ``dst`` is legal.

    Requires: ``src`` is an output, ``dst`` is an input, and the payload types
    match (``PortType.ANY`` on either end is a wildcard, so the if-else and
    special nodes accept any upstream and pass it through). Cycle and
    single-wire-per-input checks live in :func:`would_create_cycle` / the scene,
    which know the whole slice.
    """
    if src.is_input or not dst.is_input:
        return False
    if PortType.ANY in (src.type, dst.type):
        return True
    return src.type == dst.type


def clone_node(node: Node, pos: Tuple[float, float]) -> Node:
    """A copy of ``node`` with fresh ids and no connections.

    Params are deep-ish copied (one level), ports get new ids so the clone
    wires independently. Used by the board's Duplicate action.
    """
    return Node(
        id=new_id("node"),
        stage=node.stage,
        role=node.role,
        op_key=node.op_key,
        title=node.title,
        params=dict(node.params),
        inputs=[Port(new_id("p"), p.name, p.type, p.is_input) for p in node.inputs],
        outputs=[Port(new_id("p"), p.name, p.type, p.is_input) for p in node.outputs],
        pos=pos,
        enabled=node.enabled,
        bridge_id=None,
        category=node.category,
        shape_kind=node.shape_kind,
    )


def would_create_cycle(
    sl: GraphSlice, src_node: str, dst_node: str
) -> bool:
    """True if adding ``src_node -> dst_node`` would introduce a cycle.

    A cycle appears iff ``src_node`` is already reachable *from* ``dst_node``
    by following existing edges forward (so the new back-edge closes a loop).
    """
    if src_node == dst_node:
        return True
    seen: set[str] = set()
    stack = [dst_node]
    while stack:
        cur = stack.pop()
        if cur == src_node:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(e.dst_node for e in sl.outgoing(cur))
    return False
