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
    CHANNEL = "channel"  # V1.48 channel-flow layer: rainbow ports + channel wires


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
    CHANNEL = "channel"  # V1.48 channel source nodes (one per channel + "All")
    CHECKPOINT = "checkpoint"  # V1.53 checkpoint node (white; freezes upstream)


class ShapeKind(Enum):
    """The body silhouette a node is drawn with."""

    RECT = "rect"          # default rounded rectangle
    TRIANGLE = "triangle"  # upright triangle (if-else: 1 in top, 2 out bottom)
    HEXAGON = "hexagon"    # special action nodes
    PILL = "pill"          # V1.48 channel source nodes (small rounded capsule)
    GEM = "gem"            # V1.77 Prism node — upright faceted 2.5D gem / prism


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


# ── edge kinds (V1.49) ───────────────────────────────────────────────────────
# A structural edge is the normal top→bottom pipeline wire. A *loop* edge is the
# V1.49 back-edge: it exits a node's bottom and returns to the top of the same
# node or an upstream node, defining an iterative loop region. Loop edges are
# **invisible** to every DAG codepath (recipe linearization, channel propagation,
# topological order, GraphRunner, cycle detection) — only the loop executor sees
# them — so a graph without loops behaves exactly as before.
STRUCTURAL_KIND = "structural"
LOOP_KIND = "loop"

# V1.68 — per-edge analysis **scope** (Frame / Object toggle). Stored on
# ``Edge.params["scope"]`` (a free-form dict that already round-trips through
# ``to_dict``/``from_dict``), so scope persists in the saved pipeline with **no
# schema bump**. The key is meaningful only on an edge whose source node produces
# objects (a 3D-mask / track / analysis label node); absent ⇒ ``whole_frame`` ⇒
# every existing graph behaves exactly as before.
SCOPE_WHOLE = "whole_frame"
SCOPE_OBJECTS = "objects"
_EDGE_SCOPE_KEY = "scope"

# V1.77 — per-edge **view-only** flag (the dotted-wire toggle). Stored on
# ``Edge.params["view_only"]`` (like the scope lever, so it round-trips through
# ``to_dict``/``from_dict`` with **no schema bump**). A view-only edge feeds its
# channel(s) to the *viewers only* — analysis channel propagation
# (:func:`~nd2studios.pipeline_graph.executor.channel_sets`) skips it entirely, so a
# view-only channel can never enter an analysis node's computation nor propagate
# downstream as an analysis channel. Absent ⇒ a normal (solid, analysis) edge, so
# every existing graph behaves exactly as before. The edge still counts for run
# ordering (``GraphRunner`` gates on structural edges regardless of this flag), so the
# upstream producer of a view-only overlay still runs before its consumer.
_EDGE_VIEW_ONLY_KEY = "view_only"


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
    # V1.49: ``kind`` is ``"structural"`` (default) or ``"loop"``. ``params`` holds
    # the loop configuration (sweep / stop condition / combine rule) for a loop
    # edge; it is empty for structural edges.
    kind: str = STRUCTURAL_KIND
    params: Dict[str, Any] = field(default_factory=dict)

    def is_loop(self) -> bool:
        return self.kind == LOOP_KIND

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "src_node": self.src_node,
            "src_port": self.src_port,
            "dst_node": self.dst_node,
            "dst_port": self.dst_port,
            "kind": self.kind,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        return cls(
            id=d["id"],
            src_node=d["src_node"],
            src_port=d["src_port"],
            dst_node=d["dst_node"],
            dst_port=d["dst_port"],
            kind=str(d.get("kind", STRUCTURAL_KIND)),
            params=dict(d.get("params", {})),
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

    # ── structural-only queries (V1.49) ───────────────────────────────────
    # DAG traversal (recipe linearization, channel propagation, GraphRunner)
    # must ignore loop edges — these variants filter them out.
    def structural_edges(self) -> List[Edge]:
        return [e for e in self.edges.values() if e.kind != LOOP_KIND]

    def loop_edges(self) -> List[Edge]:
        return [e for e in self.edges.values() if e.kind == LOOP_KIND]

    def structural_incoming(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges.values()
                if e.dst_node == node_id and e.kind != LOOP_KIND]

    def structural_outgoing(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges.values()
                if e.src_node == node_id and e.kind != LOOP_KIND]

    def structural_edge_into_port(
        self, node_id: str, port_id: str
    ) -> Optional[Edge]:
        # V1.77: prefer the ANALYSIS wire — a view-only (overlay) wire coexisting on
        # the same port is display-only and must not be a recipe/run predecessor. So a
        # port with one analysis + N view-only wires still linearizes off the analysis
        # one. (A port with only a view-only wire returns it as a last resort so the
        # node still has a source.)
        fallback: Optional[Edge] = None
        for e in self.edges.values():
            if (e.dst_node == node_id and e.dst_port == port_id
                    and e.kind != LOOP_KIND):
                if not edge_view_only(e):
                    return e
                fallback = fallback or e
        return fallback

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
    # Must match ``io.PIPELINE_VERSION`` so a freshly-created doc round-trips
    # cleanly. V6 (V1.61): Processing folded into the merged slice AND the
    # intermediate bridge nodes dropped — one connected chain from a universal
    # input through processing into analysis to output nodes. V4 (V1.49) added
    # edge ``kind``; V3 (V1.45) the Track Objects node.
    schema_version: int = 6

    @classmethod
    def empty(cls) -> "PipelineDoc":
        return cls(
            processing=GraphSlice(Stage.PROCESSING),
            analysis=GraphSlice(Stage.ANALYSIS),
            results=GraphSlice(Stage.RESULTS),
        )

    def slice_for(self, stage: Stage) -> GraphSlice:
        # V1.61 merge: Processing is folded into the merged (analysis) slice, so
        # both PROCESSING and ANALYSIS resolve to it — the page edits one graph
        # across what used to be two sub-tabs. Node ``stage`` still distinguishes
        # enhancement nodes from analysis nodes for execution dispatch. The empty
        # ``processing`` slice is retained for serialization back-compat (older
        # two-slice docs fold into ``analysis`` on load via io._migrate_v4_to_v5).
        if stage is Stage.RESULTS:
            return self.results
        return self.analysis

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

    V1.48: the ``CHANNEL`` flavor is a separate layer — a channel wire pairs only
    with another ``CHANNEL`` port (a rainbow port). It never connects to a
    structural IMAGE/BINARY/DATA port, and not even to an ``ANY`` wildcard, so
    channel wires must target a rainbow port explicitly.
    """
    if src.is_input or not dst.is_input:
        return False
    if PortType.CHANNEL in (src.type, dst.type):
        return src.type is PortType.CHANNEL and dst.type is PortType.CHANNEL
    if PortType.ANY in (src.type, dst.type):
        return True
    return src.type == dst.type


# ── edge scope (V1.68 Frame / Object toggle) ────────────────────────────────

def edge_scope(edge: Edge) -> str:
    """Return an edge's analysis scope: :data:`SCOPE_WHOLE` (default) or
    :data:`SCOPE_OBJECTS`.

    Reads ``edge.params["scope"]``. Absent ⇒ ``whole_frame`` so legacy graphs
    (and every structural edge that never toggled the lever) run exactly as
    before. Any unrecognized value also degrades to ``whole_frame``.
    """
    val = (edge.params or {}).get(_EDGE_SCOPE_KEY, SCOPE_WHOLE)
    return SCOPE_OBJECTS if val == SCOPE_OBJECTS else SCOPE_WHOLE


def set_edge_scope(edge: Edge, scope: str) -> None:
    """Write an edge's scope into ``edge.params`` (persists via ``to_dict``).

    ``whole_frame`` clears the key (keeping the saved params tidy and legacy-
    identical); ``objects`` writes it. No ``schema_version`` change is needed.
    """
    if edge.params is None:
        edge.params = {}
    if scope == SCOPE_OBJECTS:
        edge.params[_EDGE_SCOPE_KEY] = SCOPE_OBJECTS
    else:
        edge.params.pop(_EDGE_SCOPE_KEY, None)


def edge_view_only(edge: Edge) -> bool:
    """Return whether ``edge`` is a **view-only** overlay wire (V1.77).

    Reads ``edge.params["view_only"]``. Absent / falsey ⇒ a normal analysis edge, so
    legacy graphs (and every edge that never toggled the dotted-wire lever) behave
    exactly as before. A view-only edge feeds channels to the viewers only — it is
    skipped by analysis channel propagation but still gates run ordering.
    """
    return bool((edge.params or {}).get(_EDGE_VIEW_ONLY_KEY, False))


def set_edge_view_only(edge: Edge, on: bool) -> None:
    """Set / clear an edge's view-only flag (persists via ``to_dict``, V1.77).

    ``True`` writes the key; ``False`` clears it (keeping saved params tidy and
    legacy-identical). No ``schema_version`` change is needed.
    """
    if edge.params is None:
        edge.params = {}
    if on:
        edge.params[_EDGE_VIEW_ONLY_KEY] = True
    else:
        edge.params.pop(_EDGE_VIEW_ONLY_KEY, None)


def is_channel_port(port: Port) -> bool:
    """True for a rainbow / channel-flow port (V1.48)."""
    return port.type is PortType.CHANNEL


def is_loop_edge(edge: Edge) -> bool:
    """True for a V1.49 loop / iteration back-edge."""
    return edge.kind == LOOP_KIND


def can_connect_loop(src: Port, dst: Port) -> bool:
    """True if a **loop** wire from output ``src`` to input ``dst`` is legal.

    A loop edge reuses the node's existing *structural* ports: it leaves a
    bottom output and returns to a top input. So ``src`` must be an output,
    ``dst`` an input, and neither may be a CHANNEL (rainbow) port — channels
    flow on their own layer. Unlike :func:`can_connect`, payload types need not
    match (a loop just says "re-run this region"), and unlike a structural wire
    the cycle check is deliberately skipped by the caller.
    """
    if src.is_input or not dst.is_input:
        return False
    if PortType.CHANNEL in (src.type, dst.type):
        return False
    return True


def structural_input_port(node: Node) -> Optional[Port]:
    """The node's first **structural** (non-channel) input port, if any.

    ``node.input_port()`` returns ``inputs[0]``, which stays the structural port
    because rainbow (CHANNEL) inputs are always appended after it. This helper is
    explicit for callers that must skip rainbow ports."""
    for p in node.inputs:
        if p.type is not PortType.CHANNEL:
            return p
    return None


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
