"""Graph evaluation for the Pipelines tab (V1.45) — Qt-free.

Phase 1 wires the **Processing** slice. A Processing slice is a branching DAG
of ``Image -> Image`` enhancement nodes between one INPUT node (the loaded
channels) and one or more OUTPUT nodes (each a processed-image bridge). Because
every action node has exactly one input port, walking backward from any node
follows a *unique* chain to the INPUT — so any node linearizes to an ordered
``List[(plugin_name, params)]`` recipe, exactly the shape
``RecipeWorker`` / :class:`EnhancedDataset` consume.

This module stays Qt-free and dependency-light: it talks only to the pure
plugin registry and ``backend.normalization``. The Qt-side Apply path reuses
``EnhancedDataset`` + ``RecipeStage`` directly; here we keep an independent
``apply_recipe`` so preview compute never imports the session/stage machinery.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from nd2studios.core.plugin_registry import PluginBase
from nd2studios.pipeline_graph.model import GraphSlice, Node, NodeRole, PortType
from nd2studios.pipeline_graph.registry_adapter import (
    CHANNEL_ALL_OP_KEY,
    CHANNEL_PREFIX,
    INPUT_OP_KEY,
    OUTPUT_OP_KEY,
    channel_name_for_op_key,
    plugin_name_for_op_key,
)

Recipe = List[Tuple[str, Dict[str, Any]]]
ProgressCb = Optional[Callable[[int], None]]
CancelledCb = Optional[Callable[[], bool]]


# ── recipe linearization ────────────────────────────────────────────────────

def predecessor(sl: GraphSlice, node_id: str) -> Optional[Node]:
    """The node feeding ``node_id``'s single input port, if wired."""
    node = sl.nodes.get(node_id)
    if node is None:
        return None
    port = node.input_port()
    if port is None:
        return None
    # V1.49: a loop back-edge into this node's top input is *not* its recipe
    # predecessor — follow structural edges only.
    edge = sl.structural_edge_into_port(node_id, port.id)
    if edge is None:
        return None
    return sl.nodes.get(edge.src_node)


def node_has_channel_input(sl: GraphSlice, node: Node) -> bool:
    """True if a channel wire feeds one of ``node``'s CHANNEL (rainbow) input
    ports.

    V1.48/49: a channel pill wired into a process's rainbow port *is* that
    process's image source (that channel). So a process fed only by a channel
    wire — with no structural Input→process wire — is still legitimately
    "connected to a source", and the recipe chain terminates there. This lets the
    user wire ``channel → process → Output`` without also wiring the main Input
    node (the two-layer model would otherwise require a redundant structural
    wire, and Apply would fail with "not connected back to an input")."""
    chan_ports = {p.id for p in node.inputs if p.type is PortType.CHANNEL}
    if not chan_ports:
        return False
    return any(e.dst_port in chan_ports for e in sl.incoming(node.id))


def recipe_for_node(sl: GraphSlice, node_id: str) -> Recipe:
    """Linearize the unique chain INPUT -> ... -> ``node_id`` into a recipe.

    Action nodes contribute ``(plugin_name, params)``; the INPUT node and an
    OUTPUT node contribute nothing (the OUTPUT just bridges its predecessor's
    image). Disabled nodes pass through (skipped). Raises ``ValueError`` if the
    chain is broken (a node with no upstream that isn't the INPUT).
    """
    chain: Recipe = []
    visited: set[str] = set()
    cur = sl.nodes.get(node_id)
    while cur is not None:
        if cur.id in visited:
            raise ValueError("Cycle detected while linearizing recipe.")
        visited.add(cur.id)

        if cur.role is NodeRole.ACTION and cur.op_key not in (
            INPUT_OP_KEY, OUTPUT_OP_KEY
        ):
            if cur.enabled:
                chain.append(
                    (plugin_name_for_op_key(cur.op_key), dict(cur.params))
                )
        elif cur.role is NodeRole.INPUT:
            break

        prev = predecessor(sl, cur.id)
        if prev is None:
            if cur.role is NodeRole.INPUT:
                break
            # V1.49: a channel wire (rainbow port) is a valid image source — the
            # channel pill feeds this process, so the chain terminates here even
            # without a structural Input→process wire.
            if node_has_channel_input(sl, cur):
                break
            raise ValueError(
                f"Node '{cur.title}' is not connected back to an input."
            )
        cur = prev

    chain.reverse()
    return chain


def output_nodes(sl: GraphSlice) -> List[Node]:
    """OUTPUT nodes in stable insertion order."""
    return [n for n in sl.nodes.values() if n.role is NodeRole.OUTPUT]


def input_node(sl: GraphSlice) -> Optional[Node]:
    return next((n for n in sl.nodes.values() if n.role is NodeRole.INPUT), None)


# ── channel-flow layer (V1.48) ────────────────────────────────────────────────
# A separate CHANNEL wire layer decides which channels flow through each process.
# Channel source nodes (``channel:<name>`` / ``channel:__all__``) feed rainbow
# ports; channels then propagate downstream along the structural wires. The
# committed processing becomes per-channel (a channel gets only the enhancement
# steps at/after where it enters); unwired channels stay raw.

def channel_source_channels(node: Node, all_names: List[str]) -> Set[str]:
    """The channel(s) a channel-source node emits (``{name}`` or all for "All")."""
    if node.op_key == CHANNEL_ALL_OP_KEY:
        return set(all_names)
    if node.op_key.startswith(CHANNEL_PREFIX):
        nm = channel_name_for_op_key(node.op_key)
        return {nm} if nm else set(all_names)
    return set()


def _edge_src_is_channel(sl: GraphSlice, edge) -> bool:
    p = sl.find_port(edge.src_port)
    return p is not None and p.type is PortType.CHANNEL


def has_channel_wiring(sl: GraphSlice) -> bool:
    """True if any channel wire exists (a CHANNEL output feeds a rainbow port).

    When False the slice is *legacy*: every process runs on all channels."""
    return any(_edge_src_is_channel(sl, e) for e in sl.edges.values())


def channel_sets(sl: GraphSlice, all_names: List[str]) -> Dict[str, Set[str]]:
    """Effective channel set per node (``node_id -> {channel names}``).

    Propagation (topological): a node's set is the union of the channel sets of
    **every** incoming edge's source — both channel wires (from a channel source
    or another node's rainbow output) and structural wires (so a channel wired
    upstream "follows" downstream automatically). Channel source nodes seed their
    own channels.

    Legacy fallback: with no channel wiring anywhere, every node maps to all
    channels (the pre-V1.48 behavior)."""
    all_set = set(all_names)
    if not has_channel_wiring(sl):
        return {nid: set(all_set) for nid in sl.nodes}
    sets: Dict[str, Set[str]] = {nid: set() for nid in sl.nodes}
    for nid, n in sl.nodes.items():
        if n.op_key.startswith(CHANNEL_PREFIX):
            sets[nid] = channel_source_channels(n, all_names)
    for nid in topological_order(sl):
        n = sl.nodes.get(nid)
        if n is None or n.op_key.startswith(CHANNEL_PREFIX):
            continue
        acc: Set[str] = set()
        for e in sl.structural_incoming(nid):  # V1.49: ignore loop back-edges
            acc |= sets.get(e.src_node, set())
        sets[nid] = acc
    return sets


def structural_chain(sl: GraphSlice, node_id: str) -> List[Node]:
    """Action nodes on the unique structural chain INPUT→…→``node_id``, in order.

    Mirrors :func:`recipe_for_node`'s walk (follows the structural input port, so
    channel wires are ignored) but keeps the :class:`Node` objects so callers can
    consult per-node channel sets. Excludes INPUT / OUTPUT / channel-source nodes.
    """
    chain: List[Node] = []
    visited: set = set()
    cur = sl.nodes.get(node_id)
    while cur is not None:
        if cur.id in visited:
            break
        visited.add(cur.id)
        if (cur.role is NodeRole.ACTION
                and cur.op_key not in (INPUT_OP_KEY, OUTPUT_OP_KEY)
                and not cur.op_key.startswith(CHANNEL_PREFIX)):
            chain.append(cur)
        elif cur.role is NodeRole.INPUT:
            break
        prev = predecessor(sl, cur.id)
        if prev is None:
            break
        cur = prev
    chain.reverse()
    return chain


def channel_recipes(
    sl: GraphSlice, output_id: str, all_names: List[str]
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """Per-channel recipe for the Processing output ``output_id``.

    Each channel's recipe is the ordered enhancement steps on the structural
    chain whose node set contains that channel — i.e. the steps at/after the node
    the channel first enters (sets are monotonic down the chain). Channels that
    never enter the chain are absent from the result (⇒ raw). With no channel
    wiring at all, every channel gets the full chain (legacy)."""
    sets = channel_sets(sl, all_names)
    chain = structural_chain(sl, output_id)
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for ch in all_names:
        steps: List[Tuple[str, Dict[str, Any]]] = []
        for node in chain:
            if not node.enabled:
                continue
            if ch in sets.get(node.id, set()):
                steps.append((plugin_name_for_op_key(node.op_key), dict(node.params)))
        if steps:
            out[ch] = steps
    return out


def edge_channels(sl: GraphSlice, all_names: List[str]) -> Dict[str, List[str]]:
    """Channels flowing through each edge (for the channel-colored overlay lines).

    An edge carries the channel set of its source node. Returns ``{}`` when the
    slice has no channel wiring (legacy) so no rainbow lines are drawn."""
    if not has_channel_wiring(sl):
        return {}
    sets = channel_sets(sl, all_names)
    return {e.id: sorted(sets.get(e.src_node, set()))
            for e in sl.structural_edges()}  # V1.49: no strands on loop edges


# ── graph Run traversal (V1.45 merge — Qt-free) ──────────────────────────────

def topological_order(sl: GraphSlice) -> List[str]:
    """Node ids in a valid execution order (Kahn's algorithm).

    The graph is a DAG (the scene rejects cycles), so this always succeeds; on a
    malformed graph any leftover nodes are appended in insertion order.
    """
    indeg: Dict[str, int] = {nid: 0 for nid in sl.nodes}
    for e in sl.structural_edges():  # V1.49: loop edges don't add indegree
        if e.dst_node in indeg:
            indeg[e.dst_node] += 1
    ready = [nid for nid, d in indeg.items() if d == 0]
    order: List[str] = []
    seen: set = set()
    while ready:
        nid = ready.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        order.append(nid)
        for e in sl.structural_outgoing(nid):
            if e.dst_node in indeg:
                indeg[e.dst_node] -= 1
                if indeg[e.dst_node] == 0:
                    ready.append(e.dst_node)
    for nid in sl.nodes:  # any cycle remnants — append so callers see every node
        if nid not in seen:
            order.append(nid)
    return order


class GraphRunner:
    """Pure traversal state machine for a Run (Qt-free, no compute, no Qt).

    Drives a topological walk of a merged-Analysis slice with **branch pruning**:
    the page asks for the :meth:`next_ready` node, executes it (sync or via a
    background job / dialog), then calls :meth:`complete` — passing the pruned
    output port(s) for an if-else so only the taken branch is followed. Nodes on
    an un-taken branch are never reached, so they stay "shaded" in the UI.

    A node becomes ready once every *reached* predecessor (via a non-pruned edge)
    is done — so a reconverging node runs as soon as its live upstream finishes,
    ignoring predecessors stranded on a dead branch.
    """

    def __init__(self, sl: GraphSlice, start: Optional[str] = None,
                 frozen: Optional[set] = None):
        self.slice = sl
        if start is None:
            inp = input_node(sl)
            start = inp.id if inp is not None else (
                next(iter(sl.nodes), None))
        self._start = start
        self._reached: set = set()
        self._done: set = set()
        self._dispatched: set = set()
        self._pruned_edges: set = set()
        if self._start is not None:
            self._reached.add(self._start)
        # V1.53 checkpoint resume: nodes whose output is already frozen (the
        # structural ancestors of a checkpoint). Treat each as already complete —
        # reached + done + dispatched — and reach its structural successors, so the
        # walk skips them entirely but the checkpoint (their successor) still runs
        # and reconverging downstream nodes still gate on them. ``frozen`` empty ⇒
        # identical to the pre-V1.53 behavior.
        self._frozen: set = set(frozen or ())
        for nid in self._frozen:
            self._reached.add(nid)
            self._done.add(nid)
            self._dispatched.add(nid)
            for e in sl.structural_outgoing(nid):
                self._reached.add(e.dst_node)
        # V1.48/49: channel-source pills are additional roots. A process fed only
        # by a channel wire (no structural Input→process edge) must still be
        # reached and run. Pills carry no computation, so mark them done up front
        # and reach their successors — they never appear as a run step.
        self._channel_sources: set = {
            nid for nid, n in sl.nodes.items()
            if n.op_key.startswith(CHANNEL_PREFIX)
        }
        for nid in self._channel_sources:
            self._reached.add(nid)
            self._done.add(nid)
            self._dispatched.add(nid)
            for e in sl.structural_outgoing(nid):
                self._reached.add(e.dst_node)

    def reachable_nodes(self) -> set:
        """All nodes reachable from the roots (Input + channel pills) via
        structural edges — the initial shaded set. Channel pills themselves are
        roots, not runnable steps, so they are excluded from the shaded set."""
        out: set = set()
        roots = set(self._channel_sources)
        if self._start is not None:
            roots.add(self._start)
        stack = list(roots)
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            for e in self.slice.structural_outgoing(cur):  # V1.49: skip loops
                stack.append(e.dst_node)
        return out - self._channel_sources

    def _active_incoming(self, nid: str) -> List:
        # V1.49: loop back-edges never gate readiness (structural only).
        return [e for e in self.slice.structural_incoming(nid)
                if e.id not in self._pruned_edges and e.src_node in self._reached]

    def active_incoming(self, nid: str) -> List:
        """Public: a node's live (un-pruned, reached) incoming edges — used by the
        page to scope per-branch rows to whichever branch feeds the node."""
        return self._active_incoming(nid)

    def next_ready(self) -> Optional[str]:
        """The next node whose live predecessors are all done; marks it
        dispatched. ``None`` when nothing is runnable right now (waiting on an
        in-flight node, or finished)."""
        for nid in self._reached:
            if nid in self._dispatched:
                continue
            active = self._active_incoming(nid)
            if all(e.src_node in self._done for e in active):
                self._dispatched.add(nid)
                return nid
        return None

    def complete(self, nid: str, *, prune_ports: Optional[set] = None) -> None:
        """Mark ``nid`` done and reach its successors, skipping any edge whose
        source port is in ``prune_ports`` (the if-else's un-taken branch)."""
        self._done.add(nid)
        self._dispatched.add(nid)
        prune_ports = set(prune_ports or ())
        for e in self.slice.structural_outgoing(nid):  # V1.49: skip loop edges
            if e.src_port in prune_ports:
                self._pruned_edges.add(e.id)
                continue
            self._reached.add(e.dst_node)

    def is_finished(self) -> bool:
        """True once every reached node is done (nothing left to run)."""
        return self._reached.issubset(self._done)

    @property
    def done(self) -> set:
        return set(self._done)


def evaluate_simple_condition(params: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    """Phase-1 if-else condition over measurement ``rows`` (list of dicts).

    Honors ``metric`` / ``aggregate`` / ``comparator`` / ``value``. ``object_count``
    tests the number of rows; other metrics aggregate the per-object field
    (any / all / mean / median / count) before the comparison. Missing data
    evaluates False. Qt-free so the executor/self-test can exercise it.
    """
    metric = str(params.get("metric", "object_count"))
    comparator = str(params.get("comparator", ">"))
    try:
        value = float(params.get("value", 0.0))
    except (TypeError, ValueError):
        value = 0.0

    def _cmp(lhs: float) -> bool:
        if comparator == ">":
            return lhs > value
        if comparator == ">=":
            return lhs >= value
        if comparator == "<":
            return lhs < value
        if comparator == "<=":
            return lhs <= value
        if comparator == "==":
            return lhs == value
        if comparator == "!=":
            return lhs != value
        return False

    rows = rows or []
    if metric == "object_count":
        return _cmp(float(len(rows)))

    # mean_intensity has per-channel column names — match the first such field.
    def _field(row: Dict[str, Any]) -> Optional[float]:
        if metric == "mean_intensity":
            for k, v in row.items():
                if k.startswith("mean_intensity"):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
            return None
        v = row.get(metric)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    vals = [x for x in (_field(r) for r in rows) if x is not None]
    if not vals:
        return False
    aggregate = str(params.get("aggregate", "any"))
    if aggregate == "any":
        return any(_cmp(v) for v in vals)
    if aggregate == "all":
        return all(_cmp(v) for v in vals)
    if aggregate == "count":
        return _cmp(float(sum(1 for v in vals if _cmp(v))))
    if aggregate == "median":
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        med = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
        return _cmp(med)
    # mean (default)
    return _cmp(sum(vals) / len(vals))


# ── caching keys ────────────────────────────────────────────────────────────

def recipe_hash(recipe: Recipe, normalized: bool) -> str:
    """Stable hash of a recipe + normalize flag, for preview coalescing tags."""
    payload = json.dumps(
        {"normalized": bool(normalized),
         "recipe": [[n, p] for (n, p) in recipe]},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# ── recipe execution (mirrors RecipeWorker / EnhancedDataset) ────────────────

def apply_recipe(
    channels: Dict[str, Any],
    recipe: Recipe,
    normalized: bool,
    frame: Optional[int] = None,
    progress_cb: ProgressCb = None,
    cancelled_cb: CancelledCb = None,
    recipe_by_channel: Optional[Dict[str, Recipe]] = None,
) -> Dict[str, np.ndarray]:
    """Run ``recipe`` over each channel and return ``{name: (T, H, W)}``.

    Mirrors :meth:`EnhancedDataset.materialize_channel`: materialize the lazy
    proxy, optionally frame-mean normalize, then apply each enhancement plugin
    in order. ``frame`` (if given) restricts work to a single ``(1, H, W)``
    plane for fast previews. ``cancelled_cb`` is polled between channels and
    steps so a coalesced preview unwinds promptly.

    V1.48: when ``recipe_by_channel`` is given, each channel uses its own recipe
    (a channel absent / empty ⇒ raw, no normalize) instead of the single
    ``recipe`` — the channel-wire pipeline where unwired channels stay raw.
    """
    names = list(channels.keys())
    out: Dict[str, np.ndarray] = {}
    total = max(1, len(names))

    for ci, name in enumerate(names):
        if cancelled_cb is not None and cancelled_cb():
            break
        rec = (recipe_by_channel.get(name, []) if recipe_by_channel is not None
               else recipe)
        raw = channels[name]
        materialize = getattr(raw, "materialize", None)
        if callable(materialize):
            arr = materialize()
        else:
            arr = np.asarray(raw)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if frame is not None:
            f = max(0, min(int(frame), arr.shape[0] - 1))
            arr = arr[f:f + 1]

        # Unwired channels (per-channel mode, empty recipe) stay raw — no
        # normalization either.
        do_norm = normalized and (recipe_by_channel is None or bool(rec))
        if do_norm:
            try:
                from nd2studios.backend.normalization import normalize_timeseries
                arr = normalize_timeseries(arr)
            except Exception:  # noqa: BLE001 — best-effort, like RecipeWorker
                pass

        for (plugin_name, params) in rec:
            if cancelled_cb is not None and cancelled_cb():
                break
            plugin_cls = PluginBase.get_plugin("enhancement", plugin_name)
            if plugin_cls is None:
                continue
            arr = plugin_cls().execute(arr, params, progress_cb=None)

        out[name] = arr
        if progress_cb is not None:
            progress_cb(int((ci + 1) / total * 100))

    return out


class PinnedProcessedVolume:
    """Lazy volume that serves a **chosen set of pre-processed planes**, raw else.

    Wraps a raw volume (``LazyND2Volume`` / ``MaterializedDataset``) and mirrors
    the ``get_frame`` read protocol the ``MultiAxisViewer`` uses. The pinned
    ``(M, T)`` planes — the frame the user is on, or every frame in a multi-frame
    selection — return the supplied **already-processed** channel frames
    (``planes``: ``{(m, t): {channel_name: (H, W)}}``, computed in a background
    job); every other read returns the **raw** plane untouched.

    So only the chosen frames are recipe-processed (an isolated processed version
    laid over the raw data) — navigating elsewhere shows raw, and the frame strip
    / LUT sampling / background pre-render all read raw. The set **follows** the
    user: :meth:`set_planes` re-points it in place (after a fresh background job)
    to whatever frame(s) they click / select, so the viewer's selection and LUTs
    are preserved (no ``set_volume`` churn).

    It deliberately does **not** expose a ``channels`` attribute, so the viewer
    takes its ``get_frame`` path (exactly as for a bare ``LazyND2Volume``).
    """

    # V1.49: this wrapper serves per-frame processed data, so the raw
    # multi-resolution pyramid (built from the unprocessed volume) must NOT serve
    # its frames — doing so bypasses processing back to raw. The viewer checks
    # this flag and skips the pyramid short-circuit for the wrapper.
    bypass_pyramid = True

    def __init__(self, raw_volume: Any,
                 planes: Dict[Tuple[int, int], Dict[str, np.ndarray]]):
        self._raw = raw_volume
        self._planes = {(int(m), int(t)): dict(ch)
                        for (m, t), ch in (planes or {}).items()}
        self.channel_names = list(getattr(raw_volume, "channel_names", []))
        self.n_multipoints = int(getattr(raw_volume, "n_multipoints", 1))
        self.n_timepoints = int(getattr(raw_volume, "n_timepoints", 1))
        self.n_zslices = int(getattr(raw_volume, "n_zslices", 1))
        self.height = int(getattr(raw_volume, "height", 0))
        self.width = int(getattr(raw_volume, "width", 0))
        self.dtype = getattr(raw_volume, "dtype", np.dtype("uint16"))
        self.pixel_size_um = float(getattr(raw_volume, "pixel_size_um", 1.0) or 1.0)
        self.z_step_um = float(getattr(raw_volume, "z_step_um", 1.0) or 1.0)

    def set_planes(self,
                   planes: Dict[Tuple[int, int], Dict[str, np.ndarray]]) -> None:
        """Replace the processed plane set in place (follows the user's frames).

        Only these planes are served processed; any previously-processed frame
        not in the new set reverts to raw.
        """
        self._planes = {(int(m), int(t)): dict(ch)
                        for (m, t), ch in (planes or {}).items()}

    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "max", **kwargs) -> np.ndarray:
        ch = self._planes.get((int(m), int(t)))
        if ch is not None:
            name = (self.channel_names[c]
                    if 0 <= c < len(self.channel_names) else None)
            frame = ch.get(name)
            if frame is not None:
                return frame
        return np.asarray(
            self._raw.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs)
        )


class ProcessedFrameVolume:
    """Lazy volume that applies a recipe to **every displayed frame on demand**.

    Wraps a raw volume (``LazyND2Volume`` / ``MaterializedDataset``), mirroring the
    ``get_frame`` read protocol the ``MultiAxisViewer`` uses. Unlike
    :class:`PinnedProcessedVolume` (which serves a fixed processed plane set),
    this applies the committed recipe to whatever plane is read — so the whole
    stack reads as **processed** and stays navigable, with no up-front
    materialization. Used for the Analysis/Results *base* image after a recipe is
    committed (Apply), so e.g. a Background Subtract shows there instead of raw.
    The per-frame recipe runs in the read path, so it's best for light recipes;
    an empty recipe is a pass-through.

    No ``channels`` attribute → the viewer uses its ``get_frame`` path.
    """

    _CACHE_MAX = 24  # processed planes kept (≈ a few channels × a few frames)
    # V1.49: the raw multi-resolution pyramid must NOT serve this wrapper's frames
    # (it would bypass the per-channel recipe back to raw — the "Analysis viewer
    # shows raw under a crop / zoom-out" bug). The viewer skips the pyramid here.
    bypass_pyramid = True

    def __init__(self, raw_volume: Any, recipe: Recipe, normalized: bool = False,
                 recipe_by_channel: Optional[Dict[str, Recipe]] = None):
        self._raw = raw_volume
        self._recipe = [(n, dict(p)) for (n, p) in (recipe or [])]
        self._normalized = bool(normalized)
        # V1.48: per-channel recipes. When set, each channel uses its own recipe
        # (a channel absent / empty ⇒ raw), overriding the single ``recipe``.
        self._recipe_by_channel: Optional[Dict[str, Recipe]] = (
            {k: [(n, dict(p)) for (n, p) in v] for k, v in recipe_by_channel.items()}
            if recipe_by_channel is not None else None
        )
        # Small bounded cache of processed planes so re-renders of the same frame
        # (e.g. an overlay refresh) don't re-run the recipe.
        self._cache: Dict[tuple, np.ndarray] = {}
        self._cache_order: List[tuple] = []
        self.channel_names = list(getattr(raw_volume, "channel_names", []))
        self.n_multipoints = int(getattr(raw_volume, "n_multipoints", 1))
        self.n_timepoints = int(getattr(raw_volume, "n_timepoints", 1))
        self.n_zslices = int(getattr(raw_volume, "n_zslices", 1))
        self.height = int(getattr(raw_volume, "height", 0))
        self.width = int(getattr(raw_volume, "width", 0))
        self.dtype = getattr(raw_volume, "dtype", np.dtype("uint16"))
        self.pixel_size_um = float(getattr(raw_volume, "pixel_size_um", 1.0) or 1.0)
        self.z_step_um = float(getattr(raw_volume, "z_step_um", 1.0) or 1.0)

    def _recipe_for(self, name: str) -> Recipe:
        """The recipe for channel ``name`` — its per-channel recipe when set
        (absent ⇒ raw), else the single recipe (legacy)."""
        if self._recipe_by_channel is not None:
            return self._recipe_by_channel.get(name) or []
        return self._recipe

    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "max", **kwargs) -> np.ndarray:
        name = (self.channel_names[c]
                if 0 <= c < len(self.channel_names) else "_c")
        recipe = self._recipe_for(name)
        if not recipe:  # no processing for this channel ⇒ raw (unwired stays raw)
            return np.asarray(
                self._raw.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs))
        key = (int(c), int(m), int(t), int(z), str(z_mode))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        frame = np.asarray(
            self._raw.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs)
        )
        res = frame
        try:
            out = apply_recipe({name: frame[None, ...]}, recipe,
                               self._normalized)
            got = out.get(name)
            if got is not None and got.shape[0] > 0:
                res = got[0]
        except Exception:  # noqa: BLE001 — never break the read path
            res = frame
        self._cache[key] = res
        self._cache_order.append(key)
        if len(self._cache_order) > self._CACHE_MAX:
            self._cache.pop(self._cache_order.pop(0), None)
        return res


class CroppedVolume:
    """Lazy volume that serves a fixed XY sub-rectangle of a wrapped volume.

    Wraps any volume exposing the ``get_frame`` read protocol the
    ``MultiAxisViewer`` uses (a raw ``LazyND2Volume`` / ``MaterializedDataset``,
    or one of the processed wrappers above). Every read is sliced to
    ``[y:y+h, x:x+w]`` and ``height`` / ``width`` report the crop size, so
    display, segmentation and measurement all operate on the crop region only.

    Used by the Pipelines-page **preview crop** — a preview-only scoping tool. A
    full Run never wraps in this (crop is ignored while running).

    No ``channels`` attribute → the viewer uses its ``get_frame`` path.
    """

    # V1.49: a raw pyramid can't serve the crop region (it is built for the full
    # raw volume in full-frame coordinates) — using it would show a raw, full-
    # frame image under a crop. The viewer skips the pyramid for this wrapper.
    bypass_pyramid = True

    def __init__(self, raw_volume: Any, rect: Tuple[int, int, int, int]):
        self._raw = raw_volume
        x, y, w, h = (int(v) for v in rect)
        self._x, self._y, self._w, self._h = x, y, w, h
        self.channel_names = list(getattr(raw_volume, "channel_names", []))
        self.n_multipoints = int(getattr(raw_volume, "n_multipoints", 1))
        self.n_timepoints = int(getattr(raw_volume, "n_timepoints", 1))
        self.n_zslices = int(getattr(raw_volume, "n_zslices", 1))
        self.height = h
        self.width = w
        self.dtype = getattr(raw_volume, "dtype", np.dtype("uint16"))
        self.pixel_size_um = float(getattr(raw_volume, "pixel_size_um", 1.0) or 1.0)
        self.z_step_um = float(getattr(raw_volume, "z_step_um", 1.0) or 1.0)

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self._x, self._y, self._w, self._h)

    def get_frame(self, c: int = 0, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "max", **kwargs) -> np.ndarray:
        frame = np.asarray(
            self._raw.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs))
        return frame[self._y:self._y + self._h, self._x:self._x + self._w]


class RegisteredFrameVolume:
    """Lazy volume that applies per-frame **registration** transforms on read.

    Wraps any volume exposing the ``get_frame`` read protocol (a raw volume or one
    of the processed wrappers above) and, for each read, applies the drift-
    correction transform stored for that multipoint ``m`` at frame ``t`` (from a
    Registration node). An M without a transform reads through unchanged.

    Used by the Pipelines-page so the *displayed base* stays drift-corrected under
    a preview crop (the Registration node reconstructs the viewer; arming the crop
    must not revert it to the un-registered image). Wrap this **before**
    :class:`CroppedVolume` so the crop is taken from the registered frame — a crop
    of the registered image, matching what the user sees.

    No ``channels`` attribute → the viewer uses its ``get_frame`` path.
    """

    # The raw pyramid serves un-registered, full-frame data — it must not bypass
    # this wrapper (same reasoning as the Processed / Cropped wrappers).
    bypass_pyramid = True

    # Warping a full-frame plane (scipy.ndimage.shift / cv2.warp) costs tens of ms;
    # without a cache, playback / scrubbing recomputes it every tick and the
    # registered image crawls while the raw (pyramid) plays fast. Cache the warped
    # display frames under a memory budget so each (c,m,t,z) is computed once — the
    # transforms are deterministic per (m,t), so caching is always safe. The budget
    # (not a fixed count) means small frames cache a whole series (instant looping)
    # while huge stacks stay bounded.
    _CACHE_BUDGET_BYTES = 512 * 1024 * 1024

    def __init__(self, base_volume: Any, by_m: Dict[int, Any],
                 interp_order: int = 1):
        self._base = base_volume
        self._by_m = {int(k): v for k, v in (by_m or {}).items()}
        self._order = int(interp_order or 1)
        self._cache: Dict[tuple, np.ndarray] = {}
        self._cache_order: List[tuple] = []
        self._cache_bytes = 0
        self.channel_names = list(getattr(base_volume, "channel_names", []))
        self.n_multipoints = int(getattr(base_volume, "n_multipoints", 1))
        self.n_timepoints = int(getattr(base_volume, "n_timepoints", 1))
        self.n_zslices = int(getattr(base_volume, "n_zslices", 1))
        self.height = int(getattr(base_volume, "height", 0))
        self.width = int(getattr(base_volume, "width", 0))
        self.dtype = getattr(base_volume, "dtype", np.dtype("uint16"))
        self.pixel_size_um = float(getattr(base_volume, "pixel_size_um", 1.0) or 1.0)
        self.z_step_um = float(getattr(base_volume, "z_step_um", 1.0) or 1.0)

    def get_frame(self, c: int = 0, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "max", **kwargs) -> np.ndarray:
        tf = self._by_m.get(int(m))
        if not tf:  # M without a transform reads straight through (no cache needed)
            return np.asarray(
                self._base.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs))
        key = (int(c), int(m), int(t), int(z), str(z_mode))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        frame = np.asarray(
            self._base.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs))
        from nd2studios.backend.registration import estimate as _est
        try:
            out = _est.apply_frame(frame, tf, int(t), interp_order=self._order)
        except Exception:  # noqa: BLE001 — never break the read path
            return frame
        self._cache[key] = out
        self._cache_order.append(key)
        self._cache_bytes += int(getattr(out, "nbytes", 0))
        while self._cache_bytes > self._CACHE_BUDGET_BYTES and self._cache_order:
            ev = self._cache.pop(self._cache_order.pop(0), None)
            if ev is not None:
                self._cache_bytes -= int(getattr(ev, "nbytes", 0))
        return out
