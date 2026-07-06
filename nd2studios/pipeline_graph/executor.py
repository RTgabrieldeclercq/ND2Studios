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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.core.plugin_registry import PluginBase
from nd2studios.pipeline_graph.model import GraphSlice, Node, NodeRole
from nd2studios.pipeline_graph.registry_adapter import (
    INPUT_OP_KEY,
    OUTPUT_OP_KEY,
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
    edge = sl.edge_into_port(node_id, port.id)
    if edge is None:
        return None
    return sl.nodes.get(edge.src_node)


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


# ── graph Run traversal (V1.45 merge — Qt-free) ──────────────────────────────

def topological_order(sl: GraphSlice) -> List[str]:
    """Node ids in a valid execution order (Kahn's algorithm).

    The graph is a DAG (the scene rejects cycles), so this always succeeds; on a
    malformed graph any leftover nodes are appended in insertion order.
    """
    indeg: Dict[str, int] = {nid: 0 for nid in sl.nodes}
    for e in sl.edges.values():
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
        for e in sl.outgoing(nid):
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

    def __init__(self, sl: GraphSlice, start: Optional[str] = None):
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

    def reachable_nodes(self) -> set:
        """All nodes reachable from the start via *any* edge — the initial
        shaded set (everything that could run before branch decisions)."""
        out: set = set()
        if self._start is None:
            return out
        stack = [self._start]
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            for e in self.slice.outgoing(cur):
                stack.append(e.dst_node)
        return out

    def _active_incoming(self, nid: str) -> List:
        return [e for e in self.slice.incoming(nid)
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
        for e in self.slice.outgoing(nid):
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
) -> Dict[str, np.ndarray]:
    """Run ``recipe`` over each channel and return ``{name: (T, H, W)}``.

    Mirrors :meth:`EnhancedDataset.materialize_channel`: materialize the lazy
    proxy, optionally frame-mean normalize, then apply each enhancement plugin
    in order. ``frame`` (if given) restricts work to a single ``(1, H, W)``
    plane for fast previews. ``cancelled_cb`` is polled between channels and
    steps so a coalesced preview unwinds promptly.
    """
    names = list(channels.keys())
    out: Dict[str, np.ndarray] = {}
    total = max(1, len(names))

    for ci, name in enumerate(names):
        if cancelled_cb is not None and cancelled_cb():
            break
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

        if normalized:
            try:
                from nd2studios.backend.normalization import normalize_timeseries
                arr = normalize_timeseries(arr)
            except Exception:  # noqa: BLE001 — best-effort, like RecipeWorker
                pass

        for (plugin_name, params) in recipe:
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

    def __init__(self, raw_volume: Any, recipe: Recipe, normalized: bool = False):
        self._raw = raw_volume
        self._recipe = [(n, dict(p)) for (n, p) in (recipe or [])]
        self._normalized = bool(normalized)
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

    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "max", **kwargs) -> np.ndarray:
        if not self._recipe:
            return np.asarray(
                self._raw.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs))
        key = (int(c), int(m), int(t), int(z), str(z_mode))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        frame = np.asarray(
            self._raw.get_frame(c=c, m=m, t=t, z=z, z_mode=z_mode, **kwargs)
        )
        name = (self.channel_names[c]
                if 0 <= c < len(self.channel_names) else "_c")
        res = frame
        try:
            out = apply_recipe({name: frame[None, ...]}, self._recipe,
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
