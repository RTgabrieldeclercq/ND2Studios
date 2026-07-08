"""Loop / iteration connector logic for the Pipelines tab (V1.49) — Qt-free.

A **loop edge** (``Edge.kind == LOOP_KIND``, see :mod:`model`) exits a node's
bottom output and returns to the top input of the same node or an upstream node,
declaring an iterative *loop region*: the structural sub-graph between the loop's
``entry`` (the edge destination) and ``exit`` (the edge source), re-run once per
iteration while parameters are swept and/or a stop condition is checked.

This module is the **pure** core (no PySide6, no app singletons): it computes the
loop region, expands the iteration plan (parameter sweep grid / fixed count /
until-condition), de-duplicates objects across iterations by mask IoU (with a
centroid-distance fallback), combines iterations under the four rules, and scores
tracking coverage. The Qt page (:mod:`pages.pipelines_page`) drives the async
per-iteration sub-runs and calls these helpers; keeping the logic here makes it
headless-testable, mirroring the backend-purity rule in CLAUDE.md.

Loop config (stored JSON-serialized in ``loop_edge.params``)::

    {
      "version": 1,
      "mode": "sweep" | "count" | "until",
      "count": 3,                 # iterations for "count" (and until fallback)
      "max_iterations": 64,       # hard safety cap (always honored)
      "combine_axes": "grid" | "zip",
      "axes": [
        {"node_id": "...", "param": "Scale",
         "start": 0.3, "stop": 0.9, "step": 0.2},
        {"node_id": "...", "param": "prob_thresh", "values": [0.3, 0.5, 0.7]},
      ],
      "stop": {condition-dict},   # for "until"; see conditions.Condition
      "combine": {
        "rule": "union_dedup" | "best" | "last" | "keep_all",
        "dedup": {"metric": "iou" | "centroid",
                  "iou_threshold": 0.3, "centroid_distance": 10.0},
        "best_metric": "object_count" | "tracking_ratio",
      },
      "multipoint": "current" | "all",
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from nd2studios.pipeline_graph.model import Edge, GraphSlice

LOOP_CONFIG_VERSION = 1

MODE_SWEEP = "sweep"
MODE_COUNT = "count"
MODE_UNTIL = "until"

COMBINE_GRID = "grid"
COMBINE_ZIP = "zip"

RULE_UNION_DEDUP = "union_dedup"
RULE_BEST = "best"
RULE_LAST = "last"
RULE_KEEP_ALL = "keep_all"

DEDUP_IOU = "iou"
DEDUP_CENTROID = "centroid"

BEST_OBJECT_COUNT = "object_count"
BEST_TRACKING_RATIO = "tracking_ratio"

_DEFAULT_MAX_ITERATIONS = 64


def default_loop_config() -> Dict[str, Any]:
    """A fresh loop config with sensible defaults (fixed 3-iteration repeat)."""
    return {
        "version": LOOP_CONFIG_VERSION,
        "mode": MODE_COUNT,
        "count": 3,
        "max_iterations": _DEFAULT_MAX_ITERATIONS,
        "combine_axes": COMBINE_GRID,
        "axes": [],
        "stop": None,
        "combine": {
            "rule": RULE_LAST,
            "dedup": {
                "metric": DEDUP_IOU,
                "iou_threshold": 0.3,
                "centroid_distance": 10.0,
            },
            "best_metric": BEST_OBJECT_COUNT,
        },
        "multipoint": "current",
        # V1.49.x: retain every iteration's result (masks + rows) so the viewer's
        # iteration dropdown can step through them. Opt-in (holds N× masks in RAM).
        "save_iterations": False,
    }


# ── loop region ──────────────────────────────────────────────────────────────

@dataclass
class LoopRegion:
    """The structural sub-graph a loop edge re-runs each iteration."""

    loop_edge_id: str
    entry: str                       # loop edge destination (top) — restart here
    exit: str                        # loop edge source (bottom) — collect here
    body: Set[str] = field(default_factory=set)  # inclusive of entry & exit

    def contains(self, node_id: str) -> bool:
        return node_id in self.body


def _forward_reachable(sl: GraphSlice, start: str) -> Set[str]:
    seen: Set[str] = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for e in sl.structural_outgoing(cur):
            stack.append(e.dst_node)
    return seen


def _backward_reachable(sl: GraphSlice, start: str) -> Set[str]:
    seen: Set[str] = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for e in sl.structural_incoming(cur):
            stack.append(e.src_node)
    return seen


def loop_region(sl: GraphSlice, loop_edge: Edge) -> LoopRegion:
    """The region a ``loop_edge`` iterates: entry = its dst, exit = its src.

    Body = nodes forward-structural-reachable from ``entry`` that can also reach
    ``exit`` (backward-structural-reachable from ``exit``), inclusive. For a
    self-loop (entry == exit) the body is that single node.
    """
    entry = loop_edge.dst_node
    exit_ = loop_edge.src_node
    fwd = _forward_reachable(sl, entry)
    bwd = _backward_reachable(sl, exit_)
    body = (fwd & bwd) | {entry, exit_}
    body &= set(sl.nodes)
    return LoopRegion(loop_edge_id=loop_edge.id, entry=entry, exit=exit_, body=body)


def loop_entry_map(sl: GraphSlice) -> Dict[str, Edge]:
    """``{entry_node_id: loop_edge}`` for every loop edge in the slice.

    The Run pump consults this to enter loop mode when a ready node is a loop
    entry. If two loop edges share an entry the last one wins (nested loops are
    out of scope for V1.49)."""
    return {e.dst_node: e for e in sl.loop_edges()}


# ── iteration plan ───────────────────────────────────────────────────────────

def expand_axis(axis: Dict[str, Any]) -> List[Any]:
    """Concrete value list for one sweep axis.

    Honors an explicit ``values`` list, else expands ``start``/``stop``/``step``
    inclusively (float-tolerant so 0.3..0.9 step 0.2 yields 0.3/0.5/0.7/0.9)."""
    vals = axis.get("values")
    if isinstance(vals, (list, tuple)) and len(vals) > 0:
        return list(vals)
    try:
        start = float(axis.get("start", 0.0))
        stop = float(axis.get("stop", 0.0))
        step = float(axis.get("step", 1.0))
    except (TypeError, ValueError):
        return []
    if step == 0:
        return [start]
    out: List[Any] = []
    n = int(round((stop - start) / step))
    if n < 0:
        return [start]
    tol = abs(step) * 1e-6
    for i in range(n + 1):
        v = start + i * step
        if (step > 0 and v > stop + tol) or (step < 0 and v < stop - tol):
            break
        out.append(round(v, 10))
    return out


Assignment = Dict[str, Dict[str, Any]]  # node_id -> {param_name: value}


def iteration_plan(config: Dict[str, Any]) -> List[Assignment]:
    """Concrete per-iteration parameter assignments for a loop config.

    * ``sweep`` — Cartesian product (``grid``) or paired (``zip``) of the axes'
      value lists; each element is ``{node_id: {param: value}}``.
    * ``count`` — ``count`` empty assignments (repeat the body unchanged).
    * ``until`` — the sweep grid when axes exist, else ``max_iterations`` empty
      assignments; the page stops early once the stop condition is met.

    All plans are truncated to ``max_iterations`` (hard safety cap).
    """
    cfg = config or {}
    mode = str(cfg.get("mode", MODE_COUNT))
    max_iter = max(1, int(cfg.get("max_iterations", _DEFAULT_MAX_ITERATIONS) or 1))
    axes = list(cfg.get("axes", []) or [])

    def _sweep() -> List[Assignment]:
        expanded = [(a, expand_axis(a)) for a in axes]
        expanded = [(a, vs) for (a, vs) in expanded if vs]
        if not expanded:
            return []
        combine = str(cfg.get("combine_axes", COMBINE_GRID))
        assignments: List[Assignment] = []
        if combine == COMBINE_ZIP:
            n = min(len(vs) for (_, vs) in expanded)
            for i in range(n):
                assignments.append(_assign([(a, vs[i]) for (a, vs) in expanded]))
        else:  # grid (Cartesian product)
            def _rec(idx: int, acc: List[Tuple[Dict[str, Any], Any]]):
                if idx == len(expanded):
                    assignments.append(_assign(acc))
                    return
                a, vs = expanded[idx]
                for v in vs:
                    _rec(idx + 1, acc + [(a, v)])
            _rec(0, [])
        return assignments

    if mode == MODE_SWEEP:
        plan = _sweep()
    elif mode == MODE_UNTIL:
        plan = _sweep()
        if not plan:
            plan = [{} for _ in range(max_iter)]
    else:  # count
        plan = [{} for _ in range(max(1, int(cfg.get("count", 1) or 1)))]

    return plan[:max_iter]


def _assign(pairs: List[Tuple[Dict[str, Any], Any]]) -> Assignment:
    out: Assignment = {}
    for axis, value in pairs:
        nid = str(axis.get("node_id", ""))
        param = str(axis.get("param", ""))
        if not nid or not param:
            continue
        out.setdefault(nid, {})[param] = value
    return out


# ── object de-duplication across iterations ──────────────────────────────────

@dataclass
class IterationResult:
    """One iteration's output — the exit node's rows + label masks."""

    index: int
    params: Assignment
    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Label masks keyed by ``(m_position, channel)`` → (T, H, W) int32.
    masks: Dict[Any, np.ndarray] = field(default_factory=dict)


def _row_frame(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("frame", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _row_m(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("m_position", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _row_centroid(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    y = row.get("centroid_y_px")
    x = row.get("centroid_x_px")
    if y is None or x is None:
        y, x = row.get("centroid_y_um"), row.get("centroid_x_um")
    try:
        return (float(y), float(x))
    except (TypeError, ValueError):
        return None


def _mask_for(masks: Dict[Any, np.ndarray], key: Any,
              frame: int) -> Optional[np.ndarray]:
    """Frame ``frame`` of the mask stored under ``key`` (``(m, channel)`` tuple,
    or a plain channel string for single-multipoint callers)."""
    arr = masks.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if not (0 <= frame < arr.shape[0]):
        return None
    return arr[frame]


def _row_bbox(row: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """``(min_row, min_col, max_row, max_col)`` from a measurement row, or None.

    ``compute_measurements`` writes ``bbox_min_row`` / ``bbox_min_col`` /
    ``bbox_max_row`` / ``bbox_max_col`` (skimage regionprops half-open bbox)."""
    try:
        return (int(row["bbox_min_row"]), int(row["bbox_min_col"]),
                int(row["bbox_max_row"]), int(row["bbox_max_col"]))
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_overlap(a: Tuple[int, int, int, int],
                  b: Tuple[int, int, int, int]) -> bool:
    """Do two half-open ``(y0, x0, y1, x1)`` bounding boxes overlap? (cheap gate)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _iou(a_mask: np.ndarray, a_label: int,
         b_mask: np.ndarray, b_label: int,
         a_bb: Optional[Tuple[int, int, int, int]] = None,
         b_bb: Optional[Tuple[int, int, int, int]] = None) -> float:
    """IoU of two labels. When both bounding boxes are known the comparison is
    cropped to their union box (a tiny region), avoiding a full-frame boolean
    allocation per pair — the difference between a sub-second dedup and a
    multi-minute GUI freeze on a dense field."""
    if a_bb is not None and b_bb is not None:
        y0, x0 = min(a_bb[0], b_bb[0]), min(a_bb[1], b_bb[1])
        y1, x1 = max(a_bb[2], b_bb[2]), max(a_bb[3], b_bb[3])
        a = a_mask[y0:y1, x0:x1] == a_label
        b = b_mask[y0:y1, x0:x1] == b_label
    else:
        a = a_mask == a_label
        b = b_mask == b_label
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def deduplicate_objects(
    results: List[IterationResult],
    *,
    metric: str = DEDUP_IOU,
    iou_threshold: float = 0.3,
    centroid_distance: float = 10.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Merge objects from every iteration, dropping cross-iteration duplicates.

    Objects are compared **per (segmentation_channel, frame)**. Two objects are
    duplicates when their mask IoU exceeds ``iou_threshold`` (``metric == "iou"``
    and both masks exist) or their centroids are within ``centroid_distance``
    (the fallback, and the sole test for ``metric == "centroid"``). Larger
    objects win ties (accepted first), so partial re-detections are absorbed.

    Returns ``(kept_rows, merged_masks)`` — the merged masks are freshly,
    consecutively re-labelled per (channel, frame) and the kept rows carry the
    new ``label_id`` so downstream nodes see one clean object set.
    """
    # Candidate objects, tagged with their source iteration.
    cands: List[Tuple[int, Dict[str, Any]]] = []
    for res in results:
        for row in res.rows:
            cands.append((res.index, dict(row)))
    # Larger objects first so a full detection beats a partial overlap.
    def _area(c: Tuple[int, Dict[str, Any]]) -> float:
        try:
            return float(c[1].get("area_px", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    cands.sort(key=_area, reverse=True)

    masks_by_iter = {res.index: res.masks for res in results}
    # Accepted objects grouped by (m, channel, frame) for fast neighbor checks —
    # m_position is in the key so objects in different multipoints never collide.
    # Each stored tuple caches the row's iteration, bbox and centroid so the inner
    # loop never re-derives them.
    accepted: Dict[Tuple[int, str, int], List[tuple]] = {}
    kept: List[Tuple[int, Dict[str, Any]]] = []

    for it_idx, row in cands:
        m = _row_m(row)
        ch = str(row.get("segmentation_channel", ""))
        fr = _row_frame(row)
        key = (m, ch, fr)
        mask_key = (m, ch)
        cent = _row_centroid(row)
        bb = _row_bbox(row)
        a_mask = None
        if metric == DEDUP_IOU:
            a_mask = _mask_for(masks_by_iter.get(it_idx, {}), mask_key, fr)
        try:
            lbl_a = int(row.get("label_id", 0))
        except (TypeError, ValueError):
            lbl_a = 0
        dup = False
        for (o_idx, o_row, o_bb, o_cent) in accepted.get(key, []):
            if metric == DEDUP_IOU and a_mask is not None:
                # Cheap bbox gate first: non-overlapping boxes can't be duplicates,
                # so we skip the IoU entirely (the common case on a dense field).
                if bb is not None and o_bb is not None and not _bbox_overlap(bb, o_bb):
                    continue
                b_mask = _mask_for(masks_by_iter.get(o_idx, {}), mask_key, fr)
                if b_mask is not None and a_mask.shape == b_mask.shape:
                    try:
                        lbl_b = int(o_row.get("label_id", 0))
                    except (TypeError, ValueError):
                        lbl_b = 0
                    if _iou(a_mask, lbl_a, b_mask, lbl_b, bb, o_bb) > iou_threshold:
                        dup = True
                        break
                    continue
            # centroid fallback / explicit centroid metric
            if cent is not None and o_cent is not None:
                d = ((cent[0] - o_cent[0]) ** 2 + (cent[1] - o_cent[1]) ** 2) ** 0.5
                if d < centroid_distance:
                    dup = True
                    break
        if not dup:
            accepted.setdefault(key, []).append((it_idx, row, bb, cent))
            kept.append((it_idx, row))

    return _rebuild_merged(kept, masks_by_iter)


def _rebuild_merged(
    kept: List[Tuple[int, Dict[str, Any]]],
    masks_by_iter: Dict[int, Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Paint kept objects into fresh consecutive-labelled masks + relabel rows.

    Masks are keyed by ``(m_position, channel)``; returned merged masks use the
    same key so the page can regroup them per multipoint."""
    # Discover mask shape per (m, channel) from any iteration that has masks.
    shape_by_key: Dict[Tuple[int, str], Tuple[int, int, int]] = {}
    for masks in masks_by_iter.values():
        for mkey, arr in masks.items():
            a = np.asarray(arr)
            if a.ndim == 2:
                a = a[None, ...]
            shape_by_key.setdefault(mkey, a.shape[:3])

    merged: Dict[Tuple[int, str], np.ndarray] = {
        mkey: np.zeros(shp, dtype=np.int32) for mkey, shp in shape_by_key.items()
    }
    next_label: Dict[Tuple[int, str, int], int] = {}
    out_rows: List[Dict[str, Any]] = []

    # Preserve m/channel/frame order for stable, readable label ids.
    kept_sorted = sorted(
        kept, key=lambda kr: (_row_m(kr[1]),
                              str(kr[1].get("segmentation_channel", "")),
                              _row_frame(kr[1])))
    for it_idx, row in kept_sorted:
        new = dict(row)
        m = _row_m(row)
        ch = str(row.get("segmentation_channel", ""))
        fr = _row_frame(row)
        mkey = (m, ch)
        merged_arr = merged.get(mkey)
        if merged_arr is not None and 0 <= fr < merged_arr.shape[0]:
            src = _mask_for(masks_by_iter.get(it_idx, {}), mkey, fr)
            if src is not None and src.shape == merged_arr.shape[1:]:
                try:
                    old_label = int(row.get("label_id", 0))
                except (TypeError, ValueError):
                    old_label = 0
                nl = next_label.get((m, ch, fr), 1)
                # Paint only within the object's bbox (a tiny slice) instead of
                # allocating a full-frame boolean per object.
                bb = _row_bbox(row)
                if bb is not None:
                    y0, x0, y1, x1 = bb
                    dst = merged_arr[fr, y0:y1, x0:x1]
                    dst[src[y0:y1, x0:x1] == old_label] = nl
                else:
                    merged_arr[fr][src == old_label] = nl
                new["label_id"] = nl
                next_label[(m, ch, fr)] = nl + 1
        out_rows.append(new)
    return out_rows, merged


# ── combine iterations ───────────────────────────────────────────────────────

def combine_iterations(
    results: List[IterationResult],
    rule: str,
    *,
    dedup: Optional[Dict[str, Any]] = None,
    best_metric: str = BEST_OBJECT_COUNT,
    n_frames: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Combine per-iteration results into ``(rows, masks)`` for the exit output.

    * ``union_dedup`` — merge all objects and drop overlap duplicates.
    * ``best`` — keep the single iteration maximizing ``best_metric``.
    * ``last`` — keep the final iteration.
    * ``keep_all`` — concatenate every iteration's rows, tagged with
      ``loop_iteration``; masks come from the last iteration (label collisions
      across iterations make a single merged mask meaningless here).
    """
    results = [r for r in results if r is not None]
    if not results:
        return [], {}

    if rule == RULE_UNION_DEDUP:
        d = dedup or {}
        return deduplicate_objects(
            results,
            metric=str(d.get("metric", DEDUP_IOU)),
            iou_threshold=float(d.get("iou_threshold", 0.3)),
            centroid_distance=float(d.get("centroid_distance", 10.0)),
        )

    if rule == RULE_BEST:
        best = max(results, key=lambda r: _score(r, best_metric, n_frames))
        return list(best.rows), dict(best.masks)

    if rule == RULE_KEEP_ALL:
        rows: List[Dict[str, Any]] = []
        for res in results:
            for row in res.rows:
                r = dict(row)
                r["loop_iteration"] = res.index
                rows.append(r)
        return rows, dict(results[-1].masks)

    # last (default)
    return list(results[-1].rows), dict(results[-1].masks)


def _score(res: IterationResult, metric: str, n_frames: int) -> float:
    if metric == BEST_TRACKING_RATIO:
        ratio, _ = tracking_ratio(res.rows, n_frames)
        return ratio
    # object count (default) — number of measured objects
    return float(len(res.rows))


# ── tracking coverage ────────────────────────────────────────────────────────

def tracking_ratio(
    rows: List[Dict[str, Any]], n_frames: int = 0
) -> Tuple[float, int]:
    """(coverage ratio, longest continuous run) of well-tracked frames.

    A frame is "tracked" when it contains at least one object carrying a
    ``track_id``. ``ratio`` = tracked frames / ``n_frames`` (or / observed frame
    count when ``n_frames`` ≤ 0). The second value is the longest run of
    consecutive tracked frames — feeding the ``tracking_coverage`` stop block
    ("≥ X% over ≥ K continuous frames").
    """
    rows = rows or []
    frames_with_track: Set[int] = set()
    all_frames: Set[int] = set()
    for r in rows:
        f = _row_frame(r)
        all_frames.add(f)
        if r.get("track_id") is not None:
            frames_with_track.add(f)
    total = int(n_frames) if n_frames and n_frames > 0 else (
        (max(all_frames) + 1) if all_frames else 0)
    if total <= 0:
        return 0.0, 0
    ratio = len(frames_with_track) / total
    # longest continuous run of tracked frames over [0, total)
    best = cur = 0
    for f in range(total):
        if f in frames_with_track:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return ratio, best
