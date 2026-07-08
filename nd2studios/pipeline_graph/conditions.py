"""If-else condition model + evaluator for the Pipelines tab (V1.45, Phase 2).

Qt-free (backend-purity rule). A :class:`Condition` is a flat list of
:class:`ConditionBlock`s combined with a single boolean operator (``ALL`` = and,
``ANY`` = or); each block may be negated (``NOT``). Blocks span three families —
**results-number**, **object-population**, and **timelapse / tracking** — and all
evaluate over the measurement rows produced by ``results_engine`` (timelapse
blocks additionally need ``track_id`` from ``object_tracker.link_objects``).

The condition is stored JSON-serialized in an if-else node's
``params["condition"]`` and edited by ``widgets/node_board`` 's
``ConditionBuilderDialog``. ``evaluate_condition`` is the Run executor's gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

CMP = [">", ">=", "<", "<=", "==", "!="]
COMBINE = ["ALL", "ANY"]  # ALL = and, ANY = or

# If-else "lens": whether the condition routes the whole result (frame) or splits
# the objects (each object → true/false). In object lens, ``group_by`` decides
# whether a whole tracked cell or a single object-row is routed as a unit.
LENS_FRAME = "Whole frame"
LENS_OBJECT = "Each object"
GROUP_TRACK = "Per track"
GROUP_ROW = "Per row"


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class ConditionBlock:
    kind: str                              # key into BLOCK_KINDS
    params: Dict[str, Any] = field(default_factory=dict)
    negate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params),
                "negate": bool(self.negate)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConditionBlock":
        return cls(kind=d["kind"], params=dict(d.get("params", {})),
                   negate=bool(d.get("negate", False)))


@dataclass
class Condition:
    combine: str = "ALL"
    blocks: List[ConditionBlock] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"combine": self.combine,
                "blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Condition":
        if not d:
            return default_condition()
        return cls(
            combine=str(d.get("combine", "ALL")).upper(),
            blocks=[ConditionBlock.from_dict(b) for b in d.get("blocks", [])],
        )


# ── numeric helpers ──────────────────────────────────────────────────────────

def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _cmp(lhs: float, op: str, rhs: float) -> bool:
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    return False


def _aggregate(vals: List[float], how: str) -> Optional[float]:
    if not vals:
        return None
    if how == "mean":
        return sum(vals) / len(vals)
    if how == "min":
        return min(vals)
    if how == "max":
        return max(vals)
    if how == "count":
        return float(len(vals))
    if how == "median":
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
    return sum(vals) / len(vals)


def _objects_per_frame(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for r in rows:
        f = int(r.get("frame", 0) or 0)
        out[f] = out.get(f, 0) + 1
    return out


def _safe_channel(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name))


def _intensity_key(rows: List[Dict[str, Any]], channel: str) -> Optional[str]:
    """The ``mean_intensity_<safe>`` column for ``channel`` (first such if blank)."""
    if channel:
        key = f"mean_intensity_{_safe_channel(channel)}"
        for r in rows:
            if key in r:
                return key
        return None
    for r in rows:
        for k in r:
            if k.startswith("mean_intensity"):
                return k
    return None


def _metric_value(row: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "mean_intensity":
        for k, v in row.items():
            if k.startswith("mean_intensity"):
                return _num(v)
        return None
    return _num(row.get(metric))


# ── per-block evaluators (each robust: missing data -> False) ─────────────────

def _eval_metric(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    metric = str(p.get("metric", "area_um2"))
    how = str(p.get("aggregate", "any"))
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    vals = [v for v in (_metric_value(r, metric) for r in rows) if v is not None]
    if not vals:
        return False
    if how == "any":
        return any(_cmp(v, op, val) for v in vals)
    if how == "all":
        return all(_cmp(v, op, val) for v in vals)
    agg = _aggregate(vals, how)
    return agg is not None and _cmp(agg, op, val)


def _eval_object_count(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    how = str(p.get("aggregate", "max_per_frame"))
    opf = _objects_per_frame(rows)
    counts = list(opf.values())
    if how == "total":
        n = float(len(rows))
    elif not counts:
        n = 0.0
    elif how == "mean_per_frame":
        n = sum(counts) / len(counts)
    elif how == "min_per_frame":
        n = float(min(counts))
    else:
        n = float(max(counts))
    return _cmp(n, op, val)


def _eval_empty_field(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    # Without the full frame list we treat "empty" as "no objects at all".
    return len(rows) == 0


def _eval_marker_positive(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    cutoff = _num(p.get("cutoff", 0.0)) or 0.0
    measure = str(p.get("measure", "count"))
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    key = _intensity_key(rows, str(p.get("channel", "")))
    if key is None or not rows:
        return False
    pos = sum(1 for r in rows
              if (_num(r.get(key)) is not None and _num(r.get(key)) >= cutoff))
    if measure == "fraction":
        return _cmp(pos / len(rows), op, val)
    return _cmp(float(pos), op, val)


def _eval_colocalization(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    measure = str(p.get("measure", "count"))
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    ka = _intensity_key(rows, str(p.get("channel_a", "")))
    kb = _intensity_key(rows, str(p.get("channel_b", "")))
    ca = _num(p.get("cutoff_a", 0.0)) or 0.0
    cb = _num(p.get("cutoff_b", 0.0)) or 0.0
    if ka is None or kb is None or not rows:
        return False
    pos = 0
    for r in rows:
        va, vb = _num(r.get(ka)), _num(r.get(kb))
        if va is not None and vb is not None and va >= ca and vb >= cb:
            pos += 1
    if measure == "fraction":
        return _cmp(pos / len(rows), op, val)
    return _cmp(float(pos), op, val)


def _eval_track_count(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    ids = {r.get("track_id") for r in rows if r.get("track_id") is not None}
    return _cmp(float(len(ids)), op, val)


def _track_frame_counts(rows: List[Dict[str, Any]]) -> Dict[Any, int]:
    """Per-track frame-appearance count — how many frames each ``track_id`` spans
    in ``rows`` (one measurement row per object per frame). Independent of the
    ``track_length`` column so it works even when Cell-Tracker metrics never ran."""
    out: Dict[Any, int] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None:
            continue
        out[tid] = out.get(tid, 0) + 1
    return out


def _eval_track_persistence(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    """Per-track persistence: keep a track by how long it lasts. ``mode="min"``
    passes when every track spans **≥ frames**; ``mode="max"`` when every track
    spans **≤ frames**. In the object lens (Per track) each group is one track, so
    this reads as "keep this track if it lasts at least / at most N frames"."""
    mode = str(p.get("mode", "At least")).lower()
    is_max = mode.startswith("at most") or mode == "max"
    frames = _num(p.get("frames", 2)) or 0.0
    counts = list(_track_frame_counts(rows).values())
    if not counts:
        return False
    if is_max:
        return all(float(c) <= frames for c in counts)
    return all(float(c) >= frames for c in counts)


def _eval_count_change(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    direction = str(p.get("direction", "increase"))
    mode = str(p.get("mode", "absolute"))
    op = str(p.get("comparator", ">="))
    val = _num(p.get("value", 0.0)) or 0.0
    opf = _objects_per_frame(rows)
    if len(opf) < 2:
        return False
    frames = sorted(opf)
    first, last = opf[frames[0]], opf[frames[-1]]
    delta = (last - first) if direction == "increase" else (first - last)
    if mode == "percent":
        base = first if direction == "increase" else last
        if base == 0:
            return False
        delta = delta / base * 100.0
    return _cmp(float(delta), op, val)


def _eval_nondup_object_count(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    """Object count after de-duplication (V1.49 loop stop condition).

    The loop applies its overlap de-dup to ``rows`` *before* calling the
    evaluator, so here this is simply a total-count comparison — expressing
    "iterate until ≥ N non-duplicate objects are found"."""
    op = str(p.get("comparator", ">="))
    val = _num(p.get("value", 0.0)) or 0.0
    return _cmp(float(len(rows or [])), op, val)


def _eval_tracking_coverage(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    """Tracking coverage (V1.49): ratio ≥ X% over ≥ K continuous frames.

    Backed by :func:`loop.tracking_ratio`. ``n_frames`` (0 = infer from the data)
    is the denominator (e.g. 33). Passes when the fraction of frames containing a
    tracked object is ≥ ``ratio``% **and** the longest run of consecutive tracked
    frames is ≥ ``min_continuous_frames``."""
    from nd2studios.pipeline_graph.loop import tracking_ratio
    ratio_pct = _num(p.get("ratio", 90.0)) or 0.0
    min_run = int(_num(p.get("min_continuous_frames", 20)) or 0)
    n_frames = int(_num(p.get("n_frames", 0)) or 0)
    ratio, longest = tracking_ratio(rows or [], n_frames)
    return (ratio * 100.0) >= ratio_pct and longest >= min_run


def _track_points(rows: List[Dict[str, Any]]) -> Dict[Any, List]:
    """Group ``rows`` by ``track_id`` into frame-ordered ``(frame, x, y, row)``
    point lists. Positions use ``centroid_*_um`` (falling back to ``centroid_*_px``
    per row) so displacements are physical when µm centroids are present. Rows
    without a track id or usable centroid are skipped. Each list is sorted by
    frame so consecutive entries are frame-to-frame steps."""
    tracks: Dict[Any, List] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None:
            continue
        y = _num(r.get("centroid_y_um"))
        x = _num(r.get("centroid_x_um"))
        if y is None or x is None:
            y, x = _num(r.get("centroid_y_px")), _num(r.get("centroid_x_px"))
        if y is None or x is None:
            continue
        tracks.setdefault(tid, []).append((int(r.get("frame", 0) or 0), x, y, r))
    for pts in tracks.values():
        pts.sort(key=lambda t: t[0])
    return tracks


def _basis_kind(p: Dict[str, Any]) -> str:
    """Normalize the ``basis`` param to ``"net"`` / ``"cumulative"`` / ``"per_frame"``."""
    b = str(p.get("basis", "net")).lower()
    if "cumul" in b:
        return "cumulative"
    if "per-frame" in b or "per frame" in b or "step" in b:
        return "per_frame"
    return "net"


def _reject_frame_only(p: Dict[str, Any]) -> bool:
    """True when the block should drop only the outlier frame (not the object)."""
    return "frame" in str(p.get("outlier", "")).lower()


def _step(a: List, b: List) -> float:
    """Euclidean distance between two ``(frame, x, y, row)`` points."""
    return ((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5


def _track_metric_value(pts: List, basis: str) -> Optional[float]:
    """Per-track scalar displacement by ``basis`` (needs ≥ 2 points).

    * ``"net"`` — straight-line distance first → last.
    * ``"cumulative"`` — total path length = Σ frame-to-frame steps.
    * ``"per_frame"`` — the largest single frame-to-frame step (so ``max > value``
      is exactly "any step > value").
    """
    if len(pts) < 2:
        return None
    if basis == "net":
        return _step(pts[0], pts[-1])
    steps = [_step(pts[i - 1], pts[i]) for i in range(1, len(pts))]
    if not steps:
        return None
    return sum(steps) if basis == "cumulative" else max(steps)


def _eval_track_displacement(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    how = str(p.get("aggregate", "mean"))
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    basis = _basis_kind(p)
    disps = [v for v in
             (_track_metric_value(pts, basis) for pts in _track_points(rows).values())
             if v is not None]
    if not disps:
        return False
    agg = _aggregate(disps, how if how in ("mean", "median", "min", "max") else "mean")
    return agg is not None and _cmp(agg, op, val)


def _outlier_frame_row_ids(p: Dict[str, Any], rows: List[Dict[str, Any]]) -> set:
    """``id()`` of each outlier frame row for a per-frame "reject frame only" block.

    Each frame's step is measured from the last **retained** frame (not the raw
    previous frame), so a single spike — a point that jumps out and back, where
    both the out- and return-steps exceed the threshold — is healed by dropping
    only the spike apex: after the apex is rejected the return frame is a normal
    step from the pre-spike position and is kept. A frame whose step from the last
    kept position satisfies the block comparator vs ``value`` is the outlier."""
    op = str(p.get("comparator", ">"))
    val = _num(p.get("value", 0.0)) or 0.0
    out: set = set()
    for pts in _track_points(rows).values():
        if len(pts) < 2:
            continue
        last_kept = pts[0]
        for i in range(1, len(pts)):
            if _cmp(_step(last_kept, pts[i]), op, val):
                out.add(id(pts[i][3]))   # outlier — drop, keep last_kept as anchor
            else:
                last_kept = pts[i]
    return out


def scrub_outlier_frames(cond: Condition,
                         rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove outlier frame rows requested by per-frame "reject frame only" blocks.

    For every ``track_displacement`` block with ``basis = Per-frame step`` and
    ``outlier = Reject frame only``, the frame rows whose incoming step exceeds the
    threshold are dropped, so the outlier frame is removed while the track survives
    on its remaining (in-range) frames. Rows are compared by identity, so callers
    keep the same row objects. Returns ``rows`` unchanged if nothing matched.

    This is inherently object-lens cleaning; see :func:`partition_rows`."""
    rows = rows or []
    drop_ids: set = set()
    for blk in cond.blocks:
        if blk.kind != "track_displacement":
            continue
        if _basis_kind(blk.params) != "per_frame" or not _reject_frame_only(blk.params):
            continue
        drop_ids |= _outlier_frame_row_ids(blk.params, rows)
    if not drop_ids:
        return list(rows)
    return [r for r in rows if id(r) not in drop_ids]


# ── block catalog (UI schema + evaluator) ─────────────────────────────────────
# Each entry: family, label, summary(params)->str, params (UI schema list), and
# an eval(params, rows)->bool. A param schema item: {name, label, type, default,
# choices?}. type ∈ {choice, float, int, channel}. ``channel`` choices are
# injected at dialog time with the live channel names.

# Tracking / Cell-Tracker-derived per-cell columns (from
# ``object_tracker.link_objects`` → ``track_length`` and the Cell-Tracker Metrics
# node → the rest). Listed so an if-else placed after those nodes can branch on
# track length, motion (``speed`` magnitude + ``velocity_x/y`` components), local
# crowding (``cell_density`` / ``neighbor_dist_*``), flow (``local_divergence`` /
# ``local_curl``) and ``self_fold`` directly.
TRACKING_METRICS = ["track_length", "speed", "velocity_x", "velocity_y",
                    "speed_um", "velocity_x_um", "velocity_y_um",
                    "cell_density", "neighbor_dist_mean", "neighbor_dist_std",
                    "local_divergence", "local_curl", "self_fold"]

# Every per-object / per-track numeric column a "Metric comparison" block can
# read (resolved via ``_metric_value`` → ``row.get(metric)``). Covers the base
# measurements, the change-over-time delta, and the tracking / Cell-Tracker
# metrics. "Cells/frame" lives in the separate ``object_count`` block; track
# counts in the tracking-family blocks. The builder unions this with the live
# columns actually present upstream (e.g. per-channel ``mean_intensity_<ch>``).
_METRICS = ["area_um2", "area_px", "volume_um3", "perimeter", "circularity",
            "eccentricity", "solidity", "mean_intensity",
            "delta_area_um2", "delta_area_px"] + TRACKING_METRICS
_OBJ_AGG = ["max_per_frame", "mean_per_frame", "min_per_frame", "total"]


def metric_choices(extra: Optional[List[str]] = None) -> List[str]:
    """The metric-comparison choices: the base ``_METRICS`` plus any ``extra``
    live columns (de-duplicated, order-stable). The builder passes the columns
    discovered upstream so all results from the connected nodes are selectable."""
    out = list(_METRICS)
    seen = set(out)
    for m in extra or []:
        if m and m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _p(name, label, ptype, default, choices=None):
    return {"name": name, "label": label, "type": ptype, "default": default,
            "choices": list(choices or [])}


BLOCK_KINDS: Dict[str, Dict[str, Any]] = {
    "metric": {
        "family": "Results number",
        "label": "Metric comparison",
        "eval": _eval_metric,
        "summary": lambda p: (
            f"{p.get('aggregate','any')} {p.get('metric','area_um2')} "
            f"{p.get('comparator','>')} {p.get('value',0)}"),
        "params": [
            _p("metric", "Metric", "choice", "area_um2", _METRICS),
            _p("aggregate", "Across objects", "choice", "any",
               ["any", "all", "mean", "median", "min", "max", "count"]),
            _p("comparator", "Is", "choice", ">", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "object_count": {
        "family": "Object population",
        "label": "Object count",
        "eval": _eval_object_count,
        "summary": lambda p: (
            f"objects/{p.get('aggregate','max_per_frame')} "
            f"{p.get('comparator','>')} {p.get('value',0)}"),
        "params": [
            _p("aggregate", "Count", "choice", "max_per_frame", _OBJ_AGG),
            _p("comparator", "Is", "choice", ">", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "empty_field": {
        "family": "Object population",
        "label": "Empty field (no objects)",
        "eval": _eval_empty_field,
        "summary": lambda p: "no objects detected",
        "params": [],
    },
    "marker_positive": {
        "family": "Object population",
        "label": "Marker-positive objects",
        "eval": _eval_marker_positive,
        "summary": lambda p: (
            f"{p.get('measure','count')} of {p.get('channel','?')}+ "
            f"(>= {p.get('cutoff',0)}) {p.get('comparator','>')} {p.get('value',0)}"),
        "params": [
            _p("channel", "Channel", "channel", ""),
            _p("cutoff", "Intensity cutoff", "float", 0.0),
            _p("measure", "Measure", "choice", "count", ["count", "fraction"]),
            _p("comparator", "Is", "choice", ">", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "colocalization": {
        "family": "Object population",
        "label": "Co-localization (double-positive)",
        "eval": _eval_colocalization,
        "summary": lambda p: (
            f"{p.get('measure','count')} {p.get('channel_a','?')}+&"
            f"{p.get('channel_b','?')}+ {p.get('comparator','>')} {p.get('value',0)}"),
        "params": [
            _p("channel_a", "Channel A", "channel", ""),
            _p("cutoff_a", "Cutoff A", "float", 0.0),
            _p("channel_b", "Channel B", "channel", ""),
            _p("cutoff_b", "Cutoff B", "float", 0.0),
            _p("measure", "Measure", "choice", "count", ["count", "fraction"]),
            _p("comparator", "Is", "choice", ">", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "track_count": {
        "family": "Timelapse / tracking",
        "label": "Track count",
        "eval": _eval_track_count,
        "summary": lambda p: f"# tracks {p.get('comparator','>')} {p.get('value',0)}",
        "params": [
            _p("comparator", "Is", "choice", ">", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "track_persistence": {
        "family": "Timelapse / tracking",
        "label": "Track persistence",
        "eval": _eval_track_persistence,
        "object_lens_only": True,
        "summary": lambda p: (
            f"track lasts {str(p.get('mode', 'At least')).lower()} "
            f"{p.get('frames', 2)} frame(s)"),
        "params": [
            _p("mode", "Keep tracks lasting", "choice", "At least",
               ["At least", "At most"]),
            _p("frames", "Frames", "int", 2),
        ],
    },
    "count_change": {
        "family": "Timelapse / tracking",
        "label": "Object-count change over time",
        "eval": _eval_count_change,
        "summary": lambda p: (
            f"count {p.get('direction','increase')} ({p.get('mode','absolute')}) "
            f"{p.get('comparator','>=')} {p.get('value',0)}"),
        "params": [
            _p("direction", "Direction", "choice", "increase",
               ["increase", "decrease"]),
            _p("mode", "Measure", "choice", "absolute", ["absolute", "percent"]),
            _p("comparator", "Is", "choice", ">=", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "nondup_object_count": {
        "family": "Object population",
        "label": "Non-duplicate object count (loop)",
        "eval": _eval_nondup_object_count,
        "summary": lambda p: (
            f"non-duplicate objects {p.get('comparator','>=')} {p.get('value',0)}"),
        "params": [
            _p("comparator", "Is", "choice", ">=", CMP),
            _p("value", "Value", "float", 0.0),
        ],
    },
    "tracking_coverage": {
        "family": "Timelapse / tracking",
        "label": "Tracking coverage (loop)",
        "eval": _eval_tracking_coverage,
        "summary": lambda p: (
            f"tracked ≥ {p.get('ratio',90)}% over ≥ "
            f"{p.get('min_continuous_frames',20)} continuous frames"),
        "params": [
            _p("ratio", "Coverage %", "float", 90.0),
            _p("min_continuous_frames", "Min continuous frames", "int", 20),
            _p("n_frames", "Total frames (0 = auto)", "int", 0),
        ],
    },
    "track_displacement": {
        "family": "Timelapse / tracking",
        "label": "Track displacement / motility",
        "eval": _eval_track_displacement,
        "summary": lambda p: (
            f"{p.get('aggregate','mean')} {_basis_kind(p).replace('_', '-')} track "
            f"displacement {p.get('comparator','>')} {p.get('value',0)}"
            + (" (drop outlier frame)"
               if _basis_kind(p) == "per_frame" and _reject_frame_only(p) else "")),
        "params": [
            _p("basis", "Measure", "choice", "Net (first→last)",
               ["Net (first→last)", "Cumulative path", "Per-frame step"]),
            _p("aggregate", "Across tracks", "choice", "mean",
               ["mean", "median", "min", "max"]),
            _p("comparator", "Is", "choice", ">", CMP),
            _p("value", "Value (um)", "float", 0.0),
            _p("outlier", "Per-frame exceed", "choice", "Reject object",
               ["Reject object", "Reject frame only"]),
        ],
    },
}

# Stable family → ordered block-kind list (for the builder's grouped menu).
FAMILY_ORDER = ["Results number", "Object population", "Timelapse / tracking"]


def families() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {fam: [] for fam in FAMILY_ORDER}
    for kind, spec in BLOCK_KINDS.items():
        out.setdefault(spec["family"], []).append(kind)
    return out


def block_label(kind: str) -> str:
    spec = BLOCK_KINDS.get(kind)
    return spec["label"] if spec else kind


def is_object_lens_only(kind: str) -> bool:
    """True for blocks that only make sense in an object lens (Each object / Per
    track) if-else — e.g. per-track persistence. The builder hides these from a
    whole-frame if-else so they never appear where they can't work."""
    return bool((BLOCK_KINDS.get(kind) or {}).get("object_lens_only"))


def block_param_schema(kind: str) -> List[Dict[str, Any]]:
    spec = BLOCK_KINDS.get(kind)
    return [dict(p) for p in spec["params"]] if spec else []


def default_block_params(kind: str) -> Dict[str, Any]:
    return {p["name"]: p["default"] for p in block_param_schema(kind)}


def make_block(kind: str) -> ConditionBlock:
    return ConditionBlock(kind=kind, params=default_block_params(kind))


def default_condition() -> Condition:
    return Condition(combine="ALL", blocks=[make_block("object_count")])


# ── evaluation + description ──────────────────────────────────────────────────

def evaluate_condition(cond: Condition, rows: List[Dict[str, Any]]) -> bool:
    """Evaluate ``cond`` over measurement ``rows`` → True/False.

    No blocks → True (a freshly-added if-else defaults to the true branch).
    Each block is evaluated robustly (missing data → False) then negated if
    flagged; results are combined with ALL (and) / ANY (or).
    """
    rows = rows or []
    if not cond.blocks:
        return True
    results: List[bool] = []
    for blk in cond.blocks:
        fn: Optional[Callable] = (BLOCK_KINDS.get(blk.kind, {}) or {}).get("eval")
        try:
            val = bool(fn(blk.params, rows)) if fn else False
        except Exception:  # noqa: BLE001 — a bad block never breaks a Run
            val = False
        results.append((not val) if blk.negate else val)
    return all(results) if cond.combine.upper() == "ALL" else any(results)


def _partition_groups(rows: List[Dict[str, Any]], group_by: str) -> List[List[Dict[str, Any]]]:
    """Group ``rows`` for object-lens routing. ``"row"`` → one group per row;
    otherwise group by ``(m_position, track_id)`` so a whole tracked cell routes
    as a unit, with untracked rows (no ``track_id``) kept as singletons."""
    if group_by == GROUP_ROW:
        return [[r] for r in rows]
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    singles: List[List[Dict[str, Any]]] = []
    order: List[Any] = []
    for r in rows:
        tid = r.get("track_id")
        if tid is None:
            singles.append([r])
            continue
        key = (r.get("m_position", 0), tid)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [groups[k] for k in order] + singles


def partition_rows(
    cond: Condition,
    rows: List[Dict[str, Any]],
    group_by: str = GROUP_TRACK,
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Split ``rows`` into ``(pass_rows, fail_rows)`` for an object-lens if-else.

    Each group (a tracked cell, or a single row in ``"Per row"`` mode) is routed
    as a unit by reusing :func:`evaluate_condition` over the group's rows — so a
    metric block's ``any`` / ``all`` / ``mean`` aggregate naturally means "across
    the cell's frames" (e.g. ``any speed_um > 20`` flags the whole track). With no
    blocks every row passes. Order within each branch is preserved.

    Before routing, per-frame "reject frame only" outlier frames are scrubbed via
    :func:`scrub_outlier_frames`, so those rows land in **neither** branch (the
    outlier frame is dropped while its track survives on its remaining frames).
    """
    rows = rows or []
    if not cond.blocks:
        return list(rows), []
    rows = scrub_outlier_frames(cond, rows)
    pass_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []
    for group in _partition_groups(rows, group_by):
        (pass_rows if evaluate_condition(cond, group) else fail_rows).extend(group)
    return pass_rows, fail_rows


def describe_block(blk: ConditionBlock) -> str:
    spec = BLOCK_KINDS.get(blk.kind)
    if not spec:
        return blk.kind
    try:
        text = spec["summary"](blk.params)
    except Exception:  # noqa: BLE001
        text = spec["label"]
    return f"NOT ({text})" if blk.negate else text


def describe_condition(cond: Condition) -> str:
    if not cond.blocks:
        return "always true"
    joiner = " AND " if cond.combine.upper() == "ALL" else " OR "
    return joiner.join(describe_block(b) for b in cond.blocks)
