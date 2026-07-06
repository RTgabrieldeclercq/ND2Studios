"""
Object tracker for the Pipelines / Results tabs.

Implements a frame-to-frame Hungarian centroid linker that assigns stable
track IDs to objects across consecutive T frames.  The per-track reference
centroid (and area) **updates every frame** — it follows the object as it
moves rather than anchoring to the first detection — and a track can survive a
short detection gap (occlusion / missed segmentation) before it is retired.
Pure NumPy / SciPy — no Qt imports.

Driven by the "Track Objects" pipeline node (see
``link_objects_with_params``) and called implicitly after
``compute_measurements`` so downstream Review / if-else steps have track ids.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

# Progress callback: fraction in [0, 1] plus a short status message. Kept
# Qt-free so the backend stays importable without PySide6; the GUI's
# ``_TrackJob`` adapts it onto its ``ProgressReporter``.
ProgressCB = Callable[[float, str], None]


# Tracking methods exposed by the "Track Objects" node.  The linker dispatches on
# this so methods slot in as new entries without changing call sites.
#   * METHOD_CENTROID    — Hungarian nearest-neighbor on object centroids.
#   * METHOD_SERIALTRACK — SerialTrack topology PTV (scale/rotation invariant);
#     vendored under ``nd2studios.backend.serialtrack`` and driven via its
#     ``track_coordinates`` path (the object centroids are the pre-detected
#     particles, so no image re-detection happens).
#   * METHOD_CT_TOPOLOGY / METHOD_CT_FINGERPRINT — CellTracker's topology-Hungarian
#     and spatial-fingerprint linkers, vendored under
#     ``nd2studios.backend.celltracker`` (pandas-DataFrame trackers; the per-group
#     ``_link_group_celltracker`` bridges the row-dicts to/from that shape).
METHOD_CENTROID = "Centroid (nearest-neighbor)"
METHOD_SERIALTRACK = "SerialTrack (topology PTV)"
METHOD_CT_TOPOLOGY = "Cell-Tracker: Topology (Hungarian)"
METHOD_CT_FINGERPRINT = "Cell-Tracker: Spatial Fingerprint"
TRACKING_METHODS: List[str] = [
    METHOD_CENTROID, METHOD_SERIALTRACK, METHOD_CT_TOPOLOGY, METHOD_CT_FINGERPRINT,
]


def link_objects(
    rows: List[Dict[str, Any]],
    max_displacement_px: float = 100.0,
    min_track_length: int = 2,
    min_circularity: float = 0.0,
    max_eccentricity: float = 1.0,
    max_size_diff_frac: float = 1.0,
    max_frame_gap: int = 0,
    method: str = METHOD_CENTROID,
    st_mode: str = "Incremental",
    st_n_neighbors: int = 25,
    st_solver: str = "Regularization",
    st_loc_solver: str = "Topology",
    st_n_neighbors_min: int = 1,
    st_smoothness: float = 0.1,
    st_outlier_threshold: float = 5.0,
    st_max_iter: int = 20,
    st_iter_stop_threshold: float = 1e-2,
    st_dist_missing: float = 5.0,
    st_use_prev_results: bool = False,
    ct_n_neighbors: int = 5,
    ct_topo_weight: float = 0.3,
    ct_area_weight: float = 0.3,
    ct_max_gap: int = 3,
    progress_cb: Optional[ProgressCB] = None,
) -> List[Dict[str, Any]]:
    """Assign track_id, track_length, and track_validation to every row.

    Mutates *rows* in-place and returns the same list.

    Parameters
    ----------
    rows:
        List of measurement dicts produced by compute_measurements().
        Each dict must have: segmentation_channel, frame,
        centroid_y_px, centroid_x_px, area_px.
        m_position is optional (defaults to 0 when absent).
    max_displacement_px:
        Maximum centroid displacement (Euclidean, pixels) between linked
        detections.  Objects farther apart than this are not the same track.
    min_track_length:
        Minimum number of frames a track must span to be kept.  Shorter tracks
        receive track_id = None and track_validation = None.
    min_circularity:
        Objects whose circularity (4π·area/perimeter²) is below this value are
        excluded from tracking.  Range 0–1; default 0.0 disables the filter.
    max_eccentricity:
        Objects whose eccentricity exceeds this value are excluded from
        tracking.  Range 0–1; default 1.0 disables the filter.
    max_size_diff_frac:
        Maximum fractional change in object area between linked detections,
        measured as ``|area_a - area_b| / max(area_a, area_b)`` (range 0–1).
        Detections whose size changes by more than this are not linked.
        1.0 disables the size gate.
    max_frame_gap:
        Number of consecutive missed frames a track may bridge before it is
        retired.  0 = the track must be re-detected in the very next frame;
        2 = it may skip up to two frames and re-link afterwards.
    method:
        Tracking method — ``METHOD_CENTROID`` (default) or ``METHOD_SERIALTRACK``.
    st_mode:
        SerialTrack mode, ``"Incremental"`` (link each frame to the previous) or
        ``"Cumulative"`` (link every frame to the first).  Ignored for the
        centroid method.
    st_n_neighbors:
        SerialTrack topology-descriptor neighbor count (``n_neighbors_max``).
        Ignored for the centroid method.
    st_solver:
        SerialTrack global-step solver: ``"MLS"`` (mesh-free moving least
        squares), ``"Regularization"`` (scatter→grid smoothing; default), or
        ``"ADMM"`` (augmented-Lagrangian with automatic L-curve α — most faithful
        to the paper, costlier).  SerialTrack only.
    st_loc_solver:
        SerialTrack local matcher: ``"Topology"`` or
        ``"Histogram then Topology"``.  SerialTrack only.
    st_n_neighbors_min:
        Floor for the exponential neighbor-count decay across iterations
        (``n_neighbors_min``).  SerialTrack only.
    st_smoothness:
        Global smoothing strength (the ``α/µ`` knob; used by Regularization and
        ADMM).  SerialTrack only.
    st_outlier_threshold:
        Westerweel normalized-median-residual cutoff (``0`` disables).
        SerialTrack only.
    st_max_iter:
        Max ADMM iterations per frame pair.  SerialTrack only.
    st_iter_stop_threshold:
        ADMM convergence threshold on the displacement-update norm.  SerialTrack
        only.
    st_dist_missing:
        Ghost-particle cull distance ``ε_d`` (px), active in late iterations.
        SerialTrack only.
    st_use_prev_results:
        Enable the data-driven initial-guess predictor (warm start) for frames
        ≥3.  The POD-GPR stage (frames ≥7) needs scikit-learn.  SerialTrack only.
    ct_n_neighbors:
        CellTracker topology-Hungarian neighbor count for the rotation-invariant
        descriptor.  Used only by ``METHOD_CT_TOPOLOGY``.
    ct_topo_weight:
        CellTracker topology cost weight (0–1) blended with raw distance.  Used
        only by ``METHOD_CT_TOPOLOGY``.
    ct_area_weight:
        CellTracker fingerprint area-similarity weight (0–1) vs. distance.  Used
        only by ``METHOD_CT_FINGERPRINT``.
    ct_max_gap:
        CellTracker fingerprint gap-filling budget — frames a track may vanish
        and still re-link.  Used only by ``METHOD_CT_FINGERPRINT``.

    Returns
    -------
    The same list, with three new keys added to every dict:

    track_id          int | None  — None if excluded from tracking or
                                    track_length < min_track_length
    track_length      int         — frames the track spans
    track_validation  str | None  — "unvalidated" if track_id is not None,
                                    else None
    """
    min_track_length = max(1, int(min_track_length))
    max_frame_gap = max(0, int(max_frame_gap))

    # ── Initialise all rows ───────────────────────────────────────────────────
    for r in rows:
        r["track_id"] = None
        r["track_length"] = 1
        r["track_validation"] = None

    if not rows:
        return rows

    # ── Morphology filter — ineligible rows keep track_id = None ─────────────
    def _passes(r: Dict[str, Any]) -> bool:
        circ = r.get("circularity")
        if circ is not None and float(circ) < min_circularity:
            return False
        ecc = r.get("eccentricity")
        if ecc is not None and float(ecc) > max_eccentricity:
            return False
        return True

    # ── Group eligible rows by (channel, m_position) ─────────────────────────
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not _passes(r):
            continue
        key = (
            str(r.get("segmentation_channel", "")),
            int(r.get("m_position", 0)),
        )
        groups[key].append(r)

    _next_track_id = [1]  # mutable counter shared across groups

    n_groups = max(1, len(groups))
    log.info(
        "link_objects: method=%r, %d eligible detections in %d (channel, m) "
        "group(s)", method, sum(len(g) for g in groups.values()), n_groups,
    )
    _t_start = time.perf_counter()

    for gi, group_rows in enumerate(groups.values()):
        # Map each linker's own 0..1 progress into this group's slice of the
        # overall bar, so multi-group runs still advance smoothly and a
        # single-group run (the common case) passes progress straight through.
        def _group_cb(frac: float, msg: str, _gi: int = gi) -> None:
            if progress_cb is not None:
                progress_cb((_gi + max(0.0, min(1.0, frac))) / n_groups, msg)

        if method == METHOD_SERIALTRACK:
            _link_group_serialtrack(
                group_rows, max_displacement_px, st_mode, st_n_neighbors,
                _next_track_id,
                solver_str=st_solver, loc_solver_str=st_loc_solver,
                n_neighbors_min=st_n_neighbors_min, smoothness=st_smoothness,
                outlier_threshold=st_outlier_threshold, max_iter=st_max_iter,
                iter_stop_threshold=st_iter_stop_threshold,
                dist_missing=st_dist_missing, use_prev_results=st_use_prev_results,
                progress_cb=_group_cb,
            )
        elif method in (METHOD_CT_TOPOLOGY, METHOD_CT_FINGERPRINT):
            _link_group_celltracker(
                group_rows, method, max_displacement_px,
                ct_n_neighbors, ct_topo_weight, ct_area_weight, ct_max_gap,
                _next_track_id, progress_cb=_group_cb,
            )
        else:
            _link_group(
                group_rows, max_displacement_px, max_size_diff_frac,
                max_frame_gap, _next_track_id, progress_cb=_group_cb,
            )

    log.info("link_objects: linking finished in %.2fs (%d tracks assigned)",
             time.perf_counter() - _t_start, _next_track_id[0] - 1)
    if progress_cb is not None:
        progress_cb(1.0, "Tracking done")

    # ── Promote track_length and track_validation ─────────────────────────────
    track_frames: Dict[int, int] = defaultdict(int)
    for r in rows:
        tid = r["track_id"]
        if tid is not None:
            track_frames[tid] += 1

    for r in rows:
        tid = r["track_id"]
        if tid is None:
            continue
        n = track_frames[tid]
        if n < min_track_length:
            r["track_id"] = None
            r["track_length"] = n
        else:
            r["track_length"] = n
            r["track_validation"] = "unvalidated"

    return rows


def link_objects_with_params(
    rows: List[Dict[str, Any]],
    params: Dict[str, Any],
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> List[Dict[str, Any]]:
    """Run :func:`link_objects` from a "Track Objects" node param dict.

    Translates the node's user-facing knobs into :func:`link_objects` arguments,
    converting a µm distance threshold to pixels via *pixel_size_um* when
    available.  Knobs:

    * shared — ``method``, ``max_distance`` with ``distance_unit`` of pixels/µm
      (becomes ``max_displacement_px``; SerialTrack and the CellTracker linkers
      use it as the field of search / max link distance), ``min_track_length``.
    * centroid only — ``max_size_diff``, ``max_frame_gap``.
    * SerialTrack only — ``st_mode``, ``st_n_neighbors``, ``st_solver``,
      ``st_loc_solver``, ``st_n_neighbors_min``, ``st_smoothness``,
      ``st_outlier_threshold``, ``st_max_iter``, ``st_iter_stop_threshold``,
      ``st_dist_missing``, ``st_use_prev_results``.
    * Cell-Tracker topology only — ``ct_n_neighbors``, ``ct_topo_weight``.
    * Cell-Tracker fingerprint only — ``ct_area_weight``, ``ct_max_gap``.

    Params that don't apply to the chosen method are passed through to
    :func:`link_objects` but ignored by the active code path.
    """
    params = params or {}
    max_distance = float(params.get("max_distance", 100.0))
    unit = str(params.get("distance_unit", "pixels"))
    if unit == "µm" and pixel_size_um:
        max_distance = max_distance / float(pixel_size_um)
    return link_objects(
        rows,
        max_displacement_px=max_distance,
        min_track_length=int(params.get("min_track_length", 2)),
        max_size_diff_frac=float(params.get("max_size_diff", 1.0)),
        max_frame_gap=int(params.get("max_frame_gap", 0)),
        method=str(params.get("method", METHOD_CENTROID)),
        st_mode=str(params.get("st_mode", "Incremental")),
        st_n_neighbors=int(params.get("st_n_neighbors", 25)),
        st_solver=str(params.get("st_solver", "Regularization")),
        st_loc_solver=str(params.get("st_loc_solver", "Topology")),
        st_n_neighbors_min=int(params.get("st_n_neighbors_min", 1)),
        st_smoothness=float(params.get("st_smoothness", 0.1)),
        st_outlier_threshold=float(params.get("st_outlier_threshold", 5.0)),
        st_max_iter=int(params.get("st_max_iter", 20)),
        st_iter_stop_threshold=float(params.get("st_iter_stop_threshold", 1e-2)),
        st_dist_missing=float(params.get("st_dist_missing", 5.0)),
        st_use_prev_results=bool(params.get("st_use_prev_results", False)),
        ct_n_neighbors=int(params.get("ct_n_neighbors", 5)),
        ct_topo_weight=float(params.get("ct_topo_weight", 0.3)),
        ct_area_weight=float(params.get("ct_area_weight", 0.3)),
        ct_max_gap=int(params.get("ct_max_gap", 3)),
        progress_cb=progress_cb,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

# active[track_id] layout: (cy, cx, area, last_frame)
_Active = Tuple[float, float, float, int]


def _link_group(
    group_rows: List[Dict[str, Any]],
    max_displacement_px: float,
    max_size_diff_frac: float,
    max_frame_gap: int,
    next_id: List[int],
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link objects within a single (channel, m_position) group.

    Walks the detected frames in order.  Each track keeps its *last matched*
    centroid + area + frame, so the reference moves with the object.  A track
    not re-detected this frame is carried forward (up to ``max_frame_gap``
    missed frames) instead of being dropped, which lets it re-link across a gap.
    """
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in group_rows:
        frames[int(r.get("frame", 0))].append(r)

    sorted_frames = sorted(frames.keys())
    T = len(sorted_frames)
    if T < 2:
        return

    # Seed first frame with fresh track IDs.
    active: Dict[int, _Active] = {}
    for r in frames[sorted_frames[0]]:
        r["track_id"] = next_id[0]
        active[next_id[0]] = (_cy(r), _cx(r), _area(r), sorted_frames[0])
        next_id[0] += 1

    for i, fr in enumerate(sorted_frames[1:], start=1):
        if progress_cb is not None:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
        curr_rows = frames[fr]
        if not curr_rows:
            continue

        # Retire tracks that have now exceeded the allowed gap.
        active = {
            tid: v for tid, v in active.items()
            if (fr - v[3] - 1) <= max_frame_gap
        }

        prev_track_ids = list(active.keys())
        if not prev_track_ids:
            for r in curr_rows:
                r["track_id"] = next_id[0]
                active[next_id[0]] = (_cy(r), _cx(r), _area(r), fr)
                next_id[0] += 1
            continue

        # Cost = Euclidean centroid distance, gated by distance + size change.
        prev_vals = [active[tid] for tid in prev_track_ids]
        prev_centroids = np.array([[v[0], v[1]] for v in prev_vals], dtype=np.float64)
        prev_areas = np.array([v[2] for v in prev_vals], dtype=np.float64)
        curr_centroids = np.array([(_cy(r), _cx(r)) for r in curr_rows], dtype=np.float64)
        curr_areas = np.array([_area(r) for r in curr_rows], dtype=np.float64)

        diff = prev_centroids[:, None, :] - curr_centroids[None, :, :]  # (P, C, 2)
        cost = np.sqrt((diff ** 2).sum(axis=2))                          # (P, C)

        # Size-difference gate: |Δarea| / max(area) must stay within the
        # threshold, else the pair cannot be the same object.
        area_diff = np.abs(prev_areas[:, None] - curr_areas[None, :])
        area_max = np.maximum.outer(prev_areas, curr_areas)
        area_max[area_max <= 0.0] = 1.0
        size_diff = area_diff / area_max
        blocked = size_diff > max_size_diff_frac
        cost[blocked] = max_displacement_px + 1.0

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_curr: set = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] <= max_displacement_px:
                tid = prev_track_ids[ri]
                curr_rows[ci]["track_id"] = tid
                active[tid] = (_cy(curr_rows[ci]), _cx(curr_rows[ci]),
                               _area(curr_rows[ci]), fr)
                matched_curr.add(ci)

        # Unmatched current objects start new tracks.  Unmatched *previous*
        # tracks stay in `active` with their old last_frame, so they remain
        # eligible to re-link until the gap budget runs out.
        for ci, r in enumerate(curr_rows):
            if ci not in matched_curr:
                r["track_id"] = next_id[0]
                active[next_id[0]] = (_cy(r), _cx(r), _area(r), fr)
                next_id[0] += 1


def _link_group_serialtrack(
    group_rows: List[Dict[str, Any]],
    f_o_s: float,
    mode_str: str,
    n_neighbors_max: int,
    next_id: List[int],
    *,
    solver_str: str = "Regularization",
    loc_solver_str: str = "Topology",
    n_neighbors_min: int = 1,
    smoothness: float = 0.1,
    outlier_threshold: float = 5.0,
    max_iter: int = 20,
    iter_stop_threshold: float = 1e-2,
    dist_missing: float = 5.0,
    use_prev_results: bool = False,
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link objects within one (channel, m_position) group via SerialTrack.

    The object centroids are fed to SerialTrack's ``track_coordinates`` path as
    pre-detected particles (no image re-detection).  ``f_o_s`` is the field of
    search (max neighbor radius, px) — the node's "Max distance" knob.  The
    remaining keyword args are SerialTrack's own tunables (global/local solver,
    neighbor-count decay, smoothing, outlier + ghost-cull thresholds, ADMM
    iteration budget, warm-start predictor).  Each detection's track id is chained
    from the per-frame ``track_b2a`` index maps; ``min_track_length`` filtering
    happens in the caller's post-pass.

    The SerialTrack library (numba JIT) is imported lazily so a missing optional
    dependency surfaces only for this method and is caught by the node handlers.
    """
    from nd2studios.backend.serialtrack.config import (
        DetectionConfig, GlobalSolver, LocalSolver, TrackingConfig, TrackingMode,
    )
    from nd2studios.backend.serialtrack.tracking import SerialTracker

    # Group by frame, keep only frames that actually have detections, in order.
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in group_rows:
        frames[int(r.get("frame", 0))].append(r)
    sorted_frames = [f for f in sorted(frames.keys()) if frames[f]]
    if len(sorted_frames) < 2:
        return  # nothing to link (mirrors _link_group's early-out)

    # coords_list[i] aligns row-for-row with row_refs[i] (same object order).
    coords_list: List[np.ndarray] = []
    row_refs: List[List[Dict[str, Any]]] = []
    for fr in sorted_frames:
        rows_f = frames[fr]
        coords_list.append(
            np.array([[_cy(r), _cx(r)] for r in rows_f], dtype=np.float64)
        )
        row_refs.append(rows_f)

    mode = (TrackingMode.CUMULATIVE if mode_str == "Cumulative"
            else TrackingMode.INCREMENTAL)
    global_solver = {
        "MLS": GlobalSolver.MLS,
        "Regularization": GlobalSolver.REGULARIZATION,
        "ADMM": GlobalSolver.ADMM,
    }.get(solver_str, GlobalSolver.REGULARIZATION)
    local_solver = {
        "Topology": LocalSolver.TOPOLOGY,
        "Histogram then Topology": LocalSolver.HISTOGRAM_THEN_TOPOLOGY,
    }.get(loc_solver_str, LocalSolver.TOPOLOGY)
    n_max = max(2, int(n_neighbors_max))
    trk = TrackingConfig(
        mode=mode,
        f_o_s=float(f_o_s),
        n_neighbors_max=n_max,
        n_neighbors_min=min(n_max, max(1, int(n_neighbors_min))),
        loc_solver=local_solver,
        solver=global_solver,
        smoothness=max(0.0, float(smoothness)),
        outlier_threshold=max(0.0, float(outlier_threshold)),
        max_iter=max(1, int(max_iter)),
        iter_stop_threshold=max(0.0, float(iter_stop_threshold)),
        dist_missing=max(0.0, float(dist_missing)),
        use_prev_results=bool(use_prev_results),
        strain_n_neighbors=0,      # skip per-frame strain (not needed for ids)
    )
    # SerialTrack runs as one opaque (numba-JIT) call — no intra-call progress
    # hook — so we mark the start; per-frame updates follow in the chaining loop.
    if progress_cb is not None:
        progress_cb(0.0, f"SerialTrack linking {len(sorted_frames)} frames…")
    try:
        session = SerialTracker(
            DetectionConfig(), trk).track_coordinates(coords_list)
    except ImportError as exc:
        # The only optional import on this path is scikit-learn, pulled in by the
        # POD-GPR warm start (frames ≥7) when use_prev_results is on.
        if use_prev_results:
            raise RuntimeError(
                "SerialTrack 'Use previous results' (POD-GPR warm start) needs "
                "scikit-learn for sequences of 7+ frames — install it "
                "(pip install scikit-learn) or turn the option off."
            ) from exc
        raise

    # Seed the reference (first) frame with fresh ids, then chain forward.
    ids_per_frame: List[List[int]] = [[] for _ in row_refs]
    for r in row_refs[0]:
        r["track_id"] = next_id[0]
        ids_per_frame[0].append(next_id[0])
        next_id[0] += 1

    n_pairs = max(1, len(session.frame_results))
    for k, res in enumerate(session.frame_results):
        if progress_cb is not None:
            progress_cb((k + 1) / n_pairs, f"Chaining frame {k + 2}/{len(row_refs)}")
        # res is the pair whose B is coords_list[k + 1].
        prev_ids = ids_per_frame[0] if mode == TrackingMode.CUMULATIVE \
            else ids_per_frame[k]
        t_b2a = np.asarray(res.track_b2a)
        b_rows = row_refs[k + 1]
        b_ids: List[int] = []
        for j, rrow in enumerate(b_rows):
            a = int(t_b2a[j]) if j < len(t_b2a) else -1
            if 0 <= a < len(prev_ids):
                tid = prev_ids[a]
            else:                       # appeared / untracked → new track
                tid = next_id[0]
                next_id[0] += 1
            rrow["track_id"] = tid
            b_ids.append(tid)
        ids_per_frame[k + 1] = b_ids


def _link_group_celltracker(
    group_rows: List[Dict[str, Any]],
    method: str,
    max_displacement_px: float,
    ct_n_neighbors: int,
    ct_topo_weight: float,
    ct_area_weight: float,
    ct_max_gap: int,
    next_id: List[int],
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link objects within one (channel, m_position) group via CellTracker.

    Bridges ND2Studios' row-dicts to CellTracker's DataFrame convention: each row
    becomes a (``frame``, ``label``, ``centroid_y``, ``centroid_x``, ``area``)
    record (``label`` = the per-frame ``label_id``), the chosen vendored linker
    runs, and the resulting per-call ``track_id`` (1-based) is remapped onto the
    shared global ``next_id`` counter so ids never collide across groups.  The
    caller's post-pass handles ``track_length`` / ``min_track_length``.

    pandas / CellTracker are imported lazily so a missing optional dependency
    surfaces only for these methods and is caught by the node handlers.
    """
    import pandas as pd

    from nd2studios.backend.celltracker.tracking import (
        track_fingerprint, track_timeseries,
    )

    # Build the DataFrame; keep a parallel handle from (frame, label) back to the
    # originating row so the assigned id can be written in place.
    records: List[Dict[str, Any]] = []
    row_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in group_rows:
        fr = int(r.get("frame", 0))
        lbl = int(r.get("label_id", 0))
        records.append({
            "frame": fr,
            "label": lbl,
            "centroid_y": _cy(r),
            "centroid_x": _cx(r),
            "area": _area(r),
        })
        row_by_key[(fr, lbl)] = r

    df = pd.DataFrame(records)
    if df.empty or df["frame"].nunique() < 2:
        return  # nothing to link (mirrors _link_group's early-out)

    if method == METHOD_CT_FINGERPRINT:
        tracked = track_fingerprint(
            df, max_dist=float(max_displacement_px),
            area_weight=float(ct_area_weight), max_gap=max(0, int(ct_max_gap)),
            progress_cb=progress_cb,
        )
    else:  # METHOD_CT_TOPOLOGY
        tracked = track_timeseries(
            df, max_dist=float(max_displacement_px),
            n_neighbors=max(1, int(ct_n_neighbors)),
            use_topology=True, topo_weight=float(ct_topo_weight),
            progress_cb=progress_cb,
        )

    # Remap local (per-call) track ids to the shared global counter, skipping the
    # -1 "unassigned" sentinel, then write onto the originating rows.
    local_to_global: Dict[int, int] = {}
    for rec in tracked.itertuples(index=False):
        local = int(getattr(rec, "track_id"))
        if local < 0:
            continue
        if local not in local_to_global:
            local_to_global[local] = next_id[0]
            next_id[0] += 1
        row = row_by_key.get((int(rec.frame), int(rec.label)))
        if row is not None:
            row["track_id"] = local_to_global[local]


def _cy(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_y_px") or 0.0)


def _cx(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_x_px") or 0.0)


def _area(row: Dict[str, Any]) -> float:
    """Object area in pixels, used by the size-difference gate (0 if absent)."""
    return float(row.get("area_px") or 0.0)
