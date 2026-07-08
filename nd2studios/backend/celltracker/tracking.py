"""Cell tracking: Hungarian LAP (with birth/death), topology, and mask overlap.

Vendored from CellTracker ``backend/tracking.py`` (topology features inspired by
SerialTrack's matching.py). Only the headless ``track_timeseries`` (topology
Hungarian) and ``track_fingerprint`` (position + area, gap filling) linkers and
their helpers are copied — CellTracker's ``track_serialtrack`` is omitted (it
carries a hardcoded path and is already covered by
:mod:`nd2studios.backend.serialtrack`).

Operates on pandas DataFrames with columns ``frame``, ``label``, ``centroid_y``,
``centroid_x`` (and ``area`` for the fingerprint linker); returns a copy with a
``track_id`` column added.

V1.58 — the per-frame assignment now uses a Jaqaman-style LAP with birth/death
"no-match" nodes (:func:`solve_lap`) rather than a bare ``linear_sum_assignment``.
A bare complete matching is forced to link the smaller side in full, which (a)
manufactures spurious long links on count imbalance (high ``max_dist``) and (b)
shuffles a whole neighborhood onto each other's targets when a true partner is
just out of range ("tracking currents"; low ``max_dist``). Letting a detection
stay unmatched at a fixed cost removes both. V1.58 also adds :func:`track_overlap`
— a mask-IoU linker that consumes the StarDist label masks directly, the robust
default for dense, slow-moving nuclei where centroid distance is ambiguous.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)

# A progress callback reports a fraction in [0, 1] and a short status message.
# Kept Qt-free so the backend stays importable without PySide6.
ProgressCB = Callable[[float, str], None]

# A per-frame cost matrix larger than this (N × M entries) makes the O(n³)
# Hungarian assignment the dominant cost; we log a one-time warning so a
# dense-field slowdown is explained rather than mysterious.
_BIG_COST_MATRIX = 4_000_000

# Finite stand-in for a forbidden pair inside the augmented cost matrix. The
# optimum never selects one (a birth+death pair is always cheaper), so its only
# job is to dominate any achievable real assignment total without overflowing.
_BIG = 1.0e12


# ═══════════════════════════════════════════════════════════════
#  Assignment: Hungarian LAP with birth/death (no-match) nodes
# ═══════════════════════════════════════════════════════════════

def _solve_lap_block(cost: np.ndarray, no_match_cost: float) -> List[Tuple[int, int]]:
    """Optimal assignment of one dense cost block, births/deaths allowed.

    Builds the Jaqaman (2008) augmented matrix and returns the accepted
    ``(row, col)`` links. ``cost`` may contain ``np.inf`` for forbidden pairs;
    those are never returned. Intended for a single connected component, so the
    block stays small even on a dense field (see :func:`solve_lap`).
    """
    N, M = cost.shape
    d = float(no_match_cost)
    size = N + M
    aug = np.full((size, size), _BIG, dtype=np.float64)
    # Q1 — real linking costs (forbidden → _BIG so they're never chosen).
    aug[:N, :M] = np.where(np.isfinite(cost), cost, _BIG)
    # Q2 — each prev may "die" on its own dummy column (diagonal = d).
    death = np.full((N, N), _BIG, dtype=np.float64)
    np.fill_diagonal(death, d)
    aug[:N, M:] = death
    # Q3 — each curr may be "born" on its own dummy row (diagonal = d).
    birth = np.full((M, M), _BIG, dtype=np.float64)
    np.fill_diagonal(birth, d)
    aug[N:, :M] = birth
    # Q4 — dummy↔dummy pairings are free; they only absorb the leftover
    # birth-rows / death-cols left over once real links are chosen.
    aug[N:, M:] = 0.0

    row_ind, col_ind = linear_sum_assignment(aug)
    out: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and np.isfinite(cost[r, c]):
            out.append((int(r), int(c)))
    return out


def solve_lap(
    cost: np.ndarray,
    no_match_cost: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve one frame-to-frame assignment allowing births and deaths.

    The Jaqaman et al. (2008) linear-assignment formulation used by u-track /
    TrackMate: the ``N×M`` linking-cost block is augmented with dummy "no-match"
    nodes so a detection may stay **unmatched at a fixed cost** (``no_match_cost``)
    instead of being force-linked. Without them a bare ``linear_sum_assignment``
    must return a complete matching of the smaller side, which forces spurious
    long links on count imbalance and shuffles neighborhoods onto each other's
    targets ("tracking currents") when a true partner is just out of range. An
    honest birth/death is cheaper than either, so the dummies remove both.

    For speed on dense fields the gated cost matrix (most pairs ``inf``) is split
    into connected components of finite-cost edges and each small component is
    solved independently — exact, because a node's only non-linking option is its
    own birth/death dummy, so components never couple.

    Parameters
    ----------
    cost : (N, M) float array
        Linking cost per prev→curr pair; forbidden pairs marked ``np.inf`` are
        never accepted.
    no_match_cost : float
        Cost ``d`` of leaving a detection unmatched (a birth or a death). A link
        is preferred only when cheaper than ``d``, so ``d`` is also a soft gate
        on top of the hard ``inf`` gate.

    Returns
    -------
    matches : list of (prev_idx, curr_idx)
    unmatched_prev, unmatched_curr : list of indices
    """
    N, M = cost.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    unmatched_prev = set(range(N))
    unmatched_curr = set(range(M))
    matches: List[Tuple[int, int]] = []

    finite = np.isfinite(cost)
    if not finite.any():
        return [], list(range(N)), list(range(M))

    # Connected components over the bipartite graph of finite (allowed) edges.
    # Prev node i → graph node i; curr node j → graph node N + j. Nodes with no
    # allowed edge fall out as singletons and stay unmatched (birth / death).
    pr, cu = np.nonzero(finite)
    n_nodes = N + M
    data = np.ones(pr.size, dtype=np.int8)
    adj = coo_matrix((data, (pr, N + cu)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj, directed=False)

    for comp in range(n_comp):
        node_ids = np.nonzero(labels == comp)[0]
        rows = node_ids[node_ids < N]
        cols = node_ids[node_ids >= N] - N
        if rows.size == 0 or cols.size == 0:
            continue  # a lone prev (death) or lone curr (birth)
        sub = cost[np.ix_(rows, cols)]
        for r, c in _solve_lap_block(sub, no_match_cost):
            pi, ci = int(rows[r]), int(cols[c])
            matches.append((pi, ci))
            unmatched_prev.discard(pi)
            unmatched_curr.discard(ci)

    return matches, sorted(unmatched_prev), sorted(unmatched_curr)


# ═══════════════════════════════════════════════════════════════
#  Topology features
# ═══════════════════════════════════════════════════════════════

def compute_topology_features(
    centroids: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    For each cell, compute rotation-invariant topology features:
    sorted distances and angular gaps to K nearest neighbors.

    Parameters
    ----------
    centroids : (N, 2) array of [y, x] positions
    n_neighbors : int, number of neighbors to use

    Returns
    -------
    features : (N, 2*K) array — first K columns are sorted distances,
               next K columns are sorted angular gaps
    """
    N = len(centroids)
    if N < 2:
        return np.zeros((N, 2 * n_neighbors))

    K = min(n_neighbors, N - 1)
    tree = cKDTree(centroids)
    dists, indices = tree.query(centroids, k=K + 1)  # +1 for self

    # Remove self (first column)
    dists = dists[:, 1:]       # (N, K)
    indices = indices[:, 1:]   # (N, K)

    features = np.zeros((N, 2 * n_neighbors))

    # Vectorized equivalent of the original per-cell loop. cKDTree returns a
    # uniform K neighbors for every cell (k == K here), so the whole thing is
    # array ops. The ``k > 1`` guard preserves the original behavior of leaving
    # the angular-gap block as zeros when there is only a single neighbor.
    k = K
    # Sorted neighbor distances (cKDTree already returns them ascending).
    features[:, :k] = dists[:, :k]

    if k > 1:
        # Neighbor offset vectors, (N, k, 2) as [y, x].
        neighbors = centroids[indices[:, :k]] - centroids[:, None, :]
        angles = np.arctan2(neighbors[:, :, 0], neighbors[:, :, 1])  # (N, k)
        angles_sorted = np.sort(angles, axis=1)
        gaps = np.diff(angles_sorted, axis=1)                        # (N, k-1)
        wrap = (2 * np.pi + angles_sorted[:, 0] - angles_sorted[:, -1])[:, None]
        gaps = np.concatenate([gaps, wrap], axis=1)                  # (N, k)
        gaps_sorted = np.sort(gaps, axis=1)
        features[:, n_neighbors:n_neighbors + k] = gaps_sorted

    return features


# ═══════════════════════════════════════════════════════════════
#  Frame-to-frame linking
# ═══════════════════════════════════════════════════════════════

def link_frames(
    centroids_prev: np.ndarray,
    centroids_curr: np.ndarray,
    max_dist: float = 30.0,
    topo_prev: Optional[np.ndarray] = None,
    topo_curr: Optional[np.ndarray] = None,
    topo_weight: float = 0.3,
    no_match_cost: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Link detections between two consecutive frames with a birth/death LAP.

    Parameters
    ----------
    centroids_prev, centroids_curr : (N, 2) and (M, 2) arrays
    max_dist : float, maximum linking distance in pixels (hard gate)
    topo_prev, topo_curr : topology feature arrays (optional)
    topo_weight : float 0-1, weight of topology cost vs distance cost
    no_match_cost : float, optional
        Cost of leaving a detection unmatched (birth/death). Defaults to
        ``max_dist``. See :func:`solve_lap` — this is what stops a cell whose
        true partner is out of range from being force-linked onto a neighbor
        (the "tracking currents" failure mode).

    Returns
    -------
    matches : list of (prev_idx, curr_idx)
    unmatched_prev : list of indices
    unmatched_curr : list of indices
    """
    N = len(centroids_prev)
    M = len(centroids_curr)

    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    # Distance cost
    dist_cost = cdist(centroids_prev, centroids_curr)

    # Topology cost (if available)
    if topo_prev is not None and topo_curr is not None and topo_weight > 0:
        topo_cost = cdist(topo_prev, topo_curr, metric="euclidean")
        # Normalize topology cost to same scale as distance
        topo_scale = np.median(dist_cost[dist_cost < max_dist]) if np.any(dist_cost < max_dist) else 1.0
        topo_norm = np.median(topo_cost) if topo_cost.size > 0 else 1.0
        if topo_norm > 0:
            topo_cost = topo_cost * (topo_scale / topo_norm)
        cost = (1 - topo_weight) * dist_cost + topo_weight * topo_cost
    else:
        cost = dist_cost.copy()

    # Hard gate: forbid links beyond max_dist (marked inf → never linked). The
    # gate is on raw distance, not the topology-blended cost, so topology only
    # ranks the *reachable* candidates.
    cost[dist_cost > max_dist] = np.inf

    # LAP with birth/death: an unmatched cell costs `no_match_cost` (default
    # max_dist) instead of being force-linked onto a neighbor.
    d = float(max_dist if no_match_cost is None else no_match_cost)
    return solve_lap(cost, d)


# ═══════════════════════════════════════════════════════════════
#  Full timeseries tracking
# ═══════════════════════════════════════════════════════════════

def track_timeseries(
    df: pd.DataFrame,
    max_dist: float = 30.0,
    n_neighbors: int = 5,
    use_topology: bool = True,
    topo_weight: float = 0.3,
    no_match_cost: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> pd.DataFrame:
    """
    Track cells across all frames using nearest-neighbor linking.

    Parameters
    ----------
    df : DataFrame with columns: frame, label, centroid_y, centroid_x
    max_dist : float, max linking distance
    n_neighbors : int, neighbors for topology features
    use_topology : bool, whether to use topology-augmented cost
    topo_weight : float, weight of topology in cost matrix
    progress_cb : callable(fraction_0_1, message), reports linking progress

    Returns
    -------
    df : same DataFrame with added 'track_id' column
    """
    # Group by frame ONCE (dict of per-frame sub-frames) instead of re-scanning
    # the whole DataFrame with a boolean mask on every iteration — that repeated
    # mask was O(T² · cells) and a major slice of the wall-clock on dense fields.
    by_frame = {int(f): sub for f, sub in df.groupby("frame", sort=True)}
    frames = sorted(by_frame)
    T = len(frames)
    n_det = len(df)
    log.info(
        "CT topology tracking: %d detections across %d frames (~%.0f/frame), "
        "topology=%s, max_dist=%.1f",
        n_det, T, (n_det / T if T else 0), use_topology, max_dist,
    )

    next_id = 1
    track_ids: dict = {}
    t_link = 0.0
    t_topo = 0.0
    max_cost_cells = 0

    # First frame: every cell gets a new track.
    first = by_frame[frames[0]]
    for lbl in first["label"].to_numpy():
        track_ids[(frames[0], lbl)] = next_id
        next_id += 1

    for i in range(1, T):
        pf, cf = frames[i - 1], frames[i]
        prev = by_frame[pf]
        curr = by_frame[cf]

        c_prev = prev[["centroid_y", "centroid_x"]].values
        c_curr = curr[["centroid_y", "centroid_x"]].values
        l_prev = prev["label"].values
        l_curr = curr["label"].values

        max_cost_cells = max(max_cost_cells, len(c_prev) * len(c_curr))

        # Topology features
        topo_prev = topo_curr = None
        if use_topology and len(c_prev) > n_neighbors and len(c_curr) > n_neighbors:
            _t0 = time.perf_counter()
            topo_prev = compute_topology_features(c_prev, n_neighbors)
            topo_curr = compute_topology_features(c_curr, n_neighbors)
            t_topo += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        matches, _, unmatched_curr = link_frames(
            c_prev, c_curr, max_dist,
            topo_prev, topo_curr, topo_weight,
            no_match_cost=no_match_cost,
        )
        t_link += time.perf_counter() - _t0

        for prev_idx, curr_idx in matches:
            prev_key = (pf, l_prev[prev_idx])
            curr_key = (cf, l_curr[curr_idx])
            track_ids[curr_key] = track_ids[prev_key]

        for curr_idx in unmatched_curr:
            track_ids[(cf, l_curr[curr_idx])] = next_id
            next_id += 1

        if progress_cb:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")

    if max_cost_cells > _BIG_COST_MATRIX:
        log.warning(
            "CT topology: largest per-frame cost matrix is %d entries — the "
            "Hungarian assignment is O(n³), so this dense field is inherently "
            "slow. Consider a smaller max_dist or fewer detections.",
            max_cost_cells,
        )

    # Assign track_id column. A vectorized dict lookup over zipped numpy arrays
    # replaces the old ``df.apply(..., axis=1)`` (a Python call + Series build
    # per row, which alone cost tens of seconds at 100k+ detections).
    _t0 = time.perf_counter()
    df = df.copy()
    frame_arr = df["frame"].to_numpy()
    label_arr = df["label"].to_numpy()
    df["track_id"] = [
        track_ids.get((int(f), int(lbl)), -1)
        for f, lbl in zip(frame_arr, label_arr)
    ]
    t_assign = time.perf_counter() - _t0

    log.info(
        "CT topology done: %d tracks; link %.2fs, topology %.2fs, assign %.2fs",
        next_id - 1, t_link, t_topo, t_assign,
    )
    if progress_cb:
        progress_cb(1.0, "Tracking done")

    return df


# ═══════════════════════════════════════════════════════════════
#  Spatial fingerprint tracker with gap filling
# ═══════════════════════════════════════════════════════════════

def fingerprint_cost_matrix(
    prev_centroids: np.ndarray,
    curr_centroids: np.ndarray,
    prev_areas: np.ndarray,
    curr_areas: np.ndarray,
    max_dist: float = 30.0,
    area_weight: float = 0.3,
) -> np.ndarray:
    """
    Build a cost matrix combining spatial distance and area similarity.

    Cost = (1 - area_weight) * spatial_distance + area_weight * area_mismatch

    area_mismatch is scaled so that a 2x area difference roughly equals
    max_dist in cost.
    """
    N, M = len(prev_centroids), len(curr_centroids)
    if N == 0 or M == 0:
        return np.zeros((N, M))

    dist = cdist(prev_centroids, curr_centroids)

    # Area mismatch: |log(a1/a2)| scaled to distance units
    pa = prev_areas[:, None].astype(np.float64)
    ca = curr_areas[None, :].astype(np.float64)
    pa = np.clip(pa, 1, None)
    ca = np.clip(ca, 1, None)
    area_ratio = np.abs(np.log(pa / ca))  # 0 = same size, ln(2)=0.69 = 2x mismatch
    area_cost = area_ratio * (max_dist / 0.7)  # normalize so 2x area ~ max_dist

    cost = (1 - area_weight) * dist + area_weight * area_cost
    cost[dist > max_dist] = np.inf  # hard gate; solve_lap treats inf as forbidden

    return cost


def track_fingerprint(
    df: pd.DataFrame,
    max_dist: float = 30.0,
    area_weight: float = 0.3,
    max_gap: int = 3,
    no_match_cost: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> pd.DataFrame:
    """
    Track cells using spatial position + size fingerprinting with gap filling.

    Designed for cells that:
    - Don't move far between frames
    - May disappear for a few frames (missed detections)
    - Maintain similar size across frames

    Parameters
    ----------
    df : DataFrame with columns: frame, label, centroid_y, centroid_x, area
    max_dist : float, max linking distance in pixels
    area_weight : float 0-1, weight of area similarity vs distance
    max_gap : int, max frames a cell can disappear and still be re-linked
    progress_cb : callable(fraction_0_1, message), reports linking progress

    Returns
    -------
    df with 'track_id' column
    """
    # Group by frame once (see track_timeseries — avoids the O(T² · cells)
    # repeated boolean mask).
    by_frame = {int(f): sub for f, sub in df.groupby("frame", sort=True)}
    frames = sorted(by_frame)
    T = len(frames)
    n_det = len(df)
    log.info(
        "CT fingerprint tracking: %d detections across %d frames (~%.0f/frame), "
        "max_dist=%.1f, area_weight=%.2f, max_gap=%d",
        n_det, T, (n_det / T if T else 0), max_dist, area_weight, max_gap,
    )
    _t_start = time.perf_counter()
    next_id = 1
    track_ids = {}  # (frame, label) -> track_id

    # Active tracks: track_id -> {last_frame, last_y, last_x, last_area}
    active_tracks = {}

    # First frame
    first = by_frame[frames[0]]
    for lbl, cy, cx, ar in zip(
        first["label"].to_numpy(), first["centroid_y"].to_numpy(),
        first["centroid_x"].to_numpy(), first["area"].to_numpy(),
    ):
        tid = next_id
        next_id += 1
        track_ids[(frames[0], lbl)] = tid
        active_tracks[tid] = {"last_frame": frames[0], "y": cy, "x": cx, "area": ar}

    for i in range(1, T):
        cf = frames[i]
        curr = by_frame[cf]

        if curr.empty:
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        c_curr = curr[["centroid_y", "centroid_x"]].values
        a_curr = curr["area"].values
        l_curr = curr["label"].values

        # Gather all active tracks (including those with gaps)
        alive_tids = []
        alive_centroids = []
        alive_areas = []

        for tid, info in active_tracks.items():
            gap = cf - info["last_frame"]
            if gap <= max_gap + 1:  # +1 because consecutive frames have gap=1
                alive_tids.append(tid)
                alive_centroids.append([info["y"], info["x"]])
                alive_areas.append(info["area"])

        if not alive_tids:
            # No active tracks — all cells start new tracks
            for ci in range(len(l_curr)):
                tid = next_id
                next_id += 1
                track_ids[(cf, l_curr[ci])] = tid
                active_tracks[tid] = {
                    "last_frame": cf,
                    "y": c_curr[ci, 0],
                    "x": c_curr[ci, 1],
                    "area": a_curr[ci],
                }
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        prev_centroids = np.array(alive_centroids)
        prev_areas = np.array(alive_areas)

        # Build cost matrix with spatial + area fingerprint
        cost = fingerprint_cost_matrix(
            prev_centroids, c_curr, prev_areas, a_curr,
            max_dist=max_dist, area_weight=area_weight,
        )

        # LAP with birth/death (see solve_lap). Unmatched alive tracks are left
        # in `active_tracks` for gap re-linking; unmatched detections below start
        # new tracks — no forced link onto a neighbor.
        d = float(max_dist if no_match_cost is None else no_match_cost)
        matches, _, _ = solve_lap(cost, d)

        matched_curr = set()
        for r, c in matches:
            tid = alive_tids[r]
            track_ids[(cf, l_curr[c])] = tid
            active_tracks[tid] = {
                "last_frame": cf,
                "y": c_curr[c, 0],
                "x": c_curr[c, 1],
                "area": a_curr[c],
            }
            matched_curr.add(c)

        # Unmatched detections start new tracks
        for ci in range(len(l_curr)):
            if ci not in matched_curr:
                tid = next_id
                next_id += 1
                track_ids[(cf, l_curr[ci])] = tid
                active_tracks[tid] = {
                    "last_frame": cf,
                    "y": c_curr[ci, 0],
                    "x": c_curr[ci, 1],
                    "area": a_curr[ci],
                }

        # Prune dead tracks (gap exceeded)
        dead = [tid for tid, info in active_tracks.items()
                if cf - info["last_frame"] > max_gap + 1]
        for tid in dead:
            del active_tracks[tid]

        if progress_cb:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")

    # Assign track_id column (vectorized dict lookup — see track_timeseries).
    df = df.copy()
    frame_arr = df["frame"].to_numpy()
    label_arr = df["label"].to_numpy()
    df["track_id"] = [
        track_ids.get((int(f), int(lbl)), -1)
        for f, lbl in zip(frame_arr, label_arr)
    ]

    log.info("CT fingerprint done: %d tracks in %.2fs",
             next_id - 1, time.perf_counter() - _t_start)
    if progress_cb:
        progress_cb(1.0, "Tracking done")

    return df


# ═══════════════════════════════════════════════════════════════
#  Mask-overlap (IoU) tracking — the robust default for segmentation
# ═══════════════════════════════════════════════════════════════

def _frame_footprints(
    mask_frame: np.ndarray,
    keep: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Per-label flat pixel indices for one label image.

    Returns ``(labels, areas, flats)`` where object ``labels[i]`` covers
    ``areas[i]`` pixels at flat indices ``flats[i]`` (into the raveled frame).
    One stable argsort over the foreground pixels, so cost is ∝ foreground area
    rather than ``n_labels × frame_pixels``. ``keep`` restricts output to those
    label ids — used to ignore mask objects the measurement stage filtered out
    of the DataFrame (so they can't steal an overlap link).
    """
    flat = np.asarray(mask_frame).reshape(-1)
    fg = np.flatnonzero(flat)
    if fg.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), []
    labs = flat[fg]
    order = np.argsort(labs, kind="stable")
    fg = fg[order]
    labs = labs[order]
    uniq, starts, counts = np.unique(labs, return_index=True, return_counts=True)
    labels: List[int] = []
    areas: List[int] = []
    flats: List[np.ndarray] = []
    for i in range(len(uniq)):
        lab = int(uniq[i])
        if keep is not None and lab not in keep:
            continue
        s = int(starts[i]); n = int(counts[i])
        labels.append(lab)
        areas.append(n)
        flats.append(fg[s:s + n])
    return np.array(labels, np.int64), np.array(areas, np.int64), flats


def track_overlap(
    df: pd.DataFrame,
    masks: np.ndarray,
    min_iou: float = 0.1,
    max_gap: int = 1,
    no_match_cost: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> pd.DataFrame:
    """Track segmented objects by **mask overlap (IoU)** with a birth/death LAP.

    For dense, slowly-moving nuclei (e.g. StarDist H2B masks) overlap is far more
    discriminative than centroid distance: an object overlaps its own previous
    mask heavily and a neighbor's barely at all, so neither a wide gate (spurious
    long links) nor a tight one (neighbor "currents") is needed. Each consecutive
    frame pair is linked on ``cost = 1 - IoU`` (pairs below ``min_iou`` forbidden)
    via :func:`solve_lap`, so an object with no real overlap starts/ends a track
    instead of being force-linked. A track keeps its last footprint for up to
    ``max_gap`` missed frames so a dropped detection can still re-link by overlap.

    Parameters
    ----------
    df : DataFrame with columns ``frame``, ``label`` (``label`` must equal the
        integer id in ``masks`` for that frame; extra columns are preserved).
    masks : (T, H, W) int label image for this group's segmentation channel.
    min_iou : float, minimum intersection-over-union to allow a link.
    max_gap : int, frames a track may vanish and still re-link by overlap.
    no_match_cost : float, optional; birth/death cost. Defaults to 1.0 (the cost
        of zero overlap), so any above-``min_iou`` overlap beats starting anew.
    progress_cb : callable(fraction_0_1, message).

    Returns
    -------
    df : copy with a ``track_id`` column (``-1`` where unassigned).
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"track_overlap needs (T,H,W) masks, got {masks.shape!r}")
    n_masks, H, W = masks.shape

    by_frame = {int(f): sub for f, sub in df.groupby("frame", sort=True)}
    frames = sorted(by_frame)
    T = len(frames)
    d = float(1.0 if no_match_cost is None else no_match_cost)

    log.info(
        "CT overlap tracking: %d detections across %d frames, min_iou=%.2f, "
        "max_gap=%d", len(df), T, min_iou, max_gap,
    )
    _t0 = time.perf_counter()

    next_id = 1
    track_ids: dict = {}
    active: dict = {}                       # tid -> {last_frame, flat, area}
    carry = np.zeros(H * W, dtype=np.int32)  # reused prev-footprint paint buffer

    if T == 0:
        df = df.copy()
        df["track_id"] = pd.Series([], dtype=int)
        return df

    def _seed(frame: int, labels, areas, flats) -> None:
        nonlocal next_id
        for lab, ar, fl in zip(labels, areas, flats):
            tid = next_id
            next_id += 1
            track_ids[(frame, int(lab))] = tid
            active[tid] = {"last_frame": frame, "flat": fl, "area": int(ar)}

    # First frame: every measured object starts a track.
    f0 = frames[0]
    keep0 = {int(x) for x in by_frame[f0]["label"].to_numpy()}
    if f0 < n_masks:
        _seed(f0, *_frame_footprints(masks[f0], keep0))
    if progress_cb:
        progress_cb(1.0 / T, f"Linking frame 1/{T}")

    for i in range(1, T):
        cf = frames[i]
        if cf >= n_masks:
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue
        keep = {int(x) for x in by_frame[cf]["label"].to_numpy()}
        cur_labels, cur_areas, cur_flats = _frame_footprints(masks[cf], keep)
        C = len(cur_labels)
        if C == 0:
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        # Tracks still within the gap budget are candidates for re-link.
        alive = [(tid, info) for tid, info in active.items()
                 if cf - info["last_frame"] <= max_gap + 1]
        if not alive:
            _seed(cf, cur_labels, cur_areas, cur_flats)
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        # Paint each alive track's last footprint into the carry buffer, older
        # first so a more-recent footprint wins any pixel contested after motion.
        alive.sort(key=lambda kv: kv[1]["last_frame"])
        K = len(alive)
        prev_area = np.empty(K, np.int64)
        idx2tid = [0] * K
        painted: List[np.ndarray] = []
        for k, (tid, info) in enumerate(alive):
            fl = info["flat"]
            carry[fl] = k + 1
            painted.append(fl)
            prev_area[k] = info["area"]
            idx2tid[k] = tid

        # Overlap crosstab: for each current object, tally shared pixels against
        # every alive footprint it touches, then convert to IoU cost.
        cost = np.full((K, C), np.inf, dtype=np.float64)
        for c in range(C):
            ka = carry[cur_flats[c]]
            hit = ka > 0
            if not hit.any():
                continue
            kk, ov = np.unique(ka[hit], return_counts=True)
            for k1, o in zip(kk, ov):
                k = int(k1) - 1
                union = prev_area[k] + int(cur_areas[c]) - int(o)
                iou = (o / union) if union > 0 else 0.0
                if iou >= min_iou:
                    cost[k, c] = 1.0 - iou

        for fl in painted:              # reset only touched pixels (cheap)
            carry[fl] = 0

        matches, _, _ = solve_lap(cost, d)
        matched_c = set()
        for k, c in matches:
            tid = idx2tid[k]
            track_ids[(cf, int(cur_labels[c]))] = tid
            active[tid] = {"last_frame": cf, "flat": cur_flats[c],
                           "area": int(cur_areas[c])}
            matched_c.add(c)

        for c in range(C):              # unmatched detections start new tracks
            if c not in matched_c:
                tid = next_id
                next_id += 1
                track_ids[(cf, int(cur_labels[c]))] = tid
                active[tid] = {"last_frame": cf, "flat": cur_flats[c],
                               "area": int(cur_areas[c])}

        dead = [tid for tid, info in active.items()
                if cf - info["last_frame"] > max_gap + 1]
        for tid in dead:
            del active[tid]

        if progress_cb:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")

    df = df.copy()
    frame_arr = df["frame"].to_numpy()
    label_arr = df["label"].to_numpy()
    df["track_id"] = [
        track_ids.get((int(f), int(lbl)), -1)
        for f, lbl in zip(frame_arr, label_arr)
    ]
    log.info("CT overlap done: %d tracks in %.2fs",
             next_id - 1, time.perf_counter() - _t0)
    if progress_cb:
        progress_cb(1.0, "Tracking done")

    return df
