"""Cell tracking: nearest-neighbor Hungarian with optional topology matching.

Vendored from CellTracker ``backend/tracking.py`` (topology features inspired by
SerialTrack's matching.py). Only the headless ``track_timeseries`` (topology
Hungarian) and ``track_fingerprint`` (position + area, gap filling) linkers and
their helpers are copied — CellTracker's ``track_serialtrack`` is omitted (it
carries a hardcoded path and is already covered by
:mod:`nd2studios.backend.serialtrack`).

Operates on pandas DataFrames with columns ``frame``, ``label``, ``centroid_y``,
``centroid_x`` (and ``area`` for the fingerprint linker); returns a copy with a
``track_id`` column added.
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

log = logging.getLogger(__name__)

# A progress callback reports a fraction in [0, 1] and a short status message.
# Kept Qt-free so the backend stays importable without PySide6.
ProgressCB = Callable[[float, str], None]

# A per-frame cost matrix larger than this (N × M entries) makes the O(n³)
# Hungarian assignment the dominant cost; we log a one-time warning so a
# dense-field slowdown is explained rather than mysterious.
_BIG_COST_MATRIX = 4_000_000


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
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Link detections between two consecutive frames using the
    Hungarian algorithm on a cost matrix.

    Parameters
    ----------
    centroids_prev, centroids_curr : (N, 2) and (M, 2) arrays
    max_dist : float, maximum linking distance in pixels
    topo_prev, topo_curr : topology feature arrays (optional)
    topo_weight : float 0-1, weight of topology cost vs distance cost

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
        cost = dist_cost

    # Mask out impossible links
    cost[dist_cost > max_dist] = 1e6

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    unmatched_prev = set(range(N))
    unmatched_curr = set(range(M))

    for r, c in zip(row_ind, col_ind):
        if dist_cost[r, c] < max_dist:
            matches.append((r, c))
            unmatched_prev.discard(r)
            unmatched_curr.discard(c)

    return matches, list(unmatched_prev), list(unmatched_curr)


# ═══════════════════════════════════════════════════════════════
#  Full timeseries tracking
# ═══════════════════════════════════════════════════════════════

def track_timeseries(
    df: pd.DataFrame,
    max_dist: float = 30.0,
    n_neighbors: int = 5,
    use_topology: bool = True,
    topo_weight: float = 0.3,
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
    cost[dist > max_dist] = 1e6

    return cost


def track_fingerprint(
    df: pd.DataFrame,
    max_dist: float = 30.0,
    area_weight: float = 0.3,
    max_gap: int = 3,
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

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_curr = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e5:
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
