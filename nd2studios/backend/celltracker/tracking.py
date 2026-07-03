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

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


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

    for i in range(N):
        k = min(K, dists.shape[1])
        # Sorted distances (already sorted by cKDTree)
        features[i, :k] = dists[i, :k]

        # Angular gaps between neighbors
        neighbors = centroids[indices[i, :k]] - centroids[i]
        angles = np.arctan2(neighbors[:, 0], neighbors[:, 1])
        angles_sorted = np.sort(angles)

        if k > 1:
            gaps = np.diff(angles_sorted)
            gaps = np.append(gaps, 2 * np.pi + angles_sorted[0] - angles_sorted[-1])
            gaps_sorted = np.sort(gaps)
            features[i, n_neighbors:n_neighbors + len(gaps_sorted)] = gaps_sorted

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
    progress_cb=None,
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
    progress_cb : callable(int), progress 0-100

    Returns
    -------
    df : same DataFrame with added 'track_id' column
    """
    frames = sorted(df["frame"].unique())
    T = len(frames)
    next_id = 1
    track_ids = {}

    # First frame: every cell gets a new track
    first = df[df["frame"] == frames[0]]
    for _, row in first.iterrows():
        track_ids[(frames[0], row["label"])] = next_id
        next_id += 1

    for i in range(1, T):
        pf, cf = frames[i - 1], frames[i]
        prev = df[df["frame"] == pf]
        curr = df[df["frame"] == cf]

        c_prev = prev[["centroid_y", "centroid_x"]].values
        c_curr = curr[["centroid_y", "centroid_x"]].values
        l_prev = prev["label"].values
        l_curr = curr["label"].values

        # Topology features
        topo_prev = topo_curr = None
        if use_topology and len(c_prev) > n_neighbors and len(c_curr) > n_neighbors:
            topo_prev = compute_topology_features(c_prev, n_neighbors)
            topo_curr = compute_topology_features(c_curr, n_neighbors)

        matches, _, unmatched_curr = link_frames(
            c_prev, c_curr, max_dist,
            topo_prev, topo_curr, topo_weight,
        )

        for prev_idx, curr_idx in matches:
            prev_key = (pf, l_prev[prev_idx])
            curr_key = (cf, l_curr[curr_idx])
            track_ids[curr_key] = track_ids[prev_key]

        for curr_idx in unmatched_curr:
            track_ids[(cf, l_curr[curr_idx])] = next_id
            next_id += 1

        if progress_cb:
            progress_cb(int((i + 1) / T * 100))

    # Assign track_id column
    df = df.copy()
    df["track_id"] = df.apply(
        lambda row: track_ids.get((int(row["frame"]), int(row["label"])), -1),
        axis=1,
    )

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
    progress_cb=None,
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
    progress_cb : callable(int), progress 0-100

    Returns
    -------
    df with 'track_id' column
    """
    frames = sorted(df["frame"].unique())
    T = len(frames)
    next_id = 1
    track_ids = {}  # (frame, label) -> track_id

    # Active tracks: track_id -> {last_frame, last_y, last_x, last_area}
    active_tracks = {}

    # First frame
    first = df[df["frame"] == frames[0]]
    for _, row in first.iterrows():
        tid = next_id
        next_id += 1
        track_ids[(frames[0], row["label"])] = tid
        active_tracks[tid] = {
            "last_frame": frames[0],
            "y": row["centroid_y"],
            "x": row["centroid_x"],
            "area": row["area"],
        }

    for i in range(1, T):
        cf = frames[i]
        curr = df[df["frame"] == cf]

        if curr.empty:
            if progress_cb:
                progress_cb(int((i + 1) / T * 100))
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
                progress_cb(int((i + 1) / T * 100))
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
            progress_cb(int((i + 1) / T * 100))

    # Assign track_id column
    df = df.copy()
    df["track_id"] = df.apply(
        lambda row: track_ids.get((int(row["frame"]), int(row["label"])), -1),
        axis=1,
    )

    return df
