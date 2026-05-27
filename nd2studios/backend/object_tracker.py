"""
Object tracker for the Results tab.

Implements a frame-to-frame Hungarian linker that assigns stable track IDs
to objects whose centroids remain within a displacement threshold across
consecutive T frames.  Pure NumPy / SciPy — no Qt imports.

Called from ResultsPage._on_compute() after compute_measurements().
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def link_objects(
    rows: List[Dict[str, Any]],
    max_displacement_px: float = 100.0,
    min_track_length: int = 2,
    min_circularity: float = 0.0,
    max_eccentricity: float = 1.0,
) -> List[Dict[str, Any]]:
    """Assign track_id, track_length, and track_validation to every row.

    Mutates *rows* in-place and returns the same list.

    Parameters
    ----------
    rows:
        List of measurement dicts produced by compute_measurements().
        Each dict must have: segmentation_channel, frame,
        centroid_y_px, centroid_x_px, bbox_min/max_row/col.
        m_position is optional (defaults to 0 when absent).
    max_displacement_px:
        Maximum centroid displacement (Euclidean, pixels) between
        consecutive frames for two objects to be linked as the same
        track.  Objects farther apart start new tracks.
    min_track_length:
        Minimum number of consecutive frames a track must span to be
        considered a valid tracked object.  Tracks shorter than this
        threshold receive track_id = None and track_validation = None.
    min_circularity:
        Objects whose circularity (4π·area/perimeter²) is below this
        value are excluded from tracking.  Range 0–1; default 0.0
        disables the filter.
    max_eccentricity:
        Objects whose eccentricity exceeds this value are excluded from
        tracking.  Range 0–1; default 1.0 disables the filter.

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

    for group_rows in groups.values():
        _link_group(group_rows, max_displacement_px, _next_track_id)

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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _link_group(
    group_rows: List[Dict[str, Any]],
    max_displacement_px: float,
    next_id: List[int],
) -> None:
    """Link objects within a single (channel, m_position) group."""
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in group_rows:
        frames[int(r.get("frame", 0))].append(r)

    sorted_frames = sorted(frames.keys())
    if len(sorted_frames) < 2:
        return

    # Seed first frame with fresh track IDs.
    for r in frames[sorted_frames[0]]:
        r["track_id"] = next_id[0]
        next_id[0] += 1

    # Track last-known centroid + bbox per active track_id.
    # Tuple layout: (cy, cx, bbox_min_row, bbox_max_row, bbox_min_col, bbox_max_col)
    active: Dict[int, Tuple[float, float, float, float, float, float]] = {
        r["track_id"]: (_cy(r), _cx(r), *_bbox(r))
        for r in frames[sorted_frames[0]]
    }

    for i in range(1, len(sorted_frames)):
        curr_rows = frames[sorted_frames[i]]
        if not curr_rows:
            continue

        prev_track_ids = list(active.keys())

        if not prev_track_ids:
            for r in curr_rows:
                r["track_id"] = next_id[0]
                next_id[0] += 1
                active[r["track_id"]] = (_cy(r), _cx(r), *_bbox(r))
            continue

        # Build cost matrix (prev_tracks × curr_objects).
        prev_vals = [active[tid] for tid in prev_track_ids]
        prev_centroids = np.array([[v[0], v[1]] for v in prev_vals], dtype=np.float64)
        prev_bboxes    = np.array([[v[2], v[3], v[4], v[5]] for v in prev_vals], dtype=np.float64)
        curr_centroids = np.array([(_cy(r), _cx(r)) for r in curr_rows], dtype=np.float64)
        curr_bboxes    = np.array([_bbox(r) for r in curr_rows], dtype=np.float64)

        # Euclidean pairwise distance matrix.
        diff = prev_centroids[:, None, :] - curr_centroids[None, :, :]  # (P, C, 2)
        cost = np.sqrt((diff ** 2).sum(axis=2))                          # (P, C)

        # Two detections whose bounding boxes do not spatially overlap cannot
        # be the same physical object — block them by exceeding the threshold.
        no_overlap = ~(
            (prev_bboxes[:, None, 0] < curr_bboxes[None, :, 1])    # prev r0 < curr r1
            & (prev_bboxes[:, None, 1] > curr_bboxes[None, :, 0])  # prev r1 > curr r0
            & (prev_bboxes[:, None, 2] < curr_bboxes[None, :, 3])  # prev c0 < curr c1
            & (prev_bboxes[:, None, 3] > curr_bboxes[None, :, 2])  # prev c1 > curr c0
        )
        cost[no_overlap] = max_displacement_px + 1.0

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_curr: set = set()
        new_active: Dict[int, Tuple[float, float, float, float, float, float]] = {}

        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] <= max_displacement_px:
                tid = prev_track_ids[ri]
                curr_rows[ci]["track_id"] = tid
                new_active[tid] = (_cy(curr_rows[ci]), _cx(curr_rows[ci]), *_bbox(curr_rows[ci]))
                matched_curr.add(ci)

        # Unmatched current objects start new tracks.
        for ci, r in enumerate(curr_rows):
            if ci not in matched_curr:
                r["track_id"] = next_id[0]
                next_id[0] += 1
                new_active[r["track_id"]] = (_cy(r), _cx(r), *_bbox(r))

        active = new_active


def _cy(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_y_px") or 0.0)


def _cx(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_x_px") or 0.0)


def _bbox(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return (min_row, max_row, min_col, max_col) from a measurement row.

    Degenerate bboxes (height or width ≤ 0) are expanded to a 1-px region
    centred on the centroid so the overlap check never blocks a valid link.
    """
    r0 = float(row.get("bbox_min_row") or 0.0)
    r1 = float(row.get("bbox_max_row") or 0.0)
    c0 = float(row.get("bbox_min_col") or 0.0)
    c1 = float(row.get("bbox_max_col") or 0.0)
    if r1 <= r0:
        cy = _cy(row)
        r0, r1 = cy - 0.5, cy + 0.5
    if c1 <= c0:
        cx = _cx(row)
        c0, c1 = cx - 0.5, cx + 0.5
    return r0, r1, c0, c1
