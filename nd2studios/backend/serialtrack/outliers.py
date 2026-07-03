"""
SerialTrack Python — Outlier detection & missing-particle culling
==================================================================
    serialtrack/outliers.py

Replaces: removeOutlierTPT.m  (~60 lines)

Implements Westerweel & Scarano (2005) universal outlier detection,
missing-particle detection, and adaptive f_o_s update.

Fixes from v1
-------------
- REMOVED circular import: ``from .matching import ...`` was unused
  and created a circular dependency (matching → outliers → matching).
- Fixed logger name: was "serialtrack.matching", now "serialtrack.outliers".
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree
import logging

log = logging.getLogger("serialtrack.outliers")


# ═══════════════════════════════════════════════════════════════
#  Westerweel universal outlier detection
# ═══════════════════════════════════════════════════════════════

def remove_outliers(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    track_a2b: np.ndarray,
    threshold: float = 5.0,
    n_neighbors: int = 27,
) -> np.ndarray:
    """Universal outlier detection for PTV data.

    Implements the normalised median residual test from:
        Westerweel & Scarano, "Universal outlier detection for PIV data",
        Exp. Fluids 39(6), 2005.

    Replaces ``removeOutlierTPT.m``.

    Parameters
    ----------
    coords_a    : (Na, D) — reference positions
    coords_b    : (Nb, D) — deformed positions
    track_a2b   : (Na,) int64 — index map (-1 = untracked)
    threshold   : float — normalised residual cutoff (typ. 2–5)
    n_neighbors : int — neighbors for median computation (typ. 27)

    Returns
    -------
    track_a2b : (Na,) int64 — updated with outliers set to -1
    """
    track = track_a2b.copy()
    tracked_mask = track >= 0
    tracked_idx = np.where(tracked_mask)[0]

    if len(tracked_idx) < n_neighbors + 1:
        return track  # too few points for statistics

    # Positions and displacements of tracked particles
    x0 = coords_a[tracked_idx]
    x1 = coords_b[track[tracked_idx]]
    u = x1 - x0  # (M, D)
    ndim = u.shape[1]

    # KNN among tracked particles
    tree = cKDTree(x0)
    K = min(n_neighbors, len(x0) - 1)
    _, knn_idx = tree.query(x0, k=K + 1)  # includes self at col 0

    # Fluctuation floor (Westerweel: epsilon ≈ 0.1 px)
    eps = 0.075

    is_outlier = np.zeros(len(tracked_idx), dtype=np.bool_)

    for d in range(ndim):
        u_d = u[:, d]
        # Gather neighbor displacements: (M, K+1)
        u_neigh = u_d[knn_idx]

        # Median displacement from all neighbors (including self)
        u_med = np.median(u_neigh, axis=1)

        # Median absolute deviation of neighbors from their median
        resid_neigh = np.abs(u_neigh - u_med[:, np.newaxis])
        r_med = np.median(resid_neigh, axis=1) + eps

        # Normalised residual for each particle
        rn = np.abs(u_d - u_med) / r_med

        is_outlier |= rn > threshold

    # Mark outliers as untracked
    track[tracked_idx[is_outlier]] = -1

    n_removed = int(np.sum(is_outlier))
    if n_removed > 0:
        log.info("Outlier removal: %d / %d particles flagged",
                 n_removed, len(tracked_idx))

    return track


# ═══════════════════════════════════════════════════════════════
#  Missing-particle detection (used late in ADMM iterations)
# ═══════════════════════════════════════════════════════════════

def find_not_missing(
    coords_a: np.ndarray,
    coords_b_warped: np.ndarray,
    dist_threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify particles that have plausible matches after warping.

    Replaces the missing-particle culling logic in the MATLAB ADMM loop
    that fires when n_neighbors < 4.

    Parameters
    ----------
    coords_a      : reference particle coords
    coords_b_warped : deformed coords after global-step warp
    dist_threshold  : max allowed nearest-neighbor distance [px]

    Returns
    -------
    not_missing_a : indices of A particles with nearby B partners
    not_missing_b : indices of B particles with nearby A partners
    """
    dist_thresh = max(2.0, dist_threshold)

    tree_b = cKDTree(coords_b_warped)
    tree_a = cKDTree(coords_a)

    # A → B: for each A particle, nearest B
    dist_ab, _ = tree_b.query(coords_a, k=1)
    not_missing_a = np.where(dist_ab < dist_thresh)[0]

    # B → A: for each B particle, nearest A
    dist_ba, _ = tree_a.query(coords_b_warped, k=1)
    not_missing_b = np.where(dist_ba < dist_thresh)[0]

    return not_missing_a, not_missing_b


# ═══════════════════════════════════════════════════════════════
#  Adaptive f_o_s update (used between ADMM iterations)
# ═══════════════════════════════════════════════════════════════

def update_f_o_s(
    disp_update: np.ndarray,
    f_o_s_current: float = 60.0,
) -> float:
    """Shrink field-of-search based on displacement quantiles.

    Replaces the MATLAB quantile-based f_o_s update in the ADMM loop.
    Uses median + 0.5*IQR per component; takes the max across components.

    The floor is the larger of 2 px and 10% of the current f_o_s,
    preventing the search window from collapsing to zero while still
    allowing meaningful shrinkage.

    Parameters
    ----------
    disp_update : (N, D) displacement update from the global step
    f_o_s_current : current field of search value

    Returns
    -------
    new_f_o_s : float
    """
    if len(disp_update) == 0:
        return f_o_s_current

    ndim = disp_update.shape[1]
    vals = []
    for d in range(ndim):
        q25, q50, q75 = np.percentile(disp_update[:, d], [25, 50, 75])
        vals.append(q50 + 0.5 * (q75 - q25))

    # Floor: max(2 px, 10% of current f_o_s) — prevents collapse
    floor = max(2.0, 0.1 * f_o_s_current)
    return max(floor, *vals)
