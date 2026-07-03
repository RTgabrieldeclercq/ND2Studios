"""
SerialTrack Python — Chunk 2
=============================
Split this file into two modules:
    serialtrack/matching.py
    serialtrack/outliers.py

Dependencies (same as Chunk 1):
    pip install numpy scipy numba
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  FILE 1: serialtrack/matching.py — Topology matching & linking  ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import numba as nb
from scipy.spatial import cKDTree
import logging

from .outliers import remove_outliers

log = logging.getLogger("serialtrack.matching")


# ═══════════════════════════════════════════════════════════════
#  Numba kernels — rotation-invariant topology features
# ═══════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _cross3(a, b):
    """Cross product of two 3-vectors."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


@nb.njit(cache=True)
def _norm3(v):
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


@nb.njit(cache=True)
def _dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@nb.njit(parallel=True, cache=True)
def _build_features_3d(coords, neighbor_idx, n_neighbors):
    """Build rotation-invariant topology features for 3-D particles.

    For each particle, we:
      1. Get the K nearest neighbors (excluding self) from neighbor_idx.
      2. Build a rotation-invariant (RI) local frame:
         - ex = direction to nearest neighbor
         - ez = cross(r1, r2), oriented so dot(ez, r3) > 0
         - ey = cross(ez, ex)
      3. Transform neighbor offsets into RI frame.
      4. Compute spherical coords {r, phi, theta}.
      5. Reorder by phi starting from the nearest neighbor.
      6. Store features: r (distances), phi_diff (angular gaps), theta.

    Parameters
    ----------
    coords : (N, 3) float64 — particle positions
    neighbor_idx : (N, K+1) int64 — KNN indices (col 0 = self)
    n_neighbors : int — K

    Returns
    -------
    feat_r     : (N, K) float64 — reordered distances
    feat_phi   : (N, K) float64 — reordered angular differences in xy-plane
    feat_theta : (N, K) float64 — reordered polar angles
    """
    N = coords.shape[0]
    K = n_neighbors
    feat_r = np.zeros((N, K), dtype=np.float64)
    feat_phi = np.zeros((N, K), dtype=np.float64)
    feat_theta = np.zeros((N, K), dtype=np.float64)

    for i in nb.prange(N):
        # ----- 1. Neighbor offsets -----
        dx = np.empty((K, 3), dtype=np.float64)
        for k in range(K):
            j = neighbor_idx[i, k + 1]  # skip self at col 0
            dx[k, 0] = coords[j, 0] - coords[i, 0]
            dx[k, 1] = coords[j, 1] - coords[i, 1]
            dx[k, 2] = coords[j, 2] - coords[i, 2]

        # ----- 2. Build rotation-invariant frame -----
        r1 = dx[0]
        nr1 = _norm3(r1)
        if nr1 < 1e-30:
            continue
        ex = r1 / nr1

        # ez = cross(r1, r2), normalised
        if K >= 2:
            ez = _cross3(dx[0], dx[1])
        else:
            # Fallback: pick arbitrary orthogonal
            if abs(ex[0]) < 0.9:
                ez = _cross3(ex, np.array([1.0, 0.0, 0.0]))
            else:
                ez = _cross3(ex, np.array([0.0, 1.0, 0.0]))
        nez = _norm3(ez)
        if nez < 1e-30:
            # r1 ∥ r2 — use fallback
            if abs(ex[0]) < 0.9:
                ez = _cross3(ex, np.array([1.0, 0.0, 0.0]))
            else:
                ez = _cross3(ex, np.array([0.0, 1.0, 0.0]))
            nez = _norm3(ez)
            if nez < 1e-30:
                continue
        ez = ez / nez

        # Orient ez so 3rd neighbor is in +ez hemisphere
        if K >= 3 and _dot3(ez, dx[2]) < 0.0:
            ez = -ez

        # ey = cross(ez, ex)
        ey = _cross3(ez, ex)
        ney = _norm3(ey)
        if ney < 1e-30:
            continue
        ey = ey / ney

        # ----- 3. Transform into RI frame -----
        dx_ri = np.empty((K, 3), dtype=np.float64)
        for k in range(K):
            dx_ri[k, 0] = _dot3(dx[k], ex)
            dx_ri[k, 1] = _dot3(dx[k], ey)
            dx_ri[k, 2] = _dot3(dx[k], ez)

        # ----- 4. Spherical coordinates -----
        r_arr = np.empty(K, dtype=np.float64)
        phi_arr = np.empty(K, dtype=np.float64)
        theta_arr = np.empty(K, dtype=np.float64)
        for k in range(K):
            r_arr[k] = np.sqrt(
                dx_ri[k,0]**2 + dx_ri[k,1]**2 + dx_ri[k,2]**2
            )
            phi_arr[k] = np.arctan2(dx_ri[k, 1], dx_ri[k, 0])
            rxy = np.sqrt(dx_ri[k,0]**2 + dx_ri[k,1]**2)
            theta_arr[k] = np.arctan2(dx_ri[k, 2], rxy)

        # ----- 5. Reorder by phi, starting from nearest (idx 0) -----
        # Sort phi to get order
        order = np.argsort(phi_arr)
        # Find where the nearest neighbor (original index 0) lands
        start = 0
        for k in range(K):
            if order[k] == 0:
                start = k
                break
        # Build reordered arrays starting from nearest neighbor
        for k in range(K):
            idx = order[(start + k) % K]
            feat_r[i, k] = r_arr[idx]
            theta_arr_val = theta_arr[idx]
            feat_theta[i, k] = theta_arr_val

        # phi_diff: consecutive angle differences (circular)
        phi_reord = np.empty(K, dtype=np.float64)
        for k in range(K):
            idx = order[(start + k) % K]
            phi_reord[k] = phi_arr[idx]
        for k in range(K):
            diff = phi_reord[(k + 1) % K] - phi_reord[k]
            if diff < 0.0:
                diff += 2.0 * np.pi
            feat_phi[i, k] = diff

    return feat_r, feat_phi, feat_theta


@nb.njit(parallel=True, cache=True)
def _build_features_2d(coords, neighbor_idx, n_neighbors):
    """Build topology features for 2-D particles.

    Same logic as 3-D but without theta and without the RI frame
    (2-D only needs r and phi_diff — phi_diff is already rotation-invariant
    because it measures relative angles between neighbors).

    Returns
    -------
    feat_r   : (N, K) float64 — reordered distances
    feat_phi : (N, K) float64 — reordered angular differences
    """
    N = coords.shape[0]
    K = n_neighbors
    feat_r = np.zeros((N, K), dtype=np.float64)
    feat_phi = np.zeros((N, K), dtype=np.float64)

    for i in nb.prange(N):
        dx = np.empty((K, 2), dtype=np.float64)
        for k in range(K):
            j = neighbor_idx[i, k + 1]
            dx[k, 0] = coords[j, 0] - coords[i, 0]
            dx[k, 1] = coords[j, 1] - coords[i, 1]

        r_arr = np.empty(K, dtype=np.float64)
        phi_arr = np.empty(K, dtype=np.float64)
        for k in range(K):
            r_arr[k] = np.sqrt(dx[k, 0]**2 + dx[k, 1]**2)
            phi_arr[k] = np.arctan2(dx[k, 1], dx[k, 0])

        order = np.argsort(phi_arr)
        start = 0
        for k in range(K):
            if order[k] == 0:
                start = k
                break

        for k in range(K):
            idx = order[(start + k) % K]
            feat_r[i, k] = r_arr[idx]

        phi_reord = np.empty(K, dtype=np.float64)
        for k in range(K):
            idx = order[(start + k) % K]
            phi_reord[k] = phi_arr[idx]
        for k in range(K):
            diff = phi_reord[(k + 1) % K] - phi_reord[k]
            if diff < 0.0:
                diff += 2.0 * np.pi
            feat_phi[i, k] = diff

    return feat_r, feat_phi


# ═══════════════════════════════════════════════════════════════
#  Numba kernel — brute-force topology match (parallelised)
# ═══════════════════════════════════════════════════════════════

@nb.njit(parallel=True, cache=True)
def _match_features_3d(
    coords_a, feat_r_a, feat_phi_a, feat_theta_a,
    coords_b, feat_r_b, feat_phi_b, feat_theta_b,
    cand_idx,       # (Na, max_cands) int64, -1 padded
    cand_counts,    # (Na,) int64, how many valid candidates per A particle
    f_o_s,
):
    """Find topology matches: A → B.

    For each particle in A, search its candidate list in B.
    A match is accepted when all three SSE channels (r, phi, theta)
    minimise at the same candidate AND displacement < f_o_s.

    Returns
    -------
    match_a : (M,) int64 — indices into A
    match_b : (M,) int64 — indices into B
    """
    Na = coords_a.shape[0]
    # Pre-allocate max possible (one match per A particle)
    out_a = np.full(Na, -1, dtype=np.int64)
    out_b = np.full(Na, -1, dtype=np.int64)

    for i in nb.prange(Na):
        nc = cand_counts[i]
        if nc == 0:
            continue

        # SSE for each candidate
        best_r = np.int64(-1);   min_r = np.inf
        best_p = np.int64(-1);   min_p = np.inf
        best_t = np.int64(-1);   min_t = np.inf

        for ci in range(nc):
            j = cand_idx[i, ci]
            sse_r = 0.0; sse_p = 0.0; sse_t = 0.0
            for k in range(feat_r_a.shape[1]):
                dr = feat_r_a[i, k] - feat_r_b[j, k]
                dp = feat_phi_a[i, k] - feat_phi_b[j, k]
                dt = feat_theta_a[i, k] - feat_theta_b[j, k]
                sse_r += dr * dr
                sse_p += dp * dp
                sse_t += dt * dt
            if sse_r < min_r:
                min_r = sse_r; best_r = j
            if sse_p < min_p:
                min_p = sse_p; best_p = j
            if sse_t < min_t:
                min_t = sse_t; best_t = j

        # All three must agree
        if best_r == best_p and best_p == best_t and best_r >= 0:
            # Check displacement < f_o_s
            d2 = 0.0
            for d in range(3):
                dd = coords_a[i, d] - coords_b[best_r, d]
                d2 += dd * dd
            if np.sqrt(d2) < f_o_s:
                out_a[i] = i
                out_b[i] = best_r

    return out_a, out_b


@nb.njit(parallel=True, cache=True)
def _match_features_2d(
    coords_a, feat_r_a, feat_phi_a,
    coords_b, feat_r_b, feat_phi_b,
    cand_idx, cand_counts, f_o_s,
):
    """2-D topology match: only r and phi channels.

    Match accepted when both r and phi minimise at same candidate.
    """
    Na = coords_a.shape[0]
    out_a = np.full(Na, -1, dtype=np.int64)
    out_b = np.full(Na, -1, dtype=np.int64)

    for i in nb.prange(Na):
        nc = cand_counts[i]
        if nc == 0:
            continue

        best_r = np.int64(-1);  min_r = np.inf
        best_p = np.int64(-1);  min_p = np.inf

        for ci in range(nc):
            j = cand_idx[i, ci]
            sse_r = 0.0; sse_p = 0.0
            for k in range(feat_r_a.shape[1]):
                dr = feat_r_a[i, k] - feat_r_b[j, k]
                dp = feat_phi_a[i, k] - feat_phi_b[j, k]
                sse_r += dr * dr
                sse_p += dp * dp
            if sse_r < min_r:
                min_r = sse_r; best_r = j
            if sse_p < min_p:
                min_p = sse_p; best_p = j

        if best_r == best_p and best_r >= 0:
            d2 = 0.0
            for d in range(2):
                dd = coords_a[i, d] - coords_b[best_r, d]
                d2 += dd * dd
            if np.sqrt(d2) < f_o_s:
                out_a[i] = i
                out_b[i] = best_r

    return out_a, out_b


# ═══════════════════════════════════════════════════════════════
#  High-level matcher classes
# ═══════════════════════════════════════════════════════════════

class TopologyMatcher:
    """Scale & rotation invariant particle matcher.

    This is the core of SerialTrack — it replaces:
        - ``f_track_neightopo_match3.m``   (3-D)
        - ``f_track_neightopo_match.m``    (2-D)

    Algorithm
    ---------
    1. Build a cKDTree for each point set (vectorised bulk KNN).
    2. Extract rotation-invariant topology features via Numba JIT.
    3. Build candidate lists (B particles near each A particle).
    4. Compare feature vectors in parallel via Numba.
    5. Accept matches where ALL feature channels agree.

    Examples
    --------
    >>> matcher = TopologyMatcher(n_neighbors=20, f_o_s=60.0)
    >>> matches = matcher.match(coords_a, coords_b)   # (M, 2) array
    """

    def __init__(self, n_neighbors: int = 20, f_o_s: float = 60.0):
        self.n_neighbors = n_neighbors
        self.f_o_s = f_o_s

    def match(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
    ) -> np.ndarray:
        """Find topology-based matches from A → B.

        Parameters
        ----------
        coords_a : (Na, D) float64 — particle coords in image A
        coords_b : (Nb, D) float64 — particle coords in image B

        Returns
        -------
        matches : (M, 2) int64 — column 0 = index into A, column 1 = index into B
        """
        ndim = coords_a.shape[1]
        K = self.n_neighbors
        Na, Nb = len(coords_a), len(coords_b)

        if Na == 0 or Nb == 0:
            return np.empty((0, 2), dtype=np.int64)

        # Clamp K to available particles (minus self)
        K = min(K, Na - 1, Nb - 1)
        if K < 1:
            return np.empty((0, 2), dtype=np.int64)

        # 1. KNN for feature extraction (self-neighbors)
        tree_a = cKDTree(coords_a)
        tree_b = cKDTree(coords_b)
        _, knn_a = tree_a.query(coords_a, k=K + 1)  # includes self
        _, knn_b = tree_b.query(coords_b, k=K + 1)

        knn_a = np.ascontiguousarray(knn_a.astype(np.int64))
        knn_b = np.ascontiguousarray(knn_b.astype(np.int64))

        # 2. Build features
        if ndim == 3:
            feat_r_a, feat_phi_a, feat_theta_a = _build_features_3d(
                coords_a, knn_a, K
            )
            feat_r_b, feat_phi_b, feat_theta_b = _build_features_3d(
                coords_b, knn_b, K
            )
        else:
            feat_r_a, feat_phi_a = _build_features_2d(coords_a, knn_a, K)
            feat_r_b, feat_phi_b = _build_features_2d(coords_b, knn_b, K)

        # 3. Build candidate lists: for each A particle, which B particles
        #    are within sqrt(ndim)*f_o_s ?
        cand_idx, cand_counts = self._build_candidates(
            coords_a, tree_b, K, ndim
        )

        # 4. Match
        if ndim == 3:
            out_a, out_b = _match_features_3d(
                coords_a, feat_r_a, feat_phi_a, feat_theta_a,
                coords_b, feat_r_b, feat_phi_b, feat_theta_b,
                cand_idx, cand_counts, self.f_o_s,
            )
        else:
            out_a, out_b = _match_features_2d(
                coords_a, feat_r_a, feat_phi_a,
                coords_b, feat_r_b, feat_phi_b,
                cand_idx, cand_counts, self.f_o_s,
            )

        # 5. Collect valid matches
        valid = out_a >= 0
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack((out_a[valid], out_b[valid]))

    def _build_candidates(
        self, coords_a, tree_b, K, ndim
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build padded candidate index array using ball query.

        Falls back to KNN with distance filter when f_o_s is finite.
        When f_o_s is inf, all B particles are candidates.
        """
        Na = len(coords_a)
        Nb = tree_b.n

        if self.f_o_s == np.inf or self.f_o_s <= 0:
            # All B particles are candidates
            max_c = Nb
            cand_idx = np.empty((Na, max_c), dtype=np.int64)
            cand_counts = np.full(Na, Nb, dtype=np.int64)
            row = np.arange(Nb, dtype=np.int64)
            for i in range(Na):
                cand_idx[i, :] = row
            return cand_idx, cand_counts

        # Use ball_point query for radius search
        radius = np.sqrt(ndim) * self.f_o_s
        ball_results = tree_b.query_ball_point(coords_a, r=radius)

        # Determine max candidate count for padding
        max_c = max(len(r) for r in ball_results) if len(ball_results) > 0 else 0
        max_c = max(max_c, 1)  # at least 1 column
        cand_idx = np.full((Na, max_c), -1, dtype=np.int64)
        cand_counts = np.zeros(Na, dtype=np.int64)

        for i, indices in enumerate(ball_results):
            n = len(indices)
            cand_counts[i] = n
            for ci in range(n):
                cand_idx[i, ci] = indices[ci]

        return cand_idx, cand_counts


class NearestNeighborMatcher:
    """Simple nearest-neighbor fallback matcher.

    Used when n_neighbors ≤ 2 (insufficient for topology matching).
    Replaces ``f_track_nearest_neighbour3.m``.
    """

    def __init__(self, f_o_s: float = 60.0):
        self.f_o_s = f_o_s

    def match(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
    ) -> np.ndarray:
        """One-to-one nearest neighbor matching.

        Returns (M, 2) int64 array of (idx_a, idx_b) pairs.
        """
        if len(coords_a) == 0 or len(coords_b) == 0:
            return np.empty((0, 2), dtype=np.int64)

        tree_b = cKDTree(coords_b)
        dists, idx_b = tree_b.query(coords_a, k=1)

        # Filter by f_o_s
        mask = dists < self.f_o_s
        idx_a = np.where(mask)[0]
        return np.column_stack((idx_a, idx_b[mask]))


# ═══════════════════════════════════════════════════════════════
#  Displacement computation from matches
# ═══════════════════════════════════════════════════════════════

def compute_displacement(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    matches: np.ndarray,
    outlier_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build track_A2B index array and compute displacements.

    Replaces ``funCompDisp3.m``.

    Parameters
    ----------
    coords_a : (Na, D) — reference particle positions
    coords_b : (Nb, D) — deformed particle positions
    matches  : (M, 2) int — (index_a, index_b) pairs
    outlier_threshold : float — if > 0, apply Westerweel outlier removal

    Returns
    -------
    track_a2b : (Na,) int64 — track_a2b[i] = index into B, or -1
    disp_a2b  : (M', D) float64 — displacements for tracked particles
    """
    Na = len(coords_a)
    ndim = coords_a.shape[1]
    track_a2b = np.full(Na, -1, dtype=np.int64)

    if len(matches) == 0:
        return track_a2b, np.empty((0, ndim))

    for ia, ib in matches:
        track_a2b[ia] = ib

    # Outlier removal
    if outlier_threshold > 0:
        track_a2b = remove_outliers(
            coords_a, coords_b, track_a2b, outlier_threshold
        )

    # Compute clean displacements
    tracked = track_a2b >= 0
    idx_a = np.where(tracked)[0]
    idx_b = track_a2b[idx_a]
    disp = coords_b[idx_b] - coords_a[idx_a]

    return track_a2b, disp


# ═══════════════════════════════════════════════════════════════
#  Convenience: adaptive matcher selection
# ═══════════════════════════════════════════════════════════════

def match_particles(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    n_neighbors: int,
    f_o_s: float,
    outlier_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-call convenience function: detect matches + compute disp.

    Automatically selects topology matching (n_neighbors > 2) or
    nearest-neighbor fallback (n_neighbors ≤ 2), matching the
    behaviour of the MATLAB ADMM inner loop.

    Parameters
    ----------
    coords_a, coords_b : particle coordinates
    n_neighbors : current ADMM neighbor count
    f_o_s : field of search
    outlier_threshold : Westerweel threshold (0 = skip)

    Returns
    -------
    matches   : (M, 2) int64
    track_a2b : (Na,) int64
    disp_a2b  : (M', D) float64
    """
    if n_neighbors > 2:
        matcher = TopologyMatcher(n_neighbors=n_neighbors, f_o_s=f_o_s)
    else:
        matcher = NearestNeighborMatcher(f_o_s=f_o_s)

    matches = matcher.match(coords_a, coords_b)

    track_a2b, disp_a2b = compute_displacement(
        coords_a, coords_b, matches, outlier_threshold
    )

    return matches, track_a2b, disp_a2b
