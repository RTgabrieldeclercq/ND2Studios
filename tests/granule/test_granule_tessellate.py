"""Headless tests for ``nd2studios.backend.analysis.granule_tessellate`` (V1.70 P3).

Pure ``scipy``/``numpy`` — no Qt. Covers both tessellation modes and the RAG
density-merge. All point clouds are generated with a fixed RNG for determinism.

Coordinate note: the module takes points in ``(z, y, x)`` voxel order and does its
geometry in world ``(x, y, z)`` µm. With ``voxel_size_um = (1, 1, 1)`` the world
coordinate is just the reversed input column order, so ``arr[..., ::-1]`` maps an
input ``(z, y, x)`` row to the internal ``(x, y, z)`` the boundaries/Delaunay use.
Convex-hull volumes are permutation-invariant, so ``ConvexHull(input)`` and
``ConvexHull(internal)`` agree and the test can use either.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.spatial import ConvexHull  # noqa: E402

from nd2studios.backend.analysis.granule_tessellate import (  # noqa: E402
    tessellate_granules,
    tetra_circumradii,
)
from nd2studios.backend.analysis.granule_types import (  # noqa: E402
    GranuleBoundary,
    GranuleTessellation,
    NOISE_LABEL,
)

VOX = (1.0, 1.0, 1.0)


def _to_xyz(pts_zyx: np.ndarray) -> np.ndarray:
    """Input ``(z,y,x)`` rows → internal ``(x,y,z)`` (valid because VOX is unit)."""
    return np.asarray(pts_zyx, dtype=float)[..., ::-1].copy()


def _blob(rng, center_zyx, sigma, n) -> np.ndarray:
    """``n`` normal points (N,3) in ``(z,y,x)`` order about ``center_zyx``."""
    return rng.normal(loc=np.asarray(center_zyx, float), scale=sigma, size=(n, 3))


def _box(rng, lo_zyx, hi_zyx, n) -> np.ndarray:
    """``n`` uniform points (N,3) in ``(z,y,x)`` order inside the given box.

    Uniform sampling fixes the region volume, so ``density = n / hull_volume`` truly
    tracks point count — unlike a Gaussian blob, whose hull expands with ``n`` as the
    tails fill in, making density nearly ``n``-independent.
    """
    return rng.uniform(np.asarray(lo_zyx, float), np.asarray(hi_zyx, float),
                       size=(n, 3))


def _check_invariants(tess: GranuleTessellation, n_points: int) -> None:
    """Structural invariants every result must satisfy."""
    assert isinstance(tess, GranuleTessellation)
    assert tess.point_labels.shape == (n_points,)
    assert tess.voxel_size_um == VOX
    present = {int(v) for v in np.unique(tess.point_labels) if int(v) != NOISE_LABEL}
    # Every non-noise label has a boundary, and vice-versa.
    assert present == set(tess.boundaries.keys())
    assert set(tess.merged_from.keys()) == set(tess.boundaries.keys())
    for gid, b in tess.boundaries.items():
        assert isinstance(b, GranuleBoundary)
        assert b.granule_id == gid
        assert b.vertices_um.ndim == 2 and b.vertices_um.shape[1] == 3
        assert b.faces.ndim == 2 and b.faces.shape[1] == 3
        assert b.n_points == int((tess.point_labels == gid).sum())
        assert b.enclosed_volume_um3 > 0.0
        assert np.isfinite(b.density) and b.density > 0.0


# ── Mode A: alpha-shape geometry ────────────────────────────────────────────────
def test_large_alpha_matches_convex_hull_volume():
    rng = np.random.default_rng(0)
    pts = _blob(rng, (0, 0, 0), 3.0, 80)
    labels = np.zeros(len(pts), dtype=int)

    tess = tessellate_granules(pts, labels, VOX,
                               {"tess_mode": "alpha_shape", "alpha": 1e6,
                                "merge_tol": 0.0, "min_granule_points": 4})
    _check_invariants(tess, len(pts))
    assert tess.mode == "alpha_shape"
    assert len(tess.boundaries) == 1
    b = tess.boundaries[0]
    hull_vol = ConvexHull(_to_xyz(pts)).volume
    # A large alpha keeps every Delaunay tetra → sum equals the convex-hull volume.
    assert np.isclose(b.enclosed_volume_um3, hull_vol, rtol=1e-6)
    assert b.delaunay is not None                       # kept for P4's find_simplex


def test_small_alpha_carves_concavity():
    """Dumbbell (two blobs, one granule): small alpha must exclude the gap that the
    convex hull bridges."""
    rng = np.random.default_rng(1)
    # Blobs separated along the x column (input column index 2).
    blob_a = rng.normal((0, 0, 0.0), 0.8, size=(60, 3))
    blob_b = rng.normal((0, 0, 12.0), 0.8, size=(60, 3))
    pts = np.vstack([blob_a, blob_b])
    labels = np.zeros(len(pts), dtype=int)
    alpha = 3.0

    tess = tessellate_granules(pts, labels, VOX,
                               {"tess_mode": "alpha_shape", "alpha": alpha,
                                "merge_tol": 0.0, "min_granule_points": 4})
    _check_invariants(tess, len(pts))
    b = tess.boundaries[0]

    hull_vol = ConvexHull(_to_xyz(pts)).volume
    # Concavity carved out → alpha-shape volume well below the convex hull.
    assert b.enclosed_volume_um3 < hull_vol

    tri = b.delaunay
    circum = tetra_circumradii(tri)

    # A point in the gap: inside the convex hull, but in a DROPPED tetra → excluded.
    mid = _to_xyz(np.array([[0.0, 0.0, 6.0]]))[0]
    s_mid = int(tri.find_simplex(mid))
    assert s_mid >= 0                                   # convex hull includes it
    assert circum[s_mid] > alpha                        # tetra dropped → outside shape

    # A point inside a dense blob sits in a surviving tetra → inside the shape.
    dense = _to_xyz(np.array([[0.0, 0.0, 0.0]]))[0]
    s_dense = int(tri.find_simplex(dense))
    assert s_dense >= 0
    assert circum[s_dense] <= alpha


# ── RAG density-merge ────────────────────────────────────────────────────────────
def _split_by_median_x(pts_zyx: np.ndarray, low_label: int, high_label: int):
    """Split a blob into two labels by the median of its x column (input col 2)."""
    x = pts_zyx[:, 2]
    med = np.median(x)
    lab = np.where(x < med, low_label, high_label)
    return lab.astype(int)


def test_equal_density_neighbours_merge():
    """Two equal-density blobs, each deliberately over-split into two labels (4 total)
    → RAG merge collapses adjacent equal-density halves back to 2 granules."""
    rng = np.random.default_rng(2)
    blob1 = _box(rng, (0, 0, 0), (5, 5, 10), 120)      # x spans [0,10]
    blob2 = _box(rng, (0, 0, 30), (5, 5, 40), 120)     # far away along x
    pts = np.vstack([blob1, blob2])
    labels = np.concatenate([
        _split_by_median_x(blob1, 0, 1),
        _split_by_median_x(blob2, 2, 3),
    ])
    assert len(np.unique(labels)) == 4

    tess = tessellate_granules(pts, labels, VOX,
                               {"tess_mode": "alpha_shape", "alpha": 1e6,
                                "merge_tol": 0.4, "adj_dist_um": 6.0,
                                "min_granule_points": 4})
    _check_invariants(tess, len(pts))
    assert len(tess.boundaries) == 2
    # Each surviving granule folded in exactly two of the original GMM ids.
    for members in tess.merged_from.values():
        assert len(members) == 2
    # The two blobs stayed distinct (no cross-blob point relabelled together).
    b1 = set(tess.point_labels[:120]); b2 = set(tess.point_labels[120:])
    assert b1.isdisjoint(b2)


def test_different_density_neighbours_stay_separate():
    """Two *adjacent* (touching) blobs with very different densities are NOT merged —
    the density-ratio gate keeps them apart despite adjacency."""
    rng = np.random.default_rng(3)
    dense = _box(rng, (0, 0, 0), (5, 5, 5), 120)    # 120 pts in a 5x5x5 box
    sparse = _box(rng, (0, 0, 5), (5, 5, 10), 24)   # 24 pts in the adjacent box
    pts = np.vstack([dense, sparse])
    labels = np.concatenate([np.zeros(120, int), np.ones(24, int)])

    tess = tessellate_granules(pts, labels, VOX,
                               {"tess_mode": "alpha_shape", "alpha": 1e6,
                                "merge_tol": 0.3, "adj_dist_um": 6.0,
                                "min_granule_points": 4})
    _check_invariants(tess, len(pts))
    assert len(tess.boundaries) == 2
    d0, d1 = tess.boundaries[0].density, tess.boundaries[1].density
    assert abs(d0 - d1) / max(d0, d1) > 0.3     # ratio really did exceed merge_tol


# ── Mode B: Voronoi ──────────────────────────────────────────────────────────────
def test_voronoi_mode_per_point_densities():
    """Voronoi mode: a dense jittered grid split into two granules yields bounded
    per-point cells, positive granule densities, and a well-formed tessellation."""
    rng = np.random.default_rng(4)
    coords = np.array([(z, y, x) for z in range(5) for y in range(5)
                       for x in range(5)], dtype=float)
    coords += rng.normal(0.0, 0.1, size=coords.shape)      # jitter avoids degeneracy
    labels = np.where(coords[:, 2] < 2.0, 0, 1).astype(int)  # split on x column

    tess = tessellate_granules(coords, labels, VOX,
                               {"tess_mode": "voronoi", "merge_tol": 0.0,
                                "min_granule_points": 4})
    _check_invariants(tess, len(coords))
    assert tess.mode == "voronoi"
    assert len(tess.boundaries) == 2                    # merge_tol 0 → no merge
    for b in tess.boundaries.values():
        assert b.delaunay is None                       # Voronoi boundaries: P4 rebuilds
        assert b.density > 0.0 and np.isfinite(b.density)
        # density == n_points / Σ(finite cell volumes) is well-defined & positive.
        assert np.isclose(b.density, b.n_points / b.enclosed_volume_um3)


def test_empty_cloud_returns_empty_tessellation():
    tess = tessellate_granules(np.zeros((0, 3)), np.zeros((0,), int), VOX,
                               {"tess_mode": "alpha_shape"})
    assert isinstance(tess, GranuleTessellation)
    assert tess.boundaries == {}
    assert tess.point_labels.shape == (0,)
    assert tess.merged_from == {}
