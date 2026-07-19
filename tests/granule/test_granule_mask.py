"""Headless tests for ``nd2studios.backend.analysis.granule_mask`` (pure scipy).

Builds synthetic spherical :class:`GranuleBoundary` objects directly from the P0
``granule_types`` dataclasses (no page / Qt) and voxelizes them, checking:

* analytic voxel count ≈ ``(4/3 π r³) / (dz·dy·dx)`` within tolerance;
* inside/outside correctness at voxel centers;
* anisotropic ``(dz, dy, dx)`` ``floor`` z-plane assignment lands the right planes;
* SDF-Gaussian smoothing (``smooth_sigma > 0``) does not shrink the volume;
* two touching granules → a clean int32 label seam (higher-density wins) and each
  per-granule bool volume passes ``object_scope.iter_objects`` as ONE 3-D object.
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.granule_mask import build_granule_masks
from nd2studios.backend.analysis.granule_types import (
    COMBINED_LABELS_KEY,
    GranuleBoundary,
    GranuleTessellation,
)
from nd2studios.backend.analysis.object_scope import iter_objects

# A typical anisotropic confocal spacing: thick Z, fine XY (µm).
DZ, DY, DX = 2.0, 0.5, 0.5


def _fibonacci_sphere(n: int, r: float, center_xyz) -> np.ndarray:
    """``n`` near-uniform points on a sphere, world ``(x, y, z)`` µm."""
    i = np.arange(n, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = golden * i
    unit = np.column_stack([np.cos(theta) * rho, np.sin(theta) * rho, z])
    return unit * float(r) + np.asarray(center_xyz, dtype=float)


def _sphere_boundary(gid: int, r: float, center_xyz, density: float,
                     n: int = 600, with_delaunay: bool = True) -> GranuleBoundary:
    """A :class:`GranuleBoundary` approximating a sphere (convex-hull mesh)."""
    from scipy.spatial import ConvexHull, Delaunay

    verts = _fibonacci_sphere(n, r, center_xyz)
    hull = ConvexHull(verts)
    tri = Delaunay(verts) if with_delaunay else None
    return GranuleBoundary(
        granule_id=int(gid),
        vertices_um=verts,
        faces=hull.simplices.astype(np.int64),
        enclosed_volume_um3=float(hull.volume),
        n_points=int(n),
        density=float(density),
        delaunay=tri,
    )


def _tess(boundaries) -> GranuleTessellation:
    return GranuleTessellation(
        mode="alpha_shape",
        boundaries={b.granule_id: b for b in boundaries},
        point_labels=np.zeros(0, dtype=int),
        voxel_size_um=(DZ, DY, DX),
        merged_from={},
    )


def test_sphere_voxel_count_matches_analytic():
    r = 6.0
    center = (10.0, 10.0, 12.0)               # world (x, y, z) µm
    shape = (16, 48, 48)                       # (Z, H, W)
    tess = _tess([_sphere_boundary(1, r, center, density=1.0)])

    masks, combined = build_granule_masks(tess, shape, (DZ, DY, DX), {})
    assert set(masks.keys()) == {1}
    vol = masks[1]
    assert vol.shape == shape and vol.dtype == bool

    expected = (4.0 / 3.0 * np.pi * r ** 3) / (DZ * DY * DX)
    count = int(vol.sum())
    assert count == pytest.approx(expected, rel=0.15)

    # Combined int32 mirrors the single granule.
    assert combined.dtype == np.int32 and combined.shape == shape
    assert np.array_equal(combined > 0, vol)
    assert set(np.unique(combined).tolist()) == {0, 1}


def test_inside_outside_at_voxel_centers():
    r = 6.0
    center = (10.0, 10.0, 12.0)
    shape = (16, 48, 48)
    tess = _tess([_sphere_boundary(1, r, center, density=1.0)])
    vol = build_granule_masks(tess, shape, (DZ, DY, DX), {})[0][1]

    # Voxel whose center is exactly the sphere center → inside.
    zc, yc, xc = int(round(center[2] / DZ)), int(round(center[1] / DY)), int(round(center[0] / DX))
    assert vol[zc, yc, xc]
    # A far corner voxel → outside.
    assert not vol[0, 0, 0]
    assert not vol[shape[0] - 1, shape[1] - 1, shape[2] - 1]


def test_anisotropic_floor_z_assignment():
    # Sphere spans world z in [cz - r, cz + r] = [6, 18]; floor(z/DZ) => planes 3..9.
    r = 6.0
    center = (10.0, 10.0, 12.0)
    shape = (16, 48, 48)
    tess = _tess([_sphere_boundary(1, r, center, density=1.0)])
    vol = build_granule_masks(tess, shape, (DZ, DY, DX), {})[0][1]

    per_plane = vol.reshape(vol.shape[0], -1).sum(axis=1)
    # Nothing below plane 3 (world z < 6) or above plane 9 (world z > 18).
    assert per_plane[:3].sum() == 0
    assert per_plane[10:].sum() == 0
    # The equatorial planes (nearest world z = 12) are populated.
    assert per_plane[5] > 0 and per_plane[6] > 0 and per_plane[7] > 0
    # The equator (plane 6, world z = 12) is the widest cross-section.
    assert per_plane[6] == per_plane.max()


def test_delaunay_absent_matches_prebuilt():
    # Voronoi-mode boundary (delaunay=None) must rebuild one internally and give
    # the same voxelization as a boundary that carried its Delaunay.
    r = 5.0
    center = (9.0, 9.0, 10.0)
    shape = (14, 40, 40)
    with_tri = _tess([_sphere_boundary(1, r, center, 1.0, with_delaunay=True)])
    no_tri = _tess([_sphere_boundary(1, r, center, 1.0, with_delaunay=False)])
    a = build_granule_masks(with_tri, shape, (DZ, DY, DX), {})[0][1]
    b = build_granule_masks(no_tri, shape, (DZ, DY, DX), {})[0][1]
    assert np.array_equal(a, b)


def test_smoothing_does_not_shrink_volume():
    r = 7.0
    center = (12.0, 12.0, 14.0)
    shape = (18, 56, 56)
    tess = _tess([_sphere_boundary(1, r, center, density=1.0)])
    base = build_granule_masks(tess, shape, (DZ, DY, DX), {})[0][1]
    smoothed = build_granule_masks(
        tess, shape, (DZ, DY, DX), {"smooth_sigma": 1.0})[0][1]

    n0, n1 = int(base.sum()), int(smoothed.sum())
    assert n1 > 0
    # SDF-Gaussian smoothing is non-shrinking to within tolerance (no erosion).
    assert n1 >= n0 * 0.90
    assert n1 <= n0 * 1.15
    # Result stays a single solid object.
    assert len(iter_objects(smoothed)) == 1


def test_min_object_voxels_drops_specks():
    # A big granule (id 1, hundreds of voxels) beside a tiny one (id 2, ~9 voxels).
    shape = (16, 64, 64)
    big = _sphere_boundary(1, 5.0, (10.0, 10.0, 12.0), density=1.0)
    speck = _sphere_boundary(2, 1.0, (24.0, 24.0, 20.0), density=1.0)
    tess = _tess([big, speck])

    # min_object_voxels=1 keeps both granules.
    kept = build_granule_masks(tess, shape, (DZ, DY, DX),
                               {"min_object_voxels": 1})[0]
    assert set(kept.keys()) == {1, 2}
    assert 0 < int(kept[2].sum()) < 20

    # A threshold above the speck's size drops granule 2 entirely (omitted from the
    # dict and painted nowhere in the combined volume); granule 1 is untouched.
    masks, combined = build_granule_masks(tess, shape, (DZ, DY, DX),
                                          {"min_object_voxels": 20})
    assert set(masks.keys()) == {1}
    assert len(iter_objects(masks[1])) == 1
    assert 2 not in np.unique(combined)


def test_two_touching_granules_clean_seam():
    r = 5.0
    shape = (16, 56, 56)
    # Centers 8 µm apart in X (< 2r = 10) → the spheres overlap.
    b1 = _sphere_boundary(1, r, (10.0, 12.0, 12.0), density=1.0)
    b2 = _sphere_boundary(2, r, (18.0, 12.0, 12.0), density=2.0)
    tess = _tess([b1, b2])

    masks, combined = build_granule_masks(tess, shape, (DZ, DY, DX), {})
    assert set(masks.keys()) == {1, 2}

    # Combined int32: exactly {0, 1, 2}, both granules present.
    assert combined.dtype == np.int32
    assert set(np.unique(combined).tolist()) == {0, 1, 2}
    assert (combined == 1).any() and (combined == 2).any()

    # There IS an overlap, and the higher-density granule (id 2) owns every
    # contested voxel — a clean, deterministic seam.
    overlap = masks[1] & masks[2]
    assert overlap.any()
    assert np.all(combined[overlap] == 2)
    # Outside the overlap each granule keeps its own label.
    only1 = masks[1] & ~masks[2]
    assert np.all(combined[only1] == 1)

    # Each per-granule bool volume enumerates as ONE 3-D object.
    assert len(iter_objects(masks[1])) == 1
    assert len(iter_objects(masks[2])) == 1


def test_lower_id_wins_on_equal_density():
    r = 5.0
    shape = (16, 56, 56)
    b1 = _sphere_boundary(1, r, (10.0, 12.0, 12.0), density=1.0)
    b2 = _sphere_boundary(2, r, (18.0, 12.0, 12.0), density=1.0)   # equal density
    tess = _tess([b1, b2])
    masks, combined = build_granule_masks(tess, shape, (DZ, DY, DX), {})
    overlap = masks[1] & masks[2]
    assert overlap.any()
    assert np.all(combined[overlap] == 1)     # tie → lower id wins


def test_combined_labels_key_constant_is_stable():
    # Guard the reserved key P4/P5 use for the combined volume beside per-gid masks.
    assert COMBINED_LABELS_KEY == "_labels"
