"""Headless tests for ``overlays.merge_surface_fields`` — the V1.74 "All granules"
composite that concatenates per-granule :class:`SurfaceField`\\s into one mesh
(pure numpy, no display).
"""
from __future__ import annotations

import numpy as np

from nd2studios.backend.viz3d.overlays import SurfaceField, merge_surface_fields


def _sf(verts, faces, scalars, upar=None):
    v = np.asarray(verts, dtype=float)
    return SurfaceField(
        vertices_um=v,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_normals=np.tile(np.array([0.0, 0.0, 1.0]), (v.shape[0], 1)),
        scalars={k: np.asarray(a, dtype=float) for k, a in scalars.items()},
        u_par_vec=(np.asarray(upar, dtype=float) if upar is not None
                   else np.zeros_like(v)),
        centroid_um=v.mean(axis=0) if v.shape[0] else np.zeros(3),
        default_scalar="disp_mag", dim=3, n_object_voxels=int(v.shape[0]))


def _tri(base=0.0):
    verts = np.array([[base, 0, 0], [base + 1, 0, 0], [base, 1, 0]], float)
    return verts, [[0, 1, 2]]


def test_merge_two_offsets_faces_and_concats_scalars():
    a_v, a_f = _tri(0.0)
    b_v, b_f = _tri(0.0)
    a = _sf(a_v, a_f, {"disp_mag": [1.0, 1.0, 1.0]})
    b = _sf(b_v, b_f, {"disp_mag": [2.0, 2.0, 2.0]})
    m = merge_surface_fields([a, b])
    assert m.vertices_um.shape == (6, 3)
    # Second field's faces must be offset by the first field's vertex count.
    assert np.array_equal(m.faces, np.array([[0, 1, 2], [3, 4, 5]]))
    assert np.array_equal(m.scalars["disp_mag"], [1, 1, 1, 2, 2, 2])
    assert m.mdm is None            # composite MDM is intentionally dropped
    assert m.n_object_voxels == 6


def test_merge_applies_per_field_offsets():
    a_v, a_f = _tri()
    b_v, b_f = _tri()
    a = _sf(a_v, a_f, {"disp_mag": [0, 0, 0]})
    b = _sf(b_v, b_f, {"disp_mag": [0, 0, 0]})
    # Offset B by (+10 x, +20 y, +30 z) world µm.
    m = merge_surface_fields([a, b], offsets_um=[(0, 0, 0), (10, 20, 30)])
    assert np.allclose(m.vertices_um[:3], a_v)                 # A unchanged
    assert np.allclose(m.vertices_um[3:], b_v + np.array([10, 20, 30]))


def test_merge_scalar_union_fills_nan_for_missing_keys():
    a_v, a_f = _tri()
    b_v, b_f = _tri()
    a = _sf(a_v, a_f, {"disp_mag": [1, 1, 1], "u_perp": [0.5, 0.5, 0.5]})
    b = _sf(b_v, b_f, {"disp_mag": [2, 2, 2]})   # no u_perp
    m = merge_surface_fields([a, b])
    assert set(m.scalars) == {"disp_mag", "u_perp"}
    # B's vertices get NaN for the key it lacks.
    assert np.array_equal(m.scalars["disp_mag"], [1, 1, 1, 2, 2, 2])
    up = m.scalars["u_perp"]
    assert np.allclose(up[:3], 0.5)
    assert np.all(np.isnan(up[3:]))


def test_merge_skips_empty_fields_and_keeps_offset_alignment():
    empty = SurfaceField(
        vertices_um=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int64),
        vertex_normals=np.zeros((0, 3)), scalars={}, u_par_vec=np.zeros((0, 3)),
        centroid_um=np.zeros(3))
    a_v, a_f = _tri()
    a = _sf(a_v, a_f, {"disp_mag": [0, 0, 0]})
    # offsets are index-aligned with `fields` BEFORE empties are dropped, so the
    # empty's slot must not shift A's offset.
    m = merge_surface_fields([empty, a], offsets_um=[(99, 99, 99), (5, 0, 0)])
    assert m.vertices_um.shape == (3, 3)
    assert np.allclose(m.vertices_um, a_v + np.array([5, 0, 0]))


def test_merge_all_empty_returns_empty():
    empty = SurfaceField(
        vertices_um=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int64),
        vertex_normals=np.zeros((0, 3)), scalars={}, u_par_vec=np.zeros((0, 3)),
        centroid_um=np.zeros(3))
    m = merge_surface_fields([empty, None])
    assert m.is_empty


def test_merge_single_field_with_offset_translates():
    a_v, a_f = _tri()
    a = _sf(a_v, a_f, {"disp_mag": [1, 1, 1]})
    m = merge_surface_fields([a], offsets_um=[(2, 3, 4)])
    assert np.allclose(m.vertices_um, a_v + np.array([2, 3, 4]))
    assert m.mdm is None
    # Single field, no offset → returned as-is.
    assert merge_surface_fields([a]) is a


def test_all_granules_composite_from_real_surfaces_is_separated():
    """End-to-end: two real per-granule surfaces (marching cubes + sampled field)
    offset by their crop origins compose into one spatially-separated mesh — the
    "All granules" 3-D view path."""
    from nd2studios.backend.viz3d.overlays import dvc_object_surface
    from tests.viz3d.test_surface import FakeDVC, _sphere_mask

    A = np.eye(3) * 0.1 - np.eye(3) * 0.0
    mask = _sphere_mask(8.0)                          # same small sphere for both
    a = dvc_object_surface(FakeDVC(A), mask, (1.0, 1.0, 1.0), smooth_iterations=0)
    b = dvc_object_surface(FakeDVC(A), mask, (1.0, 1.0, 1.0), smooth_iterations=0)
    assert not a.is_empty and not b.is_empty

    span = float(np.ptp(a.vertices_um[:, 0]))
    gap = span * 5.0                                  # push B well clear of A in x
    m = merge_surface_fields([a, b], offsets_um=[(0.0, 0.0, 0.0), (gap, 0.0, 0.0)])

    assert m.vertices_um.shape[0] == a.n_vertices + b.n_vertices
    assert m.faces.shape[0] == a.faces.shape[0] + b.faces.shape[0]
    # Faces of component B index only B's vertices (offset applied, no cross-links).
    assert m.faces[a.faces.shape[0]:].min() >= a.n_vertices
    # The two components are disjoint in x (a real separation, not overlapping).
    assert float(np.ptp(m.vertices_um[:, 0])) > span + gap * 0.5
    assert "disp_mag" in m.scalars
    assert m.scalars["disp_mag"].shape[0] == m.vertices_um.shape[0]
