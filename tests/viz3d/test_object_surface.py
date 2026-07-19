"""Headless tests for the V1.68 DVC-on-object **surface** adapters in
``nd2studios.backend.viz3d.overlays`` — ``dvc_object_surface`` + ``unwrap_surface``
(pure numpy, no display).
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.viz3d import overlays
from tests.viz3d.test_surface import FakeDVC, _sphere_mask


class FakeDVC2D:
    """A 2-D DIC result (dim=2): grid + displacement in native (y, x) voxels."""

    def __init__(self, ax=0.2, ay=0.1, shape=(40, 40), step=4.0, px=1.0):
        H, W = shape
        gy = np.arange(0, H, step)
        gx = np.arange(0, W, step)
        GY, GX = np.meshgrid(gy, gx, indexing="ij")
        self.grid_coords = np.stack([GY, GX], axis=-1)          # native (y,x) voxels
        # world disp: u_x = ax*x, u_y = ay*y ; store native (y,x) voxels
        uy = ay * (GY * px) / px
        ux = ax * (GX * px) / px
        self.displacement_field = np.stack([uy, ux], axis=-1)
        self.voxel_size_um = (px, px)
        self.dim = 2
        self.strain_field = None
        self.qfactor = None


def test_dvc_object_surface_recovers_mdm_and_scalars():
    A = np.diag([1.0, 1.0 / 1.5, 1.5]) - np.eye(3)
    mask = _sphere_mask(12.0)
    sf = overlays.dvc_object_surface(FakeDVC(A), mask, (1.0, 1.0, 1.0),
                                     smooth_iterations=5, with_metrics=True)
    assert isinstance(sf, overlays.SurfaceField)
    assert not sf.is_empty
    assert sf.mdm is not None
    assert np.allclose(sf.mdm.F, np.eye(3) + A, atol=1e-6)
    assert sf.mdm.J == pytest.approx(1.0, abs=1e-6)
    # Superset of Phase-2 scalars + the new normal/tangential decomposition.
    for key in ("disp_mag", "u_x", "u_y", "u_z", "u_perp", "u_par"):
        assert key in sf.scalars
        assert sf.scalars[key].shape[0] == sf.n_vertices
    # disp_mag is consistent with the sampled components.
    comp = np.sqrt(sf.scalars["u_x"] ** 2 + sf.scalars["u_y"] ** 2
                   + sf.scalars["u_z"] ** 2)
    assert np.allclose(sf.scalars["disp_mag"], comp, atol=1e-9)


def test_dvc_object_surface_2d_dic_broadcasts_over_z():
    mask = _sphere_mask(10.0)
    sf = overlays.dvc_object_surface(FakeDVC2D(), mask, (1.0, 1.0, 1.0),
                                     smooth_iterations=0, with_metrics=True)
    assert not sf.is_empty
    assert sf.dim == 2
    # 2-D field has no z-displacement.
    assert np.allclose(sf.scalars["u_z"], 0.0, atol=1e-9)
    # u_x grows with world x (linear field) → non-constant.
    assert np.ptp(sf.scalars["u_x"]) > 0.0


def test_dvc_object_surface_with_interior_attaches_masked_field():
    mask = _sphere_mask(10.0)
    sf = overlays.dvc_object_surface(FakeDVC(np.eye(3) * 0.1 - np.eye(3) * 0.0),
                                     mask, (1.0, 1.0, 1.0), with_interior=True)
    assert sf.interior is not None
    assert isinstance(sf.interior, overlays.MaskedField)
    assert not sf.interior.is_empty


def test_dvc_object_surface_empty_mask():
    sf = overlays.dvc_object_surface(FakeDVC(np.zeros((3, 3))),
                                     np.zeros((10, 10, 10), bool), (1, 1, 1))
    assert sf.is_empty
    assert sf.mdm is None


def test_unwrap_mollweide_and_equirectangular_cover_the_sphere():
    A = np.diag([1.0, 1.0 / 1.5, 1.5]) - np.eye(3)
    sf = overlays.dvc_object_surface(FakeDVC(A), _sphere_mask(12.0), (1, 1, 1),
                                     smooth_iterations=5)
    moll = overlays.unwrap_surface(sf, "u_perp", projection="mollweide",
                                   width=128, height=64)
    eq = overlays.unwrap_surface(sf, "u_perp", projection="equirectangular",
                                 width=128, height=64)
    assert not moll.is_empty and not eq.is_empty
    # Equirectangular fills the whole (lon,lat) rectangle; Mollweide fills the
    # inscribed ellipse (≈ π/4 of the frame).
    assert eq.coverage.mean() > 0.85
    assert 0.5 < moll.coverage.mean() < 0.95
    # Values sit in the u_perp range and the tangential field is populated.
    assert np.nanmin(moll.values) < np.nanmax(moll.values)
    assert moll.vec_u.shape == (64, 128)


def test_unwrap_missing_scalar_falls_back_to_default():
    sf = overlays.dvc_object_surface(FakeDVC(np.zeros((3, 3)) + 0.01),
                                     _sphere_mask(10.0), (1, 1, 1))
    um = overlays.unwrap_surface(sf, "nonexistent_scalar", width=64, height=32)
    # Falls back to the surface's default scalar rather than returning empty.
    assert um.scalar == sf.default_scalar


def test_unwrap_degenerate_surface_returns_empty_not_crash():
    # A <3-vertex "surface" (even with the scalar present) must not reach griddata.
    class TinySF:
        vertices_um = np.zeros((2, 3), dtype=float)
        faces = np.zeros((0, 3), dtype=np.int64)
        scalars = {"u_perp": np.zeros(2)}
        u_par_vec = np.zeros((2, 3))
        centroid_um = np.zeros(3)
        default_scalar = "u_perp"
    um = overlays.unwrap_surface(TinySF(), "u_perp", width=64, height=32)
    assert um.is_empty


def test_sample_displacement_degenerate_grid_returns_zeros():
    from nd2studios.backend.viz3d import surface as sm

    class BadDVC:
        dim = 3
        # A single-node grid → RegularGridInterpolator build will fail; must not
        # propagate (the sibling scalar sampler already guards).
        grid_coords = np.zeros((1, 1, 1, 3), dtype=float)
        displacement_field = np.zeros((1, 1, 1, 3), dtype=float)
        voxel_size_um = (1.0, 1.0, 1.0)
    surf = sm.build_object_surface(_sphere_mask(9.0), (1, 1, 1), smooth_iterations=0)
    u = sm.sample_displacement_on_surface(BadDVC(), surf)
    assert u.shape == (surf.n_vertices, 3)
    assert np.all(u == 0.0)          # graceful zero, no exception
