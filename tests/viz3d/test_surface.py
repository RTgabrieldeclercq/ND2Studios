"""Headless tests for ``nd2studios.backend.viz3d.surface`` (pure numpy, no display).

Covers marching-cubes + Taubin surface construction, the divergence-theorem
enclosed volume, on-surface displacement sampling (exact on a linear field) and
the normal/tangential decomposition.
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.viz3d import surface


class FakeDVC:
    """Minimal DVCResult stand-in with a LINEAR displacement field ``u = A x``.

    Grid coords + displacement are in native ``(z, y, x)`` voxel order (as the
    real ``DVCResult``); the field is built so the world displacement is exactly
    ``A @ x_world``.
    """

    def __init__(self, A, shape=(40, 40, 40), step=4.0, voxel=(1.0, 1.0, 1.0)):
        A = np.asarray(A, dtype=np.float64)
        Z, H, W = shape
        gz = np.arange(0, Z, step)
        gy = np.arange(0, H, step)
        gx = np.arange(0, W, step)
        GZ, GY, GX = np.meshgrid(gz, gy, gx, indexing="ij")
        self.grid_coords = np.stack([GZ, GY, GX], axis=-1)     # native (z,y,x) voxels
        vz, vy, vx = voxel
        Xw = np.stack([GX * vx, GY * vy, GZ * vz], axis=-1)    # world (x,y,z) µm
        Uw = np.einsum("ij,...j->...i", A, Xw)                 # world disp µm
        # store native (z,y,x) displacement in VOXELS (so *voxel → µm reproduces Uw)
        self.displacement_field = np.stack(
            [Uw[..., 2] / vz, Uw[..., 1] / vy, Uw[..., 0] / vx], axis=-1)
        self.voxel_size_um = tuple(float(v) for v in voxel)
        self.dim = 3
        self.strain_field = None
        self.qfactor = None


def _sphere_mask(r=12.0, n=40):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    c = n // 2
    return ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= r ** 2


def test_sphere_enclosed_volume_and_normals():
    r = 12.0
    surf = surface.build_object_surface(_sphere_mask(r), (1.0, 1.0, 1.0),
                                        smooth_iterations=10)
    assert not surf.is_empty
    v_true = (4.0 / 3.0) * np.pi * r ** 3
    # Marching-cubes discretization of a radius-12 sphere is within a few percent.
    assert surf.enclosed_volume_um3 == pytest.approx(v_true, rel=0.05)
    assert surf.enclosed_volume_um3 > 0                       # outward-oriented
    n = np.linalg.norm(surf.vertex_normals, axis=1)
    assert np.allclose(n, 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(surf.face_normals, axis=1), 1.0, atol=1e-6)


def test_surface_is_closed_manifold():
    """Every edge is shared by exactly two triangles (closed, genus-0 → V−E+F=2)."""
    surf = surface.build_object_surface(_sphere_mask(10.0), (1.0, 1.0, 1.0),
                                        smooth_iterations=0)
    f = surf.faces
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    _uniq, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)                                # watertight
    V, E, F = surf.n_vertices, len(_uniq), surf.n_faces
    assert V - E + F == 2                                     # genus 0


def test_anisotropic_voxel_scales_volume():
    mask = _sphere_mask(10.0)
    iso = surface.build_object_surface(mask, (1.0, 1.0, 1.0), smooth_iterations=0)
    aniso = surface.build_object_surface(mask, (3.0, 1.0, 1.0), smooth_iterations=0)
    # Tripling dz roughly triples the enclosed volume.
    assert aniso.enclosed_volume_um3 == pytest.approx(3.0 * iso.enclosed_volume_um3,
                                                      rel=0.05)


def test_empty_mask_returns_empty_surface():
    surf = surface.build_object_surface(np.zeros((10, 10, 10), bool))
    assert surf.is_empty
    assert surf.n_vertices == 0 and surf.n_faces == 0


def test_2d_mask_promoted_to_single_plane():
    m = np.zeros((16, 16), bool)
    m[4:12, 4:12] = True
    surf = surface.build_object_surface(m, (1.0, 1.0, 1.0), smooth_iterations=0)
    assert not surf.is_empty                                  # a thin slab surface


def test_sample_linear_field_is_exact():
    A = np.array([[0.0, 0.2, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.3]])
    mask = _sphere_mask(12.0)
    surf = surface.build_object_surface(mask, (1.0, 1.0, 1.0), smooth_iterations=5)
    u = surface.sample_displacement_on_surface(FakeDVC(A), surf)
    expected = (A @ surf.vertices_um.T).T
    assert np.allclose(u, expected, atol=1e-6)


def test_decompose_normal_tangential():
    rng = np.random.default_rng(0)
    n = rng.standard_normal((50, 3))
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    # Build u = a*n + t where t ⟂ n (t = arbitrary minus its normal part).
    a = rng.standard_normal(50)
    t = rng.standard_normal((50, 3))
    t = t - np.einsum("ij,ij->i", t, n)[:, None] * n
    u = a[:, None] * n + t
    u_perp, u_par_vec, u_par_mag = surface.decompose_surface_displacement(u, n)
    assert np.allclose(u_perp, a, atol=1e-9)
    assert np.allclose(u_par_vec, t, atol=1e-9)
    assert np.allclose(np.einsum("ij,ij->i", u_par_vec, n), 0.0, atol=1e-9)
    assert np.allclose(u_par_mag, np.linalg.norm(t, axis=1), atol=1e-9)


def test_taubin_preserves_volume_better_than_shrinking():
    """Taubin (λ|μ) is non-shrinking: volume stays close after many iterations."""
    mask = _sphere_mask(12.0)
    s0 = surface.build_object_surface(mask, (1.0, 1.0, 1.0), smooth_iterations=0)
    s50 = surface.build_object_surface(mask, (1.0, 1.0, 1.0), smooth_iterations=50)
    # Non-shrinking smoothing keeps the volume within ~10% (a pure Laplacian
    # would collapse the sphere).
    assert s50.enclosed_volume_um3 == pytest.approx(s0.enclosed_volume_um3, rel=0.10)
