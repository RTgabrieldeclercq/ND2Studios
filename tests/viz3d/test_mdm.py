"""Headless tests for ``nd2studios.backend.viz3d.mdm`` — Mean Deformation Metrics
(Stout et al. 2016), pure numpy, no display.

Validates the paper's kinematic suite on analytic cases: the discrete surface
integral (divergence theorem) must recover ``⟨F⟩`` **exactly** for any linear
displacement field, so the canonical stretch / rotation / shear cases (Fig 1A–C)
come out to machine precision.
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.viz3d import mdm


def _unit_cube_surface():
    """A closed axis-aligned unit-cube triangulation (12 triangles).

    Returns a tiny surface-like stand-in exposing exactly what ``mdm`` reads:
    ``faces``, ``face_normals``, ``face_areas``, ``enclosed_volume_um3`` — plus
    ``vertices`` so tests can build a linear per-vertex displacement.
    """
    from nd2studios.backend.viz3d.surface import (
        _face_geometry, _signed_volume,
    )

    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],          # bottom (z=0), outward = -z
        [4, 5, 6], [4, 6, 7],          # top (z=1), outward = +z
        [0, 1, 5], [0, 5, 4],          # y=0, outward = -y
        [2, 3, 7], [2, 7, 6],          # y=1, outward = +y
        [1, 2, 6], [1, 6, 5],          # x=1, outward = +x
        [0, 4, 7], [0, 7, 3],          # x=0, outward = -x
    ], dtype=np.int64)
    if _signed_volume(v, faces) < 0:
        faces = faces[:, ::-1].copy()
    fn, fa, _fc = _face_geometry(v, faces)

    class _S:
        pass
    s = _S()
    s.vertices = v
    s.faces = faces
    s.face_normals = fn
    s.face_areas = fa
    s.enclosed_volume_um3 = abs(_signed_volume(v, faces))
    return s


def _mdm_for_linear_A(A: np.ndarray) -> mdm.MDMResult:
    """MDM for a linear displacement field ``u = A x`` on the unit cube."""
    s = _unit_cube_surface()
    u_vert = (A @ s.vertices.T).T
    grad = mdm.mean_displacement_gradient(s, u_vert)
    return mdm.deformation_metrics(grad, dim=3)


def test_zero_field_is_identity():
    res = _mdm_for_linear_A(np.zeros((3, 3)))
    assert np.allclose(res.F, np.eye(3), atol=1e-12)
    assert res.J == pytest.approx(1.0, abs=1e-12)
    assert res.theta_deg == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(res.stretches, [1, 1, 1], atol=1e-9)


def test_simple_stretch_recovered_exactly():
    # Paper Fig 1A: λ = [1, 1/1.5, 1.5].
    A = np.diag([1.0, 1.0 / 1.5, 1.5]) - np.eye(3)
    res = _mdm_for_linear_A(A)
    assert np.allclose(res.F, np.eye(3) + A, atol=1e-12)
    assert res.J == pytest.approx(1.0, abs=1e-12)           # 1 * (1/1.5) * 1.5
    # Pure stretch → no rotation. θ comes via arccos(trace) which is numerically
    # sensitive near 0, so ~1e-6° round-off is expected (F itself is exact above).
    assert res.theta_deg == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(sorted(res.stretches), sorted([1.0, 1 / 1.5, 1.5]), atol=1e-9)


def test_axial_rotation_45deg():
    # Paper Fig 1B: rigid rotation θ = 45° about z. F = R, U = I, J = 1.
    th = np.deg2rad(45.0)
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0],
                  [0, 0, 1]], dtype=np.float64)
    res = _mdm_for_linear_A(R - np.eye(3))
    assert np.allclose(res.F, R, atol=1e-12)
    assert res.J == pytest.approx(1.0, abs=1e-12)
    assert res.theta_deg == pytest.approx(45.0, abs=1e-6)
    assert np.allclose(res.stretches, [1, 1, 1], atol=1e-9)   # no stretch
    assert np.allclose(res.R, R, atol=1e-9)


def test_simple_shear_recovered_exactly():
    # Paper Fig 1C: simple shear F = [[1,k,0],[0,1,0],[0,0,1]], det = 1.
    k = 0.3
    A = np.array([[0, k, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    res = _mdm_for_linear_A(A)
    assert np.allclose(res.F, np.eye(3) + A, atol=1e-12)
    assert res.J == pytest.approx(1.0, abs=1e-12)
    # Shear carries both stretch (off-identity U) and a rotation.
    assert res.theta_deg > 0.0
    assert res.stretches[0] < 1.0 < res.stretches[-1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_linear_field_recovered_exactly(seed):
    """Core claim: ⟨F⟩ from the surface integral is exact for ANY linear field."""
    rng = np.random.default_rng(seed)
    A = 0.2 * rng.standard_normal((3, 3))
    res = _mdm_for_linear_A(A)
    assert np.allclose(res.F, np.eye(3) + A, atol=1e-10)
    assert res.J == pytest.approx(np.linalg.det(np.eye(3) + A), abs=1e-10)


def test_polar_decomposition_proper_rotation():
    rng = np.random.default_rng(7)
    A = 0.3 * rng.standard_normal((3, 3))
    res = mdm.deformation_metrics(A, dim=3)
    # R is a proper rotation, U symmetric, and R @ U == F.
    assert np.linalg.det(res.R) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(res.R @ res.R.T, np.eye(3), atol=1e-9)
    assert np.allclose(res.U, res.U.T, atol=1e-9)
    assert np.allclose(res.R @ res.U, res.F, atol=1e-9)


def test_cumulative_rotation_trapezoidal():
    # ∫|θ|dτ over θ=[0,10,20] at t=[0,1,2] = 5 + 15 = 20.
    assert mdm.cumulative_rotation([0, 10, 20], [0, 1, 2]) == pytest.approx(20.0)
    # Uses |θ|; negative angles contribute their magnitude.
    assert mdm.cumulative_rotation([0, -10, -20], [0, 1, 2]) == pytest.approx(20.0)
    # Unit spacing when times omitted.
    assert mdm.cumulative_rotation([0, 10, 20]) == pytest.approx(20.0)
    # Degenerate (fewer than two samples) → 0.
    assert mdm.cumulative_rotation([5.0]) == 0.0
    assert mdm.cumulative_rotation([]) == 0.0


def test_empty_gradient_from_degenerate_surface():
    class _Empty:
        faces = np.zeros((0, 3), dtype=np.int64)
        face_normals = np.zeros((0, 3))
        face_areas = np.zeros((0,))
        enclosed_volume_um3 = 0.0
    grad = mdm.mean_displacement_gradient(_Empty(), np.zeros((0, 3)))
    assert np.allclose(grad, 0.0)
