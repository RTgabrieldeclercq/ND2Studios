"""viz3d.mdm — Mean Deformation Metrics (MDM) on a closed object surface.

Pure numpy. Implements the *kinematic* post-processing of Stout, Bar-Kochba,
Estrada, Toyjanova, Kesari, Reichner & Franck, "Mean deformation metrics for
quantifying 3D cell–matrix interactions without requiring information about
matrix material properties," PNAS 113(11):2898–2903 (2016). See
``Research/mean_deformation_metrics.md``.

The paper converts the *mean* deformation gradient over an object volume ``V0``
into a **surface integral over the object boundary ``∂V0``** via the divergence
theorem (Eq. 6)::

    ⟨∇u⟩ = (1/vol(V0)) ∮_∂V0  u ⊗ n  dA
    ⟨F⟩  = I + ⟨∇u⟩                                    (Eq. 7)

From ``⟨F⟩`` the metrics follow::

    ⟨J⟩          = det⟨F⟩                              (volume-change ratio)
    ⟨F⟩          = ⟨R⟩⟨U⟩                              (right polar decomposition)
    ⟨λ_i⟩,⟨N_i⟩  = eigenvalues/vectors of ⟨U⟩          (stretches + principal dirs)
    cos⟨θ⟩       = (tr⟨R⟩ − 1)/2                        (mean rotation, Eq. 8)
    ⟨Θ⟩          = ∫₀ᵗ |⟨θ(τ)⟩| dτ                      (cumulative rotation, Eq. 9)

This module consumes only an :class:`~nd2studios.backend.viz3d.surface.ObjectSurface`-
like object (``faces``, ``face_normals``, ``face_areas``, ``enclosed_volume_um3``)
plus a per-vertex displacement array. It imports **nothing** from
``backend/dvc/`` — the DVC computation is never touched (the user's caution).
Everything here is the paper's material-property-free adaptation of an
already-measured displacement field onto a surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MDMResult:
    """Mean deformation metrics over a closed object surface (paper Eqs. 6–9).

    All tensors are ``(3, 3)`` even for 2-D DIC input (the extra z row/column is
    then ~0 / identity); ``dim`` records the source field dimensionality.
    """

    grad_u: np.ndarray            # ⟨∇u⟩ (3,3), dimensionless
    F: np.ndarray                 # ⟨F⟩  (3,3) = I + ⟨∇u⟩
    J: float                      # det⟨F⟩ (mean volume-change ratio)
    R: np.ndarray                 # ⟨R⟩  (3,3) proper rotation (polar decomposition)
    U: np.ndarray                 # ⟨U⟩  (3,3) right stretch tensor (symmetric)
    stretches: np.ndarray         # ⟨λ_i⟩ (3,) eigenvalues of ⟨U⟩, ascending
    principal_dirs: np.ndarray    # ⟨N_i⟩ (3,3), columns aligned with `stretches`
    theta_deg: float              # ⟨θ⟩ mean rotation angle (degrees, 0..180)
    dim: int = 3
    notes: str = ""


def mean_displacement_gradient(surface: Any, u_vert: np.ndarray) -> np.ndarray:
    """Discrete surface-integral form of Eq. 6 → ``⟨∇u⟩`` ``(3, 3)``.

    ``⟨∇u⟩_ij = (1/V) Σ_faces (ū_f)_i (n_f)_j A_f`` with ``ū_f`` the mean of the
    face's three vertex displacements, ``n_f`` the **outward** unit face normal,
    ``A_f`` the face area and ``V = surface.enclosed_volume_um3``. Requires an
    outward-oriented triangulation (guaranteed by
    :func:`~nd2studios.backend.viz3d.surface.build_object_surface`).
    """
    faces = np.asarray(surface.faces, dtype=np.int64)
    u = np.asarray(u_vert, dtype=np.float64)
    normals = np.asarray(surface.face_normals, dtype=np.float64)
    areas = np.asarray(surface.face_areas, dtype=np.float64)
    vol = float(surface.enclosed_volume_um3)
    if faces.size == 0 or u.shape[0] == 0 or abs(vol) < 1e-12:
        return np.zeros((3, 3), dtype=np.float64)
    # ū_f : mean displacement over each face's three vertices → (Nf, 3).
    ubar = u[faces].mean(axis=1)
    # Σ_f (ū_f ⊗ n_f) A_f — area-weighted sum of outer products over faces.
    grad = np.einsum("fi,fj,f->ij", ubar, normals, areas)
    return grad / vol


def _polar_decomposition(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Right polar decomposition ``F = R U`` via SVD, with ``R`` a proper rotation.

    ``F = W Σ Vᵀ`` ⇒ ``R = W diag(1,1,det(WVᵀ)) Vᵀ`` (Kabsch correction, so
    ``det R = +1`` even for a near-degenerate ``F``) and ``U = Rᵀ F`` symmetrized.
    For a physical deformation (``det F > 0``) the correction is inert and
    ``U`` is symmetric positive-definite.
    """
    W, S, Vt = np.linalg.svd(F)
    d = np.sign(np.linalg.det(W @ Vt))
    if d == 0.0:
        d = 1.0
    D = np.diag([1.0, 1.0, float(d)])
    R = W @ D @ Vt
    U = R.T @ F
    U = 0.5 * (U + U.T)
    return R, U


def deformation_metrics(grad_u: np.ndarray, *, dim: int = 3,
                        notes: str = "") -> MDMResult:
    """Assemble the MDM suite from ``⟨∇u⟩`` (paper Eqs. 7, 8)."""
    grad_u = np.asarray(grad_u, dtype=np.float64).reshape(3, 3)
    F = np.eye(3) + grad_u                        # Eq. 7
    J = float(np.linalg.det(F))
    R, U = _polar_decomposition(F)
    evals, evecs = np.linalg.eigh(U)              # ascending, orthonormal columns
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))   # Eq. 8
    theta_deg = float(np.degrees(np.arccos(cos_theta)))
    return MDMResult(grad_u=grad_u, F=F, J=J, R=R, U=U,
                     stretches=evals, principal_dirs=evecs,
                     theta_deg=theta_deg, dim=int(dim), notes=notes)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """``np.trapezoid`` (numpy ≥ 2) with a ``np.trapz`` fallback (numpy < 2)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def cumulative_rotation(thetas_deg: Sequence[float],
                        times_s: Optional[Sequence[float]] = None) -> float:
    """Cumulative rotation ``⟨Θ⟩ = ∫₀ᵗ |⟨θ⟩| dτ`` (Eq. 9), trapezoidal.

    ``thetas_deg`` are the per-frame mean rotation magnitudes; ``times_s`` the
    matching timestamps (defaults to unit frame spacing, and falls back to it if
    the lengths disagree). Fewer than two samples ⇒ 0 (nothing accumulated yet).
    """
    th = np.abs(np.asarray(list(thetas_deg), dtype=np.float64))
    if th.size < 2:
        return 0.0
    if times_s is None:
        t = np.arange(th.size, dtype=np.float64)
    else:
        t = np.asarray(list(times_s), dtype=np.float64)
        if t.size != th.size:
            t = np.arange(th.size, dtype=np.float64)
    return _trapezoid(th, t)
