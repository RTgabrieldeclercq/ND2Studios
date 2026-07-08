"""
Stage 7 — strain tensor from the compatible displacement field.

Builds the displacement gradient ``G = ∇û`` (finite differences on the subset
grid), the deformation gradient ``F = I + G``, and one of four strain measures.
Mirrors FranckLab ``ComputeStrain3`` and parallels
:meth:`nd2studios.backend.serialtrack.fields.DisplacementField.gradient`
(``G[i, j] = ∂u_i/∂x_j``, mesh axis order).

Physical (anisotropic-voxel) scaling: converting both displacement components and
spatial axes to micrometers rescales each gradient entry by
``voxel_i / voxel_j`` — the cross-axis rescaling required for non-cubic voxels
(confocal ``z`` step ≠ ``xy`` pixel).

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def displacement_gradient(
    u_grid: np.ndarray, grid_step: np.ndarray,
    voxel_size: Optional[np.ndarray] = None, smooth_sigma: float = 0.0,
) -> np.ndarray:
    """``(ndim, *grid) → (ndim, ndim, *grid)`` displacement gradient ``∂u_i/∂x_j``.

    ``grid_step`` is the node spacing (voxels) per axis. If ``voxel_size`` is
    given, the result is in physical (dimensionless-strain) units with the
    ``voxel_i/voxel_j`` cross-axis rescaling applied.
    """
    ndim = u_grid.shape[0]
    step = np.asarray(grid_step, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)
    if smooth_sigma and smooth_sigma > 0:
        u = np.stack([gaussian_filter(u[i], sigma=float(smooth_sigma))
                      for i in range(ndim)])
    G = np.zeros((ndim, ndim, *u.shape[1:]), dtype=np.float64)
    gshape = u.shape[1:]
    for i in range(ndim):
        for j in range(ndim):
            # np.gradient needs >=2 samples along the axis; a singleton axis
            # (e.g. a 1-node-deep Z grid) contributes zero gradient there.
            if gshape[j] < 2:
                continue
            G[i, j] = np.gradient(u[i], float(step[j]), axis=j)
    if voxel_size is not None:
        v = np.asarray(voxel_size, dtype=np.float64)
        for i in range(ndim):
            for j in range(ndim):
                G[i, j] *= v[i] / v[j]
    return G


def strain_from_gradient(G: np.ndarray, strain_type: str = "infinitesimal") -> np.ndarray:
    """Strain tensor ``(ndim, ndim, *grid)`` from the displacement gradient ``G``.

    ``strain_type`` ∈ {infinitesimal, green-lagrange, almansi, hencky}.
    """
    ndim = G.shape[0]

    def _t(A):  # transpose the two tensor axes, keep grid axes
        return A.transpose(1, 0, *range(2, A.ndim))

    st = str(strain_type).lower().replace("_", "-")
    if st in ("infinitesimal", "small", "engineering"):
        return 0.5 * (G + _t(G))

    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (G.ndim - 2)))
    F = G + eye                                       # deformation gradient
    if st in ("green-lagrange", "green", "lagrange"):
        FtF = np.einsum("ki...,kj...->ij...", F, F)   # Fᵀ F
        return 0.5 * (FtF - eye)
    if st in ("almansi", "euler-almansi", "eulerian-almansi"):
        Finv = _inv_tensor_field(F)
        FiTFi = np.einsum("ki...,kj...->ij...", Finv, Finv)   # F⁻ᵀ F⁻¹
        return 0.5 * (eye - FiTFi)
    if st in ("hencky", "log", "logarithmic"):
        return _hencky(F)
    raise ValueError(f"unknown strain_type: {strain_type!r}")


def _inv_tensor_field(F: np.ndarray) -> np.ndarray:
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    inv = np.linalg.inv(moved)
    return np.moveaxis(inv, (-2, -1), (0, 1))


def _hencky(F: np.ndarray) -> np.ndarray:
    """Hencky (logarithmic) strain ``½ ln(FᵀF)`` via eigen-decomposition."""
    ndim = F.shape[0]
    C = np.einsum("ki...,kj...->ij...", F, F)         # right Cauchy-Green
    Cm = np.moveaxis(C, (0, 1), (-2, -1))
    w, V = np.linalg.eigh(Cm)
    w = np.clip(w, 1e-12, None)
    logw = 0.5 * np.log(w)
    E = np.einsum("...ik,...k,...jk->...ij", V, logw, V)
    return np.moveaxis(E, (-2, -1), (0, 1))


def compute_strain(
    u_grid: np.ndarray, grid_step: np.ndarray,
    voxel_size: Optional[np.ndarray] = None, *,
    strain_type: str = "infinitesimal", smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(F_def, strain)`` — deformation gradient ``I+∇u`` and the strain
    tensor of the requested measure, both ``(ndim, ndim, *grid)``."""
    G = displacement_gradient(u_grid, grid_step, voxel_size, smooth_sigma)
    ndim = G.shape[0]
    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (G.ndim - 2)))
    strain = strain_from_gradient(G, strain_type)
    return G + eye, strain
