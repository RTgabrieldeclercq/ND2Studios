"""
SerialTrack Python — Displacement & strain field computation
=============================================================
    serialtrack/fields.py

Replaces: funDerivativeOp3 (post-processing usage),
          funCompDefGrad3 (strain gauge), postprocessing sections.

Fixes from v1
-------------
- Fixed logger name: was "serialtrack.regularization", now "serialtrack.fields".
- Fixed ``DisplacementField.gradient()``: was fragile inference of grid
  spacing from meshgrid diff.  Now uses the actual grid axis spacing
  directly from the meshgrid arrays (simple ``np.gradient`` with axis spacing).
- Fixed ``DisplacementField.velocity``: broadcasting was fragile for 2D
  (hardcoded 4 trailing ``None`` dimensions).  Now uses generic reshaping.
- Fixed ``StrainField``: added ``pixel_steps`` attribute for serialization
  compatibility.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass, field as dc_field
import logging

from .regularization import scatter_to_grid_multi

log = logging.getLogger("serialtrack.fields")


# ═══════════════════════════════════════════════════════════════
#  Displacement field
# ═══════════════════════════════════════════════════════════════

@dataclass
class DisplacementField:
    """Gridded displacement field with metadata.

    Stores displacement components on a regular grid and provides
    gradient (strain) computation via ``np.gradient``.

    Attributes
    ----------
    grids : tuple of D ndarray
        Meshgrid coordinate arrays (one per spatial dimension).
    components : (D, *grid_shape) ndarray
        Displacement components on the grid.
    pixel_steps : (D,) ndarray
        Physical size per pixel in each dimension.
    time_step : float
        Time between frames (default 1.0).
    """
    grids: Tuple[np.ndarray, ...]
    components: np.ndarray
    pixel_steps: np.ndarray
    time_step: float = 1.0

    @property
    def ndim(self) -> int:
        return len(self.grids)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.grids[0].shape

    @property
    def velocity(self) -> np.ndarray:
        """Displacement / time_step → velocity field, in physical units."""
        ndim = self.ndim
        # Build a shape like (D, 1, 1, ...) for broadcasting
        ps_shape = [ndim] + [1] * ndim
        ps = self.pixel_steps.reshape(ps_shape)
        return self.components * ps / self.time_step

    def _axis_spacings(self) -> list:
        """Extract grid spacing along each axis from the meshgrid arrays.

        For axis ``d``, we extract the unique 1-D coordinates from
        ``self.grids[d]`` and compute the step.  Falls back to
        ``pixel_steps[d]`` if the grid has only one point along that axis.
        """
        spacings = []
        for d in range(self.ndim):
            # Extract the unique coordinates along axis d
            # (meshgrid arrays repeat; taking a 1-D slice is cheapest)
            idx = [0] * self.ndim
            idx[d] = slice(None)
            axis_coords = self.grids[d][tuple(idx)]
            if len(axis_coords) > 1:
                step = float(np.median(np.diff(axis_coords)))
                spacings.append(step)
            else:
                spacings.append(float(self.pixel_steps[d]))
        return spacings

    def gradient(self) -> np.ndarray:
        """Compute deformation gradient tensor F = ∂u/∂x.

        Returns
        -------
        F : (D, D, *grid_shape) — F[i, j] = ∂u_i / ∂x_j
        """
        ndim = self.ndim
        spacings = self._axis_spacings()
        F = np.zeros((ndim, ndim, *self.shape), dtype=np.float64)

        for i in range(ndim):
            # np.gradient returns a list of arrays (one per axis) for ndim > 1
            grads = np.gradient(self.components[i], *spacings)
            if ndim == 1:
                grads = [grads]
            for j in range(ndim):
                F[i, j] = grads[j]
        return F

    def strain(self) -> np.ndarray:
        """Infinitesimal strain tensor ε = 0.5*(F + F^T).

        Returns (D, D, *grid_shape).
        """
        F = self.gradient()
        return 0.5 * (F + F.transpose(1, 0, *range(2, 2 + self.ndim)))

    def to_physical(self) -> 'DisplacementField':
        """Return a new field with displacements in physical units."""
        ndim = self.ndim
        phys = self.components.copy()
        for d in range(ndim):
            phys[d] *= self.pixel_steps[d]
        return DisplacementField(
            grids=self.grids,
            components=phys,
            pixel_steps=self.pixel_steps,
            time_step=self.time_step,
        )


# ═══════════════════════════════════════════════════════════════
#  Strain field
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrainField:
    """Pre-computed strain field with components.

    Attributes
    ----------
    grids : tuple of D ndarray
        Meshgrid coordinate arrays.
    F_tensor : (D, D, *grid_shape) ndarray
        Deformation gradient tensor.
    eps_tensor : (D, D, *grid_shape) ndarray
        Infinitesimal strain tensor.
    pixel_steps : (D,) ndarray
        Physical size per pixel (for serialization round-trip).
    """
    grids: Tuple[np.ndarray, ...]
    F_tensor: np.ndarray
    eps_tensor: np.ndarray
    pixel_steps: np.ndarray = dc_field(default_factory=lambda: np.ones(3))


# ═══════════════════════════════════════════════════════════════
#  Scattered MLS strain gauge
# ═══════════════════════════════════════════════════════════════

def compute_strain_mls(
    disp: np.ndarray,
    coords: np.ndarray,
    f_o_s: float,
    n_neighbors: int,
    pixel_steps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving least-squares strain gauge at each particle.

    Replaces ``funCompDefGrad3.m`` for post-processing strain
    computation.  Fits u(x) = u0 + F·(x - x0) at each particle
    using its neighbors.

    Parameters
    ----------
    disp : (N, D) — displacement at each particle
    coords : (N, D) — particle positions
    f_o_s, n_neighbors : search parameters
    pixel_steps : optional physical scaling

    Returns
    -------
    U : (N, D) — fitted displacement (smoothed)
    F_tensor : (N, D, D) — deformation gradient at each particle
    valid : (N,) bool — which particles had a valid fit
    """
    ndim = coords.shape[1]
    N = len(coords)
    U = np.full((N, ndim), np.nan)
    F_tensor = np.full((N, ndim, ndim), np.nan)
    valid = np.zeros(N, dtype=bool)

    if N < ndim + 1:
        return U, F_tensor, valid

    tree = cKDTree(coords)
    K = min(n_neighbors, N - 1)
    radius = np.sqrt(ndim) * f_o_s
    dd, ii = tree.query(coords, k=K + 1)

    for p in range(N):
        # Neighbors within radius
        mask = dd[p] < radius
        idx = ii[p][mask]
        if len(idx) < ndim + 1:
            U[p] = disp[p]
            continue

        # Design matrix: [1, (x-x0), (y-y0), (z-z0)]
        dx = coords[idx] - coords[p]
        A = np.column_stack([np.ones(len(idx)), dx])

        try:
            for d in range(ndim):
                params, _, _, _ = np.linalg.lstsq(A, disp[idx, d], rcond=None)
                U[p, d] = params[0]
                F_tensor[p, d, :] = params[1:]  # ∂u_d/∂x_j
            valid[p] = True
        except np.linalg.LinAlgError:
            U[p] = disp[p]

    # Apply physical scaling if provided
    if pixel_steps is not None:
        ps = np.asarray(pixel_steps)
        for i in range(ndim):
            for j in range(ndim):
                F_tensor[valid, i, j] *= ps[i] / ps[j]

    return U, F_tensor, valid


# ═══════════════════════════════════════════════════════════════
#  Full post-processing pipeline
# ═══════════════════════════════════════════════════════════════

def compute_gridded_strain(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 1e-3,
    pixel_steps: Optional[np.ndarray] = None,
) -> Tuple[DisplacementField, StrainField]:
    """Full post-processing: scatter → grid → gradient → strain.

    Replaces the postprocessing sections of
    ``run_Serial_MPT_3D_hardpar_accum.m``.

    Returns both the gridded displacement field and strain field.
    """
    ndim = coords.shape[1]
    ps = np.ones(ndim) if pixel_steps is None else np.asarray(pixel_steps)

    grids, disp_grid = scatter_to_grid_multi(
        coords, disp, grid_step, smoothness
    )

    dfield = DisplacementField(
        grids=grids,
        components=disp_grid,
        pixel_steps=ps,
    )

    F = dfield.gradient()
    eps = 0.5 * (F + F.transpose(1, 0, *range(2, 2 + ndim)))

    sfield = StrainField(grids=grids, F_tensor=F, eps_tensor=eps,
                         pixel_steps=ps)

    return dfield, sfield
