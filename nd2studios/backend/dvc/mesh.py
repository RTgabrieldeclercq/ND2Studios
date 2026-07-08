"""
DVC mesh + DOF bookkeeping — the single source of truth for coordinate and
degree-of-freedom conventions in the ALDVC port.

Why this module is isolated
---------------------------
The single largest correctness risk when porting FranckLab's MATLAB ALDVC is
index/order scrambling (1-based vs 0-based, column- vs row-major flattening,
``ndgrid`` axis order, the swapped ``(v, u)`` interpolation argument order). We
sidestep the MATLAB conventions entirely and adopt clean, internally-consistent
**numpy** conventions, centralized here so every other DVC module inherits them:

* **Axis order == array axis order.** A 2D image is ``(y, x)``; a 3D volume is
  ``(z, y, x)``. Grid coordinates and displacement components use the *same*
  slowest-first order — component ``0`` is ``z`` in 3D / ``y`` in 2D. This is the
  order the ND2Studios executor's ``get_frame(z_mode="none")`` yields and the
  order :mod:`nd2studios.backend.serialtrack.fields` already uses
  (``F[i, j] = ∂u_i/∂x_j``).
* **Regular grid of subset centers**, spaced ``subset_spacing`` voxels apart and
  inset by ``subset_size // 2`` from every border so each subset window fits.
* **Flat DOF layout matches** :func:`nd2studios.backend.serialtrack.regularization._build_gradient_operator`
  so we can reuse its sparse finite-difference operator ``D`` verbatim in the
  global step:

  - displacement vector ``u_vec``: ``u_vec[ndim*p + c] = u[c]`` at grid node ``p``
    (``p`` is the C-order linear index of the grid node, ``c`` the component).
  - deformation-gradient vector ``F_vec``:
    ``F_vec[ndim² * p + (j*ndim + i)] = ∂u_i/∂x_j`` — i.e. component ``i`` varies
    fastest within a node, then derivative axis ``j``. This is exactly the layout
    ``_build_gradient_operator`` produces, so ``D @ u_vec ≈ F_vec``.

Pure numpy — no PySide6 (backend-purity rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Grid:
    """A regular grid of subset centers over an image/volume.

    Attributes
    ----------
    axes : list of ``ndim`` 1-D arrays
        Per-axis center coordinates (in voxels), slowest axis first.
    coords : (*grid_shape, ndim) float64
        Center coordinate of every node, ``coords[..., c]`` the ``c``-th axis
        (``ij`` meshgrid → axis order matches ``grid_shape``).
    grid_shape : tuple[int, ...]
        Number of centers along each axis.
    step : (ndim,) float64
        Spacing between adjacent centers, per axis (voxels). Equals
        ``subset_spacing`` except on axes too small to hold >1 center.
    ndim : int
    """
    axes: List[np.ndarray]
    coords: np.ndarray
    grid_shape: Tuple[int, ...]
    step: np.ndarray
    ndim: int

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.grid_shape)) if self.grid_shape else 0

    def coords_flat(self) -> np.ndarray:
        """``(n_nodes, ndim)`` C-order flattened center coordinates."""
        return self.coords.reshape(-1, self.ndim)


def build_grid(shape: Tuple[int, ...], subset_size: int, subset_spacing: int) -> Grid:
    """Build a :class:`Grid` of subset centers for a volume of ``shape``.

    Centers are ``subset_spacing`` apart, inset by ``subset_size // 2`` so every
    subset window lies fully inside the volume. Tiny axes fall back to a single
    center at the axis midpoint.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    half = max(1, int(subset_size) // 2)
    step = max(1, int(subset_spacing))
    axes: List[np.ndarray] = []
    steps: List[float] = []
    for n in shape:
        half_a = min(half, max(0, (n - 1) // 2))   # can't inset past the axis
        start = half_a
        stop = max(start + 1, n - half_a)          # exclusive upper bound
        c = np.arange(start, stop, step, dtype=np.float64)
        if c.size == 0:
            c = np.asarray([n / 2.0], dtype=np.float64)
        elif c.size == 1 and (stop - 1) > start:
            # Force >=2 centers where the extent allows: the finite-difference
            # gradient operator + np.gradient (strain) are undefined on a size-1
            # axis, so a single grid node along an axis would break the global
            # solve. Two evenly-placed centers keep a thin (e.g. shallow-Z) grid
            # well-formed.
            c = np.round(np.linspace(start, stop - 1, 2)).astype(np.float64)
        axes.append(c)
        steps.append(float(np.median(np.diff(c))) if c.size > 1 else float(step))
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack(mesh, axis=-1).astype(np.float64)
    return Grid(
        axes=axes,
        coords=coords,
        grid_shape=tuple(a.size for a in axes),
        step=np.asarray(steps, dtype=np.float64),
        ndim=ndim,
    )


# ─────────────────────────────────────────────────────────────────────────
#  DOF pack / unpack  (layout matches serialtrack _build_gradient_operator)
# ─────────────────────────────────────────────────────────────────────────

def pack_u(disp_grid: np.ndarray) -> np.ndarray:
    """``(ndim, *grid) → (ndim*n_nodes,)`` — ``u_vec[ndim*p + c] = disp_grid[c].flat[p]``."""
    ndim = disp_grid.shape[0]
    n = int(np.prod(disp_grid.shape[1:]))
    out = np.empty(ndim * n, dtype=np.float64)
    for c in range(ndim):
        out[c::ndim] = np.asarray(disp_grid[c], dtype=np.float64).ravel(order="C")
    return out


def unpack_u(u_vec: np.ndarray, grid_shape: Tuple[int, ...], ndim: int) -> np.ndarray:
    """``(ndim*n,) → (ndim, *grid)`` — inverse of :func:`pack_u`."""
    out = np.empty((ndim, *grid_shape), dtype=np.float64)
    for c in range(ndim):
        out[c] = u_vec[c::ndim].reshape(grid_shape)
    return out


def pack_F(F_grid: np.ndarray) -> np.ndarray:
    """``(ndim, ndim, *grid) → (ndim²*n,)``.

    ``F_grid[i, j]`` is ``∂u_i/∂x_j``; packed at per-node offset ``j*ndim + i``
    (component ``i`` fastest), matching ``_build_gradient_operator``'s ``F_vec``.
    """
    ndim = F_grid.shape[0]
    n = int(np.prod(F_grid.shape[2:]))
    out = np.empty(ndim * ndim * n, dtype=np.float64)
    for j in range(ndim):          # derivative axis
        for i in range(ndim):      # displacement component
            out[(j * ndim + i)::(ndim * ndim)] = \
                np.asarray(F_grid[i, j], dtype=np.float64).ravel(order="C")
    return out


def unpack_F(F_vec: np.ndarray, grid_shape: Tuple[int, ...], ndim: int) -> np.ndarray:
    """``(ndim²*n,) → (ndim, ndim, *grid)`` — inverse of :func:`pack_F` (``[i, j]``)."""
    out = np.empty((ndim, ndim, *grid_shape), dtype=np.float64)
    for j in range(ndim):
        for i in range(ndim):
            out[i, j] = F_vec[(j * ndim + i)::(ndim * ndim)].reshape(grid_shape)
    return out
