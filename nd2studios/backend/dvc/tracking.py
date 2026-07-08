"""
Incremental → cumulative accumulation (ALDVC Section, main_ALDVC.m lines 631–706).

In **incremental** tracking mode ALDVC correlates each consecutive frame pair
(N-1 → N), which keeps every correlated step small (robust for large total motion),
then **composes** the per-step increments into a cumulative displacement field by
Lagrangian point-tracking: the reference grid points are advanced through the
sequence — at each step the increment is interpolated at the points' *current*
(drifted) positions and added — and the cumulative displacement is
``U_accum = coordCurr − coord``. Strain is then computed from ``U_accum``.

This module ports that accumulation. It replaces the MATLAB ``interp3(...,'makima')``
+ median±σ clip + ``inpaint_nans3`` with a :class:`RegularGridInterpolator` +
finite-value guard (the grids are regular, so a grid interpolator is exact and
cheaper than scattered interpolation).

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nd2studios.core.dvc_registry import DVCResult
from nd2studios.backend.dvc.mesh import Grid
from nd2studios.backend.dvc.strain import compute_strain


def accumulate_incremental(
    grid: Grid, increments: List[Tuple[int, np.ndarray]],
) -> List[Tuple[int, np.ndarray]]:
    """Compose ordered per-step increment fields into cumulative displacements.

    Parameters
    ----------
    grid : the (shared) subset grid the increments live on.
    increments : ``[(t, u_grid), ...]`` in ascending frame order, each ``u_grid``
        an ``(ndim, *grid)`` incremental displacement (frame ``t-1`` → ``t``), in
        voxels.

    Returns ``[(t, u_accum_grid), ...]`` — the cumulative displacement from the
    reference frame at each ``t`` (``(ndim, *grid)``), by tracking the reference
    grid points through the increments.
    """
    ndim = grid.ndim
    axes = [np.asarray(a, dtype=np.float64) for a in grid.axes]
    coords0 = grid.coords_flat().astype(np.float64)     # (N, ndim), reference
    cur = coords0.copy()
    out: List[Tuple[int, np.ndarray]] = []
    for t, u_grid in increments:
        u_grid = np.asarray(u_grid, dtype=np.float64)
        disp = np.zeros_like(cur)
        for c in range(ndim):
            interp = RegularGridInterpolator(
                axes, u_grid[c], method="linear",
                bounds_error=False, fill_value=None)   # extrapolate at borders
            disp[:, c] = np.nan_to_num(interp(cur), nan=0.0,
                                       posinf=0.0, neginf=0.0)
        cur = cur + disp
        u_accum = (cur - coords0).reshape(*grid.grid_shape, ndim)
        out.append((int(t), np.moveaxis(u_accum, -1, 0)))   # (ndim, *grid)
    return out


def build_accumulated_results(
    grid: Grid, increment_results: List[Tuple[int, DVCResult]],
    voxel_size_um: Tuple[float, ...], *, strain_type: str = "infinitesimal",
    strain_smooth: float = 0.0,
) -> Dict[int, DVCResult]:
    """Turn ordered incremental :class:`DVCResult`s into cumulative ones.

    Accumulates the increments (:func:`accumulate_incremental`), then rebuilds a
    :class:`DVCResult` per frame with the cumulative displacement + strain
    recomputed from it — the field ALDVC's incremental mode actually reports.
    """
    ndim = grid.ndim
    incr = [(t, np.moveaxis(np.asarray(r.displacement_field), -1, 0))
            for t, r in increment_results]     # (ndim,*grid) each
    accum = accumulate_incremental(grid, incr)
    voxel = (np.asarray(voxel_size_um, dtype=np.float64)
             if voxel_size_um and len(voxel_size_um) == ndim else np.ones(ndim))
    out: Dict[int, DVCResult] = {}
    base_by_t = dict(increment_results)
    for t, u_grid in accum:
        base = base_by_t[t]
        _F, strain = compute_strain(u_grid, grid.step, voxel_size=voxel,
                                    strain_type=strain_type,
                                    smooth_sigma=strain_smooth)
        disp = np.moveaxis(u_grid, 0, -1)                    # (*grid, ndim)
        strain_field = np.moveaxis(strain, (0, 1), (-2, -1))
        diag = dict(base.diagnostics)
        diag["accumulated_from_incremental"] = True
        out[t] = DVCResult(
            dim=base.dim, grid_coords=base.grid_coords, displacement_field=disp,
            voxel_size_um=base.voxel_size_um, strain_field=strain_field,
            strain_type=strain_type, qfactor=base.qfactor,
            converged=base.converged, iterations=base.iterations,
            mu=base.mu, beta=base.beta, method="ALDVC (cumulative from incremental)",
            notes=base.notes + " · accumulated to cumulative", diagnostics=diag)
    return out
