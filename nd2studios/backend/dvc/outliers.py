"""
Stage 2 — outlier detection + NaN inpainting for displacement fields.

Two guards, both standard in the DIC/DVC literature and in FranckLab's
``RemoveOutliers3`` / ``inpaint_nans3``:

* **cc threshold** — drop subsets whose correlation confidence is too low.
* **normalized median test** (Westerweel & Scarano, *Exp. Fluids* 2005) — drop
  vectors that disagree with their neighborhood median beyond a robust,
  self-scaling threshold.

Flagged nodes become NaN, then :func:`inpaint_nans` fills them (nearest-value via
Euclidean distance transform — always converges as long as one finite value
exists), so a handful of bad subsets never poison the field or the downstream
global solve. Grid axis order follows :mod:`nd2studios.backend.dvc.mesh`.

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter


def normalized_median_flags(
    u_grid: np.ndarray, *, eps: float = 0.1, threshold: float = 2.0,
    size: int = 3,
) -> np.ndarray:
    """Boolean mask of nodes failing the normalized median test (any component).

    ``u_grid`` is ``(ndim, *grid)``. ``eps`` is the noise floor (voxels) that
    keeps the test from firing on uniform fields; ``threshold`` ~2 is typical.
    """
    ndim = u_grid.shape[0]
    flags = np.zeros(u_grid.shape[1:], dtype=bool)
    for c in range(ndim):
        comp = np.asarray(u_grid[c], dtype=np.float64)
        finite = np.nan_to_num(comp, nan=0.0)
        med = median_filter(finite, size=size, mode="nearest")
        res = np.abs(finite - med)
        res_med = median_filter(res, size=size, mode="nearest")
        norm_res = res / (res_med + eps)
        flags |= norm_res > threshold
    return flags


def remove_outliers(
    u_grid: np.ndarray, cc: np.ndarray, *,
    cc_thresh: float = 0.5, median_thresh: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set low-confidence / median-failing / non-finite nodes to NaN.

    Returns ``(u_clean, bad_mask)`` where ``u_clean`` is ``u_grid`` with flagged
    nodes NaN'd (a copy) and ``bad_mask`` is ``(*grid,)`` bool.
    """
    ndim = u_grid.shape[0]
    bad = ~np.all(np.isfinite(u_grid), axis=0)
    if cc is not None:
        bad |= np.asarray(cc) < float(cc_thresh)
    bad |= normalized_median_flags(u_grid, threshold=median_thresh)
    out = np.array(u_grid, dtype=np.float64, copy=True)
    out[:, bad] = np.nan
    return out, bad


def inpaint_nans(field: np.ndarray) -> np.ndarray:
    """Fill NaNs in a scalar ``(*grid,)`` array by nearest finite value.

    Uses the Euclidean distance transform to index the nearest valid sample —
    guaranteed to fill every NaN provided at least one finite value exists.
    """
    arr = np.asarray(field, dtype=np.float64)
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    if nan_mask.all():
        return np.zeros_like(arr)
    idx = distance_transform_edt(nan_mask, return_distances=False,
                                 return_indices=True)
    return arr[tuple(idx)]


def inpaint_vector(u_grid: np.ndarray) -> np.ndarray:
    """Apply :func:`inpaint_nans` to each component of a ``(ndim, *grid)`` field."""
    out = np.array(u_grid, dtype=np.float64, copy=True)
    for c in range(out.shape[0]):
        out[c] = inpaint_nans(out[c])
    return out
