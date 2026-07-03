"""Eulerian spatial-field computation.

Vendored from CellTracker ``backend/fields.py``. Bins per-cell measurements onto
a regular grid to produce spatial heatmaps of density, intensity, velocity,
divergence, curl, etc. (``compute_self_fold_change`` lives in
:mod:`nd2studios.backend.celltracker.metrics`.)

Operates on a tracked pandas DataFrame (``frame``, ``centroid_y``,
``centroid_x``, ``area``, ``track_id`` for velocity, and an intensity column for
intensity / fold-change).

All fields are built the same way as **cell density** — from the cells actually
recorded in each region, not by interpolating across the whole grid. Density is a
Gaussian-smoothed count of cells per grid bin; every value-bearing field
(``mean_area``, ``intensity``, ``velocity_*`` …) is the matching Gaussian-weighted
*local mean*: bin the per-cell value and the cell count, smooth both with the same
Gaussian, divide. Density is exactly the denominator of that ratio, so the fields
are consistent — values exist only where cells were measured and are left
undefined (``NaN``) in empty regions, instead of being invented by interpolation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def _bin_sum_count(
    cy: np.ndarray,
    cx: np.ndarray,
    values: np.ndarray,
    grid_shape: Tuple[int, int],
    grid_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate per-cell ``values`` (and a unit count) into grid bins by centroid.

    Each cell drops into the bin ``(int(y / grid_step), int(x / grid_step))`` — the
    same binning the density field uses. Cells whose centroid falls outside the
    grid, or whose value is non-finite, are skipped (so a ``NaN`` value never
    contaminates either the sum or the count). Returns ``(value_sum, count)``,
    both ``grid_shape`` float arrays.
    """
    ny, nx = grid_shape
    cy = np.asarray(cy, dtype=float)
    cx = np.asarray(cx, dtype=float)
    values = np.asarray(values, dtype=float)

    gi = np.floor(cy / grid_step).astype(int)
    gj = np.floor(cx / grid_step).astype(int)
    inb = (gi >= 0) & (gi < ny) & (gj >= 0) & (gj < nx) & np.isfinite(values)

    vsum = np.zeros((ny, nx), dtype=float)
    cnt = np.zeros((ny, nx), dtype=float)
    np.add.at(vsum, (gi[inb], gj[inb]), values[inb])
    np.add.at(cnt, (gi[inb], gj[inb]), 1.0)
    return vsum, cnt


def cell_footprint(
    cy: np.ndarray,
    cx: np.ndarray,
    grid_shape: Tuple[int, int],
    grid_step: int,
    dilate: int = 1,
) -> np.ndarray:
    """Boolean grid mask of the bins that actually contain an object.

    A value field is restricted to this footprint so the Gaussian smoothing can't
    spread a cell's value across empty space or bridge the gap between neighbouring
    cells. ``dilate`` adds a skirt of that many bins around each occupied bin (1 by
    default — one ``grid_step``) so a cell whose body spills into the next bin is
    still covered, while genuinely empty regions stay masked. This is needed only
    for the *normalised* fields (local means): unlike the density **count**, which
    fades to zero away from cells, a local mean (``sum / count``) holds the cell's
    value flat across the whole kernel, so it must be masked explicitly.
    """
    _, cnt = _bin_sum_count(cy, cx, np.ones(len(cy)), grid_shape, grid_step)
    mask = cnt > 0
    if dilate and dilate > 0 and mask.any():
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=int(dilate))
    return mask


def _binned_mean_field(
    cy: np.ndarray,
    cx: np.ndarray,
    values: np.ndarray,
    grid_shape: Tuple[int, int],
    grid_step: int,
    sigma: float,
    fill: float = np.nan,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gaussian-weighted local mean of ``values`` sampled at cell centroids.

    The density approach generalised to a value-bearing field: bin the values and
    the cell counts, smooth both with the same Gaussian, then divide. The result is
    the local (kernel-weighted) average of the recorded cells. Because that ratio
    stays flat across the whole smoothing kernel (numerator and denominator decay
    together), the field is restricted to ``mask`` — the cell footprint — so it is
    defined only where objects actually are; everywhere else (and where no cell
    lies within the kernel) is ``fill`` (``NaN`` by default). Velocity fields pass
    ``fill=0.0`` and ``mask=None`` so their divergence / curl gradients stay finite,
    and are masked by the caller afterwards.
    """
    vsum, cnt = _bin_sum_count(cy, cx, values, grid_shape, grid_step)
    ssum = gaussian_filter(vsum, sigma=sigma)
    scnt = gaussian_filter(cnt, sigma=sigma)
    out = np.full(grid_shape, float(fill), dtype=float)
    nz = scnt > 1e-9
    out[nz] = ssum[nz] / scnt[nz]
    if mask is not None:
        out[~mask] = float(fill)
    return out


def compute_spatial_fields(
    df: pd.DataFrame,
    frame: int,
    field_shape: Tuple[int, int],
    grid_step: int = 20,
    sigma: float = 2.0,
    intensity_col: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute gridded spatial fields for one frame.

    Every field is built by binning the recorded cells into the grid (the density
    method): density is the smoothed cell count per bin, and each value-bearing
    field is the Gaussian-weighted local mean of the cells (smoothed value-sum ÷
    smoothed count). Values are defined only where cells were measured — ``NaN``
    (scalar fields) or ``0`` (velocity) elsewhere — never interpolated across gaps.

    Parameters
    ----------
    df : tracked DataFrame with centroid_y, centroid_x, area, track_id, etc.
    frame : which frame to compute
    field_shape : (H, W) of the image (for grid bounds)
    grid_step : pixel spacing of the output grid
    sigma : Gaussian smoothing sigma (in grid units) for the output fields
    intensity_col : column name for intensity field (e.g. "ERK-mRuby2_mean")

    Returns
    -------
    dict with keys:
        "grid_y", "grid_x" : 2D coordinate arrays
        "density"           : cell count per grid bin (smoothed)
        "mean_area"         : local mean cell area
        "velocity_y", "velocity_x" : displacement field (if velocity available)
        "speed"             : velocity magnitude
        "divergence"        : div(v) — expansion/contraction
        "curl"              : curl(v) — local rotation
        "intensity"         : local mean intensity (if intensity_col given)
        "fold_change"       : local mean intensity / frame mean intensity
    """
    H, W = field_shape
    fdf = df[df["frame"] == frame].copy()

    if fdf.empty:
        return {}

    # Build output grid
    gy = np.arange(0, H, grid_step).astype(float)
    gx = np.arange(0, W, grid_step).astype(float)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_shape = grid_y.shape

    cy = fdf["centroid_y"].values
    cx = fdf["centroid_x"].values

    result = {"grid_y": grid_y, "grid_x": grid_x}

    # The cell footprint (occupied bins, lightly dilated) restricts every value
    # field to where objects actually are. Density is exempt — it is a count that
    # fades to zero in empty space, so it masks itself.
    footprint = cell_footprint(cy, cx, grid_shape, grid_step)

    # --- Density: smoothed count of cells per grid bin ---
    _, cnt = _bin_sum_count(cy, cx, np.ones(len(cy)), grid_shape, grid_step)
    result["density"] = gaussian_filter(cnt, sigma=sigma)

    # --- Binned scalar fields (Gaussian-weighted local mean, masked to cells) ---
    # Area
    if "area" in fdf.columns:
        result["mean_area"] = _binned_mean_field(
            cy, cx, fdf["area"].values, grid_shape, grid_step, sigma, mask=footprint)

    # Intensity
    if intensity_col and intensity_col in fdf.columns:
        vals = fdf[intensity_col].values.astype(float)
        result["intensity"] = _binned_mean_field(
            cy, cx, vals, grid_shape, grid_step, sigma, mask=footprint)
        finite = vals[np.isfinite(vals)]
        frame_mean = float(finite.mean()) if finite.size else 0.0
        if frame_mean > 0:
            result["fold_change"] = _binned_mean_field(
                cy, cx, vals / frame_mean, grid_shape, grid_step, sigma,
                mask=footprint)

    # --- Velocity field (requires consecutive-frame tracking) ---
    has_velocity = False

    # Per-track displacement: position(t) - position(t-1), binned at position(t).
    if "track_id" in fdf.columns and frame > df["frame"].min():
        prev_frame = frame - 1
        prev_df = df[df["frame"] == prev_frame]
        if not prev_df.empty:
            merged = fdf.merge(
                prev_df[["track_id", "centroid_y", "centroid_x"]],
                on="track_id", suffixes=("", "_prev"),
            )
            if len(merged) >= 1:
                vy = (merged["centroid_y"] - merged["centroid_y_prev"]).values
                vx = (merged["centroid_x"] - merged["centroid_x_prev"]).values
                mcy = merged["centroid_y"].values
                mcx = merged["centroid_x"].values

                # Filled (finite) velocity grids so the gradients below stay
                # well-defined; the footprint mask is applied to the displayed
                # fields afterwards.
                vy_grid = _binned_mean_field(
                    mcy, mcx, vy, grid_shape, grid_step, sigma, fill=0.0)
                vx_grid = _binned_mean_field(
                    mcy, mcx, vx, grid_shape, grid_step, sigma, fill=0.0)
                # Velocity has its own footprint — only cells tracked into the
                # previous frame contribute, a subset of all cells.
                vfoot = cell_footprint(mcy, mcx, grid_shape, grid_step)

                # div = dvx/dx + dvy/dy ; curl (2D) = dvx/dy - dvy/dx
                dvx_dx = np.gradient(vx_grid, grid_step, axis=1)
                dvy_dy = np.gradient(vy_grid, grid_step, axis=0)
                divergence = dvx_dx + dvy_dy
                dvx_dy = np.gradient(vx_grid, grid_step, axis=0)
                dvy_dx = np.gradient(vy_grid, grid_step, axis=1)
                curl = dvx_dy - dvy_dx

                speed = np.sqrt(vy_grid**2 + vx_grid**2)
                result["velocity_y"] = np.where(vfoot, vy_grid, np.nan)
                result["velocity_x"] = np.where(vfoot, vx_grid, np.nan)
                result["speed"] = np.where(vfoot, speed, np.nan)
                result["divergence"] = np.where(vfoot, divergence, np.nan)
                result["curl"] = np.where(vfoot, curl, np.nan)
                has_velocity = True

    return result


# Available field names for the UI
FIELD_OPTIONS = [
    ("density", "Cell Density", "Cells per grid area"),
    ("mean_area", "Mean Cell Area", "Average area of cells in each region"),
    ("intensity", "Intensity", "Mean channel intensity"),
    ("fold_change", "Fold Change (frame)", "Intensity / frame mean (1.0 = frame avg)"),
    ("self_fold", "Self Fold Change", "Each cell's intensity / its own time-average"),
    ("speed", "Speed", "Cell migration speed magnitude"),
    ("velocity_y", "Velocity Y", "Displacement in Y (vertical)"),
    ("velocity_x", "Velocity X", "Displacement in X (horizontal)"),
    ("divergence", "Divergence", "Positive = spreading, negative = converging"),
    ("curl", "Curl", "Local rotation of cell motion"),
]
