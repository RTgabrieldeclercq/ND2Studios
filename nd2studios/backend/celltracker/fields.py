"""Eulerian spatial-field computation.

Vendored from CellTracker ``backend/fields.py`` — the construction matches the
original Cell-Tracker repository. Interpolates per-cell measurements onto a
regular grid to produce spatial heatmaps of density, intensity, velocity,
divergence, curl, etc. (``compute_self_fold_change`` lives in
:mod:`nd2studios.backend.celltracker.metrics`.)

Operates on a tracked pandas DataFrame (``frame``, ``centroid_y``,
``centroid_x``, ``area``, ``track_id`` for velocity, and an intensity column for
intensity / fold-change).

Density is a Gaussian-smoothed count of cells per grid bin. Every value-bearing
field (``mean_area``, ``intensity``, ``velocity_*`` …) is built by linearly
interpolating the per-cell values across the grid with
``scipy.interpolate.griddata`` (gaps filled with the frame mean, then
Gaussian-smoothed) — the same method the original Cell-Tracker spatial page uses.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


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

    Density is a smoothed count of cells per grid bin; every value-bearing field
    is a ``griddata`` linear interpolation of the per-cell values across the grid
    (gaps filled with the frame mean, then Gaussian-smoothed).

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
        "density"           : cell count per grid cell (smoothed)
        "mean_area"         : average cell area
        "velocity_y", "velocity_x" : displacement field (if velocity available)
        "speed"             : velocity magnitude
        "divergence"        : div(v) — expansion/contraction
        "curl"              : curl(v) — local rotation
        "intensity"         : mean intensity (if intensity_col given)
        "fold_change"       : intensity / frame mean intensity
    """
    H, W = field_shape
    fdf = df[df["frame"] == frame].copy()

    if fdf.empty:
        return {}

    # Build output grid
    gy = np.arange(0, H, grid_step).astype(float)
    gx = np.arange(0, W, grid_step).astype(float)
    grid_x, grid_y = np.meshgrid(gx, gy)

    cy = fdf["centroid_y"].values
    cx = fdf["centroid_x"].values
    points = np.column_stack([cy, cx])

    result = {"grid_y": grid_y, "grid_x": grid_x}

    # --- Density: count cells in each grid bin ---
    density = np.zeros_like(grid_y)
    for y, x in zip(cy, cx):
        gi = int(y / grid_step)
        gj = int(x / grid_step)
        if 0 <= gi < density.shape[0] and 0 <= gj < density.shape[1]:
            density[gi, gj] += 1
    result["density"] = gaussian_filter(density.astype(float), sigma=sigma)

    # --- Interpolated scalar fields ---
    def _interp(values):
        if len(values) < 4:
            return np.full_like(grid_y, np.nan)
        try:
            field = griddata(points, values, (grid_y, grid_x), method="linear")
            field = np.nan_to_num(field, nan=np.nanmean(values))
            return gaussian_filter(field, sigma=sigma)
        except Exception:  # noqa: BLE001 — degenerate point set (collinear, etc.)
            return np.full_like(grid_y, np.nanmean(values))

    # Area
    if "area" in fdf.columns:
        result["mean_area"] = _interp(fdf["area"].values)

    # Intensity
    if intensity_col and intensity_col in fdf.columns:
        vals = fdf[intensity_col].values
        result["intensity"] = _interp(vals)
        frame_mean = vals.mean()
        if frame_mean > 0:
            result["fold_change"] = _interp(vals / frame_mean)

    # --- Velocity field (requires consecutive-frame tracking) ---
    has_velocity = False

    # Per-track displacement: position(t) - position(t-1).
    if "track_id" in fdf.columns and frame > df["frame"].min():
        prev_frame = frame - 1
        prev_df = df[df["frame"] == prev_frame]
        if not prev_df.empty:
            merged = fdf.merge(
                prev_df[["track_id", "centroid_y", "centroid_x"]],
                on="track_id", suffixes=("", "_prev"),
            )
            if len(merged) > 3:
                vy = (merged["centroid_y"] - merged["centroid_y_prev"]).values
                vx = (merged["centroid_x"] - merged["centroid_x_prev"]).values
                v_points = merged[["centroid_y", "centroid_x"]].values

                vy_grid = griddata(v_points, vy, (grid_y, grid_x), method="linear")
                vx_grid = griddata(v_points, vx, (grid_y, grid_x), method="linear")
                vy_grid = np.nan_to_num(vy_grid, nan=0)
                vx_grid = np.nan_to_num(vx_grid, nan=0)
                vy_grid = gaussian_filter(vy_grid, sigma=sigma)
                vx_grid = gaussian_filter(vx_grid, sigma=sigma)

                result["velocity_y"] = vy_grid
                result["velocity_x"] = vx_grid
                result["speed"] = np.sqrt(vy_grid**2 + vx_grid**2)
                has_velocity = True

    # --- Divergence and curl from velocity field ---
    if has_velocity:
        vy_g = result["velocity_y"]
        vx_g = result["velocity_x"]

        # div = dvx/dx + dvy/dy
        dvx_dx = np.gradient(vx_g, grid_step, axis=1)
        dvy_dy = np.gradient(vy_g, grid_step, axis=0)
        result["divergence"] = dvx_dx + dvy_dy

        # curl (2D) = dvx/dy - dvy/dx
        dvx_dy = np.gradient(vx_g, grid_step, axis=0)
        dvy_dx = np.gradient(vy_g, grid_step, axis=1)
        result["curl"] = dvx_dy - dvy_dx

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
