"""Generic interpolated spatial maps from measurement rows (V1.45).

Unlike :mod:`nd2studios.backend.celltracker.fields` (a fixed menu of Eulerian
cell-migration fields), this module is *value-agnostic*: it takes **any** numeric
measurement column carried on the ND2Studios row-dicts (``area_px``,
``mean_intensity_{channel}``, ``speed``, a custom metric, …) and linearly
interpolates that value, sampled at each object's centroid, onto a regular grid —
one spatial map per ``(m_position, frame)``.

Before mapping, an optional **temporal fill** linearly interpolates each track's
value across the frames it spans (filling ``None`` / missing samples). Temporal
fill needs ``track_id`` (from an upstream Track Objects node); un-tracked rows are
left untouched.

Pure numpy / scipy / pandas-free / matplotlib (Agg via the shared render helpers
in :mod:`nd2studios.backend.celltracker_bridge`) — no Qt imports, so the
backend-purity rule holds.
"""
from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Methods griddata accepts; "linear"/"cubic" need a triangulation (≥4 points),
# "nearest" works for any non-empty point set.
_GRIDDATA_METHODS = ("linear", "nearest", "cubic")


def _value(row: Dict[str, Any], value_col: str) -> Optional[float]:
    """The chosen column as a finite float, or None when absent / non-finite."""
    try:
        f = float(row.get(value_col))
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def temporal_fill(rows: List[Dict[str, Any]], value_col: str) -> List[Dict[str, Any]]:
    """Linearly interpolate ``value_col`` across time within each track, in place.

    Rows are grouped by ``(m_position, segmentation_channel, track_id)``; within a
    track the known ``(frame, value)`` samples drive a ``np.interp`` that fills any
    row whose value is missing / non-finite but whose frame lies inside the track's
    observed frame range (no extrapolation). Rows without a ``track_id`` — or tracks
    with fewer than two known samples — are left unchanged. Returns ``rows``.
    """
    groups: Dict[Tuple[int, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tid = r.get("track_id")
        if tid is None:
            continue
        key = (int(r.get("m_position", 0)),
               str(r.get("segmentation_channel", "")), float(tid))
        groups[key].append(r)

    for group_rows in groups.values():
        known_f: List[float] = []
        known_v: List[float] = []
        for r in group_rows:
            v = _value(r, value_col)
            if v is not None:
                known_f.append(float(r.get("frame", 0)))
                known_v.append(v)
        if len(known_f) < 2:
            continue
        order = np.argsort(known_f)
        xf = np.asarray(known_f)[order]
        xv = np.asarray(known_v)[order]
        lo, hi = xf[0], xf[-1]
        for r in group_rows:
            if _value(r, value_col) is not None:
                continue
            f = float(r.get("frame", 0))
            if lo <= f <= hi:
                r[value_col] = float(np.interp(f, xf, xv))

    return rows


def compute_interp_map(
    rows: List[Dict[str, Any]],
    m: int,
    frame: int,
    field_shape: Tuple[int, int],
    value_col: str,
    grid_step: int = 20,
    sigma: float = 2.0,
    method: str = "linear",
) -> Optional[np.ndarray]:
    """Interpolate ``value_col`` at the centroids of one ``(m, frame)`` onto a grid.

    Returns a 2-D ``(H // grid_step, W // grid_step)`` array (Gaussian-smoothed), or
    ``None`` when there are no usable samples. Falls back to ``nearest`` when the
    requested ``linear`` / ``cubic`` triangulation cannot be built (fewer than four
    points), and to a flat mean field when even that fails.
    """
    H, W = field_shape
    grid_step = max(2, int(grid_step))
    method = method if method in _GRIDDATA_METHODS else "linear"

    cy: List[float] = []
    cx: List[float] = []
    vals: List[float] = []
    for r in rows:
        if int(r.get("m_position", 0)) != int(m) or int(r.get("frame", 0)) != int(frame):
            continue
        v = _value(r, value_col)
        if v is None:
            continue
        cy.append(float(r.get("centroid_y_px") or 0.0))
        cx.append(float(r.get("centroid_x_px") or 0.0))
        vals.append(v)

    if not vals:
        return None

    gy = np.arange(0, H, grid_step).astype(float)
    gx = np.arange(0, W, grid_step).astype(float)
    grid_x, grid_y = np.meshgrid(gx, gy)
    points = np.column_stack([np.asarray(cy), np.asarray(cx)])
    values = np.asarray(vals, dtype=float)

    use = method
    if method in ("linear", "cubic") and len(values) < 4:
        use = "nearest"
    try:
        field = griddata(points, values, (grid_y, grid_x), method=use)
        field = np.nan_to_num(field, nan=float(np.nanmean(values)))
    except Exception:  # noqa: BLE001 — degenerate point set (collinear, etc.)
        field = np.full_like(grid_y, float(np.mean(values)))
    return gaussian_filter(field, sigma=float(sigma))


def export_interp_maps(
    rows: List[Dict[str, Any]],
    shapes_by_m: Dict[int, Tuple[int, int]],
    params: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    """Render an interpolated map for every ``(m, frame)`` and write image files.

    ``shapes_by_m`` maps a multipoint index to its ``(H, W)`` frame size. PNG output
    is a rendered heatmap (colormap + colorbar); TIFF output is the raw float field
    array. When ``fill_time`` is set, a per-track temporal fill runs once over a copy
    of the rows first. Returns the list of written file paths.
    """
    from nd2studios.backend.celltracker_bridge import render_field_figure

    params = params or {}
    value_col = str(params.get("value_column", "") or "")
    if not value_col:
        return []
    method = str(params.get("interp_method", "linear"))
    grid_step = max(2, int(params.get("grid_step", 20)))
    sigma = float(params.get("sigma", 2.0))
    fmt = str(params.get("image_format", "PNG")).upper()

    work = [dict(r) for r in rows] if params.get("fill_time", True) else rows
    if params.get("fill_time", True):
        temporal_fill(work, value_col)

    frames_by_m: Dict[int, set] = defaultdict(set)
    for r in work:
        frames_by_m[int(r.get("m_position", 0))].add(int(r.get("frame", 0)))

    written: List[str] = []
    for m in sorted(shapes_by_m.keys()):
        field_shape = shapes_by_m[m]
        for frame in sorted(frames_by_m.get(m, [])):
            arr = compute_interp_map(
                work, m, frame, field_shape, value_col,
                grid_step=grid_step, sigma=sigma, method=method,
            )
            if arr is None:
                continue
            base = os.path.join(out_dir, f"{value_col}_M{m + 1:02d}_T{frame:04d}")
            if fmt == "TIFF":
                import tifffile
                path = base + ".tif"
                tifffile.imwrite(path, np.asarray(arr, dtype=np.float32))
            else:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                title = f"{value_col}  M{m + 1} T{frame}"
                fig = render_field_figure(arr, field_shape, title, params, None)
                FigureCanvasAgg(fig)
                path = base + ".png"
                fig.savefig(path, dpi=150, facecolor="white")
            written.append(path)

    return written
