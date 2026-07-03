"""Bridge ND2Studios measurement rows ↔ CellTracker DataFrame analyses.

ND2Studios carries measurements as a ``List[Dict]`` of row-dicts
(``frame``, ``label_id``, ``centroid_y_px``, ``centroid_x_px``, ``area_px``,
``m_position``, ``segmentation_channel``, ``track_id``,
``mean_intensity_{channel}`` …); the vendored CellTracker analyses in
:mod:`nd2studios.backend.celltracker` operate on pandas DataFrames with
CellTracker's column convention. This module adapts between the two for the
"Cell-Tracker Metrics" and "Spatial Field Maps" pipeline nodes.

Pure numpy / scipy / pandas / matplotlib (Agg) — no Qt imports, so the
backend-purity rule holds. The matplotlib object-oriented API
(``Figure`` + ``FigureCanvasAgg``) is used deliberately so rendering never
touches the GUI's global ``QtAgg`` backend.
"""
from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Columns the metrics node adds to every row (initialised even when skipped so
# the results table / export see a consistent schema). ``speed`` / ``velocity_*``
# / ``cell_density`` are ND2Studios-side derivations (the vendored
# ``compute_spatial_metrics`` computes per-cell velocity internally but drops it);
# they let a downstream if-else branch on motion / crowding.
METRIC_COLUMNS = (
    "neighbor_dist_mean", "neighbor_dist_std",
    "local_divergence", "local_curl", "self_fold",
    "speed", "velocity_y", "velocity_x", "cell_density",
    "speed_um", "velocity_y_um", "velocity_x_um",
)

# Requested field name → key returned by ``compute_spatial_fields``. ``self_fold``
# is a per-cell quantity gridded as an intensity field (see ``_field_array``).
_FIELD_RESULT_KEY = {"self_fold": "intensity"}


def _cy(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_y_px") or 0.0)


def _cx(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_x_px") or 0.0)


def _area(row: Dict[str, Any]) -> float:
    return float(row.get("area_px") or 0.0)


def _track_id(row: Dict[str, Any]):
    """track_id as a float (np.nan when untracked) so pandas groupby drops it."""
    tid = row.get("track_id")
    return float(tid) if tid is not None else np.nan


def _intensity_channel(params: Dict[str, Any]) -> Optional[str]:
    ch = str((params or {}).get("intensity_channel", "") or "")
    return None if ch in ("", "None") else ch


def _clean(v: Any) -> Optional[float]:
    """NaN/inf → None (keeps the results table tidy); else a python float."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _group_rows(rows: List[Dict[str, Any]]):
    """Group rows by (segmentation_channel, m_position), preserving order."""
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r.get("segmentation_channel", "")), int(r.get("m_position", 0)))
        groups[key].append(r)
    return groups


def _build_df(group_rows: List[Dict[str, Any]], intensity_ch: Optional[str]):
    """One DataFrame per group, row-for-row aligned with ``group_rows`` (index
    0..N-1), using CellTracker's column names."""
    import pandas as pd

    records = []
    for r in group_rows:
        rec = {
            "frame": int(r.get("frame", 0)),
            "label": int(r.get("label_id", 0)),
            "track_id": _track_id(r),
            "centroid_y": _cy(r),
            "centroid_x": _cx(r),
            "area": _area(r),
        }
        if intensity_ch is not None:
            rec["intensity"] = float(r.get(f"mean_intensity_{intensity_ch}") or 0.0)
        records.append(rec)
    return pd.DataFrame(records)


# Row keys that are structural / already mapped — never passed through as extra
# numeric "value" columns when building a CellTracker-style DataFrame.
_STRUCTURAL_COLS = frozenset({
    "frame", "label_id", "track_id", "m_position",
    "centroid_y_px", "centroid_x_px", "area_px",
})


def build_tracked_df(rows: List[Dict[str, Any]], m: Optional[int] = None):
    """A full CellTracker-style tracked DataFrame from ND2Studios measurement rows.

    Unlike :func:`_build_df` (which carries a single optional ``intensity``
    column), this keeps **every** per-channel intensity as ``{channel}_mean`` and
    passes through any other numeric measurement column (``speed``, ``self_fold``,
    a Cell-Tracker Metrics column …) under its own name — so the Spatial Maps tab
    can offer them as interpolated fields. When ``m`` is given, only that
    multipoint's rows are included. Columns:
    ``frame``, ``label``, ``track_id``, ``centroid_y``, ``centroid_x``, ``area``,
    plus ``{channel}_mean`` and any extra numeric columns present on the rows.
    """
    import pandas as pd

    records = []
    for r in rows:
        if m is not None and int(r.get("m_position", 0)) != int(m):
            continue
        rec: Dict[str, Any] = {
            "frame": int(r.get("frame", 0)),
            "label": int(r.get("label_id", 0)),
            "track_id": _track_id(r),
            "centroid_y": _cy(r),
            "centroid_x": _cx(r),
            "area": _area(r),
        }
        for k, v in r.items():
            if k.startswith("mean_intensity_"):
                rec[f"{k[len('mean_intensity_'):]}_mean"] = _clean(v)
            elif k not in _STRUCTURAL_COLS and isinstance(v, (int, float)) \
                    and not isinstance(v, bool):
                rec[k] = float(v)
        records.append(rec)
    if not records:
        # Empty (no rows for this M): keep the schema so downstream
        # ``df["frame"]`` lookups hit the empty-frame guard, not a KeyError.
        return pd.DataFrame(columns=["frame", "label", "track_id",
                                     "centroid_y", "centroid_x", "area"])
    return pd.DataFrame(records)


# ── Cell-Tracker Metrics node ────────────────────────────────────────────────

def _add_velocity_density(out, pixel_size_um=None):
    """Add per-cell ``velocity_y/x``, ``speed`` (px/frame) and ``cell_density``
    columns to a metrics DataFrame ``out`` (index-aligned with its source rows);
    when ``pixel_size_um`` is given, also the µm/frame columns ``speed_um`` /
    ``velocity_y_um`` / ``velocity_x_um``.

    Velocity is the per-track centroid displacement from the previous frame
    (``NaN`` on a track's first frame and for untracked cells, where ``track_id``
    is ``NaN`` and groupby drops it). ``cell_density`` is an inverse-area crowding
    proxy ``1 / (π · neighbor_dist_mean²)`` (cells / px²)."""
    import numpy as np

    out = out.copy()
    out["velocity_y"] = np.nan
    out["velocity_x"] = np.nan
    if "track_id" in out.columns and {"centroid_y", "centroid_x", "frame"} <= set(out.columns):
        for _tid, grp in out.groupby("track_id"):
            if len(grp) < 2:
                continue
            g = grp.sort_values("frame")
            out.loc[g.index, "velocity_y"] = g["centroid_y"].diff()
            out.loc[g.index, "velocity_x"] = g["centroid_x"].diff()
    out["speed"] = np.sqrt(out["velocity_y"] ** 2 + out["velocity_x"] ** 2)

    # µm/frame columns — let an if-else threshold motion directly in µm (e.g.
    # "frame-frame vector > 20 µm"). Skipped (left NaN) when no pixel size.
    px = None
    try:
        px = float(pixel_size_um) if pixel_size_um else None
    except (TypeError, ValueError):
        px = None
    if px and px > 0:
        out["velocity_y_um"] = out["velocity_y"] * px
        out["velocity_x_um"] = out["velocity_x"] * px
        out["speed_um"] = out["speed"] * px
    else:
        out["velocity_y_um"] = np.nan
        out["velocity_x_um"] = np.nan
        out["speed_um"] = np.nan

    nd = out["neighbor_dist_mean"] if "neighbor_dist_mean" in out.columns else None
    if nd is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = 1.0 / (np.pi * nd.to_numpy(dtype=float) ** 2)
        out["cell_density"] = np.where(nd.to_numpy(dtype=float) > 0, dens, np.nan)
    else:
        out["cell_density"] = np.nan
    return out


def augment_rows_with_metrics(
    rows: List[Dict[str, Any]],
    params: Dict[str, Any],
    pixel_size_um: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Add per-cell spatial metrics + self-fold-change columns to ``rows``.

    Mutates ``rows`` in place (and returns it). Adds
    ``neighbor_dist_mean/std``, ``local_divergence``, ``local_curl`` (from
    :func:`compute_spatial_metrics`), the per-cell motion / crowding columns
    ``speed``, ``velocity_y``, ``velocity_x``, ``cell_density`` (plus the µm/frame
    ``speed_um`` / ``velocity_*_um`` when ``pixel_size_um`` is given) from
    :func:`_add_velocity_density`, and — when an intensity channel is chosen —
    ``self_fold`` (from :func:`compute_self_fold_change`). Neighbor distances and
    ``cell_density`` are computed regardless of tracking; the velocity-derived
    metrics (``speed`` / ``velocity_*`` / divergence / curl) and self-fold are
    only meaningful once a Track Objects node has assigned ``track_id``.
    """
    from nd2studios.backend.celltracker.metrics import (
        compute_spatial_metrics, compute_self_fold_change,
    )

    params = params or {}
    n_neighbors = max(1, int(params.get("n_neighbors", 6)))
    intensity_ch = _intensity_channel(params)

    # Initialise the columns so they always exist in the schema.
    for r in rows:
        for col in METRIC_COLUMNS:
            r.setdefault(col, None)
    if not rows:
        return rows

    for group_rows in _group_rows(rows).values():
        if len(group_rows) < 2:
            continue
        df = _build_df(group_rows, intensity_ch)
        out = compute_spatial_metrics(df, n_neighbors=n_neighbors)
        if intensity_ch is not None and "intensity" in out.columns:
            out = compute_self_fold_change(out, "intensity", out_col="self_fold")
        out = _add_velocity_density(out, pixel_size_um=pixel_size_um)

        for col in METRIC_COLUMNS:
            if col not in out.columns:
                continue
            by_index = out[col].to_dict()  # original 0..N-1 index → value
            for i, row in enumerate(group_rows):
                row[col] = _clean(by_index.get(i))

    return rows


# ── Spatial Field Maps node ──────────────────────────────────────────────────

def compute_field_for_frame(
    rows: List[Dict[str, Any]],
    m: int,
    frame: int,
    field_shape: Tuple[int, int],
    params: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Compute the requested Eulerian field for one ``(m, frame)``.

    Returns ``(field_2d, velocity)`` where ``velocity`` is
    ``{"velocity_y", "velocity_x"}`` when available (for an optional quiver
    overlay), else ``None``. ``field_2d`` is ``None`` when the field can't be
    produced (e.g. velocity-derived field without tracking on that frame).
    """
    from nd2studios.backend.celltracker.fields import compute_spatial_fields
    from nd2studios.backend.celltracker.metrics import compute_self_fold_change

    params = params or {}
    field = str(params.get("field", "density"))
    grid_step = max(2, int(params.get("grid_step", 20)))
    sigma = float(params.get("sigma", 2.0))
    intensity_ch = _intensity_channel(params)

    m_rows = [r for r in rows if int(r.get("m_position", 0)) == m]
    if not m_rows:
        return None, None
    df = _build_df(m_rows, intensity_ch)

    intensity_col: Optional[str] = None
    if field in ("intensity", "fold_change"):
        intensity_col = "intensity" if intensity_ch is not None else None
    elif field == "self_fold":
        if intensity_ch is None:
            return None, None
        df = compute_self_fold_change(df, "intensity", out_col="self_fold")
        intensity_col = "self_fold"

    flds = compute_spatial_fields(
        df, frame=int(frame), field_shape=field_shape,
        grid_step=grid_step, sigma=sigma, intensity_col=intensity_col,
    )
    if not flds:
        return None, None

    arr = flds.get(_FIELD_RESULT_KEY.get(field, field))
    velocity = None
    if "velocity_y" in flds and "velocity_x" in flds:
        velocity = {"velocity_y": flds["velocity_y"], "velocity_x": flds["velocity_x"]}
    return arr, velocity


def draw_field_on_ax(
    ax,
    arr: np.ndarray,
    field_shape: Tuple[int, int],
    title: str,
    params: Dict[str, Any],
    velocity: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Draw a field heatmap (+ colorbar, and a quiver overlay for ``speed``) onto
    a matplotlib ``Axes``. Backend-agnostic — used by both the Agg export figure
    and the Qt preview canvas."""
    params = params or {}
    cmap = str(params.get("colormap", "viridis"))
    field = str(params.get("field", "density"))
    H, W = field_shape

    im = ax.imshow(arr, cmap=cmap, origin="upper", extent=[0, W, H, 0],
                   aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if field == "speed" and velocity is not None:
        vy, vx = velocity["velocity_y"], velocity["velocity_x"]
        gh, gw = vy.shape
        ys = np.linspace(0, H, gh)
        xs = np.linspace(0, W, gw)
        gx, gy = np.meshgrid(xs, ys)
        ax.quiver(gx, gy, vx, vy, color="white", alpha=0.6,
                  angles="xy", scale_units="xy")

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")


def render_field_figure(
    arr: np.ndarray,
    field_shape: Tuple[int, int],
    title: str,
    params: Dict[str, Any],
    velocity: Optional[Dict[str, np.ndarray]] = None,
):
    """Render a field array into an Agg ``Figure`` (heatmap + colorbar). Returns
    the matplotlib ``Figure``."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.0, 5.0), dpi=150)
    ax = fig.add_subplot(111)
    draw_field_on_ax(ax, arr, field_shape, title, params, velocity)
    fig.tight_layout()
    return fig


def export_field_maps(
    rows: List[Dict[str, Any]],
    shapes_by_m: Dict[int, Tuple[int, int]],
    params: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    """Render the chosen field for every ``(m, frame)`` and write image files.

    ``shapes_by_m`` maps a multipoint index to its ``(H, W)`` frame size. PNG
    output is a rendered heatmap (colormap + colorbar); TIFF output is the raw
    float field array. Returns the list of written file paths.
    """
    params = params or {}
    field = str(params.get("field", "density"))
    fmt = str(params.get("image_format", "PNG")).upper()
    written: List[str] = []

    frames_by_m: Dict[int, set] = defaultdict(set)
    for r in rows:
        frames_by_m[int(r.get("m_position", 0))].add(int(r.get("frame", 0)))

    for m in sorted(shapes_by_m.keys()):
        field_shape = shapes_by_m[m]
        for frame in sorted(frames_by_m.get(m, [])):
            arr, velocity = compute_field_for_frame(rows, m, frame, field_shape, params)
            if arr is None:
                continue
            base = os.path.join(out_dir, f"{field}_M{m + 1:02d}_T{frame:04d}")
            if fmt == "TIFF":
                import tifffile
                path = base + ".tif"
                tifffile.imwrite(path, np.asarray(arr, dtype=np.float32))
            else:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                title = f"{field}  M{m + 1} T{frame}"
                fig = render_field_figure(arr, field_shape, title, params, velocity)
                FigureCanvasAgg(fig)
                path = base + ".png"
                fig.savefig(path, dpi=150, facecolor="white")
            written.append(path)

    return written
