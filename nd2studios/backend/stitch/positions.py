"""
Coordinate placement and the preview-compatibility layer.

This module owns the **kept** orientation math: converting stage XY (µm) plus a
pixel size into per-tile top-left pixel offsets, honoring the lab's Nikon
sign/flip convention. It is the zero-overlap placement path *and* the seed for
the overlap registrar *and* the source of the ``StitchLayout`` /
``compute_tile_layout`` the GUI preview widgets and diagnostic scripts import
(re-exported from the old ``backend.exporters.stitch_exporter`` module for
backward compatibility).

Nothing here imports PySide6 or the ``Dataset`` type — it works on raw arrays so
there are no import cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── StitchLayout (unchanged public shape — previews & scripts depend on it) ──

@dataclass
class StitchLayout:
    """Result of placing M tiles on a canvas (coordinate placement)."""
    canvas_h: int
    canvas_w: int
    # offsets[i] = (y, x) corner pixel for the i-th tile on the canvas.
    offsets: List[Tuple[int, int]] = field(default_factory=list)
    tile_h: int = 0
    tile_w: int = 0
    source: str = "physical"   # 'physical' | 'grid_fallback' | 'empty'
    n_phys_cells: int = 0
    n_visits_per_cell: int = 1
    n_phys_cols: int = 0
    n_phys_rows: int = 0
    serpentine_reason: str = ""
    fill_ratio: float = 1.0


# ── Axis clustering (kept utility; used for grid inference & scripts) ──

def _cluster_axis(values: np.ndarray,
                   max_tolerance: Optional[float] = None,
                   min_tolerance: Optional[float] = None) -> List[float]:
    """Cluster nearby 1D values into ascending centroids.

    The tolerance is derived from the data: bimodal "jitter vs step" gaps split
    at the jump (≥3× ratio); uniform gaps use half the smallest gap.
    ``max_tolerance`` is an optional sanity cap (typically half the tile size).
    ``min_tolerance`` is a floor (typically a few percent of the tile size) that
    prevents a single grid line from being over-split by sub-tile stage jitter —
    without it, a jittered constant axis (e.g. a single-row scan) fractures into
    many bogus "rows" with a near-zero step, which downstream reads as spurious
    ~99% overlap.
    """
    if len(values) == 0:
        return []
    sorted_vals = np.sort(np.asarray(values, dtype=np.float64))
    if len(sorted_vals) == 1:
        return [float(sorted_vals[0])]

    diffs = np.diff(sorted_vals)
    nonzero = diffs[diffs > 1e-6]
    if len(nonzero) == 0:
        return [float(np.mean(sorted_vals))]

    sorted_gaps = np.sort(nonzero)
    if len(sorted_gaps) == 1:
        tol = float(sorted_gaps[0]) * 0.5
    else:
        ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1e-9)
        i_max = int(np.argmax(ratios))
        if ratios[i_max] > 3.0:
            tol = (float(sorted_gaps[i_max]) + float(sorted_gaps[i_max + 1])) / 2.0
        else:
            tol = float(sorted_gaps[0]) * 0.5

    if max_tolerance is not None and max_tolerance > 0:
        tol = min(tol, float(max_tolerance))
    if min_tolerance is not None and min_tolerance > 0:
        tol = max(tol, float(min_tolerance))
    tol = max(tol, 1e-6)

    centroids: List[float] = []
    current = [float(sorted_vals[0])]
    for v in sorted_vals[1:]:
        if float(v) - current[-1] <= tol:
            current.append(float(v))
        else:
            centroids.append(float(np.mean(current)))
            current = [float(v)]
    centroids.append(float(np.mean(current)))
    return centroids


def _assign_index(value: float, centroids: List[float]) -> int:
    """Return the index of the centroid closest to ``value``."""
    best_idx = 0
    best_diff = abs(value - centroids[0])
    for i in range(1, len(centroids)):
        d = abs(value - centroids[i])
        if d < best_diff:
            best_diff = d
            best_idx = i
    return best_idx


# ── Kept orientation math ──

def oriented_offsets_um(
    stage_xy_um: List[Tuple[float, float]],
    m_indices: List[int],
    flip_x: bool = True,
    flip_y: bool = False,
    swap_xy: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (row_um, col_um) offsets (≥0, origin at 0) for each requested M.

    This is the orientation convention preserved from the pre-V1.54 stitcher.
    With the defaults (flip_x=True, flip_y=False, swap_xy=False) the mapping is::

        col_um = max_stage_x - stage_x     (x flipped)
        row_um = stage_y - min_stage_y     (y not flipped)

    matching the historical placement (Nikon stage sign convention).
    """
    xs = np.array([float(stage_xy_um[m][0]) for m in m_indices], dtype=np.float64)
    ys = np.array([float(stage_xy_um[m][1]) for m in m_indices], dtype=np.float64)
    if swap_xy:
        xs, ys = ys, xs

    def axis(vals: np.ndarray, flip: bool) -> np.ndarray:
        return (vals.max() - vals) if flip else (vals - vals.min())

    col_um = axis(xs, flip_x)   # X → columns
    row_um = axis(ys, flip_y)   # Y → rows
    return row_um, col_um


def coordinate_offsets_px(
    stage_xy_um: List[Tuple[float, float]],
    m_indices: List[int],
    pixel_size_um: float,
    flip_x: bool = True,
    flip_y: bool = False,
    swap_xy: bool = False,
) -> List[Tuple[float, float]]:
    """Return sub-pixel (row, col) top-left offsets for each requested M."""
    if pixel_size_um <= 0:
        pixel_size_um = 1.0
    row_um, col_um = oriented_offsets_um(
        stage_xy_um, m_indices, flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
    return [(float(row_um[i] / pixel_size_um), float(col_um[i] / pixel_size_um))
            for i in range(len(m_indices))]


# ── compute_tile_layout — coordinate placement (preview + zero-overlap) ──

def compute_tile_layout(stage_xy_um: List[Tuple[float, float]],
                         pixel_size_um: float,
                         tile_h: int, tile_w: int,
                         m_indices: Optional[List[int]] = None,
                         cluster_tolerance_frac: float = 0.5,
                         flip_x: bool = True,
                         flip_y: bool = False,
                         swap_xy: bool = False) -> StitchLayout:
    """Place tiles on a canvas from stage coordinates (integer offsets).

    Backward-compatible with the pre-V1.54 signature so the GUI preview widgets
    and diagnostic scripts keep working unchanged. Falls back to a row-major
    sqrt grid when there is no usable stage XY data.
    """
    if pixel_size_um <= 0:
        pixel_size_um = 1.0
    if m_indices is None:
        m_indices = list(range(len(stage_xy_um)))
    if not m_indices:
        return StitchLayout(canvas_h=tile_h, canvas_w=tile_w,
                            offsets=[(0, 0)], tile_h=tile_h, tile_w=tile_w,
                            source="empty")

    # Acquisition stopped early → truncate to the M's that have coordinates.
    if len(stage_xy_um) < len(m_indices):
        m_indices = list(m_indices[:len(stage_xy_um)])

    if len(stage_xy_um) >= len(m_indices) and len(m_indices) > 0:
        try:
            xs = np.array([stage_xy_um[m][0] for m in m_indices], dtype=np.float64)
            ys = np.array([stage_xy_um[m][1] for m in m_indices], dtype=np.float64)
            spread = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
            if spread > 0.1:
                sub = coordinate_offsets_px(
                    stage_xy_um, m_indices, pixel_size_um,
                    flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
                offsets = [(int(round(r)), int(round(c))) for (r, c) in sub]

                canvas_w = max(off[1] for off in offsets) + tile_w
                canvas_h = max(off[0] for off in offsets) + tile_h
                n_unique = len(set(offsets))
                n_phys_cols = len(set(off[1] // max(tile_w, 1) for off in offsets))
                n_phys_rows = len(set(off[0] // max(tile_h, 1) for off in offsets))
                return StitchLayout(
                    canvas_h=canvas_h, canvas_w=canvas_w, offsets=offsets,
                    tile_h=tile_h, tile_w=tile_w, source="physical",
                    n_phys_cells=n_unique,
                    n_visits_per_cell=max(1, len(m_indices) // max(n_unique, 1)),
                    n_phys_cols=n_phys_cols, n_phys_rows=n_phys_rows,
                    fill_ratio=1.0,
                )
        except Exception:
            pass

    # Grid fallback (sqrt row-major).
    n = len(m_indices)
    if n <= 0:
        return StitchLayout(canvas_h=tile_h, canvas_w=tile_w, offsets=[],
                            tile_h=tile_h, tile_w=tile_w, source="grid_fallback")
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
    offsets = [((i // cols) * tile_h, (i % cols) * tile_w) for i in range(n)]
    return StitchLayout(canvas_h=rows * tile_h, canvas_w=cols * tile_w,
                        offsets=offsets, tile_h=tile_h, tile_w=tile_w,
                        source="grid_fallback")
