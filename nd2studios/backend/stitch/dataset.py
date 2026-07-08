"""
``Tile`` / ``Dataset`` — the normalization layer (spec §4).

``build_dataset`` turns a set of multipoint indices + the **kept** stage-XY
metadata into a geometry description the rest of the pipeline routes on: per-tile
grid (row, col), oriented pixel offsets, inferred grid shape, and inferred
overlap fraction. It carries no image data and no Qt — just numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.stitch.config import StitchConfig
from nd2studios.backend.stitch.positions import (
    _assign_index, _cluster_axis, oriented_offsets_um,
)


@dataclass
class Tile:
    """One multipoint field of view."""
    index: int                              # global M index
    row: Optional[int] = None               # grid row (canvas order, 0 = top)
    col: Optional[int] = None               # grid col (canvas order, 0 = left)
    stage_um: Optional[Tuple[float, float]] = None   # raw (x, y) stage µm
    offset_row_um: float = 0.0              # oriented row offset (µm, ≥0)
    offset_col_um: float = 0.0              # oriented col offset (µm, ≥0)


@dataclass
class Dataset:
    """Geometry of a multipoint set — no pixels, no Qt."""
    tiles: List[Tile]
    m_indices: List[int]
    pixel_size_um: float
    tile_h: int
    tile_w: int
    grid_shape: Optional[Tuple[int, int]] = None     # (rows, cols)
    overlap_frac: float = 0.0                         # max over axes (regime gate)
    overlap_frac_x: Optional[float] = None
    overlap_frac_y: Optional[float] = None
    stage_present: bool = False
    is_regular_grid: bool = False
    # Orientation used (echoed from config for QC / reproducibility).
    axis_flip_x: bool = True
    axis_flip_y: bool = False
    swap_xy: bool = False

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)


def _median_step(centroids: List[float]) -> Optional[float]:
    """Median positive gap between sorted cluster centroids (None if <2)."""
    if len(centroids) < 2:
        return None
    gaps = np.diff(np.sort(np.asarray(centroids, dtype=np.float64)))
    gaps = gaps[gaps > 1e-9]
    return float(np.median(gaps)) if len(gaps) else None


def build_dataset(volume,
                  stage_xy_um: List[Tuple[float, float]],
                  m_indices: List[int],
                  config: StitchConfig) -> Dataset:
    """Assemble a :class:`Dataset` from a volume + stage metadata.

    ``volume`` only needs ``height`` / ``width`` / ``pixel_size_um`` attributes.
    """
    tile_h = int(getattr(volume, "height", 0) or 0)
    tile_w = int(getattr(volume, "width", 0) or 0)
    px = config.pixel_size_um or float(getattr(volume, "pixel_size_um", 1.0) or 1.0)
    if px <= 0:
        px = 1.0

    stage_xy_um = list(stage_xy_um or [])
    m_indices = list(m_indices)
    # Truncate to M's that actually have coordinates (early-stopped acquisitions).
    if stage_xy_um and len(stage_xy_um) < len(m_indices):
        m_indices = m_indices[: len(stage_xy_um)]

    have_stage = False
    if stage_xy_um and len(stage_xy_um) >= len(m_indices) and len(m_indices) > 0:
        xs = np.array([stage_xy_um[m][0] for m in m_indices], dtype=np.float64)
        ys = np.array([stage_xy_um[m][1] for m in m_indices], dtype=np.float64)
        spread = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        have_stage = spread > 0.1

    if not have_stage:
        # No usable coordinates → tiles carry no geometry; caller falls back to a
        # grid layout. Regime is forced to zero-overlap (nothing to register on).
        tiles = [Tile(index=m) for m in m_indices]
        return Dataset(tiles=tiles, m_indices=m_indices, pixel_size_um=px,
                       tile_h=tile_h, tile_w=tile_w, stage_present=False,
                       axis_flip_x=config.axis_flip_x, axis_flip_y=config.axis_flip_y,
                       swap_xy=config.swap_xy)

    # Oriented offsets (µm) using the kept convention (config flips).
    row_um, col_um = oriented_offsets_um(
        stage_xy_um, m_indices,
        flip_x=config.axis_flip_x, flip_y=config.axis_flip_y, swap_xy=config.swap_xy)

    # Cluster each axis into grid lines. Tolerance is capped at half a tile
    # extent (so a real sub-tile overlap step is never merged) and floored at 5%
    # of a tile (so sub-tile stage jitter on a constant axis collapses to one
    # grid line instead of fracturing into many bogus rows/cols → spurious ~99%
    # overlap). The 5% floor also caps believable overlap at ~95%: a microscope
    # never steps by <5% of the FOV, so a smaller "step" is jitter/duplicates.
    tile_h_um = tile_h * px
    tile_w_um = tile_w * px
    row_centroids = _cluster_axis(row_um, max_tolerance=0.5 * tile_h_um,
                                  min_tolerance=0.05 * tile_h_um)
    col_centroids = _cluster_axis(col_um, max_tolerance=0.5 * tile_w_um,
                                  min_tolerance=0.05 * tile_w_um)

    tiles: List[Tile] = []
    for i, m in enumerate(m_indices):
        r = _assign_index(float(row_um[i]), row_centroids) if row_centroids else 0
        c = _assign_index(float(col_um[i]), col_centroids) if col_centroids else 0
        tiles.append(Tile(index=m, row=r, col=c,
                          stage_um=(float(stage_xy_um[m][0]), float(stage_xy_um[m][1])),
                          offset_row_um=float(row_um[i]), offset_col_um=float(col_um[i])))

    n_rows = max(1, len(row_centroids))
    n_cols = max(1, len(col_centroids))
    grid_shape = (n_rows, n_cols)

    # Overlap inference: 1 − step/tile_extent, per axis where a step exists.
    step_row = _median_step(row_centroids)
    step_col = _median_step(col_centroids)
    ov_y = (1.0 - step_row / tile_h_um) if (step_row and tile_h_um > 0) else None
    ov_x = (1.0 - step_col / tile_w_um) if (step_col and tile_w_um > 0) else None

    if config.overlap_frac is not None:
        overlap = float(config.overlap_frac)
    else:
        avail = [v for v in (ov_x, ov_y) if v is not None]
        overlap = max(avail) if avail else 0.0
    # Cap at 0.95: the 5% clustering floor guarantees a real step ≥ 0.05·tile,
    # so a higher "overlap" would only come from an explicit override.
    overlap = float(np.clip(overlap, -1.0, 0.95))

    # Regular grid = every (row, col) cell occupied at most once and the tile
    # count doesn't exceed the grid (revisits/duplicates → not a clean grid).
    cells = {(t.row, t.col) for t in tiles}
    is_regular = (len(cells) == len(tiles)) and (len(tiles) <= n_rows * n_cols) \
        and n_rows >= 1 and n_cols >= 1

    return Dataset(
        tiles=tiles, m_indices=m_indices, pixel_size_um=px,
        tile_h=tile_h, tile_w=tile_w, grid_shape=grid_shape,
        overlap_frac=overlap, overlap_frac_x=ov_x, overlap_frac_y=ov_y,
        stage_present=True, is_regular_grid=is_regular,
        axis_flip_x=config.axis_flip_x, axis_flip_y=config.axis_flip_y,
        swap_xy=config.swap_xy,
    )
