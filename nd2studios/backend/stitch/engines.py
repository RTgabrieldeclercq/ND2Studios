"""
Engine dispatch — turn a dataset + reference tiles into absolute tile positions.

Routing (spec §2 selector):

- zero-overlap regime           → ``coordinate`` (never register).
- overlap, explicit engine      → that engine (``m2stitch`` / ``ashlar`` /
                                   ``phase_correlation`` / ``coordinate``).
- overlap, ``auto``             → ``m2stitch`` for an inferable regular grid
                                   when importable, else built-in
                                   ``phase_correlation``; both fall back to
                                   coordinates if they find nothing to trust.

``ashlar`` is installed but its ``reg`` module hard-imports ``jnius``
(BioFormats) and needs a JDK/``JAVA_HOME``. It is offered as a selectable engine
that raises a clear message when Java is absent; ``auto`` never picks it.
"""
from __future__ import annotations

import importlib.util
from typing import Dict, Optional, Tuple

import numpy as np

from nd2studios.backend.stitch.config import (
    ENGINE_ASHLAR, ENGINE_AUTO, ENGINE_COORDINATE, ENGINE_M2STITCH,
    ENGINE_PHASE, REGIME_ZERO, StitchConfig,
)
from nd2studios.backend.stitch.dataset import Dataset
from nd2studios.backend.stitch.register import refine_positions


Positions = Dict[int, Tuple[float, float]]


def seed_positions(dataset: Dataset) -> Positions:
    """Coordinate placement seed (row, col) px per tile — the kept orientation."""
    if dataset.stage_present:
        px = max(dataset.pixel_size_um, 1e-6)
        return {t.index: (t.offset_row_um / px, t.offset_col_um / px)
                for t in dataset.tiles}
    # No stage data → row-major sqrt grid (matches compute_tile_layout fallback).
    n = dataset.n_tiles
    cols = max(1, int(np.ceil(np.sqrt(max(n, 1)))))
    return {t.index: (float((i // cols) * dataset.tile_h),
                      float((i % cols) * dataset.tile_w))
            for i, t in enumerate(dataset.tiles)}


def _m2stitch_available() -> bool:
    return importlib.util.find_spec("m2stitch") is not None


def _m2stitch_positions(dataset: Dataset, ref_frames: Dict[int, np.ndarray],
                        seed: Positions, config: StitchConfig
                        ) -> Tuple[Positions, Dict, Dict]:
    import m2stitch
    order = [t.index for t in dataset.tiles if t.index in ref_frames]
    if len(order) < 2:
        raise RuntimeError("m2stitch needs ≥2 tiles with reference frames")
    idx_by_m = {m: i for i, m in enumerate(order)}
    images = np.stack([np.asarray(ref_frames[m]) for m in order], axis=0)
    tile_by_m = {t.index: t for t in dataset.tiles}
    rows = [int(tile_by_m[m].row) for m in order]
    cols = [int(tile_by_m[m].col) for m in order]
    init = np.array([[seed[m][0], seed[m][1]] for m in order], dtype=np.float64)
    grid, _props = m2stitch.stitch_images(
        images, rows=rows, cols=cols,
        position_initial_guess=init,
        row_col_transpose=False,
        ncc_threshold=config.ncc_threshold,
    )
    positions: Positions = {}
    for m in order:
        i = idx_by_m[m]
        positions[m] = (float(grid.loc[i, "y_pos"]), float(grid.loc[i, "x_pos"]))
    # Tiles without a reference frame stay at their seed.
    for t in dataset.tiles:
        positions.setdefault(t.index, seed[t.index])
    info = {"engine": "m2stitch", "n_tiles": len(order)}
    return positions, {}, info


def _ashlar_positions(dataset, ref_frames, seed, config):
    from nd2studios.backend.stitch.ashlar_engine import ashlar_positions
    return ashlar_positions(dataset, ref_frames, seed, config)


def compute_positions(dataset: Dataset,
                      ref_frames: Optional[Dict[int, np.ndarray]],
                      config: StitchConfig,
                      regime: str) -> Tuple[Positions, Dict, Dict]:
    """Return ``(positions, confidences, info)`` (positions in row/col px)."""
    seed = seed_positions(dataset)

    # Zero-overlap or nothing to register on → coordinate placement only.
    if regime == REGIME_ZERO or not ref_frames or dataset.n_tiles < 2:
        return seed, {}, {"engine": "coordinate", "regime": regime}

    engine = config.engine
    if engine == ENGINE_COORDINATE:
        return seed, {}, {"engine": "coordinate"}
    if engine == ENGINE_ASHLAR:
        return _ashlar_positions(dataset, ref_frames, seed, config)
    if engine == ENGINE_M2STITCH:
        return _m2stitch_positions(dataset, ref_frames, seed, config)
    if engine == ENGINE_PHASE:
        return refine_positions(dataset, ref_frames, seed, config)

    # auto (overlap): prefer m2stitch on a clean grid, else phase correlation.
    if dataset.is_regular_grid and _m2stitch_available():
        try:
            return _m2stitch_positions(dataset, ref_frames, seed, config)
        except Exception as exc:
            fallback = refine_positions(dataset, ref_frames, seed, config)
            fallback[2]["m2stitch_error"] = str(exc)
            return fallback
    return refine_positions(dataset, ref_frames, seed, config)
