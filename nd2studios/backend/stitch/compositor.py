"""
Compositor — paste tiles onto a canvas at solved positions with blending.

Blend modes (spec §3 step 4):

- ``feather`` — distance-from-border alpha ramp, normalized in overlap zones.
  The standard, robust choice; hides seams under uneven illumination.
- ``average`` — mean of all tiles covering a pixel.
- ``max`` — brightest tile wins (useful for sparse fluorescence).
- ``none`` — last-drawn tile overwrites (matches the pre-V1.54 behavior).

All accumulation is done in float; the result is cast back to the source dtype
so ``uint16`` in → ``uint16`` out (spec §11.11). Gaps between tiles keep the
canvas fill value.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.stitch.config import (
    BLEND_AVERAGE, BLEND_FEATHER, BLEND_MAX, BLEND_NONE,
)


def _paste_ranges(y, x, h, w, canvas_h, canvas_w):
    """Clamp a tile placed at (y,x) to the canvas, handling negative offsets.

    Returns ((dy0, dy1, dx0, dx1), (sy0, sx0)) — destination window and the
    source-tile top-left crop — or None when the tile lies fully off-canvas.
    Without this, a negative iy/ix would index numpy from the wrong end and
    place the tile at the opposite edge.
    """
    iy, ix = int(round(y)), int(round(x))
    dy0, dx0 = max(0, iy), max(0, ix)
    dy1, dx1 = min(canvas_h, iy + h), min(canvas_w, ix + w)
    if dy1 <= dy0 or dx1 <= dx0:
        return None
    return (dy0, dy1, dx0, dx1), (dy0 - iy, dx0 - ix)


def canvas_size(positions: Dict[int, Tuple[float, float]],
                tile_h: int, tile_w: int) -> Tuple[int, int]:
    """Bounding-box canvas (H, W) that contains every placed tile."""
    if not positions:
        return tile_h, tile_w
    ys = [p[0] for p in positions.values()]
    xs = [p[1] for p in positions.values()]
    h = int(np.ceil(max(ys) - min(ys))) + tile_h
    w = int(np.ceil(max(xs) - min(xs))) + tile_w
    return max(h, tile_h), max(w, tile_w)


def normalize_positions(positions: Dict[int, Tuple[float, float]]
                        ) -> Dict[int, Tuple[float, float]]:
    """Shift positions so the minimum row/col is 0 (top-left origin)."""
    if not positions:
        return {}
    min_y = min(p[0] for p in positions.values())
    min_x = min(p[1] for p in positions.values())
    return {m: (y - min_y, x - min_x) for m, (y, x) in positions.items()}


def _feather_weights(h: int, w: int, feather_px: Optional[int]) -> np.ndarray:
    """Alpha ramp: 0 at the tile border rising linearly to 1 by ``feather_px``.

    A full-tile bilinear taper (feather_px=None) approximates a distance
    transform cheaply and always leaves the tile center at weight 1.
    """
    if feather_px is not None and feather_px <= 0:
        return np.ones((h, w), dtype=np.float32)
    ramp_y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32) + 1.0
    ramp_x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32) + 1.0
    if feather_px is not None:
        ramp_y = np.clip(ramp_y, 0, feather_px)
        ramp_x = np.clip(ramp_x, 0, feather_px)
    wy = ramp_y / max(ramp_y.max(), 1e-6)
    wx = ramp_x / max(ramp_x.max(), 1e-6)
    weights = np.outer(wy, wx).astype(np.float32)
    # Never fully zero, else a pixel covered only by one tile's edge vanishes.
    return np.clip(weights, 1e-3, 1.0)


def composite_frame(
    tiles: List[np.ndarray],
    tile_positions: List[Tuple[float, float]],
    canvas_h: int,
    canvas_w: int,
    dtype: np.dtype,
    blend: str = BLEND_FEATHER,
    feather_width_px: Optional[int] = None,
    fill_value: int = 0,
) -> np.ndarray:
    """Composite one (H, W) frame.

    ``tile_positions`` are (row, col) top-left offsets aligned with ``tiles``
    (already normalized to a 0-origin canvas).
    """
    if blend == BLEND_NONE:
        canvas = np.full((canvas_h, canvas_w), fill_value, dtype=dtype)
        for (y, x), frame in zip(tile_positions, tiles):
            h, w = frame.shape
            r = _paste_ranges(y, x, h, w, canvas_h, canvas_w)
            if r is None:
                continue
            (dy0, dy1, dx0, dx1), (sy0, sx0) = r
            canvas[dy0:dy1, dx0:dx1] = frame[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        return canvas

    if blend == BLEND_MAX:
        acc = np.full((canvas_h, canvas_w), fill_value, dtype=np.float64)
        touched = np.zeros((canvas_h, canvas_w), dtype=bool)
        for (y, x), frame in zip(tile_positions, tiles):
            h, w = frame.shape
            r = _paste_ranges(y, x, h, w, canvas_h, canvas_w)
            if r is None:
                continue
            (dy0, dy1, dx0, dx1), (sy0, sx0) = r
            sub = frame[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)].astype(np.float64)
            region = acc[dy0:dy1, dx0:dx1]
            mask = touched[dy0:dy1, dx0:dx1]
            region[:] = np.where(mask, np.maximum(region, sub), sub)
            touched[dy0:dy1, dx0:dx1] = True
        return _finalize(acc, touched, fill_value, dtype)

    # feather / average — weighted accumulation.
    num = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    den = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    for (y, x), frame in zip(tile_positions, tiles):
        h, w = frame.shape
        r = _paste_ranges(y, x, h, w, canvas_h, canvas_w)
        if r is None:
            continue
        (dy0, dy1, dx0, dx1), (sy0, sx0) = r
        sub = frame[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)].astype(np.float64)
        if blend == BLEND_FEATHER:
            wts = _feather_weights(h, w, feather_width_px)[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        else:  # average
            wts = np.ones_like(sub)
        num[dy0:dy1, dx0:dx1] += sub * wts
        den[dy0:dy1, dx0:dx1] += wts
    touched = den > 0
    out = np.divide(num, den, out=np.zeros_like(num), where=touched)
    return _finalize(out, touched, fill_value, dtype)


def _finalize(values: np.ndarray, touched: np.ndarray,
              fill_value: int, dtype: np.dtype) -> np.ndarray:
    values = np.where(touched, values, float(fill_value))
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype)
