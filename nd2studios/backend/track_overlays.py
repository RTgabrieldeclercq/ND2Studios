"""Track-colored overlays and motion-vector drawing (numpy + OpenCV, Qt-free).

Renders tracking results onto an RGB frame for the Pipelines viewer overlay tabs:

* :func:`generate_track_colormap` — a deterministic HSV-spread color per track id
  (ported from CellTracker ``backend/export.py``).
* :func:`make_colored_overlay` — fill each tracked cell as a solid colored region
  (ported from CellTracker ``widgets/cell_overlay.make_colored_overlay``).
* :func:`draw_cell_vectors` — per-cell displacement arrows (with arrowheads).
* :func:`draw_grid_velocity` — a gridded velocity quiver (with arrowheads).

Arrowheads use ``cv2.arrowedLine`` (OpenCV is already a project dependency). The
input RGB array's channel order is preserved (colors are written verbatim), so an
``(R, G, B)`` tuple paints R→channel 0, etc.
"""
from __future__ import annotations

from colorsys import hsv_to_rgb
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def generate_track_colormap(track_ids: Iterable[int], seed: int = 42) -> Dict[int, Tuple[int, int, int]]:
    """Deterministic ``{track_id: (R, G, B)}`` with maximally-spread hues.

    Mirrors CellTracker's ``generate_track_colormap``: a permuted hue sweep with
    per-track random saturation / value in ``[0.7, 1.0]``.
    """
    ids = sorted({int(t) for t in track_ids})
    n = max(len(ids), 1)
    rng = np.random.RandomState(seed)
    hues = rng.permutation(n) / n
    out: Dict[int, Tuple[int, int, int]] = {}
    for i, tid in enumerate(ids):
        s = 0.7 + rng.random() * 0.3
        v = 0.7 + rng.random() * 0.3
        r, g, b = hsv_to_rgb(float(hues[i]), s, v)
        out[tid] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def make_colored_overlay(
    rgb: np.ndarray,
    label_frame: np.ndarray,
    label_to_track: Dict[int, Optional[int]],
    colormap: Dict[int, Tuple[int, int, int]],
    long_ids: set,
    alpha: float = 1.0,
) -> np.ndarray:
    """Paint each tracked cell as a solid colored region (CellTracker style).

    Parameters
    ----------
    rgb:
        ``(H, W, 3)`` uint8 base image.
    label_frame:
        ``(H, W)`` int label image for this frame.
    label_to_track:
        ``{label_id: track_id}`` for this frame.
    colormap:
        ``{track_id: (R, G, B)}``.
    long_ids:
        track ids that should be colored (others stay as the base image).
    alpha:
        1.0 = solid replace (matches CellTracker); < 1.0 blends with the base.
    """
    out = np.ascontiguousarray(rgb).astype(np.uint8)
    lf = np.asarray(label_frame)
    max_label = int(lf.max())
    if max_label <= 0:
        return out

    lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    has = np.zeros(max_label + 1, dtype=bool)
    for lid, tid in label_to_track.items():
        if tid is None or int(lid) <= 0 or int(lid) > max_label:
            continue
        tid = int(tid)
        if tid in long_ids and tid in colormap:
            lut[int(lid)] = colormap[tid]
            has[int(lid)] = True

    colored = lut[lf]                 # (H, W, 3)
    pix = has[lf]                     # (H, W) bool — pixels belonging to a colored cell
    if not pix.any():
        return out
    if alpha >= 1.0:
        out[pix] = colored[pix]
    else:
        out[pix] = ((1.0 - alpha) * out[pix] + alpha * colored[pix]).astype(np.uint8)
    return out


def draw_cell_vectors(
    rgb: np.ndarray,
    vectors: List[Tuple[float, float, float, float]],
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 1,
    tip_length: float = 0.35,
) -> np.ndarray:
    """Draw per-cell displacement arrows. ``vectors`` = ``(y0, x0, y1, x1)`` list."""
    import cv2

    out = np.ascontiguousarray(rgb).astype(np.uint8)
    col = tuple(int(c) for c in color)
    for (y0, x0, y1, x1) in vectors:
        cv2.arrowedLine(
            out, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
            col, thickness, cv2.LINE_AA, tipLength=tip_length,
        )
    return out


def draw_grid_velocity(
    rgb: np.ndarray,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1,
    tip_length: float = 0.3,
    min_mag: float = 0.5,
) -> np.ndarray:
    """Draw a gridded velocity quiver with arrowheads at each grid node."""
    import cv2

    out = np.ascontiguousarray(rgb).astype(np.uint8)
    col = tuple(int(c) for c in color)
    gh, gw = vy.shape
    for i in range(gh):
        for j in range(gw):
            dy = float(vy[i, j]) * scale
            dx = float(vx[i, j]) * scale
            if (abs(dy) + abs(dx)) < min_mag:
                continue
            y0 = float(grid_y[i, j])
            x0 = float(grid_x[i, j])
            cv2.arrowedLine(
                out, (int(round(x0)), int(round(y0))),
                (int(round(x0 + dx)), int(round(y0 + dy))),
                col, thickness, cv2.LINE_AA, tipLength=tip_length,
            )
    return out
