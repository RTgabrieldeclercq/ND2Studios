"""
Backward-compatibility shim for the pre-V1.54 stitch exporter.

The multipoint stitching **method** now lives in
:mod:`nd2studios.backend.stitch` (regime-aware: overlap registration via
m2stitch / phase-correlation, zero-overlap coordinate placement, feather
blending, optional illumination correction, pyramidal OME-TIFF output, QC).

This module keeps the small surface that the GUI preview widgets
(``tile_layout``, ``tile_preview``) and the diagnostic scripts still import —
``StitchLayout``, ``compute_tile_layout``, ``_cluster_axis``, ``_assign_index``,
``stitch_one_frame`` — and a **deprecated** ``export_stitched_tiff`` that writes
via the new OME-TIFF writer. New code should call
:func:`nd2studios.backend.stitch.run_stitch`.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.stitch.positions import (  # re-exported for callers
    StitchLayout, _assign_index, _cluster_axis, compute_tile_layout,
)

__all__ = [
    "StitchLayout", "compute_tile_layout", "_cluster_axis", "_assign_index",
    "stitch_one_frame", "export_stitched_tiff",
]


def stitch_one_frame(tile_frames: List[np.ndarray],
                     layout: StitchLayout,
                     dtype: np.dtype) -> np.ndarray:
    """Place tiles onto a single canvas by overwrite (coordinate placement).

    Retained for tests / out-of-tree callers; the live pipeline uses the
    blending compositor in :mod:`nd2studios.backend.stitch.compositor`.
    """
    canvas = np.zeros((layout.canvas_h, layout.canvas_w), dtype=dtype)
    for (y, x), frame in zip(layout.offsets, tile_frames):
        h, w = frame.shape
        y1 = min(y + h, layout.canvas_h)
        x1 = min(x + w, layout.canvas_w)
        canvas[y:y1, x:x1] = frame[:y1 - y, :x1 - x]
    return canvas


def export_stitched_tiff(
    volume,
    layout: StitchLayout,
    m_indices: List[int],
    channel_indices: List[int],
    channel_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    filepath: str = "",
    z_mode: str = "max",
    z_index: int = 0,
    rgb: bool = True,
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """**Deprecated.** Coordinate-placement export writing the new OME-TIFF.

    Prefer :func:`nd2studios.backend.stitch.run_stitch`, which adds regime
    detection, overlap registration, blending, and a QC report. This wrapper
    composites tiles at the precomputed ``layout`` offsets (overwrite) and writes
    a pyramidal OME-TIFF so old callers keep functioning.
    """
    warnings.warn(
        "export_stitched_tiff is deprecated; use nd2studios.backend.stitch."
        "run_stitch (regime-aware pipeline).",
        DeprecationWarning, stacklevel=2,
    )
    from nd2studios.backend.stitch.config import StitchConfig
    from nd2studios.backend.stitch.writer import write_ome_tiff

    n_t = int(getattr(volume, "n_timepoints", 1) or 1)
    n_z = int(getattr(volume, "n_zslices", 1) or 1)
    preserve_z = (z_mode == "none" and n_z > 1)
    n_z_out = n_z if preserve_z else 1
    dtype = np.dtype(getattr(volume, "dtype", np.uint16))
    data = np.zeros((n_t, n_z_out, len(channel_indices),
                     layout.canvas_h, layout.canvas_w), dtype=dtype)
    total = max(1, n_t * n_z_out * len(channel_indices))
    done = 0
    for t in range(n_t):
        for z_out in range(n_z_out):
            z_req = z_out if preserve_z else z_index
            z_req_mode = "none" if preserve_z else z_mode
            for c_idx, c in enumerate(channel_indices):
                tiles = []
                for m in m_indices:
                    f = volume.get_frame(c=c, m=m, t=t, z=z_req, z_mode=z_req_mode)
                    tiles.append(f if f.ndim == 2 else f.squeeze())
                data[t, z_out, c_idx] = stitch_one_frame(tiles, layout, dtype)
                done += 1
                if progress_cb is not None:
                    progress_cb(int(done / total * 95))
    ch_names_all = list(getattr(volume, "channel_names", []) or [])
    ch_names = [ch_names_all[c] if c < len(ch_names_all) else f"Ch{c}"
                for c in channel_indices]
    write_ome_tiff(data, filepath,
                   pixel_size_um or float(getattr(volume, "pixel_size_um", 1.0) or 1.0),
                   ch_names, StitchConfig())
    if progress_cb is not None:
        progress_cb(100)
