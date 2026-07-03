"""Frame-source helpers shared by the analysis pipelines (V1.46).

A "frame source" is a per-channel ``(T, H, W)`` object that may be a fully
resident numpy array (eager) or a lazy reader (``LazyND2Channel`` / zarr / memmap
/ ``_ChannelView``). Both support ``.shape`` and ``[t]`` one-frame indexing, so
pipelines can iterate frames without materializing the whole stack. These helpers
normalize the two shape conventions (2-D single frame vs 3-D stack) without
forcing a full read.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def source_shape(source) -> Tuple[int, int, int]:
    """Return ``(T, H, W)`` for a frame source without materializing it.

    A 2-D source is treated as a single ``T=1`` frame.
    """
    shp = getattr(source, "shape", None)
    if shp is None:
        shp = np.asarray(source).shape
    if len(shp) == 2:
        return 1, int(shp[0]), int(shp[1])
    return int(shp[0]), int(shp[-2]), int(shp[-1])


def read_plane(source, t: int) -> np.ndarray:
    """Read frame ``t`` as a 2-D ``(H, W)`` ndarray (one frame only).

    Works for both 3-D stacks (``source[t]``) and a 2-D single-frame source.
    """
    shp = getattr(source, "shape", None)
    if shp is not None and len(shp) == 2:
        return np.asarray(source)
    return np.asarray(source[t])


def filter_and_relabel(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Drop objects outside ``[min_area, max_area]`` (px) and relabel ``1..K``.

    Single ``O(pixels)`` pass: object areas come from ``np.bincount`` over the
    label image, a lookup table maps each surviving label to a fresh contiguous
    id, and ``lut[mask]`` applies the filter + relabel at once. This replaces the
    old per-object ``mask[mask == label] = 0`` / ``out[mask == id] = new`` loops,
    which scanned the whole frame **once per object** — ``O(n_objects × pixels)``,
    tens of seconds on a 4096² frame with thousands of nuclei. Returns an
    ``int32`` array of the same shape.
    """
    mask = np.asarray(mask)
    if mask.size == 0 or int(mask.max()) == 0:
        return mask.astype(np.int32, copy=False)
    counts = np.bincount(mask.ravel())
    lut = np.zeros(counts.shape[0], dtype=np.int32)
    new_id = 1
    for lbl in range(1, counts.shape[0]):
        c = int(counts[lbl])
        if c and min_area <= c <= max_area:
            lut[lbl] = new_id
            new_id += 1
    return lut[mask]
