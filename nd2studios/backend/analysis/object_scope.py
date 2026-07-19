"""object_scope — enumerate objects from a mask / label volume for per-object
pipeline scoping (V1.68 Frame/Object scope toggle).

Pure numpy / scipy — **Qt-free, no DVC import**. Turns an object source (a drawn
3-D mask, a per-frame label mask, or tracked-object labels) into a list of
:class:`ObjectRegion` — each an axis-aligned crop box (conserving T/M/Z/C) plus
the object's own boolean mask within that box.

These regions drive per-object scoping (V1.68 Frame/Object lever). The **DVC**
node consumes them directly (one ALDVC run per object on its bbox crop, via
``_DVCJob(object_regions=...)`` in ``pages/pipelines_page.py``);
:class:`~nd2studios.pipeline_graph.executor.ObjectCropVolume` is the generic lazy
per-object volume wrapper for the (deferred) per-object execution of other
downstream node types.

Sources handled by :func:`iter_objects`:

- ``(Z, H, W) bool`` — a drawn 3-D object mask (e.g. ``record._mask3d_by_m[m][t]``):
  split into 3-D connected components; each component's box is Z-scoped.
- ``(T, H, W) int`` — a per-frame label mask (``AnalysisResult.label_masks`` /
  tracked-object labels): each label id becomes one object whose XY box is the
  **union over all T** (so the object keeps one stable crop across the timelapse
  and Z is not scoped — the raw stack's full Z is preserved).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ObjectRegion:
    """One object's crop box + local mask.

    - ``object_id`` — 1-based id (connected-component index or label value).
    - ``bbox`` ``(z0, z1, y0, y1, x0, x1)`` half-open. ``z1 <= z0`` means **all Z**
      (a 2-D / label-mask source that scopes only XY — full Z is preserved).
    - ``mask`` the object within its box: ``(zc, yc, xc)`` for a 3-D source or
      ``(yc, xc)`` for a 2-D/label source (``mask_out`` uses it to hard-clip).
    - ``source`` — ``"mask3d"`` | ``"label"`` (provenance, for status/reporting).
    """

    object_id: int
    bbox: Tuple[int, int, int, int, int, int]
    mask: np.ndarray
    source: str = "mask3d"

    @property
    def z_scoped(self) -> bool:
        """True when the box restricts Z (a 3-D object); False = keep full Z."""
        return self.bbox[1] > self.bbox[0]

    @property
    def rect_xywh(self) -> Tuple[int, int, int, int]:
        """XY crop as ``(x, y, w, h)`` (the convention the page's crop rects use)."""
        _z0, _z1, y0, y1, x0, x1 = self.bbox
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    def n_voxels(self) -> int:
        return int(np.asarray(self.mask).astype(bool).sum())


def _pad_bbox(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
              pad: int, shape_zyx: Tuple[int, int, int], z_scope: bool
              ) -> Tuple[int, int, int, int, int, int]:
    """Dilate a box by ``pad`` voxels, clamped to the array bounds."""
    Z, H, W = shape_zyx
    y0 = max(0, y0 - pad); y1 = min(H, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(W, x1 + pad)
    if z_scope:
        z0 = max(0, z0 - pad); z1 = min(Z, z1 + pad)
    return (z0, z1, y0, y1, x0, x1)


def _regions_from_bool_volume(mask: np.ndarray, min_voxels: int, pad: int
                              ) -> List[ObjectRegion]:
    """3-D connected components of a ``(Z, H, W)`` boolean object mask."""
    from scipy.ndimage import find_objects, label

    structure = np.ones((3, 3, 3), dtype=bool)   # 26-connectivity
    labeled, n = label(mask, structure=structure)
    Z, H, W = mask.shape
    regions: List[ObjectRegion] = []
    slices = find_objects(labeled)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        obj_id = i + 1
        sub = labeled[sl] == obj_id
        if int(sub.sum()) < int(min_voxels):
            continue
        z0, z1 = sl[0].start, sl[0].stop
        y0, y1 = sl[1].start, sl[1].stop
        x0, x1 = sl[2].start, sl[2].stop
        bbox = _pad_bbox(z0, z1, y0, y1, x0, x1, pad, (Z, H, W), z_scope=True)
        z0, z1, y0, y1, x0, x1 = bbox
        regions.append(ObjectRegion(
            object_id=obj_id, bbox=bbox,
            mask=np.ascontiguousarray(labeled[z0:z1, y0:y1, x0:x1] == obj_id),
            source="mask3d"))
    return regions


def _regions_from_label_volume(labels: np.ndarray, min_voxels: int, pad: int
                               ) -> List[ObjectRegion]:
    """Per-label objects from a ``(T, H, W)`` int label mask (XY union over T)."""
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[None, ...]
    T, H, W = arr.shape
    ids = np.unique(arr)
    ids = ids[ids > 0]
    regions: List[ObjectRegion] = []
    for obj_id in ids:
        present = np.any(arr == obj_id, axis=0)          # (H, W) union over T
        if int(present.sum()) < int(min_voxels):
            continue
        ys, xs = np.where(present)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bbox = _pad_bbox(0, 0, y0, y1, x0, x1, pad, (0, H, W), z_scope=False)
        _z0, _z1, y0, y1, x0, x1 = bbox
        regions.append(ObjectRegion(
            object_id=int(obj_id), bbox=bbox,
            mask=np.ascontiguousarray(present[y0:y1, x0:x1]),
            source="label"))
    return regions


def iter_objects(obj: np.ndarray,
                 voxel_size_um: Optional[Tuple[float, ...]] = None,
                 *, min_voxels: int = 1, pad: int = 0) -> List[ObjectRegion]:
    """Enumerate objects from a mask / label volume → ``List[ObjectRegion]``.

    Dispatch by dtype: a **boolean** array is a drawn 3-D mask (``(Z, H, W)`` →
    3-D connected components, Z-scoped boxes); an **integer** array is a per-frame
    label mask (``(T, H, W)`` → one object per label id, XY box unioned over T,
    full Z preserved). ``voxel_size_um`` is accepted for API symmetry (physical
    thresholds are the caller's concern). ``min_voxels`` drops specks; ``pad``
    dilates each box (to keep surrounding matrix context — correlation nodes like
    DVC need it). Returns ``[]`` for an empty / all-background source.
    """
    arr = np.asarray(obj)
    if arr.size == 0 or not arr.any():
        return []
    pad = max(0, int(pad))
    if arr.dtype == bool:
        if arr.ndim == 2:
            arr = arr[None, ...]
        return _regions_from_bool_volume(arr.astype(bool), min_voxels, pad)
    if np.issubdtype(arr.dtype, np.integer):
        return _regions_from_label_volume(arr, min_voxels, pad)
    # A float mask is treated as boolean (> 0).
    a = arr > 0
    if a.ndim == 2:
        a = a[None, ...]
    return _regions_from_bool_volume(a, min_voxels, pad)


def iter_objects_3d_labels(labels: np.ndarray,
                           voxel_size_um: Optional[Tuple[float, ...]] = None,
                           *, min_voxels: int = 1, pad: int = 0) -> List[ObjectRegion]:
    """Enumerate objects from a 3-D ``(Z, H, W)`` **integer label** volume (V1.70).

    Each distinct label id ``> 0`` becomes one **Z-scoped** :class:`ObjectRegion`
    (bbox via ``scipy.ndimage.find_objects``; ``mask = labels == id`` within the box).

    This is the explicit 3-D-label counterpart to :func:`iter_objects`, which reads
    an integer array as a ``(T, H, W)`` per-frame label mask (XY box unioned over T,
    full Z preserved). The Granule Volume Mask node publishes a combined
    ``(Z, H, W) int32`` label volume; feeding it to :func:`iter_objects` would
    misread Z as T, so this function treats the first axis as **Z** — matching the
    per-granule ``(Z, H, W) bool`` path. Returns ``[]`` for an empty / all-background
    volume. ``voxel_size_um`` is accepted for API symmetry.
    """
    from scipy.ndimage import find_objects

    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.size == 0 or not arr.any():
        return []
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.intp)
    pad = max(0, int(pad))
    Z, H, W = arr.shape
    regions: List[ObjectRegion] = []
    slices = find_objects(arr)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        obj_id = i + 1
        z0, z1 = sl[0].start, sl[0].stop
        y0, y1 = sl[1].start, sl[1].stop
        x0, x1 = sl[2].start, sl[2].stop
        sub = arr[sl] == obj_id
        if int(sub.sum()) < int(min_voxels):
            continue
        bbox = _pad_bbox(z0, z1, y0, y1, x0, x1, pad, (Z, H, W), z_scope=True)
        z0, z1, y0, y1, x0, x1 = bbox
        regions.append(ObjectRegion(
            object_id=obj_id, bbox=bbox,
            mask=np.ascontiguousarray(arr[z0:z1, y0:y1, x0:x1] == obj_id),
            source="granule"))
    return regions
