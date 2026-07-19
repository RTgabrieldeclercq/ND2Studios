"""3D object-mask construction from per-Z drawn shapes (pure numpy/scipy).

Backs the **3D Mask Drawing** node (``special:mask3d``). The node's editor stores
hand-drawn shapes per Z plane as ``{z_key: [shape, …]}`` where ``z_key`` is an int
Z index or the literal ``"all"`` (applies to every plane) — the same shape schema
as :mod:`nd2studios.backend.analysis.manual_mask`. This module turns that vector
description into a dense ``(Z, H, W)`` boolean **volume** mask, supporting the
node's three creation modes:

* **Manual (per-plane)** — ``build_mask_volume(..., propagate="none")``: only planes
  that carry shapes are filled.
* **Propagate across Z** — ``propagate="copy"`` extrudes the union footprint of all
  drawn planes through every Z; ``propagate="interpolate"`` morphs between the
  nearest drawn planes via a signed-distance-field blend (smooth, higher-resolution
  than the sampling) and copies the nearest drawn plane beyond the first / last
  drawn slice, so both modes fill the **whole** stack from every drawn plane.
* **Threshold seed + edit** — :func:`threshold_seed_shapes` seeds an editable
  polygon outline per Z from an intensity threshold (0 → Otsu) that the user then
  corrects; the corrected shapes flow back through :func:`build_mask_volume`.

Each shape may carry an optional ``"op"`` key (``"add"`` default, or ``"sub"`` to
erase), so hand corrections can subtract from a threshold seed.

Pure backend — **no PySide6** (backend-purity rule). Reuses the rasterization and
polygon helpers from :mod:`manual_mask` rather than duplicating them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from nd2studios.backend.analysis.manual_mask import (
    DEFAULT_EDIT_VERTEX_COUNT,
    _rasterize,
    _resample_closed_polygon,
)

PROPAGATE_NONE = "none"
PROPAGATE_COPY = "copy"
PROPAGATE_INTERPOLATE = "interpolate"


def rasterize_plane(shapes: List[Dict[str, Any]], H: int, W: int,
                    base: Optional[np.ndarray] = None) -> np.ndarray:
    """Rasterize a list of shapes into a single ``(H, W)`` boolean plane.

    ``"add"`` shapes union in; ``"sub"`` (erase) shapes are removed. Shapes are
    applied in order, so a later erase can carve out an earlier fill. ``base`` is
    an optional starting mask (e.g. the ``"all"`` base) the shapes are applied *on
    top of*, so a per-plane erase can carve the base — it is copied, not mutated.
    """
    out = (np.asarray(base, dtype=bool).copy() if base is not None
           else np.zeros((int(H), int(W)), dtype=bool))
    for shape in shapes or []:
        tmp = np.zeros((int(H), int(W)), dtype=np.int32)
        _rasterize(tmp, shape, 1, int(H), int(W))
        m = tmp > 0
        if str(shape.get("op", "add")) == "sub":
            out &= ~m
        else:
            out |= m
    return out


def _normalise_shapes_by_z(
    shapes_by_z: Dict[Any, List[Dict[str, Any]]], Z: int,
) -> tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Split a ``{z_key: [shapes]}`` dict into (``"all"`` shapes, ``{int z: shapes}``).

    Accepts str or int keys (JSON round-trips them as strings). Out-of-range and
    unparseable Z keys are dropped.
    """
    all_shapes: List[Dict[str, Any]] = []
    per_z: Dict[int, List[Dict[str, Any]]] = {}
    for k, shapes in (shapes_by_z or {}).items():
        if not shapes:
            continue
        if k == "all" or k == "ALL":
            all_shapes = list(shapes)
            continue
        try:
            zi = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= zi < int(Z):
            per_z[zi] = list(shapes)
    return all_shapes, per_z


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance field of a boolean mask (inside ≥ 0, outside < 0).

    Used to morph between two drawn outlines: linearly blending the SDFs of the
    bounding planes and thresholding at 0 yields a smooth in-between contour — the
    standard shape-interpolation trick.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, -1e6, dtype=np.float64)
    if mask.all():
        return np.full(mask.shape, 1e6, dtype=np.float64)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return inside - outside


def build_mask_volume(
    shapes_by_z: Dict[Any, List[Dict[str, Any]]],
    Z: int,
    H: int,
    W: int,
    propagate: str = PROPAGATE_INTERPOLATE,
) -> np.ndarray:
    """Rasterize per-Z shapes into a dense ``(Z, H, W)`` boolean volume.

    ``shapes_by_z`` maps a Z index (or ``"all"``) to a list of shape dicts.
    ``"all"`` shapes form a base painted on **every** plane. ``propagate`` controls
    how the planes *between* drawn slices are filled:

    - ``"none"``  — only drawn planes (plus the ``"all"`` base) are filled.
    - ``"copy"``  — the union footprint of all drawn planes is extruded through Z.
    - ``"interpolate"`` — signed-distance morph between consecutive drawn planes,
      with the ends extrapolated (the nearest drawn plane copied beyond the first /
      last drawn slice) so the object fills the whole stack, not just the gaps
      between drawn planes.

    Drawn planes always keep their exact rasterization; propagation fills every
    other plane (``"none"`` excepted) so the whole volume reflects the drawn set.
    """
    Z, H, W = int(Z), int(H), int(W)
    vol = np.zeros((Z, H, W), dtype=bool)
    if Z <= 0 or H <= 0 or W <= 0:
        return vol

    all_shapes, per_z = _normalise_shapes_by_z(shapes_by_z, Z)
    all_mask = rasterize_plane(all_shapes, H, W) if all_shapes else None
    if all_mask is not None:
        vol[:] = all_mask  # base on every plane

    drawn = sorted(per_z)
    if not drawn:
        return vol

    drawn_masks: Dict[int, np.ndarray] = {}
    for zi in drawn:
        # Apply the per-plane shapes *on top of* the base so a per-plane erase can
        # carve the 'all' base (seeding, not OR-ing after — see rasterize_plane).
        pm = rasterize_plane(per_z[zi], H, W, base=all_mask)
        drawn_masks[zi] = pm
        vol[zi] = pm  # drawn planes are exact regardless of propagate mode

    if propagate == PROPAGATE_NONE:
        return vol

    # COPY, or INTERPOLATE with a single drawn plane: extrude the union footprint of
    # the drawn plane(s) through every Z. The single-plane fallback is essential —
    # otherwise "Propagate" on one drawn slice fills nothing, because the interpolate
    # loop below needs a *pair* of consecutive drawn planes to morph between.
    if propagate == PROPAGATE_COPY or len(drawn) < 2:
        footprint = np.zeros((H, W), dtype=bool)
        for pm in drawn_masks.values():
            footprint |= pm
        if all_mask is not None:
            footprint |= all_mask
        vol[:] = footprint
        return vol

    # PROPAGATE_INTERPOLATE — SDF morph between consecutive drawn planes, then
    # extrapolate the ends. Every drawn plane's mask feeds the result, and the whole
    # stack is filled (planes beyond the first / last drawn slice copy the nearest
    # drawn plane) — "propagate across Z" fills across ALL Z, not just the gaps.
    for a, b in zip(drawn[:-1], drawn[1:]):
        if b - a <= 1:
            continue
        sdf_a = _signed_distance(drawn_masks[a])
        sdf_b = _signed_distance(drawn_masks[b])
        span = float(b - a)
        for zi in range(a + 1, b):
            alpha = (zi - a) / span
            plane = ((1.0 - alpha) * sdf_a + alpha * sdf_b) >= 0.0
            if all_mask is not None:
                plane = plane | all_mask
            vol[zi] = plane
    # Extrapolate beyond the drawn range by copying the nearest drawn plane (whose
    # mask already carries the "all" base), so a mask drawn on a middle plane still
    # reaches the top and bottom of the stack.
    first, last = drawn[0], drawn[-1]
    for zi in range(0, first):
        vol[zi] = drawn_masks[first]
    for zi in range(last + 1, Z):
        vol[zi] = drawn_masks[last]
    return vol


def threshold_seed_shapes(
    volume: np.ndarray,
    threshold: float = 0.0,
    roi: Optional[np.ndarray] = None,
    max_regions: int = 1,
    n_vertices: int = DEFAULT_EDIT_VERTEX_COUNT,
) -> Dict[int, List[Dict[str, Any]]]:
    """Seed an editable polygon outline per Z plane from an intensity threshold.

    For each Z slice of ``volume`` (``(Z, H, W)``), pixels above ``threshold``
    (``threshold <= 0`` → Otsu) form a binary mask; the ``max_regions`` largest
    connected components are traced to closed contours and resampled to
    ``n_vertices``-point polygons. Returns ``{z: [polygon_shape, …]}`` ready to
    merge into the node's ``mask_shapes`` store and hand-edit.

    ``roi`` (``(Z,H,W)`` or ``(H,W)`` bool) restricts the seed to a region of
    interest — typically the manually-drawn object footprint — so the outline
    refines the object **inside the drawn area** instead of snapping to the
    brightest blob anywhere in the plane. When given, the Otsu level is computed
    from the ROI's pixels (a local object/background split) and the binary mask is
    intersected with the ROI. Planes whose ROI is empty are skipped.
    """
    from skimage.filters import threshold_otsu

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"threshold_seed_shapes: expected (Z,H,W), got {vol.shape}")
    roi_vol = None
    if roi is not None:
        r = np.asarray(roi, dtype=bool)
        if r.ndim == 2:
            r = np.repeat(r[None, ...], vol.shape[0], axis=0)
        if r.shape == vol.shape:
            roi_vol = r
    out: Dict[int, List[Dict[str, Any]]] = {}
    for z in range(vol.shape[0]):
        plane = np.asarray(vol[z], dtype=np.float64)
        roi_plane = roi_vol[z] if roi_vol is not None else None
        if roi_plane is not None:
            if not roi_plane.any():
                continue                       # nothing drawn here → nothing to seed
            finite = plane[roi_plane & np.isfinite(plane)]
        else:
            finite = plane[np.isfinite(plane)]
        if finite.size == 0 or float(finite.max()) == float(finite.min()):
            continue
        thr = float(threshold)
        if thr <= 0.0:
            try:
                thr = float(threshold_otsu(finite))    # local level within the ROI
            except (ValueError, RuntimeError):
                thr = float(finite.mean())
        binary = plane > thr
        if roi_plane is not None:
            binary &= roi_plane                # keep the seed inside the drawn area
        shapes = _plane_to_polygons(binary, max_regions, n_vertices)
        if shapes:
            out[int(z)] = shapes
    return out


def _plane_to_polygons(binary_plane: np.ndarray, max_regions: int = 0,
                       n_vertices: int = DEFAULT_EDIT_VERTEX_COUNT
                       ) -> List[Dict[str, Any]]:
    """Trace a boolean ``(H, W)`` plane into editable polygon shapes.

    Each connected component is traced to a closed contour (marching squares) and
    resampled to ``n_vertices`` points. ``max_regions`` caps how many of the
    largest components are kept; ``max_regions <= 0`` keeps **all** of them (so a
    plane with several disjoint shapes yields one polygon per shape). Returns a
    list of ``{"type": "polygon", "vertices": [[y, x], …], "op": "add"}``.
    """
    from skimage.measure import find_contours, label as sk_label, regionprops

    binary = np.asarray(binary_plane, dtype=bool)
    if not binary.any():
        return []
    lbl = sk_label(binary)
    props = sorted(regionprops(lbl), key=lambda p: p.area, reverse=True)
    if int(max_regions) > 0:
        props = props[: int(max_regions)]
    shapes: List[Dict[str, Any]] = []
    for p in props:
        region = lbl == p.label
        # Pad so contours never run off the image edge (open paths otherwise).
        padded = np.zeros((region.shape[0] + 2, region.shape[1] + 2),
                          dtype=np.float64)
        padded[1:-1, 1:-1] = region.astype(np.float64)
        contours = find_contours(padded, 0.5)
        if not contours:
            continue
        biggest = max(contours, key=len) - 1.0  # (row, col), undo the pad
        verts = _resample_closed_polygon(biggest, int(n_vertices))
        if len(verts) >= 3:
            shapes.append({"type": "polygon", "vertices": verts, "op": "add"})
    return shapes


def volume_to_shapes(volume: np.ndarray, max_regions: int = 0,
                     n_vertices: int = DEFAULT_EDIT_VERTEX_COUNT
                     ) -> Dict[int, List[Dict[str, Any]]]:
    """Trace a boolean ``(Z, H, W)`` volume into per-plane polygon shapes.

    ``{z: [polygon_shape, …]}`` for every plane with content — used to materialize
    a propagated / interpolated mask volume back into editable shapes. By default
    (``max_regions <= 0``) **every** connected component on each plane is traced, so
    a mask made of several disjoint shapes round-trips to one polygon per shape
    (not just the largest).
    """
    vol = np.asarray(volume)
    if vol.ndim == 2:
        vol = vol[None, ...]
    out: Dict[int, List[Dict[str, Any]]] = {}
    for z in range(vol.shape[0]):
        shapes = _plane_to_polygons(vol[z].astype(bool), max_regions, n_vertices)
        if shapes:
            out[int(z)] = shapes
    return out


def mask_volume_bounds(volume: np.ndarray) -> Optional[tuple]:
    """Tight ``(z0, z1, y0, y1, x0, x1)`` bounding box of a boolean volume, or None.

    A convenience for downstream (Phase 2) consumers that only need to interpolate
    the DVC field within the object's extent.
    """
    vol = np.asarray(volume, dtype=bool)
    if not vol.any():
        return None
    zz, yy, xx = np.where(vol)
    return (int(zz.min()), int(zz.max()) + 1,
            int(yy.min()), int(yy.max()) + 1,
            int(xx.min()), int(xx.max()) + 1)
