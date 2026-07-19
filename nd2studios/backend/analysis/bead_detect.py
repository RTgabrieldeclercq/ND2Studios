"""Bead / particle detection for the V1.70 Granule Separation chain (Qt-free).

This is P1: the first node in the granule pipeline. It detects bead centroids in
one channel's raw ``(Z, H, W)`` volume and publishes them as an ``(N, 3)`` point
cloud in **``(z, y, x)`` voxel order** plus the ``List[Dict]`` DATA rows defined in
:mod:`nd2studios.backend.analysis.granule_types` (P0 §3). It writes the first
``centroid_z_px`` in the app.

Design notes
------------
* We **reuse** :class:`nd2studios.backend.serialtrack.detection.ParticleDetector`
  (LoG local-maxima with parabolic sub-voxel refine, or connected-component
  centroids with radial-symmetry sub-voxel refine) rather than reinventing blob
  detection. That detector was written for arrays whose axis order is ``(x, y, z)``
  (see the ``# (x-dim, y-dim)`` note at ``detection.py:43``): array axis 0 == x,
  axis 1 == y, axis 2 == z, and its output columns follow suit.
* Our app stores volumes as ``(Z, H, W)`` and point clouds as ``(z, y, x)``. So we
  feed the detector the volume transposed into its native ``(x, y, z)`` order and
  reverse the returned columns to ``(z, y, x)``. **This transpose/flip is the ONLY
  place the axis order changes** (P0 §5); everything downstream is ``(z, y, x)``.
* Voxel size is ``(dz, dy, dx)`` µm everywhere (``viz3d.prep.Spacing``). Anisotropy
  is handed to the detector via the ``abc`` factors (dimensionless aspect ratio, in
  the detector's ``(x, y, z)`` order) while ``dccd`` stays ``(1, 1, 1)`` so the
  sub-voxel shifts remain in **voxel** units — the returned cloud is voxel coords,
  and the µm conversion happens once, in ``make_point_rows``.

The heavy optional dependencies (``numba``/``scipy``) are gated with
``importlib.util.find_spec`` and a friendly :class:`ImportError`, matching the
cellpose / stardist precedent.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Tuple

import numpy as np

from nd2studios.backend.analysis.granule_types import make_point_rows

# ── parameter defaults (mirrored by ``param_specs_for`` when wired in P6) ─────
_DEFAULT_DETECT_MODE = "log"        # "log" | "components"
_DEFAULT_MIN_DISTANCE_PX = 5        # peak separation / NMS radius (voxels)
_DEFAULT_THRESHOLD = 0.0            # normalized [0, 1]; 0 -> auto (Otsu)
_DEFAULT_MIN_INTENSITY = 0.0        # absolute raw-intensity floor at the peak
_DEFAULT_SUBPIXEL = True
_DEFAULT_MIN_SIZE = 1               # min blob volume/area [voxels]
_DEFAULT_MAX_SIZE = 2 ** 31 - 1     # effectively unbounded


def _require_backends() -> tuple:
    """Import the SerialTrack detector, gating its optional deps.

    Returns ``(DetectionConfig, DetectionMethod, ParticleDetector)``. Raises a
    friendly :class:`ImportError` naming the missing package.
    """
    for mod in ("numpy", "scipy", "numba"):
        if importlib.util.find_spec(mod) is None:
            raise ImportError(
                f"Bead detection requires the optional dependency '{mod}'. "
                f"Install it with `pip install {mod}`."
            )
    from nd2studios.backend.serialtrack.config import (  # noqa: WPS433 (lazy)
        DetectionConfig,
        DetectionMethod,
    )
    from nd2studios.backend.serialtrack.detection import (  # noqa: WPS433 (lazy)
        ParticleDetector,
    )
    return DetectionConfig, DetectionMethod, ParticleDetector


def _otsu_threshold_norm(vol: np.ndarray) -> float:
    """Otsu threshold expressed in the detector's normalized ``[0, 1]`` space.

    The detector compares ``img / img.max()`` against ``cfg.threshold``, so we
    compute Otsu on the max-normalized histogram and return the threshold in the
    same space (numpy-only, no scikit-image import).
    """
    a = np.asarray(vol, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    vmax = float(a.max())
    if vmax <= 0.0:
        return 0.0
    an = a / vmax
    hist, edges = np.histogram(an, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) * 0.5
    wb = np.cumsum(hist)
    wf = total - wb
    cum = np.cumsum(hist * centers)
    mb = np.divide(cum, wb, out=np.zeros_like(cum), where=wb > 0)
    mf = np.divide(cum[-1] - cum, wf, out=np.zeros_like(cum), where=wf > 0)
    between = wb * wf * (mb - mf) ** 2
    idx = int(np.argmax(between))
    return float(centers[idx])


def _sample_intensity(vol_zhw: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    """Raw intensity at each (rounded, clipped) ``(z, y, x)`` voxel."""
    if coords_zyx.size == 0:
        return np.zeros((0,), dtype=float)
    z, h, w = vol_zhw.shape
    zi = np.clip(np.rint(coords_zyx[:, 0]).astype(np.int64), 0, z - 1)
    yi = np.clip(np.rint(coords_zyx[:, 1]).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(coords_zyx[:, 2]).astype(np.int64), 0, w - 1)
    return np.asarray(vol_zhw[zi, yi, xi], dtype=float)


def _suppress_close(coords_zyx: np.ndarray, intensities: np.ndarray,
                    min_distance: float) -> np.ndarray:
    """Greedy non-max suppression: keep the brightest point in each
    ``min_distance`` (voxel Euclidean) neighborhood.

    Returns the row indices (in original order) to keep.
    """
    n = coords_zyx.shape[0]
    if min_distance <= 0.0 or n <= 1:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(intensities)[::-1]
    md2 = float(min_distance) ** 2
    kept: List[int] = []
    kept_pts: List[np.ndarray] = []
    for idx in order:
        p = coords_zyx[idx]
        ok = True
        for kp in kept_pts:
            d = p - kp
            if float(d @ d) < md2:
                ok = False
                break
        if ok:
            kept.append(int(idx))
            kept_pts.append(p)
    return np.array(sorted(kept), dtype=np.int64)


def _detector_coords(detector: Any, vol_zhw: np.ndarray, is_3d: bool) -> np.ndarray:
    """Run the detector and return centroids in ``(z, y, x)`` voxel order.

    The detector's native axis order is ``(x, y, z)`` (2-D: ``(x, y)``), so we feed
    it the volume transposed into that order and reverse the output columns. THIS
    is the single point where the ``(x,y,z) -> (z,y,x)`` flip happens (P0 §5).
    """
    if is_3d:
        # (Z, H, W) -> (W, H, Z) == (x, y, z) for the detector.
        img_xyz = np.ascontiguousarray(np.transpose(vol_zhw, (2, 1, 0)))
        coords_xyz = np.asarray(detector.detect(img_xyz), dtype=float)
        if coords_xyz.size == 0:
            return np.zeros((0, 3), dtype=float)
        return coords_xyz[:, ::-1]              # (x, y, z) -> (z, y, x)
    # 2-D fallback: (H, W) -> (W, H) == (x, y); detect -> (x, y) -> (y, x); z = 0.
    img2d = vol_zhw[0] if vol_zhw.ndim == 3 else vol_zhw
    img_xy = np.ascontiguousarray(np.transpose(img2d, (1, 0)))
    coords_xy = np.asarray(detector.detect(img_xy), dtype=float)
    if coords_xy.size == 0:
        return np.zeros((0, 3), dtype=float)
    yx = coords_xy[:, ::-1]                      # (x, y) -> (y, x)
    zeros = np.zeros((yx.shape[0], 1), dtype=float)
    return np.hstack([zeros, yx])               # (0, y, x)


def detect_beads(volume_zhw, voxel_size_um, params) -> tuple[np.ndarray, list[dict]]:
    """Detect bead centroids in one raw ``(Z, H, W)`` volume.

    Parameters
    ----------
    volume_zhw : np.ndarray
        Raw single-channel volume, ``(Z, H, W)`` (float or integer). ``Z == 1``
        (or a plain ``(H, W)`` array) triggers the 2-D fallback: every
        ``centroid_z_px`` is ``0.0`` and the cloud keeps its ``(N, 3)`` shape with a
        zero z-column, so the downstream chain stays uniform (clustering degrades to
        2-D).
    voxel_size_um : tuple[float, float, float]
        Physical voxel spacing ``(dz, dy, dx)`` µm (``viz3d.prep.Spacing`` order).
    params : dict
        Plain dict read with ``.get`` defaults:

        * ``detect_mode`` (``"log"`` | ``"components"``) — LoG local-maxima vs.
          connected-component centroids.
        * ``min_distance_px`` (int) — minimum peak separation; also LoG scale.
        * ``threshold`` (float) — normalized ``[0, 1]``; ``0`` -> auto (Otsu).
        * ``min_intensity`` (float) — drop peaks below this raw intensity.
        * ``subpixel`` (bool, default ``True``) — keep sub-voxel refinement; when
          ``False`` centroids are rounded to integer voxels.
        * ``all_multipoints`` (bool) — consumed by the P6 page handler, not here.
        * ``m_position`` / ``frame`` (int) — the current ``(m, t)`` written onto the
          DATA rows (default ``0``).

    Returns
    -------
    (np.ndarray, list[dict])
        ``points_zyx`` — ``(N, 3)`` float array of centroids in ``(z, y, x)`` voxel
        order (the fast path for P2/P3), and ``rows`` — the ``List[Dict]`` DATA rows
        built by :func:`granule_types.make_point_rows` (P0 §3 schema, ``granule_id``
        left ``None`` pre-clustering).
    """
    DetectionConfig, DetectionMethod, ParticleDetector = _require_backends()

    vol = np.asarray(volume_zhw)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(
            f"detect_beads expects a (Z, H, W) or (H, W) array, got shape {vol.shape}"
        )
    is_3d = vol.shape[0] > 1

    dz = float(voxel_size_um[0])
    dy = float(voxel_size_um[1])
    dx = float(voxel_size_um[2])

    # ── read params ──────────────────────────────────────────────────────────
    detect_mode = str(params.get("detect_mode", _DEFAULT_DETECT_MODE)).lower()
    min_distance_px = float(params.get("min_distance_px", _DEFAULT_MIN_DISTANCE_PX))
    threshold = float(params.get("threshold", _DEFAULT_THRESHOLD))
    min_intensity = float(params.get("min_intensity", _DEFAULT_MIN_INTENSITY))
    subpixel = bool(params.get("subpixel", _DEFAULT_SUBPIXEL))
    min_size = int(params.get("min_size", _DEFAULT_MIN_SIZE))
    max_size = int(params.get("max_size", _DEFAULT_MAX_SIZE))
    m = int(params.get("m_position", params.get("m", 0)))
    t = int(params.get("frame", params.get("t", 0)))

    # ── map params -> DetectionConfig ─────────────────────────────────────────
    if detect_mode == "components":
        # Connected-component centroids. Sub-voxel via radial symmetry (TPT, 3-D
        # only) when requested; otherwise a pure centroid (bead_radius == 0).
        method = DetectionMethod.TPT if subpixel else DetectionMethod.TRACTRAC
        bead_radius = 0.0
    else:                                       # "log" (default)
        method = DetectionMethod.TRACTRAC
        # LoG scale: sigma ~ half the min separation so the max-filter footprint
        # (~4*sigma+1) keeps detections at least ~min_distance_px apart.
        bead_radius = max(1.0, min_distance_px / 2.0)

    thr = threshold if threshold > 0.0 else _otsu_threshold_norm(vol)

    # Anisotropy factors in the detector's (x, y, z) order; dccd stays 1 so the
    # sub-voxel shifts come out in voxel units (the cloud is voxel coords).
    vals = [v for v in (dx, dy, dz) if v > 0.0]
    aniso_min = min(vals) if vals else 1.0
    abc = (dx / aniso_min, dy / aniso_min, dz / aniso_min)

    cfg = DetectionConfig(
        method=method,
        threshold=float(thr),
        bead_radius=float(bead_radius),
        min_size=min_size,
        max_size=max_size,
        color="white",
        win_size=(5, 5, 5),
        dccd=(1.0, 1.0, 1.0),
        abc=abc,
    )
    detector = ParticleDetector(cfg)

    # ── detect + flip to (z, y, x) ────────────────────────────────────────────
    coords_zyx = _detector_coords(detector, vol, is_3d)
    if not is_3d:
        coords_zyx[:, 0] = 0.0                  # enforce centroid_z_px == 0

    # ── post-filters: min-intensity floor, then min-distance NMS ──────────────
    if coords_zyx.shape[0]:
        intens = _sample_intensity(vol, coords_zyx)
        if min_intensity > 0.0:
            keep = intens >= min_intensity
            coords_zyx = coords_zyx[keep]
            intens = intens[keep]
        if coords_zyx.shape[0] and min_distance_px > 0.0:
            keep_idx = _suppress_close(coords_zyx, intens, min_distance_px)
            coords_zyx = coords_zyx[keep_idx]

    if not subpixel and coords_zyx.shape[0]:
        coords_zyx = np.rint(coords_zyx)

    coords_zyx = np.ascontiguousarray(coords_zyx.reshape(-1, 3).astype(float))

    rows = make_point_rows(coords_zyx, m=m, t=t,
                           voxel_size_um=(dz, dy, dx), labels=None)
    return coords_zyx, rows
