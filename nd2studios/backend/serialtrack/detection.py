"""
SerialTrack Python — Chunk 1
=============================
Split this file into three modules:
    serialtrack/config.py
    serialtrack/io.py
    serialtrack/detection.py

Dependencies:
    pip install numpy scipy scikit-image numba tifffile
"""
# ╔══════════════════════════════════════════════════════════════╗
# ║  FILE 3: serialtrack/detection.py — Particle detection      ║
# ╚══════════════════════════════════════════════════════════════╝
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path
from typing import List
import logging
from scipy import ndimage
import numba as nb
from .config import DetectionConfig, DetectionMethod
# ─────────────────────────────────────────────────────────────
#  Numba-accelerated sub-pixel localization kernels
# ─────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _subpixel_poly_2d(log_img, xs, ys):
    """3-point parabola sub-pixel refinement for 2-D peaks.

    For each peak at integer coords (xs[i], ys[i]) in *log_img*,
    fit y = a + bx + cx² along each axis and return the shift to
    the parabola vertex.

    Returns (dx, dy) arrays — shifts to add to integer coords.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    h, w = log_img.shape  # note: (x-dim, y-dim) in our convention

    for i in range(n):
        xi, yi = xs[i], ys[i]
        # x-direction
        if 0 < xi < h - 1:
            a = log_img[xi - 1, yi]
            b = log_img[xi, yi]
            c = log_img[xi + 1, yi]
            d = 2.0 * (a - 2.0 * b + c)
            dx[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y-direction
        if 0 < yi < w - 1:
            a = log_img[xi, yi - 1]
            b = log_img[xi, yi]
            c = log_img[xi, yi + 1]
            d = 2.0 * (a - 2.0 * b + c)
            dy[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
    return dx, dy


@nb.njit(cache=True)
def _subpixel_poly_3d(log_img, xs, ys, zs):
    """3-point parabola sub-pixel refinement for 3-D peaks.

    Returns (dx, dy, dz) arrays.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)
    sx, sy, sz = log_img.shape

    for i in range(n):
        xi, yi, zi = xs[i], ys[i], zs[i]
        # x
        if 0 < xi < sx - 1:
            a, b, c = log_img[xi-1,yi,zi], log_img[xi,yi,zi], log_img[xi+1,yi,zi]
            d = 2.0*(a - 2.0*b + c)
            dx[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y
        if 0 < yi < sy - 1:
            a, b, c = log_img[xi,yi-1,zi], log_img[xi,yi,zi], log_img[xi,yi+1,zi]
            d = 2.0*(a - 2.0*b + c)
            dy[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
        # z
        if 0 < zi < sz - 1:
            a, b, c = log_img[xi,yi,zi-1], log_img[xi,yi,zi], log_img[xi,yi,zi+1]
            d = 2.0*(a - 2.0*b + c)
            dz[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dz[i] = 0.0
    return dx, dy, dz


@nb.njit(parallel=True, cache=True)
def _radial_symmetry_3d(patches, half_win, dccd, abc):
    """Radial-symmetry sub-voxel localization (Liu et al. 2013).

    Parameters
    ----------
    patches : (N, wx, wy, wz)  float64 array of image patches
    half_win : (3,) int array   half window sizes
    dccd : (3,) float array     pixel spacings
    abc  : (3,) float array     anisotropy factors

    Returns
    -------
    dx, dy, dz : (N,) sub-pixel shifts
    """
    N = patches.shape[0]
    wx, wy, wz = patches.shape[1], patches.shape[2], patches.shape[3]
    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)
    dz = np.zeros(N, dtype=np.float64)
    a, b, c = abc[0], abc[1], abc[2]
    dxc, dyc, dzc = dccd[0], dccd[1], dccd[2]

    for pi in nb.prange(N):
        # --- intensity-weighted centroid ---
        sx_ = 0.0; sy_ = 0.0; sz_ = 0.0; tot = 0.0
        for ix in range(wx):
            for iy in range(wy):
                for iz in range(wz):
                    v = patches[pi, ix, iy, iz]
                    sx_ += v * (ix - (wx-1)*0.5) * dxc
                    sy_ += v * (iy - (wy-1)*0.5) * dyc
                    sz_ += v * (iz - (wz-1)*0.5) * dzc
                    tot += v
        if tot < 1e-30:
            continue
        xm = sx_ / tot;  ym = sy_ / tot;  zm = sz_ / tot

        # --- build 3×3 normal system from gradient votes ---
        A00=0.;A01=0.;A02=0.;A11=0.;A12=0.;A22=0.
        B0=0.;B1=0.;B2=0.

        for ix in range(1, wx-1):
            for iy in range(1, wy-1):
                for iz in range(1, wz-1):
                    gu = (patches[pi,ix+1,iy,iz] - patches[pi,ix-1,iy,iz])/(2*dxc)
                    gv = (patches[pi,ix,iy+1,iz] - patches[pi,ix,iy-1,iz])/(2*dyc)
                    gw = (patches[pi,ix,iy,iz+1] - patches[pi,ix,iy,iz-1])/(2*dzc)
                    gm = np.sqrt(gu*gu + gv*gv + gw*gw)
                    if gm < 1e-10:
                        continue
                    gu /= gm; gv /= gm; gw /= gm

                    xp = (ix-(wx-1)*0.5)*dxc/a - xm/a
                    yp = (iy-(wy-1)*0.5)*dyc/b - ym/b
                    zp = (iz-(wz-1)*0.5)*dzc/c - zm/c
                    dd = np.sqrt(xp*xp + yp*yp + zp*zp)
                    if dd < 1e-10:
                        continue
                    q = gm*gm / dd

                    A00 += q*(1-gu*gu); A01 += q*(-gu*gv); A02 += q*(-gu*gw)
                    A11 += q*(1-gv*gv); A12 += q*(-gv*gw); A22 += q*(1-gw*gw)
                    dot = gu*xp + gv*yp + gw*zp
                    B0 += q*(xp - gu*dot)
                    B1 += q*(yp - gv*dot)
                    B2 += q*(zp - gw*dot)

        # --- solve 3×3 symmetric system via Cramer ---
        det = (A00*(A11*A22 - A12*A12)
             - A01*(A01*A22 - A02*A12)
             + A02*(A01*A12 - A02*A11))
        if abs(det) < 1e-30:
            continue
        inv = 1.0 / det
        dx[pi] = ((A11*A22-A12*A12)*B0 + (A02*A12-A01*A22)*B1 + (A01*A12-A02*A11)*B2)*inv*a
        dy[pi] = ((A02*A12-A01*A22)*B0 + (A00*A22-A02*A02)*B1 + (A01*A02-A00*A12)*B2)*inv*b
        dz[pi] = ((A01*A12-A02*A11)*B0 + (A01*A02-A00*A12)*B1 + (A00*A11-A01*A01)*B2)*inv*c

    return dx, dy, dz


# ─────────────────────────────────────────────────────────────
#  Main detector class
# ─────────────────────────────────────────────────────────────

class ParticleDetector:
    """Detect and localise particles in 2-D or 3-D images.

    Methods
    -------
    detect(img)
        Full pipeline: threshold → filter → detect → sub-pixel.
        Returns ``(N, ndim)`` coordinate array.

    Examples
    --------
    >>> cfg = DetectionConfig(threshold=0.3, bead_radius=4)
    >>> det = ParticleDetector(cfg)
    >>> coords = det.detect(image_3d)       # shape (N, 3)
    >>> coords = det.detect(image_2d)       # shape (N, 2)
    """

    def __init__(self, config: DetectionConfig):
        self.cfg = config

    # ── public API ──────────────────────────────────────────

    def detect(
        self,
        img: np.ndarray,
        roi_slices: Optional[Tuple[slice, ...]] = None,
    ) -> np.ndarray:
        """Run full detection pipeline. Returns (N, ndim) coords."""
        ndim = img.ndim

        # ROI crop
        offset = np.zeros(ndim, dtype=np.float64)
        if roi_slices is not None:
            offset = np.array([s.start or 0 for s in roi_slices], dtype=np.float64)
            img = img[roi_slices].copy()

        # Deconvolution
        if self.cfg.psf is not None:
            from skimage.restoration import richardson_lucy
            img = richardson_lucy(
                img.astype(np.float64), self.cfg.psf,
                num_iter=self.cfg.deconv_iters, clip=False,
            )

        # Invert for dark particles
        if self.cfg.color == "black":
            img = img.max() - img

        # Normalise to [0, 1]
        img = img.astype(np.float64)
        vmax = img.max()
        if vmax > 0:
            img_n = img / vmax
        else:
            return np.empty((0, ndim))

        # Dispatch
        if self.cfg.method == DetectionMethod.TRACTRAC:
            coords = self._detect_tractrac(img_n)
        else:
            coords = self._detect_tpt(img_n, img)

        # Offset back to full-image coords & clip
        if coords.size:
            coords += offset
        return coords

    # ── TracTrac method ─────────────────────────────────────

    def _detect_tractrac(self, img_n: np.ndarray) -> np.ndarray:
        """LoG → local-maximum → sub-pixel polynomial fit."""
        ndim = img_n.ndim
        bw = self._size_filtered_mask(img_n)
        img_m = img_n * bw  # masked image

        if self.cfg.bead_radius > 0:
            return self._log_detect(img_m, img_n, ndim)
        else:
            return self._centroid_detect(img_n, ndim)

    def _log_detect(self, img_m, img_n, ndim):
        sigma = self.cfg.bead_radius

        # Laplacian of Gaussian
        log_img = -ndimage.gaussian_laplace(img_m, sigma=sigma)

        # Local-maximum filter
        fp_size = int(2 * sigma * 2) + 1
        fp = np.ones((fp_size,) * ndim)
        rng = np.random.default_rng(42)
        noise = rng.random(log_img.shape) * 1e-5
        dilated = ndimage.maximum_filter(log_img + noise, footprint=fp)
        peaks = ((log_img + noise) == dilated) & (img_n > self.cfg.threshold)

        coords_int = np.asarray(np.nonzero(peaks), dtype=np.int64).T  # (N, ndim)
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Trim border
        nb_ = max(int((sigma + 2) / 2), 1)
        mask = np.ones(len(coords_int), dtype=np.bool_)
        for d in range(ndim):
            mask &= (coords_int[:, d] >= nb_) & (coords_int[:, d] < img_n.shape[d] - nb_)
        coords_int = coords_int[mask]
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Sub-pixel via parabola on log(LoG)
        log_safe = np.log(np.clip(log_img - log_img.min() + 1e-8, 1e-12, None))

        if ndim == 2:
            dx, dy = _subpixel_poly_2d(log_safe, coords_int[:, 0], coords_int[:, 1])
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
        else:
            dx, dy, dz = _subpixel_poly_3d(
                log_safe, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
            )
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5) & (np.abs(dz) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
            out[:, 2] += dz[valid]
        return out

    # ── TPT method ──────────────────────────────────────────

    def _detect_tpt(self, img_n: np.ndarray, img_raw: np.ndarray) -> np.ndarray:
        """Blob centroid → radial-symmetry sub-voxel refinement."""
        ndim = img_n.ndim
        coords = self._centroid_detect(img_n, ndim)
        if len(coords) == 0 or ndim != 3:
            return coords  # radial symmetry only for 3-D

        # Extract patches for radial-symmetry
        ws = np.array(self.cfg.win_size[:3], dtype=np.int64)
        half = ws // 2
        ci = np.round(coords).astype(np.int64)

        # Pad with reflected noise
        img_f = img_raw.astype(np.float64)
        img_f += self.cfg.rand_noise * np.random.default_rng(0).random(img_f.shape)
        img_p = np.pad(img_f, [(h, h) for h in half], mode="reflect")

        patches = np.empty((len(ci), ws[0], ws[1], ws[2]), dtype=np.float64)
        for i, c in enumerate(ci):
            cp = c + half  # padded coords
            patches[i] = img_p[
                cp[0]-half[0]:cp[0]+half[0]+1,
                cp[1]-half[1]:cp[1]+half[1]+1,
                cp[2]-half[2]:cp[2]+half[2]+1,
            ]

        dccd = np.array(self.cfg.dccd[:3], dtype=np.float64)
        abc = np.array(self.cfg.abc[:3], dtype=np.float64)

        dx, dy, dz = _radial_symmetry_3d(patches, half, dccd, abc)

        # Apply only well-behaved shifts
        ok = (np.abs(dx) < half[0]) & (np.abs(dy) < half[1]) & (np.abs(dz) < half[2])
        out = coords.copy()
        out[ok, 0] += dx[ok]
        out[ok, 1] += dy[ok]
        out[ok, 2] += dz[ok]
        return out

    # ── shared helpers ──────────────────────────────────────

    def _size_filtered_mask(self, img_n: np.ndarray) -> np.ndarray:
        """Threshold → label → keep blobs within [min_size, max_size]."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return bw
        sizes = ndimage.sum_labels(bw, labeled, range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        for i, s in enumerate(sizes, 1):
            if self.cfg.min_size <= s <= self.cfg.max_size:
                keep[i] = True
        return keep[labeled]

    def _centroid_detect(self, img_n: np.ndarray, ndim: int) -> np.ndarray:
        """Connected-component centroids, filtered by size."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return np.empty((0, ndim))
        sizes = ndimage.sum_labels(bw, labeled, range(1, n + 1))
        centroids = ndimage.center_of_mass(img_n, labeled, range(1, n + 1))
        out = np.array([
            c for c, s in zip(centroids, sizes)
            if self.cfg.min_size <= s <= self.cfg.max_size
        ])
        return out if out.size else np.empty((0, ndim))

    @staticmethod
    def clip_to_bounds(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Remove coords outside [0, shape) for each dimension."""
        if coords.size == 0:
            return coords
        mask = np.ones(len(coords), dtype=bool)
        for d in range(coords.shape[1]):
            mask &= (coords[:, d] >= 0) & (coords[:, d] < shape[d])
        return coords[mask]