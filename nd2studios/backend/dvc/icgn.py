"""
Stages 3 & 5 — local subset registration by Inverse-Compositional Gauss-Newton.

This is the numerical heart of DVC and the one piece SerialTrack has no analogue
for (SerialTrack links particle centroids; it never registers image subsets).
For each subset we refine the integer seed to a sub-voxel first-order (affine)
warp

    x'(Δx) = x0 + Δx + u + G·Δx          G[i, j] = ∂u_i/∂x_j

by IC-GN with a zero-normalized SSD (ZNSSD ≡ ZNCC) objective — robust to linear
brightness/contrast change. Following Baker–Matthews / Blaber (Ncorr):

* the steepest-descent images ``∇f · ∂W/∂p`` and the Hessian ``H = SDᵀSD`` are
  built **once per subset from the reference** and reused across iterations
  (the deformed image is never differentiated);
* the warp is updated by **inverse composition** ``W ← W ∘ ΔW⁻¹`` (a homogeneous
  ``(d+1)×(d+1)`` matrix), never additively;
* the deformed volume is **spline-prefiltered once** by the caller, so per-iter
  sampling is ``map_coordinates(order=3, prefilter=False)``.

An optional quadratic **penalty** pulls the warp toward an ADMM target
``(u→a, F→B)`` with weights ``(μ, β)`` — this is Subpb1 of ALDVC. With
``μ=β=0`` it is plain local (conventional) DVC.

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from nd2studios.backend.dvc.mesh import Grid


@dataclass
class _RefSubset:
    """Cached reference-side quantities for one subset (constant across IC-GN
    iterations and across ADMM outer iterations)."""
    c0: np.ndarray          # (ndim,) integer subset center
    dx: np.ndarray          # (ndim, n_pix) local offsets from c0
    f0: np.ndarray          # (n_pix,) zero-mean reference intensities
    f_norm: float           # ||f - mean(f)||
    sd: np.ndarray          # (n_pix, n_params) steepest-descent images
    H_img: np.ndarray       # (n_params, n_params) image Hessian SDᵀSD
    ndim: int
    n_params: int


def _subset_offsets(subset_size: int, ndim: int) -> np.ndarray:
    """``(ndim, n_pix)`` integer local offsets spanning a centered subset."""
    half = max(1, int(subset_size) // 2)
    axis = np.arange(-half, half + 1, dtype=np.float64)
    mesh = np.meshgrid(*([axis] * ndim), indexing="ij")
    return np.stack([m.ravel(order="C") for m in mesh], axis=0)


def _prepare_reference(ref: np.ndarray, c0: np.ndarray, dx: np.ndarray,
                       subset_shape: Tuple[int, ...]) -> Optional[_RefSubset]:
    """Extract the reference window at ``c0`` and precompute SD images + Hessian.

    Parameter order is ``[u_0..u_{d-1}, G_00, G_01, ..., G_{d-1,d-1}]`` (row-major
    ``G``). SD column for ``u_i`` is ``∂f/∂x_i``; for ``G_ij`` it is
    ``(∂f/∂x_i)·Δx_j``.
    """
    ndim = c0.size
    half = np.asarray(subset_shape, dtype=np.int64) // 2
    lo = c0 - half
    hi = c0 + half + 1
    if np.any(lo < 0) or np.any(hi > np.asarray(ref.shape)):
        return None                                   # window off the edge
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    f = np.asarray(ref[sl], dtype=np.float64)
    if f.std() < 1e-8:
        return None                                   # featureless subset
    grads = np.gradient(f)                            # list[ndim] (ndim>1) or array
    if ndim == 1:
        grads = [grads]
    fg = [g.ravel(order="C") for g in grads]          # ∂f/∂x_k, each (n_pix,)
    f_flat = f.ravel(order="C")
    f0 = f_flat - f_flat.mean()
    f_norm = float(np.sqrt(np.sum(f0 * f0)))
    if f_norm < 1e-8:
        return None

    n_pix = f_flat.size
    n_params = ndim + ndim * ndim
    sd = np.empty((n_pix, n_params), dtype=np.float64)
    for i in range(ndim):                             # displacement params
        sd[:, i] = fg[i]
    col = ndim
    for i in range(ndim):                             # gradient params G_ij
        for j in range(ndim):
            sd[:, col] = fg[i] * dx[j]
            col += 1
    H_img = sd.T @ sd
    return _RefSubset(c0=c0, dx=dx, f0=f0, f_norm=f_norm, sd=sd,
                      H_img=H_img, ndim=ndim, n_params=n_params)


def _warp_matrix(u: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Homogeneous ``(d+1)×(d+1)`` warp ``[[I+G, u], [0, 1]]``."""
    ndim = u.size
    W = np.eye(ndim + 1, dtype=np.float64)
    W[:ndim, :ndim] = np.eye(ndim) + G
    W[:ndim, ndim] = u
    return W


def _icgn_subset(
    ref_sub: _RefSubset,
    defm_pref: np.ndarray,
    u_init: np.ndarray,
    G_init: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    mu: float = 0.0,
    beta: float = 0.0,
    u_target: Optional[np.ndarray] = None,
    F_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Run IC-GN for one subset. Returns ``(u, G, zncc, iters)``.

    ``u`` is the center displacement (voxels, mesh axis order); ``G`` the
    displacement gradient ``∂u_i/∂x_j``. ``zncc`` in ``[-1, 1]`` (NaN on failure).
    """
    ndim = ref_sub.ndim
    dx = ref_sub.dx                                    # (ndim, n_pix)
    c0 = ref_sub.c0.astype(np.float64)
    f0, f_norm, sd = ref_sub.f0, ref_sub.f_norm, ref_sub.sd

    # Penalty Hessian (diagonal): μ on the u params, β on the G params.
    pen_diag = np.zeros(ref_sub.n_params)
    if mu > 0.0:
        pen_diag[:ndim] = mu
    if beta > 0.0:
        pen_diag[ndim:] = beta
    B_vec = None
    if beta > 0.0 and F_target is not None:
        B_vec = np.asarray(F_target, dtype=np.float64).reshape(-1)   # row-major G

    W = _warp_matrix(np.asarray(u_init, float), np.asarray(G_init, float))
    shape_arr = np.asarray(defm_pref.shape)
    zncc = np.nan
    it = 0
    for it in range(1, int(max_iter) + 1):
        u = W[:ndim, ndim]
        G = W[:ndim, :ndim] - np.eye(ndim)
        # Warp reference-subset pixels into the deformed volume and sample.
        pos = c0[:, None] + dx + u[:, None] + G @ dx        # (ndim, n_pix)
        in_lo = np.all(pos >= 0, axis=0)
        in_hi = np.all(pos <= (shape_arr[:, None] - 1), axis=0)
        valid = in_lo & in_hi
        if valid.mean() < 0.8:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)
        g = map_coordinates(defm_pref, pos, order=3, prefilter=False,
                            mode="nearest").astype(np.float64)
        g0 = g - g.mean()
        g_norm = float(np.sqrt(np.sum(g0 * g0)))
        if g_norm < 1e-8:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)
        fn = f0 / f_norm
        gn = g0 / g_norm
        zncc = float(np.dot(fn, gn))
        evec = gn - fn                                       # (n_pix,)

        rhs = f_norm * (sd.T @ evec)
        H = ref_sub.H_img
        if pen_diag.any():
            H = H + np.diag(pen_diag)
            g_pen = np.zeros(ref_sub.n_params)
            if mu > 0.0 and u_target is not None:
                g_pen[:ndim] = mu * (np.asarray(u_target, float) - u)
            if beta > 0.0 and B_vec is not None:
                g_pen[ndim:] = beta * (B_vec - G.reshape(-1))
            rhs = rhs + g_pen
        try:
            dp = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)

        du = dp[:ndim]
        dG = dp[ndim:].reshape(ndim, ndim)
        dW = _warp_matrix(du, dG)
        try:
            W = W @ np.linalg.inv(dW)
        except np.linalg.LinAlgError:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)

        # Convergence: radius-weighted parameter step (Ncorr criterion).
        half = float(np.max(np.abs(dx))) or 1.0
        step = np.sqrt(np.sum(du ** 2) + (half ** 2) * np.sum(dG ** 2))
        if step < tol:
            break

    u = W[:ndim, ndim].copy()
    G = (W[:ndim, :ndim] - np.eye(ndim)).copy()
    return u, G, zncc, it


def local_icgn(
    ref: np.ndarray,
    defm_pref: np.ndarray,
    grid: Grid,
    u0: np.ndarray,
    subset_size: int,
    *,
    tol: float = 1e-2,
    max_iter: int = 100,
    mu: float = 0.0,
    beta: float = 0.0,
    u_target: Optional[np.ndarray] = None,
    F_target: Optional[np.ndarray] = None,
    ref_cache: Optional[dict] = None,
    n_workers: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IC-GN over every subset on ``grid``.

    ``defm_pref`` must be the deformed volume **already** spline-prefiltered
    (``scipy.ndimage.spline_filter``). ``u0`` is the ``(ndim, *grid)`` seed.
    When ``mu``/``beta`` > 0, ``u_target`` / ``F_target`` (``(ndim, *grid)`` /
    ``(ndim, ndim, *grid)``) supply the per-node ADMM penalty targets (Subpb1).
    ``ref_cache`` (a dict keyed by node index) memoizes the per-subset reference
    SD/Hessian across ADMM outer iterations (serial path only).

    ``n_workers > 1`` fans the sweep across processes
    (:func:`nd2studios.backend.dvc.parallel.parallel_local_icgn`) instead of the
    cached serial loop below — the multicore win dominates the lost SD cache on
    large grids.

    Returns ``(u_grid, F_grid, zncc_grid, iters_grid)``; failed subsets are NaN.
    """
    if n_workers and int(n_workers) > 1:
        from nd2studios.backend.dvc.parallel import parallel_local_icgn
        return parallel_local_icgn(
            ref, defm_pref, grid, u0, subset_size, tol=tol, max_iter=max_iter,
            mu=mu, beta=beta, u_target=u_target, F_target=F_target,
            n_workers=int(n_workers), progress_cb=progress_cb,
            cancelled_cb=cancelled_cb, progress_lo=progress_lo,
            progress_hi=progress_hi)
    ndim = grid.ndim
    gshape = grid.grid_shape
    u_out = np.full((ndim, *gshape), np.nan)
    F_out = np.full((ndim, ndim, *gshape), np.nan)
    zncc_out = np.full(gshape, np.nan)
    iters_out = np.zeros(gshape, dtype=np.int32)
    dx = _subset_offsets(subset_size, ndim)
    subset_shape = tuple([max(1, int(subset_size) // 2) * 2 + 1] * ndim)
    if ref_cache is None:
        ref_cache = {}

    idx_list = list(np.ndindex(*gshape))
    n = len(idx_list)
    for k, idx in enumerate(idx_list):
        if cancelled_cb is not None and (k & 31) == 0 and cancelled_cb():
            raise InterruptedError("DVC cancelled")
        seed = u0[(slice(None), *idx)]
        if not np.all(np.isfinite(seed)):
            continue
        rs = ref_cache.get(idx)
        if rs is None:
            c0 = np.rint(grid.coords[idx]).astype(np.int64)
            rs = _prepare_reference(ref, c0, dx, subset_shape)
            ref_cache[idx] = rs
        if rs is None:
            continue
        ut = u_target[(slice(None), *idx)] if u_target is not None else None
        Ft = (F_target[(slice(None), slice(None), *idx)]
              if F_target is not None else None)
        u, G, zncc, it = _icgn_subset(
            rs, defm_pref, seed, np.zeros((ndim, ndim)),
            tol=tol, max_iter=max_iter, mu=mu, beta=beta,
            u_target=ut, F_target=Ft)
        u_out[(slice(None), *idx)] = u
        F_out[(slice(None), slice(None), *idx)] = G
        zncc_out[idx] = zncc
        iters_out[idx] = it
        if progress_cb is not None and (k & 31) == 0 and n > 0:
            progress_cb(int(progress_lo + (progress_hi - progress_lo) * k / n))

    return u_out, F_out, zncc_out, iters_out
