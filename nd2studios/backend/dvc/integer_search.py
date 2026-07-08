"""
Stage 1 — integer displacement seed via windowed FFT cross-correlation.

Per-subset normalized cross-correlation (the FIDVC/DIC "bigxcorr" seed): for
each subset center we extract the reference subset window and correlate it,
via FFT, against a slightly larger search window in the deformed volume,
locating the best integer offset and refining it to sub-voxel by a separable
parabolic peak fit. The peak correlation value doubles as the ``cc`` /
q-factor confidence that :mod:`nd2studios.backend.dvc.outliers` thresholds on.

This seed is deliberately coarse — it only has to land the subsequent IC-GN
(:mod:`nd2studios.backend.dvc.icgn`) inside its convergence basin. Works for 2D
images and 3D volumes; axis order follows :mod:`nd2studios.backend.dvc.mesh`
(``(y, x)`` / ``(z, y, x)``, slowest first).

Pure numpy/scipy/scikit-image — no PySide6.
"""
from __future__ import annotations

import importlib.util
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.registration import phase_cross_correlation

from nd2studios.backend.dvc.mesh import Grid, build_grid


def _get_backends(use_gpu: bool):
    """Return ``(xp, signal_module, on_device)``.

    ``(cupy, cupyx.scipy.signal, True)`` when ``use_gpu`` and CuPy is importable;
    otherwise ``(numpy, scipy.signal, False)``. The same :func:`_ncc_fft` runs on
    either backend, so the GPU path keeps the deformed/reference volumes resident
    on the device and does the per-subset FFT cross-correlation there.
    """
    if use_gpu:
        try:
            if importlib.util.find_spec("cupy") is not None:
                import cupy as cp
                import cupyx.scipy.signal as csignal
                return cp, csignal, True
        except Exception:  # noqa: BLE001 — any CuPy import/init issue → CPU
            pass
    import scipy.signal as ssignal
    return np, ssignal, False


def _to_host(a) -> np.ndarray:
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(a)


def _ncc_fft(search, template, xp, signal):
    """FFT normalized cross-correlation of ``template`` within ``search`` (valid
    positions) — the Lewis (1995) NCC that ``skimage.feature.match_template``
    computes, written against a backend module (numpy or cupy) so it runs on CPU
    or GPU. Returns the NCC map (backend array) or ``None`` for a flat template.
    """
    t0 = template - template.mean()
    tnorm = float(xp.sqrt(xp.sum(t0 * t0)))
    if tnorm < 1e-12:
        return None
    n = float(template.size)
    rev = tuple(slice(None, None, -1) for _ in range(template.ndim))
    ones = xp.ones_like(template)
    num = signal.fftconvolve(search, t0[rev], mode="valid")           # Σ I·T0
    sum_i = signal.fftconvolve(search, ones[rev], mode="valid")       # Σ I
    sum_i2 = signal.fftconvolve(search * search, ones[rev], mode="valid")  # Σ I²
    var = xp.clip(sum_i2 - (sum_i * sum_i) / n, 0.0, None)
    denom = tnorm * xp.sqrt(var)
    return xp.where(denom > 1e-12, num / denom, 0.0)


def _parabolic_subpixel(corr: np.ndarray, peak: Tuple[int, ...]) -> np.ndarray:
    """Separable 3-point parabolic peak refinement.

    Returns a per-axis fractional offset in ``[-0.5, 0.5]`` added to the integer
    ``peak`` index. Falls back to 0 on any axis where the peak is on a border or
    the curvature is degenerate.
    """
    ndim = corr.ndim
    delta = np.zeros(ndim, dtype=np.float64)
    for ax in range(ndim):
        i = peak[ax]
        if i <= 0 or i >= corr.shape[ax] - 1:
            continue
        sl_m = list(peak); sl_m[ax] = i - 1
        sl_p = list(peak); sl_p[ax] = i + 1
        cm = float(corr[tuple(sl_m)])
        c0 = float(corr[tuple(peak)])
        cp = float(corr[tuple(sl_p)])
        denom = (cm - 2.0 * c0 + cp)
        if abs(denom) > 1e-12:
            d = 0.5 * (cm - cp) / denom
            if -1.0 < d < 1.0:
                delta[ax] = d
    return delta


def _clamp_window(ctr: np.ndarray, half: np.ndarray, shape: Tuple[int, ...]):
    """Return ``(lo, hi)`` integer slice bounds for a window of half-width
    ``half`` around integer center ``ctr``, clamped to ``[0, shape)``."""
    lo = np.maximum(ctr - half, 0)
    hi = np.minimum(ctr + half + 1, np.asarray(shape))
    return lo.astype(np.int64), hi.astype(np.int64)


def integer_search(
    ref: np.ndarray,
    defm: np.ndarray,
    grid: Grid,
    subset_size: int,
    search_radius: int,
    *,
    correlation: str = "zncc",
    use_gpu: bool = False,
    u0_center: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integer + sub-voxel displacement seed on the subset grid.

    Parameters
    ----------
    ref, defm : normalized (float) volumes of identical shape (2D or 3D).
    grid : the subset-center grid from :func:`mesh.build_grid`.
    subset_size : edge length of the correlation window (voxels).
    search_radius : how far (voxels) the deformed match may sit from the
        (offset) reference position — bounds the *residual* seed displacement.
    correlation : ``"zncc"`` (masked FFT normalized cross-correlation) or
        ``"phase"`` (phase cross-correlation).
    u0_center : optional ``(ndim, *grid)`` per-subset displacement guess. The
        deformed search window is centered at ``ctr + round(u0_center)`` and the
        returned displacement carries that offset — the multigrid refinement hook.

    Returns
    -------
    u0 : (ndim, *grid_shape) sub-voxel displacement seed (voxels, mesh axis order).
    cc : (*grid_shape) peak correlation confidence in ``[-1, 1]`` (ZNCC) — the
        q-factor proxy; ``0`` marks a subset that could not be correlated.
    """
    ndim = grid.ndim
    half = np.full(ndim, max(1, int(subset_size) // 2), dtype=np.int64)
    rad = np.full(ndim, max(1, int(search_radius)), dtype=np.int64)
    gshape = grid.grid_shape
    u0 = np.zeros((ndim, *gshape), dtype=np.float64)
    cc = np.zeros(gshape, dtype=np.float64)

    ref = np.asarray(ref, dtype=np.float32)
    defm = np.asarray(defm, dtype=np.float32)
    # Backend: CPU numpy/scipy, or CuPy (volumes resident on the device for the
    # per-subset FFT NCC) when use_gpu and CuPy is present. Same _ncc_fft either way.
    xp, signal, on_dev = _get_backends(use_gpu)
    ref_x = xp.asarray(ref) if on_dev else ref
    defm_x = xp.asarray(defm) if on_dev else defm
    flat_idx = list(np.ndindex(*gshape))
    n = len(flat_idx)
    for k, idx in enumerate(flat_idx):
        if cancelled_cb is not None and (k & 63) == 0 and cancelled_cb():
            raise InterruptedError("DVC cancelled")
        ctr = np.rint(grid.coords[idx]).astype(np.int64)   # (ndim,) center voxel
        off = (np.rint(u0_center[(slice(None), *idx)]).astype(np.int64)
               if u0_center is not None else np.zeros(ndim, dtype=np.int64))
        dctr = ctr + off                                    # deformed-window center
        rlo, rhi = _clamp_window(ctr, half, ref.shape)
        rsl = tuple(slice(int(a), int(b)) for a, b in zip(rlo, rhi))
        template = ref[rsl]
        if template.size == 0 or float(template.std()) < 1e-6:
            continue                                        # flat/empty → no info

        if correlation == "phase":
            # Compare equal-size windows (ref @ ctr vs deformed @ ctr+off).
            dlo, dhi = _clamp_window(dctr, half, defm.shape)
            dsl = tuple(slice(int(a), int(b)) for a, b in zip(dlo, dhi))
            moving = defm[dsl]
            if moving.shape != template.shape or float(moving.std()) < 1e-6:
                continue
            try:
                shift, _err, _phase = phase_cross_correlation(
                    template, moving, upsample_factor=10, normalization=None)
                # residual shift maps moving→ref; ref→deformed disp = off − shift.
                u0[(slice(None), *idx)] = (off.astype(np.float64)
                                           - np.asarray(shift, dtype=np.float64))
                cc[idx] = 1.0
            except Exception:                               # noqa: BLE001
                continue
            _emit(progress_cb, progress_lo, progress_hi, k, n)
            continue

        # ZNCC: FFT normalized cross-correlation of the template within an
        # expanded deformed search window (on the GPU when on_dev), centered at
        # ctr+off so the returned displacement includes the coarse-level offset.
        slo, shi = _clamp_window(dctr, half + rad, defm.shape)
        ssl = tuple(slice(int(a), int(b)) for a, b in zip(slo, shi))
        if any((int(shi[d]) - int(slo[d])) < template.shape[d]
               for d in range(ndim)):
            continue
        corr_x = _ncc_fft(defm_x[ssl], ref_x[rsl], xp, signal)
        if corr_x is None:
            continue
        corr = _to_host(corr_x)
        if not np.isfinite(corr).any():
            continue
        peak = np.unravel_index(int(np.nanargmax(corr)), corr.shape)
        # Best-match origin of the template inside defm = slo + peak (+subvoxel).
        sub = _parabolic_subpixel(corr, peak)
        match_origin = slo.astype(np.float64) + np.asarray(peak, np.float64) + sub
        u0[(slice(None), *idx)] = match_origin - rlo.astype(np.float64)
        cc[idx] = float(corr[peak])
        _emit(progress_cb, progress_lo, progress_hi, k, n)

    return u0, cc


def _emit(cb: Optional[Callable[[int], None]], lo: int, hi: int, k: int, n: int) -> None:
    if cb is not None and n > 0 and (k & 63) == 0:
        cb(int(lo + (hi - lo) * k / n))


# ─────────────────────────────────────────────────────────────────────────
#  Multigrid (coarse-to-fine) seed  (ALDVC IntegerSearch3Multigrid)
# ─────────────────────────────────────────────────────────────────────────

def _downsample_facs(shape, f: int, subset_size: int):
    """Per-axis downsample factor: ``f`` where the axis stays usable after, else 1
    (keeps a thin Z axis from being crushed)."""
    return tuple(int(f) if (s // f) >= max(6, subset_size // 2) and s >= f * 4 else 1
                 for s in shape)


def _downsample(vol: np.ndarray, facs) -> np.ndarray:
    """Block-mean downsample per axis (crops to a multiple of the factor)."""
    a = np.asarray(vol, dtype=np.float32)
    sl = tuple(slice(0, s - (s % f)) for s, f in zip(a.shape, facs))
    a = a[sl]
    newshape = []
    for s, f in zip(a.shape, facs):
        newshape += [s // f, f]
    a = a.reshape(newshape)
    return a.mean(axis=tuple(range(1, a.ndim, 2)))


def _interp_u_to_grid(src_grid: Grid, u_src: np.ndarray, dst_grid: Grid,
                      facs) -> np.ndarray:
    """Interpolate a coarse-image displacement field ``u_src`` (``(ndim,*src)``, in
    coarse-image voxels) onto ``dst_grid`` at full resolution — evaluate at the
    dst node coords mapped into the coarse image (``/facs``) and scale each
    component back up (``×facs``)."""
    ndim = src_grid.ndim
    facs = np.asarray(facs, dtype=np.float64)
    pts = dst_grid.coords_flat() / facs
    out = np.zeros((ndim, dst_grid.n_nodes), dtype=np.float64)
    for c in range(ndim):
        interp = RegularGridInterpolator(
            [np.asarray(a, float) for a in src_grid.axes], u_src[c],
            method="linear", bounds_error=False, fill_value=None)
        out[c] = np.nan_to_num(interp(pts)) * facs[c]
    return out.reshape(ndim, *dst_grid.grid_shape)


def integer_search_multigrid(
    ref: np.ndarray, defm: np.ndarray, grid: Grid, subset_size: int,
    search_radius: int, *, levels: int = 3, correlation: str = "zncc",
    use_gpu: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0, progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse-to-fine integer seed (ALDVC ``IntegerSearch3Multigrid``).

    Seeds on a ``2^(levels-1)``-downsampled volume with a large (cheap)
    search radius to bracket **large** displacement, upsamples the estimate onto
    the full grid as a per-subset ``u0_center``, then refines at full resolution
    with a small residual radius. ``levels=1`` is the single-scale path.
    """
    levels = max(1, int(levels))
    if levels == 1:
        return integer_search(
            ref, defm, grid, subset_size, search_radius, correlation=correlation,
            use_gpu=use_gpu, progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=progress_lo, progress_hi=progress_hi)
    f = 2 ** (levels - 1)
    facs = _downsample_facs(ref.shape, f, subset_size)
    if all(x == 1 for x in facs):
        return integer_search(
            ref, defm, grid, subset_size, search_radius, correlation=correlation,
            use_gpu=use_gpu, progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=progress_lo, progress_hi=progress_hi)
    ref_c = _downsample(ref, facs)
    defm_c = _downsample(defm, facs)
    fmean = float(np.mean([x for x in facs if x > 1])) or 1.0
    coarse_spacing = max(2, int(round(float(np.mean(grid.step)) / fmean)))
    coarse_grid = build_grid(ref_c.shape, subset_size, coarse_spacing)
    coarse_rad = int(max(np.ceil(search_radius / fmean),
                         0.3 * min(ref_c.shape)))
    mid = int(progress_lo + 0.4 * (progress_hi - progress_lo))
    u0_c, _cc_c = integer_search(
        ref_c, defm_c, coarse_grid, subset_size, coarse_rad,
        correlation=correlation, use_gpu=use_gpu, progress_cb=progress_cb,
        cancelled_cb=cancelled_cb, progress_lo=progress_lo, progress_hi=mid)
    u0_center = _interp_u_to_grid(coarse_grid, u0_c, grid, facs)
    fine_rad = max(4, int(subset_size) // 2)
    return integer_search(
        ref, defm, grid, subset_size, fine_rad, correlation=correlation,
        use_gpu=use_gpu, u0_center=u0_center, progress_cb=progress_cb,
        cancelled_cb=cancelled_cb, progress_lo=mid, progress_hi=progress_hi)
