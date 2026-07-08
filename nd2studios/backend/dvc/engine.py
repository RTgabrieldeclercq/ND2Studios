"""
Top-level ALDVC orchestration.

Runs the full Augmented Lagrangian Digital Volume Correlation pipeline on one
reference→deformed pair and returns a :class:`DVCResult`:

    Stage 0  normalize both volumes, spline-prefilter the deformed one once
    Stage 1  per-subset FFT integer displacement seed        (integer_search)
    Stage 2  outlier clean + inpaint the seed                 (outliers)
    Stage 3–6  local IC-GN + ADMM global compatibility loop   (admm → icgn+global_step)
    Stage 7  strain tensor from the compatible field          (strain)

Works for 2D images (DIC, 6-DOF) and 3D volumes (DVC, 12-DOF); dimensionality is
inferred from the input. Axis order throughout is mesh order (``(y,x)`` /
``(z,y,x)``, slowest first). Pure numpy/scipy — NO PySide6 (backend-purity rule).

This replaces the ``Add-DVC`` Phase-0 global-shift stub; ``normalize_volume`` is
retained (Stage 0). See ``CodeLog/ClaudesPlan/V1.51_dvc_aldvc_node.md``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import spline_filter

from nd2studios.core.dvc_registry import DVCParams, DVCResult
from nd2studios.backend.dvc.mesh import build_grid
from nd2studios.backend.dvc.integer_search import integer_search_multigrid
from nd2studios.backend.dvc.outliers import remove_outliers, inpaint_vector
from nd2studios.backend.dvc.admm import run_admm
from nd2studios.backend.dvc.strain import compute_strain


# ──────────────────────────────────────────────────────────────────────
# Stage 0 — volume preparation
# ──────────────────────────────────────────────────────────────────────
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Affine min/max rescale to float32 in ``[0, 1]`` over the whole volume.

    Mirrors ALDVC ``funNormalizeImg3``. A degenerate (constant) volume maps to
    all-zeros rather than dividing by zero.
    """
    arr = np.asarray(vol, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _get(raw: Dict[str, Any], key: str, default):
    v = raw.get(key, default)
    return default if v is None else v


def run_aldvc(
    ref_vol: np.ndarray,
    def_vol: np.ndarray,
    voxel_size_um: Tuple[float, ...],
    params: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    u0_seed: Optional[np.ndarray] = None,
    use_fft_seed: bool = True,
) -> DVCResult:
    """Run ALDVC on a single reference/deformed pair.

    ``ref_vol``/``def_vol`` are ``(H,W)`` (2D DIC) or ``(Z,H,W)`` (3D DVC) and
    must share a shape. ``params`` is the ParamEditor dict (see
    :class:`~nd2studios.core.dvc_registry.DVCParams` plus optional
    ``search_radius``, ``seed_levels``, ``icgn_tol``, ``icgn_max_iter``,
    ``admm_tol``, ``cc_thresh``, ``median_thresh``, ``strain_smooth``).

    ``u0_seed`` (``(ndim,*grid)``) + ``use_fft_seed=False`` warm-start the seed
    from a prior frame's field (ALDVC's cross-frame ``U0``), skipping the FFT
    search; on any shape mismatch the FFT multigrid seed is used instead.
    """
    ref = np.asarray(ref_vol)
    mov = np.asarray(def_vol)
    if ref.shape != mov.shape:
        raise ValueError(f"reference and deformed volumes must match: "
                         f"{ref.shape} vs {mov.shape}")
    if ref.ndim not in (2, 3):
        raise ValueError(f"expected a 2D image or 3D volume, got ndim={ref.ndim}")

    p = DVCParams.from_dict(params)
    raw = params or {}
    dim = ref.ndim
    subset_size = int(p.subset_size)
    subset_spacing = int(p.subset_spacing)
    search_radius = int(_get(raw, "search_radius", 0)) or max(4, subset_size)
    seed_levels = max(1, int(_get(raw, "seed_levels", 3)))
    correlation = str(p.correlation)
    mu = float(p.mu)
    admm_iterations = int(p.admm_iterations)
    strain_type = str(p.strain_type)
    strain_smooth = float(_get(raw, "strain_smooth", 0.0))
    use_gpu = bool(p.use_gpu)
    n_workers = int(_get(raw, "n_workers", 0))
    icgn_tol = float(_get(raw, "icgn_tol", 1e-2))
    icgn_max_iter = int(_get(raw, "icgn_max_iter", 100))
    admm_tol = float(_get(raw, "admm_tol", 1e-2))
    cc_thresh = float(_get(raw, "cc_thresh", 0.5))
    median_thresh = float(_get(raw, "median_thresh", 2.0))

    def _tick(v: int) -> None:
        if progress_cb is not None:
            progress_cb(int(max(0, min(100, v))))

    def _cancelled() -> bool:
        return bool(cancelled_cb()) if cancelled_cb is not None else False

    _tick(2)
    ref_n = normalize_volume(ref)
    def_n = normalize_volume(mov)
    if _cancelled():
        raise InterruptedError("DVC cancelled")
    # Prefilter the deformed volume ONCE so per-iteration warping is a pure
    # cubic-B-spline evaluation (map_coordinates prefilter=False).
    defm_pref = spline_filter(def_n.astype(np.float32), order=3,
                              mode="nearest").astype(np.float32)
    grid = build_grid(ref.shape, subset_size, subset_spacing)
    # Fan the IC-GN sweep across cores by default; skip the process-pool overhead
    # on tiny grids. GPU (seed FFTs) and CPU multiprocessing (IC-GN) compose.
    if n_workers <= 0:
        from nd2studios.backend.dvc.parallel import default_workers
        n_workers = default_workers()
    if grid.n_nodes < 64:
        n_workers = 1
    _tick(8)

    # Stage 1 — integer seed. Warm-start from a prior frame's field when given
    # (ALDVC's cross-frame U0), else the multigrid FFT search; clean the FFT seed
    # (the warm-start field is already smooth).
    want_warm = (u0_seed is not None and not use_fft_seed
                 and tuple(np.shape(u0_seed)) == (dim, *grid.grid_shape))
    if want_warm:
        u0 = inpaint_vector(np.asarray(u0_seed, dtype=np.float64).copy())
        _tick(32)
    else:
        u0, cc = integer_search_multigrid(
            ref_n, def_n, grid, subset_size, search_radius, levels=seed_levels,
            correlation=correlation, use_gpu=use_gpu,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=8, progress_hi=32)
        u0, _bad = remove_outliers(u0, cc, cc_thresh=cc_thresh,
                                   median_thresh=median_thresh)
        u0 = inpaint_vector(u0)
    _tick(34)

    # Stages 3–6 — local IC-GN + ADMM compatibility loop.
    res = run_admm(
        ref_n, defm_pref, grid, u0, subset_size,
        mu=mu, admm_iterations=admm_iterations,
        icgn_tol=icgn_tol, icgn_max_iter=icgn_max_iter, admm_tol=admm_tol,
        cc_thresh=cc_thresh, median_thresh=median_thresh, n_workers=n_workers,
        progress_cb=progress_cb, cancelled_cb=cancelled_cb,
        progress_lo=34, progress_hi=90)
    _tick(92)

    # Stage 7 — strain.
    if not voxel_size_um or len(voxel_size_um) != dim:
        voxel_size_um = tuple([1.0] * dim)
    voxel = np.asarray(voxel_size_um, dtype=np.float64)
    _F_def, strain = compute_strain(
        res.u, grid.step, voxel_size=voxel,
        strain_type=strain_type, smooth_sigma=strain_smooth)
    _tick(98)

    disp = np.moveaxis(res.u, 0, -1)                  # (*grid, ndim), voxels
    strain_field = np.moveaxis(strain, (0, 1), (-2, -1))   # (*grid, ndim, ndim)

    _tick(100)
    return DVCResult(
        dim=dim,
        grid_coords=grid.coords,
        displacement_field=disp,
        voxel_size_um=tuple(float(v) for v in voxel_size_um),
        strain_field=strain_field,
        strain_type=strain_type,
        qfactor=np.asarray(res.zncc),
        converged=bool(res.converged),
        iterations=int(res.iterations),
        mu=mu,
        beta=float(res.beta),
        method="ALDVC",
        notes=(
            f"ALDVC · {dim}D · subset={subset_size} spacing={subset_spacing} · "
            f"ADMM {res.iterations}/{admm_iterations} iters "
            f"(converged={res.converged}) · β={res.beta:.3g}"
        ),
        diagnostics={
            "grid_shape": list(grid.grid_shape),
            "n_subsets": int(grid.n_nodes),
            "beta": float(res.beta),
            "admm_residuals": [float(r) for r in res.residuals],
            "median_zncc": float(np.nanmedian(res.zncc)) if res.zncc.size else 0.0,
            "search_radius": int(search_radius),
            "n_workers": int(n_workers),
            "use_gpu": bool(use_gpu),
        },
    )
