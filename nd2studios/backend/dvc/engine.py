"""
Top-level ALDVC orchestration.

**Phase 0 status:** this is a *scaffold*. :func:`run_aldvc` currently
implements only Stage 0 (volume prep) plus an honest **global-shift
stub**: it estimates a single rigid integer translation between the two
volumes by FFT phase correlation and fills the displacement grid
uniformly with it. This is NOT the ALDVC algorithm — it exists so the
registry → worker → page → result-viewer wiring can be exercised
end-to-end on real data before the numerical stages land.

The real per-stage implementations (integer search, IC-GN, ADMM global
step, strain) are filled in over Phases 1-5 per
``CodeLog/ClaudesPlan/V1.45_DVC.md``; ``run_aldvc`` will dispatch to them
and the stub will be removed.

Pure numpy/scipy — NO PySide6 (backend-purity rule).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from nd2studios.core.dvc_registry import DVCParams, DVCResult


# ──────────────────────────────────────────────────────────────────────
# Stage 0 — volume preparation
# ──────────────────────────────────────────────────────────────────────
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Affine min/max rescale to float32 in ``[0, 1]`` over the whole volume.

    Mirrors ALDVC ``funNormalizeImg3``. A degenerate (constant) volume maps
    to all-zeros rather than dividing by zero.
    """
    arr = np.asarray(vol, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def make_grid(
    shape: Tuple[int, ...], subset_size: int, subset_spacing: int
) -> np.ndarray:
    """Build a regular grid of subset-center coordinates (in voxels).

    Centers are spaced ``subset_spacing`` apart, inset by ``subset_size // 2``
    from every border so each subset window fits inside the volume.

    Returns ``(*grid, d)`` with ``d == len(shape)``; the last axis holds the
    per-axis center coordinate in the same axis order as ``shape`` (slowest
    axis first: ``y, x`` in 2D; ``z, y, x`` in 3D).
    """
    half = max(1, int(subset_size) // 2)
    step = max(1, int(subset_spacing))
    axes = []
    for n in shape:
        start = half
        stop = max(half + 1, n - half)
        coords = np.arange(start, stop, step, dtype=np.int64)
        if coords.size == 0:                      # tiny volume — at least 1 center
            coords = np.asarray([n // 2], dtype=np.int64)
        axes.append(coords)
    mesh = np.meshgrid(*axes, indexing="ij")      # ij so axis order == shape
    return np.stack(mesh, axis=-1).astype(np.float64)


def _global_phase_shift(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Estimate a single rigid integer translation ``mov ≈ ref(x - shift)``.

    Plain-numpy FFT phase correlation; works for 2D and 3D. Returned shift is
    in voxels, axis order matching the array axes (y,x or z,y,x), wrapped into
    ``[-n/2, n/2)`` per axis. (Stub-only; the real seed is the windowed,
    per-subset search in ``integer_search.py``, Phase 2.)
    """
    a = ref - ref.mean()
    b = mov - mov.mean()
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag == 0] = 1.0
    corr = np.fft.ifftn(cross / mag).real        # phase correlation
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shift = np.array(peak, dtype=np.float64)
    for i, n in enumerate(corr.shape):           # wrap to signed range
        if shift[i] > n / 2:
            shift[i] -= n
    return shift


def run_aldvc(
    ref_vol: np.ndarray,
    def_vol: np.ndarray,
    voxel_size_um: Tuple[float, ...],
    params: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> DVCResult:
    """Run (the Phase-0 stub of) ALDVC on a single reference/deformed pair.

    Args mirror :meth:`DVCMethod.run`. ``ref_vol``/``def_vol`` are ``(H,W)``
    for 2D DIC or ``(Z,H,W)`` for 3D DVC and must share a shape.
    """
    ref = np.asarray(ref_vol)
    mov = np.asarray(def_vol)
    if ref.shape != mov.shape:
        raise ValueError(
            f"reference and deformed volumes must match: "
            f"{ref.shape} vs {mov.shape}"
        )
    if ref.ndim not in (2, 3):
        raise ValueError(f"expected a 2D image or 3D volume, got ndim={ref.ndim}")

    p = DVCParams.from_dict(params)
    dim = ref.ndim

    def _tick(v: int) -> None:
        if progress_cb is not None:
            progress_cb(int(v))

    def _cancelled() -> bool:
        return bool(cancelled_cb()) if cancelled_cb is not None else False

    _tick(2)
    ref_n = normalize_volume(ref)
    def_n = normalize_volume(mov)
    _tick(15)
    if _cancelled():
        raise InterruptedError("DVC cancelled")

    grid = make_grid(ref.shape, p.subset_size, p.subset_spacing)
    _tick(35)

    # ── Phase-0 stub: uniform global integer shift ──
    shift = _global_phase_shift(ref_n, def_n)
    _tick(85)

    disp = np.empty_like(grid)
    disp[...] = shift                            # broadcast shift over the grid

    if not voxel_size_um or len(voxel_size_um) != dim:
        voxel_size_um = tuple([1.0] * dim)

    _tick(100)
    return DVCResult(
        dim=dim,
        grid_coords=grid,
        displacement_field=disp,
        voxel_size_um=tuple(float(v) for v in voxel_size_um),
        converged=True,
        iterations=0,
        mu=float(p.mu),
        method="stub-global-shift",
        notes=(
            "Phase-0 scaffold: uniform rigid global-shift estimate via FFT "
            "phase correlation — NOT the ALDVC algorithm. The local IC-GN + "
            "ADMM global step (a true per-subset field) lands in Phases 2-5."
        ),
        diagnostics={
            "global_shift_voxels": shift.tolist(),
            "grid_shape": list(grid.shape[:-1]),
            "n_subsets": int(np.prod(grid.shape[:-1])),
        },
    )
