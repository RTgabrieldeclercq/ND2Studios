"""
Stage 6 — the ALDVC ADMM outer loop (the "augmented Lagrangian" in ALDVC).

Ties the local image-correlation solve (Subpb1, :mod:`icgn`) to the global
compatibility projection (Subpb2, :mod:`global_step`) and alternates them with
scaled-dual updates until the compatible displacement stops moving. This is what
buys global-DVC accuracy at near-local-DVC cost.

Standard scaled ADMM for ``min f(u,F) s.t. u=û, F=∇û`` (``f`` = the ZNSSD image
residual solved by IC-GN):

* **Subpb1** — penalized local IC-GN, warp pulled toward ``(û − s_u, F̂ − s_F)``.
* **Subpb2** — global solve ``(μI + βDᵀD)û = μ(u+s_u) + βDᵀ(F+s_F)``.
* **duals** — ``s_u += (u − û)``, ``s_F += (F − F̂)``.
* **converge** — ``‖û_new − û_old‖₂/√N < ADMMtol``, capped at ``admm_iterations``.

Pass 0 (before the loop) is plain local IC-GN + one global solve ⇒ conventional
DVC; each subsequent iteration tightens compatibility.

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from nd2studios.backend.dvc.mesh import Grid
from nd2studios.backend.dvc.icgn import local_icgn
from nd2studios.backend.dvc.global_step import AugLagGlobalStep
from nd2studios.backend.dvc.outliers import remove_outliers, inpaint_vector, inpaint_nans


@dataclass
class ADMMResult:
    u: np.ndarray           # (ndim, *grid) compatible displacement (voxels)
    F: np.ndarray           # (ndim, ndim, *grid) compatible gradient ∇û
    zncc: np.ndarray        # (*grid) final local correlation confidence
    converged: bool
    iterations: int
    beta: float
    residuals: list         # per-iteration ‖Δû‖₂/√N


def _inpaint_tensor(F: np.ndarray) -> np.ndarray:
    out = np.array(F, dtype=np.float64, copy=True)
    ndim = out.shape[0]
    for i in range(ndim):
        for j in range(ndim):
            out[i, j] = inpaint_nans(out[i, j])
    return out


def run_admm(
    ref: np.ndarray,
    defm_pref: np.ndarray,
    grid: Grid,
    u0: np.ndarray,
    subset_size: int,
    *,
    mu: float = 1e-3,
    admm_iterations: int = 4,
    icgn_tol: float = 1e-2,
    icgn_max_iter: int = 100,
    admm_tol: float = 1e-2,
    cc_thresh: float = 0.5,
    median_thresh: float = 2.0,
    n_workers: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> ADMMResult:
    """Run conventional local DVC (pass 0) then the ADMM refinement loop.

    ``ref`` is the (normalized) reference; ``defm_pref`` the (normalized) deformed
    volume **already spline-prefiltered**. ``u0`` is the ``(ndim,*grid)`` integer
    seed. Returns an :class:`ADMMResult` with the compatible field.
    """
    ndim = grid.ndim
    ref_cache: dict = {}                       # per-subset reference SD/H reuse
    n_iters = max(0, int(admm_iterations))

    def _span(lo_frac: float, hi_frac: float):
        lo = progress_lo + (progress_hi - progress_lo) * lo_frac
        hi = progress_lo + (progress_hi - progress_lo) * hi_frac
        return int(lo), int(hi)

    # ── Pass 0: conventional local IC-GN ───────────────────────────────
    p_lo, p_hi = _span(0.0, 0.5 if n_iters else 0.9)
    u_L, F_L, zncc, _iters = local_icgn(
        ref, defm_pref, grid, u0, subset_size,
        tol=icgn_tol, max_iter=icgn_max_iter, ref_cache=ref_cache,
        n_workers=n_workers,
        progress_cb=progress_cb, cancelled_cb=cancelled_cb,
        progress_lo=p_lo, progress_hi=p_hi)

    # Clean the noisy local field before the compatibility solve.
    u_L, _bad = remove_outliers(u_L, zncc, cc_thresh=cc_thresh,
                                median_thresh=median_thresh)
    u_L = inpaint_vector(u_L)
    F_L = _inpaint_tensor(F_L)

    # ── First global solve (β via L-curve) ─────────────────────────────
    gstep = AugLagGlobalStep(grid)
    beta = gstep.tune_beta(u_L, F_L, mu)
    uhat, Fhat = gstep.solve(u_L, F_L, None, None, mu, beta)

    residuals: list = []
    converged = (n_iters == 0)
    if n_iters == 0:
        return ADMMResult(u=uhat, F=Fhat, zncc=zncc, converged=True,
                          iterations=0, beta=beta, residuals=residuals)

    # ── ADMM loop ──────────────────────────────────────────────────────
    wu = np.zeros_like(uhat)                   # scaled dual for u constraint
    wF = np.zeros_like(Fhat)                   # scaled dual for F constraint
    n_nodes = float(grid.n_nodes)
    it = 0
    for it in range(1, n_iters + 1):
        if cancelled_cb is not None and cancelled_cb():
            raise InterruptedError("DVC cancelled")
        a = uhat - wu                          # Subpb1 targets (z − dual)
        B = Fhat - wF
        li_lo, li_hi = _span(0.5 + 0.5 * (it - 1) / n_iters,
                             0.5 + 0.5 * it / n_iters)
        u1, F1, zncc, _it2 = local_icgn(
            ref, defm_pref, grid, uhat, subset_size,
            tol=icgn_tol, max_iter=icgn_max_iter,
            mu=mu, beta=beta, u_target=a, F_target=B, ref_cache=ref_cache,
            n_workers=n_workers,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=li_lo, progress_hi=li_hi)
        u1 = inpaint_vector(u1)
        F1 = _inpaint_tensor(F1)

        prev = uhat
        uhat, Fhat = gstep.solve(u1, F1, wu, wF, mu, beta)
        wu = wu + (u1 - uhat)                   # dual update (x − z)
        wF = wF + (F1 - Fhat)

        change = float(np.linalg.norm((uhat - prev).ravel()) / np.sqrt(n_nodes))
        residuals.append(change)
        if change < admm_tol:
            converged = True
            break

    return ADMMResult(u=uhat, F=Fhat, zncc=zncc, converged=converged,
                      iterations=it, beta=beta, residuals=residuals)
