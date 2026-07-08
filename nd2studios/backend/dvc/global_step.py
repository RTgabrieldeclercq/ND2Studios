"""
Stage 4 — global augmented-Lagrangian compatibility solve (ALDVC Subpb2, FD).

The local IC-GN field ``(u, F)`` is noisy and kinematically incompatible
(``F`` need not equal ``∇u``). This step projects it onto a smooth, compatible
displacement ``û`` (with ``F̂ = ∇û``) by solving the augmented-Lagrangian normal
equations

    (β·DᵀD + μ·I) û = β·Dᵀ(F − w_F) + μ·(u − w_u)

where ``D`` is the sparse finite-difference gradient operator and ``w_u`` / ``w_F``
are the ADMM scaled duals. This is exactly the FD variant of FranckLab's Subpb2.

**Operator ``D``** (:func:`_build_fd_operator`) is a hardened re-derivation of
SerialTrack's ``funDerivativeOp3`` port (``regularization._build_gradient_operator``)
— same finite-difference stencil and DOF layout (matching
:mod:`nd2studios.backend.dvc.mesh` ``pack_u`` / ``pack_F``), but it **guards
singleton grid axes** so a thin (shallow-Z) DVC grid can't overflow the operator
(SerialTrack's version assumes ≥2 nodes per axis). SerialTrack's own ``ADMMLSolver``
solves the ``μ``-only variant ``(α·DᵀD + I)û = u − v``; DVC adds the **F-coupling**
term ``β·Dᵀ(F − w_F)`` that carries the local deformation gradient into the solve.

Robustness over the MATLAB reference (which runs a near-singular solve): a small
Tikhonov term and a guarded ``poly2`` L-curve fit. The matrix is constant across
the β-sweep and the ADMM iterations (only the RHS changes), so its factorization
is cached.

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix, eye as speye
from scipy.sparse.linalg import factorized

from nd2studios.backend.dvc import mesh as _mesh
from nd2studios.backend.dvc.mesh import Grid


def _build_fd_operator(grid_shape, grid_step, ndim: int) -> csc_matrix:
    """Sparse finite-difference gradient operator ``D`` with ``F_vec = D @ u_vec``.

    Same DOF layout as :func:`mesh.pack_u` / :func:`mesh.pack_F` — ``u`` at
    ``ndim*p + comp`` and ``∂u_comp/∂x_deriv`` at ``ndim²*p + deriv*ndim + comp``
    (``p`` C-order). Central differences interior, one-sided at borders. This is a
    hardened re-derivation of SerialTrack's ``funDerivativeOp3`` port that
    additionally **guards singleton axes** (a grid axis with <2 nodes contributes
    a zero derivative instead of indexing an out-of-range neighbour), so thin
    (e.g. shallow-Z) DVC grids never overflow the operator.
    """
    grid_shape = tuple(int(s) for s in grid_shape)
    n = int(np.prod(grid_shape)) if grid_shape else 0
    n_u = ndim * n
    n_f = ndim * ndim * n
    # C-order strides.
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * grid_shape[d + 1]

    def _multi(idx):
        mi = [0] * ndim
        for d in range(ndim - 1, -1, -1):
            mi[d] = idx % grid_shape[d]
            idx //= grid_shape[d]
        return mi

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for deriv in range(ndim):
        h = float(grid_step[deriv]) or 1.0
        stride = strides[deriv]
        sz = grid_shape[deriv]
        for comp in range(ndim):
            f_off = deriv * ndim + comp
            for p in range(n):
                f_row = ndim * ndim * p + f_off
                if sz < 2:
                    continue                    # singleton axis → zero derivative
                idx_along = _multi(p)[deriv]
                if 0 < idx_along < sz - 1:       # central
                    rows += [f_row, f_row]
                    cols += [ndim * (p - stride) + comp, ndim * (p + stride) + comp]
                    vals += [-1.0 / (2 * h), 1.0 / (2 * h)]
                elif idx_along == 0:             # forward
                    rows += [f_row, f_row]
                    cols += [ndim * p + comp, ndim * (p + stride) + comp]
                    vals += [-1.0 / h, 1.0 / h]
                else:                            # backward
                    rows += [f_row, f_row]
                    cols += [ndim * (p - stride) + comp, ndim * p + comp]
                    vals += [-1.0 / h, 1.0 / h]
    return csc_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_f, n_u))


class AugLagGlobalStep:
    """Cached augmented-Lagrangian global solver for one grid.

    Build once per (grid, β, μ); reuse :meth:`solve` across ADMM iterations
    (only the RHS — i.e. the duals — changes).
    """

    def __init__(self, grid: Grid, *, tikhonov: float = 1e-6):
        self.grid = grid
        self.ndim = grid.ndim
        self.n = grid.n_nodes
        self.tikhonov = float(tikhonov)
        self.D: csc_matrix = _build_fd_operator(
            grid.grid_shape, grid.step, grid.ndim)
        self.DtD: csc_matrix = (self.D.T @ self.D).tocsc()
        self._I = speye(self.ndim * self.n, format="csc")
        self._mu: Optional[float] = None
        self._beta: Optional[float] = None
        self._factor = None

    # ── matrix / factorization cache ────────────────────────────────────
    def _ensure_factor(self, mu: float, beta: float) -> None:
        if self._factor is not None and self._mu == mu and self._beta == beta:
            return
        A = (beta * self.DtD + (mu + self.tikhonov) * self._I).tocsc()
        self._factor = factorized(A)      # splu-backed solve(rhs)
        self._mu, self._beta = mu, beta

    def _rhs(self, u_vec, F_vec, wu_vec, wF_vec, mu, beta) -> np.ndarray:
        # Standard scaled-ADMM z-update RHS: μ(u + w_u) + β·Dᵀ(F + w_F).
        # Paired in admm.py with targets (ẑ − w) and dual update (w += x − z).
        return beta * (self.D.T @ (F_vec + wF_vec)) + mu * (u_vec + wu_vec)

    # ── public solve ────────────────────────────────────────────────────
    def solve(
        self, u_grid: np.ndarray, F_grid: np.ndarray,
        wu_grid: Optional[np.ndarray], wF_grid: Optional[np.ndarray],
        mu: float, beta: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(u_hat_grid (ndim,*grid), F_hat_grid (ndim,ndim,*grid))``.

        ``F_hat = ∇û`` (the compatible gradient, ``unpack_F(D @ û)``). Duals may be
        ``None`` (treated as zero).
        """
        u_vec = _mesh.pack_u(u_grid)
        F_vec = _mesh.pack_F(F_grid)
        wu_vec = _mesh.pack_u(wu_grid) if wu_grid is not None else 0.0
        wF_vec = _mesh.pack_F(wF_grid) if wF_grid is not None else 0.0
        self._ensure_factor(mu, beta)
        rhs = self._rhs(u_vec, F_vec, wu_vec, wF_vec, mu, beta)
        uhat = self._factor(rhs)
        Fhat = self.D @ uhat
        return (_mesh.unpack_u(uhat, self.grid.grid_shape, self.ndim),
                _mesh.unpack_F(Fhat, self.grid.grid_shape, self.ndim))

    # ── β selection (L-curve, MATLAB ErrSum criterion) ──────────────────
    def tune_beta(
        self, u_grid: np.ndarray, F_grid: np.ndarray, mu: float,
        beta_list: Optional[List[float]] = None,
    ) -> float:
        """Pick β by the FranckLab ``ErrSum = ‖u−û‖ + ‖F−∇û‖·mean(step)²`` L-curve,
        with a guarded parabolic refine. Duals are zero at selection time."""
        if beta_list is None:
            base = float(np.mean(self.grid.step) ** 2) * mu
            factors = np.array([np.sqrt(1e-5), 1e-2, np.sqrt(1e-3),
                                1e-1, np.sqrt(1e-1)])
            beta_list = list(np.maximum(factors * base, 1e-12))
        u_vec = _mesh.pack_u(u_grid)
        F_vec = _mesh.pack_F(F_grid)
        w2 = float(np.mean(self.grid.step) ** 2)
        errs = np.empty(len(beta_list))
        for i, b in enumerate(beta_list):
            self._ensure_factor(mu, float(b))
            uhat = self._factor(mu * u_vec)          # duals zero → rhs = μ·u
            fid = float(np.linalg.norm(u_vec - uhat))
            smooth = float(np.linalg.norm(F_vec - (self.D @ uhat)))
            errs[i] = fid + smooth * w2
        i0 = int(np.argmin(errs))
        beta = float(beta_list[i0])
        if 0 < i0 < len(beta_list) - 1:
            lb = np.log10(np.asarray(beta_list[i0 - 1:i0 + 2]))
            y = errs[i0 - 1:i0 + 2]
            try:
                p = np.polyfit(lb, y, 2)
                if abs(p[0]) > 1e-15:
                    cand = 10 ** (-p[1] / (2 * p[0]))
                    if beta_list[i0 - 1] <= cand <= beta_list[i0 + 1]:
                        beta = float(cand)
            except Exception:                         # noqa: BLE001
                pass
        # Reset the cached factor so the next solve rebuilds at the chosen β.
        self._factor = None
        self._mu = self._beta = None
        return beta
