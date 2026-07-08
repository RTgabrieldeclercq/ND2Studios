"""
SerialTrack Python — Chunk 3
=============================
Split this file into two modules:
    serialtrack/regularization.py
    serialtrack/fields.py

Dependencies:
    pip install numpy scipy numba
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FILE 1: serialtrack/regularization.py                              ║
# ║  Global-step solvers: MLS, grid regularization, ADMM-AL             ║
# ║  Replaces: funCompDefGrad3.m, funScatter2Grid3D.m, regularizeNd.m,  ║
# ║            funDerivativeOp3.m, and the gbSolver branches in          ║
# ║            f_track_serial_match3D.m                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum
import numpy as np
from scipy.spatial import cKDTree, QhullError
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import eye as speye, diags as spdiags, kron as spkron, csc_matrix
from scipy.sparse.linalg import spsolve
import logging
from .config import GlobalSolver


log = logging.getLogger("serialtrack.regularization")


# ═══════════════════════════════════════════════════════════════
#  Scatter → Grid interpolation  (replaces funScatter2Grid3D.m
#  + the 600-line regularizeNd.m)
# ═══════════════════════════════════════════════════════════════

def scatter_to_grid(
    coords: np.ndarray,
    values: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Interpolate scattered data onto a regular grid.

    Replaces ``funScatter2Grid3D.m`` + ``regularizeNd.m`` (~650 lines).

    Parameters
    ----------
    coords : (N, D) — scattered point positions
    values : (N,)   — scalar field values at those points
    grid_step : (D,) — grid spacing per dimension
    smoothness : float — RBF smoothing parameter (0 = pure interpolation)
    grid_coords : optional pre-built meshgrid arrays

    Returns
    -------
    grids : tuple of D arrays, each with shape of the output grid
    f_grid : array with same shape — interpolated values
    """
    ndim = coords.shape[1]
    gs = np.asarray(grid_step, dtype=np.float64)

    # Build grid axes
    if grid_coords is not None:
        grids = grid_coords
        axes = tuple(np.unique(g) for g in grids)
    else:
        axes = []
        for d in range(ndim):
            lo, hi = coords[:, d].min(), coords[:, d].max()
            axes.append(np.arange(lo, hi + gs[d], gs[d]))
        axes = tuple(axes)
        grids = np.meshgrid(*axes, indexing="ij")
        grids = tuple(grids)

    query_pts = np.column_stack([g.ravel() for g in grids])

    # Minimum point requirements:
    #   LinearNDInterpolator needs ndim+1 for a simplex
    #   RBFInterpolator(degree=1) needs ndim+1 points
    min_pts = ndim + 1
    if len(coords) < min_pts:
        log.warning("scatter_to_grid: only %d points (need %d) — returning zeros",
                     len(coords), min_pts)
        f_grid = np.zeros(grids[0].shape, dtype=np.float64)
        return grids, f_grid

    if smoothness <= 0:
        # Pure linear interpolation (fast)
        f_flat = _linear_or_nearest_interpolate(coords, values, query_pts)
    else:
        # RBF with smoothing — replaces the entire regularizeNd.m
        interp = RBFInterpolator(
            coords, values,
            smoothing=smoothness,
            kernel="thin_plate_spline",
            degree=1,
        )
        f_flat = interp(query_pts)

    f_grid = f_flat.reshape(grids[0].shape)
    return grids, f_grid


def _linear_or_nearest_interpolate(
    coords: np.ndarray,
    values: np.ndarray,
    query_pts: np.ndarray,
) -> np.ndarray:
    """Try LinearNDInterpolator, fallback to nearest neighbour on degenerate input."""
    ndim = coords.shape[1]
    if len(coords) < ndim + 1:
        log.warning(
            "linear interpolation: only %d points for %dd data — using nearest neighbour",
            len(coords), ndim,
        )
        interp = NearestNDInterpolator(coords, values)
        return interp(query_pts)

    try:
        interp = LinearNDInterpolator(coords, values, fill_value=0.0)
        return interp(query_pts)
    except QhullError as exc:
        log.warning(
            "LinearNDInterpolator failed (%s); falling back to nearest neighbour",
            exc,
        )
        interp = NearestNDInterpolator(coords, values)
        return interp(query_pts)


def scatter_to_grid_multi(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Interpolate a multi-component displacement field to a grid.

    Parameters
    ----------
    coords : (N, D) — particle positions
    disp   : (N, D) — displacement vectors at those positions
    grid_step : (D,) — grid spacing
    smoothness : float

    Returns
    -------
    grids : tuple of D meshgrid arrays
    disp_grid : (D, *grid_shape) — gridded displacement components
    """
    ndim = coords.shape[1]
    grids = grid_coords          # reuse grid if provided (critical for ADMM)
    components = []
    for d in range(ndim):
        grids, fg = scatter_to_grid(
            coords, disp[:, d], grid_step, smoothness, grids
        )
        components.append(fg)
    return grids, np.array(components)  # shape (D, *grid_shape)


# ═══════════════════════════════════════════════════════════════
#  Bounded scatter → Grid  (Cell-Tracker model, no extrapolation)
#  Mirrors CellTracker/backend/fields.py: linear inside the convex
#  hull, ZERO outside it, then an optional Gaussian blur.  Used for
#  the PTV / post-processing field products, where the grid IS the
#  output and is evaluated everywhere — including the empty corners
#  and gaps that the RBF thin-plate-spline path blows up on.
# ═══════════════════════════════════════════════════════════════

def scatter_to_grid_bounded(
    coords: np.ndarray,
    values: np.ndarray,
    grid_step: np.ndarray,
    smoothing_sigma: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Interpolate scattered data onto a grid **without extrapolating**.

    This is the original Cell-Tracker approach (``CellTracker/backend/fields.py``):
    piecewise-linear interpolation inside the convex hull of the points, filled
    with ``0`` outside it (``griddata`` / ``LinearNDInterpolator`` return NaN
    there), followed by an optional Gaussian blur of ``smoothing_sigma`` grid
    cells.  Unlike :func:`scatter_to_grid`'s ``thin_plate_spline`` RBF branch,
    the output can never exceed the range of ``values`` — so it does not blow up
    in the empty corners / inter-cluster gaps of the bounding-box grid.

    Parameters
    ----------
    coords : (N, D) — scattered point positions
    values : (N,)   — scalar field values at those points
    grid_step : (D,) — grid spacing per dimension
    smoothing_sigma : float — Gaussian blur sigma in *grid cells* (0 = none)
    grid_coords : optional pre-built meshgrid arrays
    """
    ndim = coords.shape[1]
    gs = np.asarray(grid_step, dtype=np.float64)

    if grid_coords is not None:
        grids = grid_coords
    else:
        axes = []
        for d in range(ndim):
            lo, hi = coords[:, d].min(), coords[:, d].max()
            axes.append(np.arange(lo, hi + gs[d], gs[d]))
        grids = tuple(np.meshgrid(*axes, indexing="ij"))

    query_pts = np.column_stack([g.ravel() for g in grids])

    if len(coords) < ndim + 1:
        log.warning("scatter_to_grid_bounded: only %d points (need %d) — returning zeros",
                     len(coords), ndim + 1)
        return grids, np.zeros(grids[0].shape, dtype=np.float64)

    # Linear interpolation inside the hull; 0 outside it (fill_value=0.0 in
    # _linear_or_nearest_interpolate). No extrapolation, ever.
    f_grid = _linear_or_nearest_interpolate(coords, values, query_pts).reshape(
        grids[0].shape)

    if smoothing_sigma and smoothing_sigma > 0:
        f_grid = gaussian_filter(f_grid, sigma=float(smoothing_sigma))
    return grids, f_grid


def scatter_to_grid_multi_bounded(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothing_sigma: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Multi-component :func:`scatter_to_grid_bounded` (Cell-Tracker model).

    Returns ``(grids, disp_grid)`` with ``disp_grid`` of shape
    ``(D, *grid_shape)`` — bounded, never extrapolated.
    """
    ndim = coords.shape[1]
    grids = grid_coords
    components = []
    for d in range(ndim):
        grids, fg = scatter_to_grid_bounded(
            coords, disp[:, d], grid_step, smoothing_sigma, grids
        )
        components.append(fg)
    return grids, np.array(components)  # shape (D, *grid_shape)


# ═══════════════════════════════════════════════════════════════
#  Global Solver 1: Moving Least Squares (MLS)
#  Replaces funCompDefGrad3.m
# ═══════════════════════════════════════════════════════════════

def solve_mls(
    disp: np.ndarray,
    coords: np.ndarray,
    query_coords: np.ndarray,
    f_o_s: float,
    n_neighbors: int,
) -> np.ndarray:
    """Moving least-squares displacement interpolation.

    For each particle, fits a local affine model:
        u(x) = u0 + du/dx * (x - x0)
    using its K nearest neighbors within f_o_s, then evaluates
    the fitted displacement at the query point.

    Replaces the gbSolver==1 branch + ``funCompDefGrad3.m``.

    Parameters
    ----------
    disp : (M, D)  — displacement at matched particles
    coords : (M, D) — positions of those matched particles
    query_coords : (Nq, D) — positions at which to evaluate
    f_o_s : float — field of search radius
    n_neighbors : int — max neighbors to use per point

    Returns
    -------
    disp_at_query : (Nq, D) — interpolated displacements
    """
    ndim = coords.shape[1]
    Nq = len(query_coords)
    result = np.zeros((Nq, ndim), dtype=np.float64)

    if len(coords) < ndim + 1:
        return result

    tree = cKDTree(coords)
    K = min(n_neighbors, len(coords) - 1)
    radius = np.sqrt(ndim) * f_o_s

    # Bulk KNN query
    dd, ii = tree.query(query_coords, k=K + 1)

    for qi in range(Nq):
        # Filter to within radius
        mask = dd[qi] < radius
        idx = ii[qi][mask]
        if len(idx) < ndim + 1:
            # Fall back to nearest value
            if len(idx) > 0:
                result[qi] = disp[idx[0]]
            continue

        # Local coords relative to query point
        dx = coords[idx] - query_coords[qi]  # (k, D)
        # Build [1, dx1, dx2, (dx3)] design matrix
        A = np.column_stack([np.ones(len(idx)), dx])  # (k, D+1)

        # Solve for each displacement component
        for d in range(ndim):
            try:
                params, _, _, _ = np.linalg.lstsq(A, disp[idx, d], rcond=None)
                result[qi, d] = params[0]  # constant term = value at query
            except np.linalg.LinAlgError:
                result[qi, d] = np.mean(disp[idx, d])

    return result


# ═══════════════════════════════════════════════════════════════
#  Global Solver 2: Grid Regularization
#  Replaces the gbSolver==2 branch
# ═══════════════════════════════════════════════════════════════

def solve_regularization(
    disp: np.ndarray,
    coords: np.ndarray,
    query_coords: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
    """Scatter → regularised grid → interpolate back to particles.

    Replaces the gbSolver==2 branch in f_track_serial_match3D.m.

    Returns
    -------
    disp_at_query : (Nq, D)
    grids : grid coordinate arrays (cached for next iteration)
    disp_grid : (D, *grid_shape)
    """
    ndim = coords.shape[1]

    # Scatter to grid with smoothing
    grids, disp_grid = scatter_to_grid_multi(
        coords, disp, grid_step, smoothness, grid_coords
    )

    # Interpolate grid back to scattered query points
    disp_at_query = _interp_grid_to_points(grids, disp_grid, query_coords)

    return disp_at_query, grids, disp_grid


def _interp_grid_to_points(
    grids: Tuple[np.ndarray, ...],
    field: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Interpolate a gridded D-component field to scattered points.

    Uses LinearNDInterpolator for speed.
    """
    ndim = points.shape[1]
    grid_pts = np.column_stack([g.ravel() for g in grids])
    result = np.zeros((len(points), ndim), dtype=np.float64)
    for d in range(ndim):
        result[:, d] = _linear_or_nearest_interpolate(
            grid_pts, field[d].ravel(), points
        )
    return result


# ═══════════════════════════════════════════════════════════════
#  Global Solver 3: ADMM / Augmented Lagrangian
#  Replaces the gbSolver==3 branch  (most complex)
# ═══════════════════════════════════════════════════════════════

class ADMMLSolver:
    """Augmented Lagrangian solver for displacement regularisation.

    On the first ADMM outer iteration it:
      1. Scatter-interpolates to a regular grid.
      2. Builds a sparse gradient operator D (central finite differences).
      3. Tunes regularisation alpha via L-curve.
      4. Initialises dual variable v.

    On subsequent iterations it reuses the grid and dual variable,
    performing one ADMM update:
        u_hat = (alpha*D'D + I)^{-1} (u - v)
        v     = v + u_hat - u

    Replaces the gbSolver==3 branch in ``f_track_serial_match3D.m``
    plus ``funDerivativeOp3.m`` (~200 lines of sparse index logic).
    """

    def __init__(self):
        self.grids: Optional[Tuple[np.ndarray, ...]] = None
        self.v_dual: Optional[np.ndarray] = None
        self.alpha: float = 1.0
        self.D: Optional[csc_matrix] = None
        self._DtD: Optional[csc_matrix] = None
        self._n_grid: int = 0

    def solve(
        self,
        disp: np.ndarray,
        coords: np.ndarray,
        query_coords: np.ndarray,
        grid_step: np.ndarray,
        smoothness: float,
        is_first_iter: bool,
        roi_ranges: Optional[Tuple[Tuple[float,float], ...]] = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
        """One ADMM global-step update.

        Parameters
        ----------
        disp, coords : matched displacement and positions
        query_coords : all current B particle positions
        grid_step : grid spacing
        smoothness : regularisation weight (used in initial scatter)
        is_first_iter : True on first ADMM iteration (triggers L-curve)
        roi_ranges : optional ((xmin,xmax), (ymin,ymax), ...) for grid

        Returns
        -------
        disp_at_query, grids, disp_grid — same interface as solve_regularization
        """
        ndim = coords.shape[1]

        # 1. Scatter to grid
        grids, disp_grid = scatter_to_grid_multi(
            coords, disp, grid_step, smoothness, self.grids
        )
        self.grids = grids
        grid_shape = grids[0].shape
        n_pts = int(np.prod(grid_shape))

        # 2. Interleave components: [u0_x, u0_y, u0_z, u1_x, ...]
        u_vec = np.zeros(ndim * n_pts, dtype=np.float64)
        for d in range(ndim):
            u_vec[d::ndim] = disp_grid[d].ravel()

        # 3. Build gradient operator on first iteration (or if grid changed)
        need_rebuild = is_first_iter
        if (not is_first_iter
                and self._DtD is not None
                and self._DtD.shape[0] != ndim * n_pts):
            log.warning("ADMM grid shape changed (%d → %d) — rebuilding operators",
                        self._DtD.shape[0], ndim * n_pts)
            need_rebuild = True

        if need_rebuild:
            self.D = _build_gradient_operator(grid_shape, grid_step, ndim)
            self._DtD = self.D.T @ self.D
            self.v_dual = np.zeros_like(u_vec)
            self.alpha = self._tune_alpha(u_vec, ndim * n_pts)

        # 4. ADMM update
        A = self.alpha * self._DtD + speye(ndim * n_pts, format="csc")
        rhs = u_vec - self.v_dual
        u_hat = spsolve(A, rhs)

        # 5. Update dual
        self.v_dual = self.v_dual + u_hat - u_vec

        # 6. Reshape back to grid
        disp_grid_out = np.zeros_like(disp_grid)
        for d in range(ndim):
            disp_grid_out[d] = u_hat[d::ndim].reshape(grid_shape)

        # 7. Interpolate to query points
        disp_at_query = _interp_grid_to_points(grids, disp_grid_out, query_coords)

        return disp_at_query, grids, disp_grid_out

    def _tune_alpha(self, u_vec: np.ndarray, n: int) -> float:
        """L-curve method to find best regularisation alpha.

        Tests a log-spaced list and fits a parabola to find the elbow.
        """
        alpha_list = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        err_fid = np.zeros(len(alpha_list))
        err_smooth = np.zeros(len(alpha_list))
        I_n = speye(n, format="csc")

        for i, alpha in enumerate(alpha_list):
            A = alpha * self._DtD + I_n
            u_hat = spsolve(A, u_vec)  # v_dual is 0 on first call
            diff = u_hat - u_vec
            err_fid[i] = np.sqrt(diff @ diff)
            Du = self.D @ u_hat
            err_smooth[i] = np.sqrt(Du @ Du)

        # Normalise and sum
        ef = err_fid / (err_fid.max() + 1e-30)
        es = err_smooth / (err_smooth.max() + 1e-30)
        total = ef + es
        idx_best = int(np.argmin(total))

        # Parabolic refinement around minimum
        if 0 < idx_best < len(alpha_list) - 1:
            log_a = np.log10(alpha_list[idx_best - 1:idx_best + 2])
            y = total[idx_best - 1:idx_best + 2]
            try:
                p = np.polyfit(log_a, y, 2)
                if abs(p[0]) > 1e-15:
                    best = 10 ** (-p[1] / (2 * p[0]))
                    log.info("ADMM alpha tuned to %.4g", best)
                    return float(best)
            except Exception:
                pass

        best = float(alpha_list[idx_best])
        log.info("ADMM alpha selected: %.4g", best)
        return best


# ═══════════════════════════════════════════════════════════════
#  Sparse gradient operator (central finite differences)
#  Replaces funDerivativeOp3.m  (~200 lines → ~40 lines)
# ═══════════════════════════════════════════════════════════════

def _build_gradient_operator(
    grid_shape: Tuple[int, ...],
    grid_step: np.ndarray,
    ndim: int,
) -> csc_matrix:
    """Build a sparse central-difference gradient operator D.

    Such that F_vec = D @ u_vec, where:
      u_vec has ndim components interleaved per grid point
      F_vec has ndim² components interleaved per grid point

    For 3-D: output layout per point is
        [F11, F21, F31, F12, F22, F32, F13, F23, F33]
    matching the MATLAB convention in funDerivativeOp3.m.

    Uses np.gradient internally for the actual stencil logic,
    but builds a sparse matrix for use in the ADMM linear system.
    """
    n_pts = int(np.prod(grid_shape))
    n_u = ndim * n_pts          # size of u_vec
    n_f = ndim * ndim * n_pts   # size of F_vec

    # For each spatial derivative direction and each displacement component,
    # build one row-block of D using sparse difference stencils.
    rows, cols, vals = [], [], []

    strides = _compute_strides(grid_shape)  # linear index strides per dim

    for deriv_dim in range(ndim):           # which spatial direction (∂/∂x_d)
        h = float(grid_step[deriv_dim])
        stride = strides[deriv_dim]
        sz = grid_shape[deriv_dim]

        for comp in range(ndim):            # which displacement component
            # F row index within the ndim² block:
            #   MATLAB layout: F_{comp+1, deriv_dim+1}
            #   row offset = deriv_dim * ndim + comp
            f_offset = deriv_dim * ndim + comp

            for p in range(n_pts):
                f_row = ndim * ndim * p + f_offset

                # Multi-index of this grid point
                mi = _linear_to_multi(p, grid_shape)
                idx_along = mi[deriv_dim]

                # Central difference where possible, one-sided at borders
                if 0 < idx_along < sz - 1:
                    # central: (u[i+1] - u[i-1]) / (2h)
                    p_plus = p + stride
                    p_minus = p - stride
                    rows.append(f_row); cols.append(ndim * p_minus + comp); vals.append(-1.0 / (2 * h))
                    rows.append(f_row); cols.append(ndim * p_plus  + comp); vals.append( 1.0 / (2 * h))
                elif idx_along == 0:
                    # forward: (-u[i] + u[i+1]) / h  (first-order)
                    p_plus = p + stride
                    rows.append(f_row); cols.append(ndim * p       + comp); vals.append(-1.0 / h)
                    rows.append(f_row); cols.append(ndim * p_plus  + comp); vals.append( 1.0 / h)
                else:  # idx_along == sz - 1
                    # backward: (-u[i-1] + u[i]) / h
                    p_minus = p - stride
                    rows.append(f_row); cols.append(ndim * p_minus + comp); vals.append(-1.0 / h)
                    rows.append(f_row); cols.append(ndim * p       + comp); vals.append( 1.0 / h)

    D = csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(n_f, n_u),
    )
    return D


def _compute_strides(shape: Tuple[int, ...]) -> list:
    """Linear-index stride for each dimension (C-order)."""
    nd = len(shape)
    strides = [1] * nd
    for d in range(nd - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def _linear_to_multi(idx: int, shape: Tuple[int, ...]) -> list:
    """Convert a linear index to a multi-index (C-order)."""
    nd = len(shape)
    mi = [0] * nd
    for d in range(nd - 1, -1, -1):
        mi[d] = idx % shape[d]
        idx //= shape[d]
    return mi


# ═══════════════════════════════════════════════════════════════
#  Unified global-step dispatcher
# ═══════════════════════════════════════════════════════════════

class DisplacementRegularizer:
    """Unified interface to all three global-step solvers.

    Maintains state for the ADMM solver across iterations.

    Usage (inside the ADMM tracking loop)::

        reg = DisplacementRegularizer(solver=GlobalSolver.ADMM)
        for iter_num in range(max_iter):
            ...  # local step: get matched_disp, matched_coords
            smooth_disp = reg.solve(
                matched_disp, matched_coords, all_b_coords,
                grid_step, smoothness, f_o_s, n_neighbors,
                is_first_iter=(iter_num == 0),
            )
    """

    def __init__(self, solver: GlobalSolver = GlobalSolver.REGULARIZATION):
        self.solver_type = solver
        self._admm = ADMMLSolver() if solver == GlobalSolver.ADMM else None
        self.grids: Optional[Tuple[np.ndarray, ...]] = None
        self.disp_grid: Optional[np.ndarray] = None

    def solve(
        self,
        disp: np.ndarray,
        coords: np.ndarray,
        query_coords: np.ndarray,
        grid_step: np.ndarray,
        smoothness: float,
        f_o_s: float = 60.0,
        n_neighbors: int = 20,
        is_first_iter: bool = False,
    ) -> np.ndarray:
        """Run one global-step solve.

        Returns (Nq, D) displacement at query_coords.
        """
        if len(coords) == 0:
            return np.zeros_like(query_coords)

        gs = np.asarray(grid_step, dtype=np.float64)

        if self.solver_type == GlobalSolver.MLS:
            return solve_mls(disp, coords, query_coords, f_o_s, n_neighbors)

        elif self.solver_type == GlobalSolver.REGULARIZATION:
            result, self.grids, self.disp_grid = solve_regularization(
                disp, coords, query_coords, gs, smoothness, self.grids
            )
            return result

        elif self.solver_type == GlobalSolver.ADMM:
            result, self.grids, self.disp_grid = self._admm.solve(
                disp, coords, query_coords, gs, smoothness, is_first_iter
            )
            return result

        raise ValueError(f"Unknown solver: {self.solver_type}")

