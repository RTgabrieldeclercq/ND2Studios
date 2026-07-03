"""
SerialTrack Python — Chunk 4a
==============================
    serialtrack/prediction.py

Initial guess predictor for ADMM warm-starting.
Replaces: funInitGuess3.m  (~80 lines)
         funPOR_GPR.m      (~70 lines)

Uses sklearn PCA + GaussianProcessRegressor instead of custom POD + fitrgp.

Dependencies:
    pip install numpy scipy scikit-learn
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import logging

log = logging.getLogger("serialtrack.prediction")


def _interp_prev_to_current(
    prev_coords: np.ndarray,
    prev_disp: np.ndarray,
    current_coords: np.ndarray,
) -> np.ndarray:
    """Interpolate a previous displacement field to current particle positions.

    Replaces the repeated scatteredInterpolant calls in funInitGuess3.m.
    Returns (N_current, D) displacement array.
    """
    ndim = prev_coords.shape[1]
    result = np.zeros((len(current_coords), ndim), dtype=np.float64)

    for d in range(ndim):
        interp = LinearNDInterpolator(
            prev_coords, prev_disp[:, d], fill_value=0.0
        )
        result[:, d] = interp(current_coords)

    return result


class InitialGuessPredictor:
    """Predict initial displacement for the next frame.

    Implements three strategies matching the MATLAB funInitGuess3.m logic:

    1. **Frame 3** (1 history frame): Linear extrapolation → 2× previous.
    2. **Frames 4–6** (2 history frames): Linear extrapolation from 2 prior.
    3. **Frame 7+** (5+ history frames): POD-GPR prediction using sklearn.

    The frame numbering convention: ``frame_idx`` is 1-based to match MATLAB,
    where frame 1 is the reference and frame 2 is the first deformed image.

    Usage
    -----
    >>> predictor = InitialGuessPredictor()
    >>> init_disp = predictor.predict(
    ...     frame_idx=5,
    ...     current_coords=coords_b,
    ...     prev_coords_list=prev_coords,   # list of (N,D) arrays
    ...     prev_disp_list=prev_disp,        # list of (N,D) arrays
    ... )
    """

    def __init__(self, n_pod_modes: int = 3, n_history: int = 5):
        """
        Parameters
        ----------
        n_pod_modes : int
            Number of POD basis vectors for the GPR predictor.
        n_history : int
            Number of past frames to use for POD-GPR (default 5).
        """
        self.n_pod_modes = n_pod_modes
        self.n_history = n_history

    def predict(
        self,
        frame_idx: int,
        current_coords: np.ndarray,
        prev_coords_list: List[np.ndarray],
        prev_disp_list: List[np.ndarray],
    ) -> np.ndarray:
        """Compute initial displacement guess for ``current_coords``.

        Parameters
        ----------
        frame_idx : int
            1-based frame index (≥ 3 required for prediction).
        current_coords : (N, D) float64
            Particle positions in the current deformed frame.
        prev_coords_list : list of (Ni, D) arrays
            Particle positions for previous frames.
            Index ``i`` corresponds to frame ``i+2`` (0-based list).
        prev_disp_list : list of (Ni, D) arrays
            Tracked displacements for previous frames (same indexing).

        Returns
        -------
        init_disp : (N, D) float64
            Predicted displacement at ``current_coords``.
        """
        ndim = current_coords.shape[1]
        N = len(current_coords)

        # Not enough history
        n_avail = len(prev_disp_list)
        if n_avail == 0 or frame_idx < 3:
            return np.zeros((N, ndim), dtype=np.float64)

        if frame_idx == 3 and n_avail >= 1:
            return self._extrapolate_linear_1(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )

        if frame_idx <= 6 and n_avail >= 2:
            return self._extrapolate_linear_2(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )

        if frame_idx > 6 and n_avail >= self.n_history:
            return self._predict_pod_gpr(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )

        # Fallback: simple linear extrapolation from most recent
        if n_avail >= 2:
            return self._extrapolate_linear_2(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )
        return self._extrapolate_linear_1(
            current_coords, prev_coords_list, prev_disp_list, frame_idx
        )

    # ── Strategy 1: single-frame extrapolation (frame 3) ──────

    def _extrapolate_linear_1(self, current, prev_c, prev_d, fi):
        """u_init = 2 * u_{n-1} interpolated to current positions."""
        idx = fi - 3  # index into 0-based list
        idx = min(idx, len(prev_d) - 1)
        u_prev = _interp_prev_to_current(prev_c[idx], prev_d[idx], current)
        return 2.0 * u_prev

    # ── Strategy 2: two-frame linear extrapolation (frames 4-6) ──

    def _extrapolate_linear_2(self, current, prev_c, prev_d, fi):
        """u_init = 2*u_{n-1} - u_{n-2}, each interpolated to current."""
        i2 = min(fi - 3, len(prev_d) - 1)  # most recent
        i3 = min(fi - 4, len(prev_d) - 1)  # one before
        if i3 < 0:
            return self._extrapolate_linear_1(current, prev_c, prev_d, fi)

        u2 = _interp_prev_to_current(prev_c[i2], prev_d[i2], current)
        u3 = _interp_prev_to_current(prev_c[i3], prev_d[i3], current)
        return 2.0 * u2 - u3

    # ── Strategy 3: POD-GPR (frame 7+) ───────────────────────

    def _predict_pod_gpr(self, current, prev_c, prev_d, fi):
        """POD + Gaussian Process Regression prediction.

        Replaces funPOR_GPR.m + the ImgSeqNum > 6 branch.

        1. Interpolate the last ``n_history`` displacement fields to
           the current particle positions → snapshot matrix.
        2. Apply POD (PCA) to extract dominant modes.
        3. Fit a GP to each mode's temporal coefficient.
        4. Predict next time step and reconstruct.
        """
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        ndim = current.shape[1]
        N = len(current)
        nT = self.n_history
        nB = min(self.n_pod_modes, nT - 1)

        # Determine which previous frames to use
        # MATLAB: ImgSeqNum+[-5:1:-1] mapped to 0-based list indices
        start = max(0, len(prev_d) - nT)
        end_ = len(prev_d)
        indices = list(range(start, end_))
        nT_actual = len(indices)
        if nT_actual < 2:
            return self._extrapolate_linear_2(current, prev_c, prev_d, fi)

        # Build snapshot matrices: (nT, N) per displacement component
        snapshots = [np.zeros((nT_actual, N)) for _ in range(ndim)]
        t_train = np.arange(nT_actual, dtype=np.float64).reshape(-1, 1)

        for ti, idx in enumerate(indices):
            u_interp = _interp_prev_to_current(prev_c[idx], prev_d[idx], current)
            for d in range(ndim):
                snapshots[d][ti, :] = u_interp[:, d]

        # Predict next time for each component
        t_predict = np.array([[float(nT_actual)]]) 
        result = np.zeros((N, ndim), dtype=np.float64)

        for d in range(ndim):
            result[:, d] = self._pod_gpr_1d(
                snapshots[d], t_train, t_predict, nB
            )

        log.debug("POD-GPR prediction for frame %d using %d snapshots, %d modes",
                   fi, nT_actual, nB)
        return result

    @staticmethod
    def _pod_gpr_1d(
        T_snap: np.ndarray,
        t_train: np.ndarray,
        t_predict: np.ndarray,
        n_modes: int,
    ) -> np.ndarray:
        """POD-GPR for a single displacement component.

        Parameters
        ----------
        T_snap : (nT, N) — snapshot matrix
        t_train : (nT, 1) — training times
        t_predict : (1, 1) — prediction time
        n_modes : int — number of POD modes

        Returns
        -------
        u_pred : (N,) — predicted field at t_predict
        """
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, WhiteKernel, ConstantKernel
        )

        nT, N = T_snap.shape
        n_modes = min(n_modes, nT - 1, N)
        if n_modes < 1:
            return T_snap[-1]  # fallback: return last snapshot

        # --- POD via sklearn PCA ---
        # PCA centres the data (subtracts mean) automatically
        pca = PCA(n_components=n_modes)
        # a_train: (nT, n_modes) — temporal coefficients
        a_train = pca.fit_transform(T_snap)

        # --- GP regression on each mode's temporal coefficient ---
        a_pred = np.zeros((1, n_modes))

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=1e-4, noise_level_bounds=(1e-8, 1e0)
        )

        for k in range(n_modes):
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                alpha=1e-6,
            )
            gpr.fit(t_train, a_train[:, k])
            a_pred[0, k] = gpr.predict(t_predict)[0]

        # --- Reconstruct ---
        u_pred = pca.inverse_transform(a_pred)  # (1, N)
        return u_pred[0]