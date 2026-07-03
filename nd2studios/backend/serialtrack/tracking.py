"""
SerialTrack Python — Main tracking engine (v3)
================================================
    serialtrack/tracking.py

Fixes from v2
-------------
- Fixed ADMM convergence order: MATLAB checks convergence BEFORE warping;
  Python was warping BEFORE checking.  Now matches MATLAB: on convergence,
  the last global-step displacement is NOT applied.
- Fixed ``_local_step`` retry loop: the retry was a no-op because
  ``min(n_neighbors, n_max_local)`` never changes when n_max_local starts
  at n_neighbors_max.  Now correctly increases the neighbor count on retry,
  matching the MATLAB ``n_neighborsMax = round(n_neighborsMax + 5)`` logic.
- Fixed f_o_s update: now passes the current f_o_s as context so the floor
  can adapt, instead of hardcoding 60.0.
- Double-frame mode (TrackingMode.DOUBLE_FRAME) implemented.
- Trajectory building integrated into TrackingSession.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple, Callable
import time
import numpy as np
import logging

from .config import (
    DetectionConfig, TrackingConfig, TrackingMode, GlobalSolver
)
from .detection import ParticleDetector
from .matching import TopologyMatcher, NearestNeighborMatcher, compute_displacement
from .outliers import remove_outliers, find_not_missing, update_f_o_s
from .regularization import DisplacementRegularizer
from .fields import (
    DisplacementField, StrainField,
    compute_strain_mls, compute_gridded_strain,
)
from .prediction import InitialGuessPredictor
from .trajectories import (
    build_segments_incremental, build_segments_cumulative,
    merge_segments, segments_to_matrix, TrajectorySegment,
)

log = logging.getLogger("serialtrack.tracking")


# ═══════════════════════════════════════════════════════════════
#  Result containers
# ═══════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """Tracking output for a single frame pair."""
    frame_idx: int
    coords_b: np.ndarray
    disp_b2a: np.ndarray
    track_a2b: np.ndarray
    track_b2a: np.ndarray
    match_ratio: float
    n_iterations: int
    wall_time: float
    # Post-processing (deformed config)
    disp_field: Optional[DisplacementField] = None
    strain_field: Optional[StrainField] = None
    # Post-processing (reference config)
    disp_field_ref: Optional[DisplacementField] = None
    strain_field_ref: Optional[StrainField] = None


@dataclass
class TrackingSession:
    """Full tracking session results across all frames."""
    detection_config: DetectionConfig
    tracking_config: TrackingConfig
    coords_ref: np.ndarray
    frame_results: List[FrameResult] = dc_field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frame_results) + 1

    @property
    def tracking_ratios(self) -> np.ndarray:
        return np.array([r.match_ratio for r in self.frame_results])

    def get_trajectories(self) -> np.ndarray:
        """Build (N_ref, n_frames, D) trajectory matrix via naive chaining.

        For full trajectory stitching (with gap filling), use
        ``build_stitched_trajectories()`` instead.
        """
        ndim = self.coords_ref.shape[1]
        Na = len(self.coords_ref)
        nf = self.n_frames
        traj = np.full((Na, nf, ndim), np.nan)
        traj[:, 0, :] = self.coords_ref

        for fi, res in enumerate(self.frame_results, 1):
            tracked = res.track_a2b >= 0
            idx_a = np.where(tracked)[0]
            idx_b = res.track_a2b[idx_a]
            traj[idx_a, fi, :] = res.coords_b[idx_b]

        return traj

    def build_stitched_trajectories(self) -> np.ndarray:
        """Build trajectories with segment merging and gap filling.

        Uses the trajectory stitching algorithm from the MATLAB code.
        Returns (N_traj, n_frames, D) array with NaN for missing frames.
        """
        cfg = self.tracking_config

        if cfg.mode == TrackingMode.CUMULATIVE:
            coords_list = [res.coords_b for res in self.frame_results]
            track_list = [res.track_a2b for res in self.frame_results]
            segments = build_segments_cumulative(
                self.coords_ref, coords_list, track_list
            )
        else:
            # Incremental or double-frame
            coords_list = [self.coords_ref] + [
                res.coords_b for res in self.frame_results
            ]
            track_list = [res.track_a2b for res in self.frame_results]
            segments = build_segments_incremental(coords_list, track_list)

        # Merge segments
        merged = merge_segments(segments, cfg.trajectory)

        return segments_to_matrix(merged)


# ═══════════════════════════════════════════════════════════════
#  Single-frame ADMM tracker
# ═══════════════════════════════════════════════════════════════

class _ADMMFrameTracker:
    """ADMM iteration loop for a single frame pair.

    Follows the MATLAB ``f_track_serial_match3D.m`` structure exactly:
      1. Local step: topology or nearest-neighbor matching
      2. Global step: regularised displacement interpolation
      3. Convergence check (BEFORE warping — matching MATLAB)
      4. Warp B coordinates and update f_o_s
      5. Cull missing particles when n_neighbors < 4
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.regularizer = DisplacementRegularizer(solver=cfg.solver)

    def run(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
        init_disp: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Execute ADMM tracking loop.

        Returns: disp_b2a, track_a2b, track_b2a, match_ratio, n_iters
        """
        cfg = self.cfg
        ndim = coords_a.shape[1]
        Na, Nb = len(coords_a), len(coords_b)

        coords_b_curr = coords_b.copy()
        disp_b2a = np.zeros((Nb, ndim))
        not_missing_a = np.arange(Na)
        not_missing_b = np.arange(Nb)
        match_ratio_eq1_count = 0
        match_ratio = 0.0

        grid_step = np.full(ndim, min(round(0.5 * cfg.f_o_s), 20), dtype=np.float64)

        track_a2b = np.full(Na, -1, dtype=np.int64)
        track_b2a = np.full(Nb, -1, dtype=np.int64)

        # Working copy of f_o_s that updates per iteration
        working_f_o_s = float(cfg.f_o_s)

        if init_disp is not None and init_disp.shape == disp_b2a.shape:
            disp_b2a += init_disp
            coords_b_curr = coords_b + init_disp

        for iter_num in range(cfg.max_iter):
            n_neighbors = round(
                cfg.n_neighbors_min
                + np.exp(-0.5 * iter_num) * (cfg.n_neighbors_max - cfg.n_neighbors_min)
            )
            log.info("  Iter %d | n_neighbors=%d | f_o_s=%.1f",
                     iter_num + 1, n_neighbors, working_f_o_s)

            # ── LOCAL STEP ──
            matches, local_track, local_disp = self._local_step(
                coords_a, coords_b_curr,
                not_missing_a, not_missing_b,
                n_neighbors, working_f_o_s, cfg.outlier_threshold,
            )

            if len(matches) == 0:
                log.warning("  No matches at iter %d", iter_num + 1)
                break

            track_a2b = local_track
            match_ratio = np.sum(track_a2b >= 0) / max(len(not_missing_a), 1)
            log.info("  Tracking ratio: %d/%d = %.4f",
                     np.sum(track_a2b >= 0), len(not_missing_a), match_ratio)

            # ── GLOBAL STEP ──
            tracked_mask = track_a2b >= 0
            idx_a = np.where(tracked_mask)[0]
            idx_b = track_a2b[idx_a]
            matched_disp_b2a = -(coords_b_curr[idx_b] - coords_a[idx_a])

            temp_disp = self.regularizer.solve(
                disp=matched_disp_b2a,
                coords=coords_b_curr[idx_b],
                query_coords=coords_b_curr,
                grid_step=grid_step,
                smoothness=cfg.smoothness,
                f_o_s=working_f_o_s,
                n_neighbors=n_neighbors,
                is_first_iter=(iter_num == 0),
            )

            # ── Convergence check (BEFORE warping — matches MATLAB) ──
            update_norm = np.sqrt(np.sum(temp_disp**2) / max(len(temp_disp), 1))
            log.info("  Disp update norm: %.6f", update_norm)

            if match_ratio > 0.999:
                match_ratio_eq1_count += 1

            threshold = np.sqrt(ndim) * cfg.iter_stop_threshold
            if update_norm < threshold or match_ratio_eq1_count > 5:
                log.info("  Converged at iter %d", iter_num + 1)
                break

            # ── Warp B (only if NOT converged — matches MATLAB) ──
            disp_b2a += temp_disp
            coords_b_curr = coords_b + disp_b2a

            # ── Update f_o_s for next iteration ──
            if len(temp_disp) > 0:
                working_f_o_s = update_f_o_s(temp_disp, working_f_o_s)

            # ── Cull missing particles (late iterations) ──
            if n_neighbors < 4:
                not_missing_a, not_missing_b = find_not_missing(
                    coords_a, coords_b_curr, cfg.dist_missing
                )

        # Build track_b2a
        track_b2a = np.full(Nb, -1, dtype=np.int64)
        for ia in range(Na):
            ib = track_a2b[ia]
            if ib >= 0:
                track_b2a[ib] = ia

        return disp_b2a, track_a2b, track_b2a, match_ratio, iter_num + 1

    def _local_step(self, coords_a, coords_b_curr, not_missing_a, not_missing_b,
                    n_neighbors, f_o_s, outlier_threshold):
        """Local matching step with retry logic.

        On failure, increases the neighbor count (matching MATLAB:
        ``n_neighborsMax = round(n_neighborsMax + 5)``), which allows
        the topology matcher to use more neighbors for a richer
        feature vector.

        FIX: The v2 code used ``min(n_neighbors, n_max_local)`` which
        was a no-op.  Now correctly increases n_neighbors on retry.
        """
        ca = coords_a[not_missing_a]
        cb = coords_b_curr[not_missing_b]
        retry_n_neighbors = n_neighbors

        matches_raw = np.empty((0, 2), dtype=np.int64)

        max_attempts = 5
        for attempt in range(max_attempts):
            if retry_n_neighbors > 2:
                matcher = TopologyMatcher(n_neighbors=retry_n_neighbors, f_o_s=f_o_s)
            else:
                matcher = NearestNeighborMatcher(f_o_s=f_o_s)

            matches_raw = matcher.match(ca, cb)

            if len(matches_raw) == 0:
                # Increase neighbor count for richer topology features
                retry_n_neighbors = min(retry_n_neighbors + 5, len(ca) - 1)
                if retry_n_neighbors < 3:
                    break  # Can't do topology matching with so few particles
                log.debug("  Local step retry: n_neighbors → %d", retry_n_neighbors)
            else:
                break

        if len(matches_raw) == 0:
            Na = len(coords_a)
            ndim = coords_a.shape[1]
            return (
                np.empty((0, 2), dtype=np.int64),
                np.full(Na, -1, dtype=np.int64),
                np.empty((0, ndim)),
            )

        # Map local indices back to full arrays
        matches_full = np.column_stack([
            not_missing_a[matches_raw[:, 0]],
            not_missing_b[matches_raw[:, 1]],
        ])
        track_a2b, disp_a2b = compute_displacement(
            coords_a, coords_b_curr, matches_full, outlier_threshold
        )
        return matches_full, track_a2b, disp_a2b


# ═══════════════════════════════════════════════════════════════
#  Main public tracker
# ═══════════════════════════════════════════════════════════════

class SerialTracker:
    """Top-level SerialTrack particle tracking engine.

    Supports INCREMENTAL, CUMULATIVE, and DOUBLE_FRAME modes.
    """

    def __init__(self, detection_config: DetectionConfig, tracking_config: TrackingConfig):
        self.det_cfg = detection_config
        self.trk_cfg = tracking_config
        self.detector = ParticleDetector(detection_config)
        self.predictor = InitialGuessPredictor()

    def track_images(
        self,
        images: List[np.ndarray],
        progress_cb: Optional[Callable] = None,
    ) -> TrackingSession:
        """Track particles across a sequence of images."""
        if len(images) < 2:
            raise ValueError("Need at least 2 images")

        cfg = self.trk_cfg
        cfg.init_roi_from_image(images[0])

        coords_ref = self.detector.detect(images[0], cfg.roi_slices())
        coords_ref = ParticleDetector.clip_to_bounds(coords_ref, images[0].shape)
        log.info("Detected %d particles in reference image", len(coords_ref))

        all_coords = [coords_ref]
        for i in range(1, len(images)):
            c = self.detector.detect(images[i], cfg.roi_slices())
            c = ParticleDetector.clip_to_bounds(c, images[i].shape)
            all_coords.append(c)
            log.info("Detected %d particles in frame %d", len(c), i + 1)

        return self._run_tracking(all_coords, progress_cb)

    def track_coordinates(
        self,
        coords_list: List[np.ndarray],
        progress_cb: Optional[Callable] = None,
    ) -> TrackingSession:
        """Track from pre-detected particle coordinates."""
        if len(coords_list) < 2:
            raise ValueError("Need at least 2 coordinate sets")
        return self._run_tracking(coords_list, progress_cb)

    def _run_tracking(
        self,
        all_coords: List[np.ndarray],
        progress_cb: Optional[Callable],
    ) -> TrackingSession:
        cfg = self.trk_cfg
        coords_ref = all_coords[0]
        n_frames = len(all_coords)

        session = TrackingSession(
            detection_config=self.det_cfg,
            tracking_config=cfg,
            coords_ref=coords_ref,
        )

        prev_coords: List[np.ndarray] = []
        prev_disp: List[np.ndarray] = []

        for fi in range(1, n_frames):
            log.info("====== Frame %d / %d ======", fi + 1, n_frames)
            t0 = time.perf_counter()

            coords_b = all_coords[fi]

            # Decide reference for this frame based on mode
            if cfg.mode == TrackingMode.CUMULATIVE:
                coords_a = coords_ref
            elif cfg.mode == TrackingMode.DOUBLE_FRAME:
                coords_a = all_coords[fi - 1]
            else:  # INCREMENTAL
                if fi == 1:
                    coords_a = coords_ref
                else:
                    coords_a = prev_coords[-1] if prev_coords else coords_ref

            # Initial guess (not used for double-frame mode)
            init_disp = None
            if (cfg.mode != TrackingMode.DOUBLE_FRAME
                    and cfg.use_prev_results
                    and fi >= 2 and len(prev_disp) > 0):
                init_disp = self.predictor.predict(
                    frame_idx=fi + 1,
                    current_coords=coords_b,
                    prev_coords_list=prev_coords,
                    prev_disp_list=prev_disp,
                )

            # Run ADMM tracker
            admm = _ADMMFrameTracker(cfg)
            disp_b2a, track_a2b, track_b2a, ratio, n_iters = admm.run(
                coords_a, coords_b, init_disp
            )

            wall = time.perf_counter() - t0
            log.info("  Frame %d done: ratio=%.3f, iters=%d, time=%.2fs",
                     fi + 1, ratio, n_iters, wall)

            result = FrameResult(
                frame_idx=fi + 1,
                coords_b=coords_b,
                disp_b2a=disp_b2a,
                track_a2b=track_a2b,
                track_b2a=track_b2a,
                match_ratio=ratio,
                n_iterations=n_iters,
                wall_time=wall,
            )

            # Strain in both configs
            if cfg.strain_n_neighbors > 0:
                self._compute_strain(result, coords_a, cfg)

            session.frame_results.append(result)

            prev_coords.append(coords_b)
            prev_disp.append(disp_b2a)

            if progress_cb is not None:
                progress_cb(fi + 1, n_frames, result)

        return session

    def _compute_strain(self, result: FrameResult, coords_a: np.ndarray,
                        cfg: TrackingConfig):
        """Compute displacement & strain in BOTH deformed and reference configs."""
        disp_a2b = -result.disp_b2a
        coords_b = result.coords_b

        tracked = result.track_b2a >= 0
        if np.sum(tracked) < cfg.strain_n_neighbors:
            return

        ndim = coords_b.shape[1]
        grid_step = np.full(ndim, min(round(0.5 * cfg.strain_f_o_s), 20),
                            dtype=np.float64)

        # 1. Strain in deformed configuration (on B particles)
        try:
            dfield, sfield = compute_gridded_strain(
                coords=coords_b[tracked],
                disp=disp_a2b[tracked],
                grid_step=grid_step,
                smoothness=1e-3,
                pixel_steps=cfg.steps,
            )
            result.disp_field = dfield
            result.strain_field = sfield
        except Exception as e:
            log.warning("Strain (deformed config) failed: %s", e)

        # 2. Strain in reference configuration (on A particles = B - disp)
        try:
            coords_ref_config = coords_b[tracked] - disp_a2b[tracked]
            dfield_ref, sfield_ref = compute_gridded_strain(
                coords=coords_ref_config,
                disp=disp_a2b[tracked],
                grid_step=grid_step,
                smoothness=1e-3,
                pixel_steps=cfg.steps,
            )
            result.disp_field_ref = dfield_ref
            result.strain_field_ref = sfield_ref
        except Exception as e:
            log.warning("Strain (reference config) failed: %s", e)


# ═══════════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════════

def track(
    images: List[np.ndarray],
    detection_config: Optional[DetectionConfig] = None,
    tracking_config: Optional[TrackingConfig] = None,
    progress_cb: Optional[Callable] = None,
) -> TrackingSession:
    """One-call convenience function for tracking."""
    det_cfg = detection_config or DetectionConfig()
    trk_cfg = tracking_config or TrackingConfig()
    tracker = SerialTracker(det_cfg, trk_cfg)
    return tracker.track_images(images, progress_cb)
