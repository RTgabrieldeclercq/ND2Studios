"""
SerialTrack Python — Trajectory stitching & merging
=====================================================
    serialtrack/trajectories.py

Replaces the ~200-line trajectory merge/stitch section from:
    - run_Serial_MPT_3D_hardpar_inc.m  (incremental postprocessing)
    - run_Serial_MPT_3D_hardpar_accum.m (cumulative trajectory collection)

The MATLAB code:
  1. Builds trajectory segments from frame-to-frame track_A2B links
  2. Extrapolates each segment forward/backward using pchip/nearest
  3. Searches for other segments whose endpoints are near the
     extrapolated predictions
  4. Merges matching segments and fills gaps via interpolation
  5. Repeats for multiple merge passes

This Python version uses scipy.interpolate.PchipInterpolator and
vectorised numpy operations for the core logic.

Dependencies:
    pip install numpy scipy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator
import logging

from .config import TrajectoryConfig

log = logging.getLogger("serialtrack.trajectories")


# ═══════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectorySegment:
    """A single trajectory segment (possibly with NaN gaps).

    Attributes
    ----------
    coords : (n_frames, D) float64
        Particle positions per frame. NaN where not tracked.
    start_frame : int
        First non-NaN frame index (0-based).
    length : int
        Number of consecutive non-NaN frames.
    active : bool
        False if this segment has been merged into another.
    """
    coords: np.ndarray
    start_frame: int = 0
    length: int = 0
    active: bool = True

    def recompute_bounds(self):
        """Recompute start_frame and length from coords."""
        valid = ~np.isnan(self.coords[:, 0])
        indices = np.where(valid)[0]
        if len(indices) == 0:
            self.start_frame = 0
            self.length = 0
            self.active = False
        else:
            self.start_frame = int(indices[0])
            self.length = int(np.sum(valid))

    @property
    def end_frame(self) -> int:
        """Last non-NaN frame (exclusive)."""
        return self.start_frame + self.length

    def valid_coords(self) -> np.ndarray:
        """Return only non-NaN rows."""
        valid = ~np.isnan(self.coords[:, 0])
        return self.coords[valid]


# ═══════════════════════════════════════════════════════════════
#  Build trajectory segments from incremental tracking results
# ═══════════════════════════════════════════════════════════════

def build_segments_incremental(
    coords_per_frame: List[np.ndarray],
    track_a2b_per_frame: List[np.ndarray],
) -> List[TrajectorySegment]:
    """Build trajectory segments from incremental frame-to-frame links.

    Replaces the MATLAB "Compute and collect all trajectory segments"
    section in run_Serial_MPT_3D_hardpar_inc.m.

    Parameters
    ----------
    coords_per_frame : list of (Ni, D) arrays
        Detected particle coordinates per frame (0-indexed).
        coords_per_frame[0] = reference frame particles.
    track_a2b_per_frame : list of (Ni,) int arrays
        track_a2b_per_frame[i] maps frame i → frame i+1.
        Value -1 means untracked.

    Returns
    -------
    segments : list of TrajectorySegment
    """
    n_frames = len(coords_per_frame)
    ndim = coords_per_frame[0].shape[1]
    segments: List[TrajectorySegment] = []

    # For each starting frame, trace forward through the links
    for start_f in range(n_frames):
        n_particles = len(coords_per_frame[start_f])

        for p_idx in range(n_particles):
            coords = np.full((n_frames, ndim), np.nan)
            coords[start_f] = coords_per_frame[start_f][p_idx]

            current_idx = p_idx
            for f in range(start_f, n_frames - 1):
                track = track_a2b_per_frame[f]
                if current_idx < len(track) and track[current_idx] >= 0:
                    next_idx = track[current_idx]
                    coords[f + 1] = coords_per_frame[f + 1][next_idx]
                    current_idx = next_idx
                else:
                    break  # chain broken

            # Only keep if we have at least 2 valid frames
            valid_count = int(np.sum(~np.isnan(coords[:, 0])))
            if valid_count >= 2:
                seg = TrajectorySegment(coords=coords)
                seg.recompute_bounds()
                segments.append(seg)

    # Deduplicate: segments sharing the same (frame, position) are redundant
    segments = _deduplicate_segments(segments)

    log.info("Built %d trajectory segments from %d frames",
             len(segments), n_frames)
    return segments


def build_segments_cumulative(
    coords_ref: np.ndarray,
    coords_per_frame: List[np.ndarray],
    track_a2b_per_frame: List[np.ndarray],
) -> List[TrajectorySegment]:
    """Build trajectory segments from cumulative tracking results.

    In cumulative mode, every track_a2b maps reference (frame 0) → frame i.
    This is simpler — each reference particle has one trajectory.

    Parameters
    ----------
    coords_ref : (Na, D) — reference frame particles
    coords_per_frame : list of (Ni, D) — detected per deformed frame
    track_a2b_per_frame : list of (Na,) int — ref→frame_i links

    Returns
    -------
    segments : list of TrajectorySegment
    """
    n_frames = len(track_a2b_per_frame) + 1  # +1 for reference
    ndim = coords_ref.shape[1]
    Na = len(coords_ref)
    segments: List[TrajectorySegment] = []

    for p_idx in range(Na):
        coords = np.full((n_frames, ndim), np.nan)
        coords[0] = coords_ref[p_idx]

        for fi, track in enumerate(track_a2b_per_frame):
            if track[p_idx] >= 0:
                coords[fi + 1] = coords_per_frame[fi][track[p_idx]]

        valid_count = int(np.sum(~np.isnan(coords[:, 0])))
        if valid_count >= 1:
            seg = TrajectorySegment(coords=coords)
            seg.recompute_bounds()
            segments.append(seg)

    log.info("Built %d cumulative trajectories", len(segments))
    return segments


# ═══════════════════════════════════════════════════════════════
#  Trajectory segment merging
# ═══════════════════════════════════════════════════════════════

def merge_segments(
    segments: List[TrajectorySegment],
    config: TrajectoryConfig,
) -> List[TrajectorySegment]:
    """Merge trajectory segments by extrapolation and proximity matching.

    Replaces the ~150-line "Merge trajectory segments" section from
    run_Serial_MPT_3D_hardpar_inc.m.

    Algorithm (per merge pass):
      For each gap size g in [0, max_gap_length]:
        For segment lengths from longest to min_segment_length:
          For each segment S of that length:
            1. Extrapolate S forward/backward using pchip/nearest.
            2. Find shorter segments whose start/end is near the
               extrapolated prediction (within dist_threshold).
            3. Merge the best candidate into S, fill the gap.

    Parameters
    ----------
    segments : list of TrajectorySegment
    config : TrajectoryConfig

    Returns
    -------
    merged : list of TrajectorySegment (only active ones)
    """
    if not segments:
        return segments

    n_frames = segments[0].coords.shape[0]
    ndim = segments[0].coords.shape[1]

    for merge_pass in range(config.merge_passes):
        n_merged_this_pass = 0

        for gap in range(config.max_gap_length + 1):

            # Process from longest to shortest segments
            max_len = n_frames - 1
            for seg_len in range(max_len, config.min_segment_length - 1, -1):

                # Collect segments of this length
                for i, seg_i in enumerate(segments):
                    if not seg_i.active or seg_i.length != seg_len:
                        continue

                    # Try to extend in the forward direction
                    target_frame = seg_i.end_frame + gap
                    if 0 <= target_frame < n_frames:
                        pred_pos = _extrapolate_position(
                            seg_i, target_frame, config.extrap_method
                        )
                        if pred_pos is not None:
                            best_j = _find_best_candidate(
                                segments, i, target_frame,
                                pred_pos, config.dist_threshold,
                                direction="forward",
                            )
                            if best_j >= 0:
                                _merge_into(segments[i], segments[best_j],
                                            config.extrap_method)
                                n_merged_this_pass += 1

                    # Try to extend in the backward direction
                    target_frame = seg_i.start_frame - 1 - gap
                    if 0 <= target_frame < n_frames:
                        pred_pos = _extrapolate_position(
                            seg_i, target_frame, config.extrap_method
                        )
                        if pred_pos is not None:
                            best_j = _find_best_candidate(
                                segments, i, target_frame,
                                pred_pos, config.dist_threshold,
                                direction="backward",
                            )
                            if best_j >= 0:
                                _merge_into(segments[i], segments[best_j],
                                            config.extrap_method)
                                n_merged_this_pass += 1

        log.debug("Merge pass %d: merged %d segments",
                  merge_pass + 1, n_merged_this_pass)
        if n_merged_this_pass == 0:
            break  # No more merges possible

    active = [s for s in segments if s.active and s.length >= 1]
    log.info("After merging: %d active trajectories", len(active))
    return active


# ═══════════════════════════════════════════════════════════════
#  Convert to trajectory matrix
# ═══════════════════════════════════════════════════════════════

def segments_to_matrix(
    segments: List[TrajectorySegment],
) -> np.ndarray:
    """Convert segment list to (N_traj, n_frames, D) array.

    NaN entries indicate frames where the particle was not tracked.
    """
    if not segments:
        return np.empty((0, 0, 0))
    n_frames = segments[0].coords.shape[0]
    ndim = segments[0].coords.shape[1]
    active = [s for s in segments if s.active]
    mat = np.full((len(active), n_frames, ndim), np.nan)
    for i, seg in enumerate(active):
        mat[i] = seg.coords
    return mat


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════

def _extrapolate_position(
    seg: TrajectorySegment,
    target_frame: int,
    method: str,
) -> Optional[np.ndarray]:
    """Extrapolate a segment's trajectory to a target frame.

    Uses scipy PchipInterpolator for 'pchip' or nearest-neighbor
    for 'nearest' (suitable for Brownian motion).

    Returns None if extrapolation is not possible.
    """
    valid_mask = ~np.isnan(seg.coords[:, 0])
    valid_frames = np.where(valid_mask)[0]
    if len(valid_frames) < 2:
        # Can't extrapolate with fewer than 2 points
        if len(valid_frames) == 1:
            return seg.coords[valid_frames[0]].copy()
        return None

    ndim = seg.coords.shape[1]
    result = np.zeros(ndim)
    t = valid_frames.astype(np.float64)

    for d in range(ndim):
        values = seg.coords[valid_frames, d]
        if method == "pchip" and len(valid_frames) >= 2:
            try:
                interp = PchipInterpolator(t, values, extrapolate=True)
                result[d] = interp(float(target_frame))
            except Exception:
                result[d] = values[-1] if target_frame > t[-1] else values[0]
        else:
            # Nearest: use the closest endpoint
            if target_frame >= t[-1]:
                result[d] = values[-1]
            else:
                result[d] = values[0]

    return result


def _find_best_candidate(
    segments: List[TrajectorySegment],
    exclude_idx: int,
    target_frame: int,
    pred_pos: np.ndarray,
    dist_threshold: float,
    direction: str,
) -> int:
    """Find the best segment to merge at the target frame.

    For 'forward' direction: looks for segments starting at target_frame.
    For 'backward' direction: looks for segments ending at target_frame+1.

    Returns index into segments list, or -1 if none found.
    """
    best_idx = -1
    best_dist = np.inf

    for j, seg_j in enumerate(segments):
        if j == exclude_idx or not seg_j.active or seg_j.length == 0:
            continue

        if direction == "forward":
            # Candidate must start at or near target_frame
            if seg_j.start_frame != target_frame:
                continue
            cand_pos = seg_j.coords[target_frame]
        else:
            # Candidate must end at or near target_frame + 1
            if seg_j.end_frame != target_frame + 1:
                continue
            cand_pos = seg_j.coords[target_frame]

        if np.any(np.isnan(cand_pos)):
            continue

        dist = np.linalg.norm(pred_pos - cand_pos)
        if dist < dist_threshold and dist < best_dist:
            best_dist = dist
            best_idx = j

    return best_idx


def _merge_into(
    target: TrajectorySegment,
    source: TrajectorySegment,
    fill_method: str,
):
    """Merge source segment into target, then fill NaN gaps.

    Copies all non-NaN entries from source into target,
    deactivates source, then interpolates any internal gaps.
    """
    # Copy non-NaN entries from source
    valid_src = ~np.isnan(source.coords[:, 0])
    target.coords[valid_src] = source.coords[valid_src]

    # Deactivate source
    source.active = False
    source.coords[:] = np.nan
    source.length = 0

    # Fill internal gaps in the merged trajectory
    _fill_gaps(target, fill_method)

    # Recompute bounds
    target.recompute_bounds()


def _fill_gaps(seg: TrajectorySegment, method: str):
    """Interpolate internal NaN gaps in a trajectory segment.

    Replaces MATLAB's fillmissing(). Only fills gaps *between*
    the first and last valid frame (no extrapolation beyond endpoints).
    """
    valid_mask = ~np.isnan(seg.coords[:, 0])
    valid_frames = np.where(valid_mask)[0]
    if len(valid_frames) < 2:
        return

    first, last = valid_frames[0], valid_frames[-1]
    interior = np.arange(first, last + 1)
    ndim = seg.coords.shape[1]

    t = valid_frames.astype(np.float64)
    for d in range(ndim):
        values = seg.coords[valid_frames, d]
        if method == "pchip" and len(valid_frames) >= 2:
            try:
                interp = PchipInterpolator(t, values)
                seg.coords[interior, d] = interp(interior.astype(np.float64))
            except Exception:
                # Fallback: linear
                seg.coords[interior, d] = np.interp(
                    interior, valid_frames, values
                )
        else:
            seg.coords[interior, d] = np.interp(
                interior, valid_frames, values
            )


def _deduplicate_segments(
    segments: List[TrajectorySegment],
) -> List[TrajectorySegment]:
    """Remove duplicate segments that share identical trajectories.

    Two segments are duplicates if they have identical non-NaN entries
    at the same frames. Keep the longer one.
    """
    if len(segments) <= 1:
        return segments

    # Sort by length descending so we keep longer ones
    segments.sort(key=lambda s: s.length, reverse=True)

    # Use a set of (frame, rounded_coords) tuples as fingerprints
    seen_fingerprints = set()
    unique = []

    for seg in segments:
        valid = ~np.isnan(seg.coords[:, 0])
        frames = np.where(valid)[0]
        if len(frames) == 0:
            continue
        # Fingerprint: (start_frame, end_frame, first_coord, last_coord)
        fp = (
            int(frames[0]), int(frames[-1]),
            tuple(np.round(seg.coords[frames[0]], 4)),
            tuple(np.round(seg.coords[frames[-1]], 4)),
        )
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            unique.append(seg)

    return unique